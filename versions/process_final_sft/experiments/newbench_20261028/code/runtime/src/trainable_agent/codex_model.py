"""Text-only Codex CLI provider for engineering pilots, not Qwen sampling."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
from .storage import digest, DefiniteCallFailure, CallFailure

TRANSPORT = (
    "Generate only the requested stage response from the supplied messages. "
    "Do not use tools, browse, read files, consult other sessions, or execute actions. "
    "Search/Browse JSON is an action proposal; the external runtime executes it. "
    "Use only evidence visible in these messages for support assessments. "
    "Return the requested JSON object without markdown. "
    "The runtime's max_new_tokens is a target budget, not an enforced Codex CLI limit."
)

RECEIPT_POLICY = "codex_single_turn_terminal_v2"

class CodexCLIModel:
    def __init__(self, executable, model, capture_dir, *, reasoning_effort="medium", timeout=600):
        self.executable = str(Path(executable).resolve())
        if not Path(self.executable).is_file():
            raise ValueError("codex_executable_missing")
        self.model, self.reasoning_effort, self.timeout = model, reasoning_effort, timeout
        self.capture_dir = Path(capture_dir)
        self.identity = digest(dict(provider="codex_cli", model=model, reasoning_effort=reasoning_effort,
                                    transport=TRANSPORT, receipt_policy=RECEIPT_POLICY, temperature=None, seed=None))

    async def generate(self, request):
        return await asyncio.to_thread(self._generate, request)

    def _generate(self, request):
        folder = self.capture_dir / (request["stage"] + "-" + uuid.uuid4().hex)
        folder.mkdir(parents=True, exist_ok=False)
        prompt = TRANSPORT + "\n\nSUPPLIED_MESSAGES_JSON:\n" + json.dumps(request["messages"], ensure_ascii=False)
        (folder / "REQUEST.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "STDIN.txt").write_text(prompt, encoding="utf-8")
        started = datetime.now(timezone.utc).isoformat()
        clock = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="medical-codex-stage-") as workspace:
            argv = [self.executable, "exec", "--ignore-user-config", "--ephemeral",
                    "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--color", "never",
                    "--model", self.model, "-c", "model_reasoning_effort=" + json.dumps(self.reasoning_effort),
                    "-c", 'web_search="disabled"', "-c", "developer_instructions=" + json.dumps(TRANSPORT)]
            for feature in ("shell_tool", "unified_exec", "apps", "plugins", "hooks", "browser_use",
                            "browser_use_external", "computer_use", "multi_agent", "image_generation",
                            "imagegenext", "artifact", "goals", "workspace_dependencies", "memories"):
                argv += ["--disable", feature]
            argv += ["--cd", workspace, "--output-last-message", str((folder / "OUTPUT.txt").resolve()), "-"]
            metadata = dict(started_at=started, argv=argv, provider="codex_cli", model_requested=self.model,
                            sampling={"temperature": None, "seed": None},
                            target_max_new_tokens=request["max_new_tokens"], max_new_tokens_enforced=False,
                            request_sha256=digest(request), receipt_policy=RECEIPT_POLICY)
            error = None
            with (folder / "EVENTS.jsonl").open("wb") as stdout, (folder / "STDERR.log").open("wb") as stderr:
                try:
                    result = subprocess.run(argv, input=prompt.encode("utf-8"), stdout=stdout,
                                            stderr=stderr, timeout=self.timeout, check=False)
                    metadata["exit_code"] = result.returncode
                    if result.returncode:
                        error = "codex_cli_nonzero_exit"
                except subprocess.TimeoutExpired:
                    metadata["exit_code"] = None
                    error = "codex_cli_timeout"
            metadata.update(elapsed_seconds=time.monotonic()-clock, finished_at=datetime.now(timezone.utc).isoformat())
            events = []
            for line in (folder / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise ValueError("event_not_object")
                    events.append(parsed)
                except (json.JSONDecodeError, ValueError):
                    error = error or "codex_cli_invalid_event_json"
            usage, completed = None, False
            completion_indices, error_indices = [], []
            terminal_failed = False
            for event_index, event in enumerate(events):
                if event.get("type") == "turn.completed":
                    usage, completed = event.get("usage"), True
                    completion_indices.append(event_index)
                if event.get("type") == "turn.failed":
                    terminal_failed = True
                    error = error or "codex_cli_turn_failed"
                elif event.get("type") == "error":
                    error_indices.append(event_index)
                item = event.get("item", {})
                if item.get("type") in ("command_execution", "mcp_tool_call", "web_search", "file_change",
                                        "image_generation", "collab_tool_call"):
                    error = "codex_cli_unexpected_tool_use"
            if len(completion_indices) > 1:
                error = error or "codex_cli_ambiguous_completed_turns"
            if error_indices and (not completed or max(error_indices) > completion_indices[-1]):
                error = error or "codex_cli_turn_failed"
            if not completed:
                error = error or "codex_cli_missing_completed_turn"
            output = folder / "OUTPUT.txt"
            text = output.read_text(encoding="utf-8") if output.exists() else ""
            if not text.strip():
                error = error or "codex_cli_empty_output"
            metadata.update(error_code=error, usage=usage,
                            completed_turn_count=len(completion_indices), terminal_failed=terminal_failed,
                            intermediate_error_indices=error_indices,
                            recovered_transport=bool(error_indices and not error))
            (folder / "CAPTURE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if error:
                if error == "codex_cli_timeout":
                    raise CallFailure(error)  # Outcome uncertain; never silently replay.
                raise DefiniteCallFailure(error)
        return dict(text=text, model=self.model, provider="codex_cli",
                    sampling={"temperature": None, "seed": None, "top_p": None}, usage=usage,
                    token_ids=None, behavior_logprobs=None, grpo_capture_ready=False, capture_path=str(folder),
                    context_accounting="local_Qwen_tokenizer_not_Codex_token_ids", max_new_tokens_enforced=False)
