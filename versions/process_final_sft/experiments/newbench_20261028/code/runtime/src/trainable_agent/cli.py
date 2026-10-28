import argparse
import asyncio
import json
import os
from pathlib import Path

# Do not send medical trajectories to optional telemetry services.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from .storage import Store, digest
from .context import ChatCodec, ContextBuilder
from .engine import Engine, initial_state
from .model import EndpointModel
from .codex_model import CodexCLIModel
from .tools import WorkerBridge, create_server, MCPTools
from .prompts import PROMPTS
from .runtime_paths import prepare_worker


async def run(args):
    from fastmcp import Client
    from transformers import AutoTokenizer
    config = json.loads(args.config.read_text(encoding="utf8"))
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    codec = ChatCodec(tokenizer)
    context = ContextBuilder(codec, config.get("context_tokens", 10240),
                             evidence_cards=config.get("evidence_cards", True),
                             safety_margin=config.get("context_safety_margin", 128),
                             fallback_rounds=config.get("context_fallback_rounds", 3),
                             compression_max_calls=config.get('compression_max_calls',12),
                             final_context_tokens=config.get('final_context_tokens'),
                             state_context_tokens=config.get('state_context_tokens'),
                             repair_reserve_tokens=config.get('repair_reserve_tokens',384),
                             soft_context_tokens=config.get('soft_context_tokens'),
                             hard_context_tokens=config.get('hard_context_tokens'))
    # Config contains env-var NAMES, never credential values.
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    def provider(settings, role):
        settings = dict(settings)
        kind = settings.pop("kind", "endpoint")
        if kind == "codex_cli":
            return CodexCLIModel(capture_dir=work / "codex_captures" / role, **settings)
        if kind != "endpoint":
            raise ValueError("unknown_model_provider")
        return EndpointModel(capture_dir=work / "endpoint_captures" / role, **settings)
    process = provider(config["process"], "process")
    final = None if args.process_only else provider(config["final"], "final")
    compressor = provider(config["compression"], "compression") if config.get("compression") else None
    command, manifest = prepare_worker(config["retrieval"], Path(__file__).resolve().parents[2],
                                       work / "retrieval")
    from .runtime_paths import retrieval_timeouts
    deadlines = retrieval_timeouts(config["retrieval"])
    bridge = WorkerBridge(command, work / "retrieval.stderr.log", timeout=deadlines["bridge"],
                          startup_timeout=deadlines["startup"], operation_timeout=deadlines["operation"])
    store = Store(work / "trajectory.sqlite")
    try:
        async with Client(create_server(bridge)) as client:
            tools = MCPTools(client)
            import hashlib
            code = {p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.rglob("*.py")}
            tokenizer_assets = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(config["tokenizer"]).iterdir()
                                if p.is_file() and p.suffix in (".json", ".txt")}
            identity = digest({"config": config, "prompts": PROMPTS, "code": code,
                               "tokenizer_assets": tokenizer_assets, "schema": "trainable_agent_v1"})
            state = initial_state(args.run_id, args.question, identity + ":process", identity + ":final",
                                  digest(manifest), process_only=args.process_only,
                                  max_charged=config.get("tool_budget", {}).get("normal_calls", 6),
                                  max_exemptions=config.get("tool_budget", {}).get("failure_exemptions", 5),
                                  max_invalid_decisions=config.get("decision_budget", {}).get("max_rejected", 6))
            engine = Engine(store, process, final, tools, context, tools.rank, compressor=compressor,
                            compression_context_tokens=config.get("compression_context_tokens", 10240))
            result = await engine.run(state)
        for name, value in (("TRAJECTORY.json", result), ("PRE_FINAL_PACKAGE.json", result["prefinal"]),
                            ("EVENTS.json", store.events(args.run_id))):
            path = work / name
            if path.exists():
                import shutil, uuid
                archive = work / "export_history" / uuid.uuid4().hex
                archive.mkdir(parents=True)
                shutil.copy2(path, archive / name)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
            tmp.replace(path)
        print(json.dumps({"termination": result["termination"], "evidence": len(result["evidence"]),
                          "final_generated": result["final"] is not None, "output": str(work),
                          "completion_mode": result.get("completion_mode", "normal"),
                          "run_status": result.get("run_status"), "final_status": result.get("final_status")}))
        return 2 if result.get("run_status") == "failed" else 0
    finally:
        await bridge.close()
        store.db.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--process-only", action="store_true")
    raise SystemExit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
