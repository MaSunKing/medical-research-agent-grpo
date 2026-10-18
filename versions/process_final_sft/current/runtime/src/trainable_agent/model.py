import os
import json, uuid, time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from .storage import digest, DefiniteCallFailure


class EndpointModel:
    """Explicit endpoint/adapter; no hidden Agent, retry, memory or tool loop."""
    def __init__(self, base_url, model, *, temperature=1.0, seed=0, key_env=None, capture_dir=None):
        self.capture_dir = Path(capture_dir) if capture_dir else None
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.seed, self.key_env = model, temperature, seed, key_env
        self.identity = digest(dict(base_url=base_url, model=model, temperature=temperature, seed=seed,
                                    thinking=False, top_p=1.0))

    async def generate(self, request):
        headers = {}
        if self.key_env:
            headers["Authorization"] = "Bearer " + os.environ[self.key_env]
        payload = dict(model=self.model, messages=request["messages"], max_tokens=request["max_new_tokens"],
                       temperature=self.temperature, seed=self.seed, top_p=1.0,
                       extra_body=None)
        del payload["extra_body"]
        if request["stage"] != "final":
            payload["response_format"] = {"type": "json_object"}
        # Qwen3-specific thinking control, explicit instead of a framework default.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        folder = None
        if self.capture_dir:
            folder = self.capture_dir / (request["stage"] + "-" + uuid.uuid4().hex)
            folder.mkdir(parents=True, exist_ok=False)
            # Authorization headers and credentials are never written.
            (folder / "REQUEST.json").write_text(json.dumps({"request": request, "payload": payload}, ensure_ascii=False), encoding="utf8")
        started = time.monotonic()
        metadata = {"provider": "endpoint", "stage": request["stage"], "request_sha256": digest(request), "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=False, trust_env=False) as client:
                response = await client.post(self.url, json=payload, headers=headers)
                metadata["http_status"] = response.status_code
                if folder:
                    (folder / "RESPONSE.txt").write_text(response.text, encoding="utf8")
                if not response.is_success:
                    raise DefiniteCallFailure("model_http_status:" + str(response.status_code))
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise ValueError("model_missing_text")
        except Exception as exc:
            metadata["exception_type"] = type(exc).__name__
            metadata["error_code"] = getattr(exc, "error_code", "endpoint_transport_or_response_failure")
            raise
        finally:
            if folder:
                metadata["elapsed_seconds"] = time.monotonic() - started
                metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                (folder / "CAPTURE.json").write_text(json.dumps(metadata), encoding="utf8")
        return dict(text=text, model=self.model, sampling={"temperature": self.temperature,
                    "seed": self.seed, "top_p": 1.0, "thinking": False}, usage=result.get("usage"),
                    # Retokenization is not behavior logprob capture. Fail GRPO admission explicitly.
                    token_ids=None, behavior_logprobs=None, grpo_capture_ready=False)
