"""Loopback-only OpenAI-compatible server sharing one NF4 Qwen base.

Process, Final and Base requests are serialized while their adapter is active.
The --smoke-adapters option creates unsaved, untrained adapters for engineering
tests only. This server does not implement token-level GRPO capture.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RequestError(ValueError):
    """Safe, fixed-message client error; never include prompt content."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--process-adapter", type=Path)
    parser.add_argument("--final-adapter", type=Path)
    parser.add_argument("--smoke-adapters", action="store_true")
    parser.add_argument("--port", type=int, default=30112)
    parser.add_argument("--context-tokens", type=int, default=10240)
    args = parser.parse_args()
    if args.smoke_adapters == bool(args.process_adapter or args.final_adapter):
        parser.error("choose real Process adapter or engineering smoke adapters")
    if not args.smoke_adapters and not args.process_adapter:
        parser.error("--process-adapter is required; Final uses disabled adapters")
    if not 1 <= args.port <= 65535 or not 1024 <= args.context_tokens <= 40960:
        parser.error("invalid port or context limit")
    if not args.base_model.is_dir() or not (args.base_model / "model.safetensors.index.json").is_file():
        parser.error("complete local Qwen base directory required")
    if not args.smoke_adapters:
        for path in filter(None, (args.process_adapter, args.final_adapter)):
            if not path.is_dir() or not (path / "adapter_config.json").is_file() or not any(
                (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
            ):
                parser.error("adapter directory is incomplete")
    return args


def main():
    args = parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if args.smoke_adapters:
        config = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, config, adapter_name="process")
        model.add_adapter("final", config)
        # Keep Process a no-op; perturb Final only in memory for switch checks.
        for name, module in reversed(list(model.named_modules())):
            if name.endswith("o_proj") and hasattr(module, "lora_B"):
                with torch.no_grad():
                    module.lora_B["final"].weight[:128, 0] = 0.5
                break
        else:
            raise RuntimeError("synthetic Final adapter did not attach")
    else:
        model = PeftModel.from_pretrained(base, args.process_adapter, adapter_name="process",
                                          is_trainable=False, local_files_only=True)
        if args.final_adapter:
            model.load_adapter(args.final_adapter, adapter_name="final", is_trainable=False,
                               local_files_only=True)
    model.eval()
    model.requires_grad_(False)

    routes = {
        "process-policy": "process",
        "frozen-final": "final" if args.final_adapter or args.smoke_adapters else None,
        "frozen-base-selector": None,
        "qwen3-8b-base": None,
    }
    lock = threading.Lock()
    max_body_bytes = 2 * 1024 * 1024

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return  # Prompts and medical data must not enter an HTTP access log.

        def reply(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode("utf8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.reply({"error": "not_found"}, 404)
                return
            self.reply({
                "status": "ready", "pid": os.getpid(), "single_base": True,
                "precision": "NF4_double_quant_BF16_compute",
                "adapters": "synthetic_untrained" if args.smoke_adapters else "provided_frozen",
                "context_tokens": args.context_tokens, "models": list(routes),
                "role_adapters": routes, "grpo_capture_ready": False,
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            })

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply({"error": "not_found"}, 404)
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise RequestError("invalid_request_size") from exc
                if not 0 < length <= max_body_bytes:
                    raise RequestError("invalid_request_size")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise RequestError("invalid_request_object")
                name = request.get("model")
                if name not in routes:
                    raise RequestError("unknown_model_role")
                messages = request.get("messages")
                if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(m.get("content"), str) for m in messages
                ):
                    raise RequestError("invalid_messages")
                max_tokens = request.get("max_tokens")
                temperature = request.get("temperature", 0)
                top_p = request.get("top_p", 1)
                seed = request.get("seed", 0)
                if type(max_tokens) is not int or not 1 <= max_tokens <= 2400:
                    raise RequestError("invalid_max_tokens")
                if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
                    raise RequestError("invalid_temperature")
                if type(top_p) not in (int, float) or not 0 < top_p <= 1:
                    raise RequestError("invalid_top_p")
                if type(seed) is not int or not 0 <= seed < 2**63:
                    raise RequestError("invalid_seed")
                response_format = request.get("response_format")
                if response_format not in (None, {"type": "json_object"}):
                    raise RequestError("unsupported_response_format")
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt",
                ).to("cuda:0")
                prompt_tokens = int(ids.shape[1])
                if prompt_tokens + max_tokens > args.context_tokens:
                    raise RequestError("context_budget_exceeded")
                adapter = routes[name]
                with lock, torch.inference_mode():
                    set_seed(seed)
                    kwargs = dict(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=max_tokens, use_cache=True,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p if temperature > 0 else None,
                        top_k=0 if temperature > 0 else None,
                        pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.0, num_beams=1,
                    )
                    if adapter is None:
                        with model.disable_adapter():
                            output = model.generate(**kwargs)
                    else:
                        model.set_adapter(adapter)
                        model.requires_grad_(False)
                        output = model.generate(**kwargs)
                generated = output[0, prompt_tokens:]
                content = tokenizer.decode(generated, skip_special_tokens=True)
                self.reply({
                    "model": name, "adapter": adapter, "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": int(generated.numel())},
                    "json_grammar_enforced": False,
                })
            except RequestError as exc:
                self.reply({"error": str(exc)}, 400)
            except json.JSONDecodeError:
                self.reply({"error": "invalid_json"}, 400)
            except Exception as exc:
                # Do not expose paths, model output or private prompt contents.
                print("GENERATION_FAILURE", type(exc).__name__, flush=True)
                self.reply({"error": type(exc).__name__}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(json.dumps({"status": "ready", "port": args.port, "pid": os.getpid(),
                      "synthetic_adapters": args.smoke_adapters, "single_base": True}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
