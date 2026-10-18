"""Validate host/worker paths and worker credential presence without live calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trainable_agent.runtime_paths import prepare_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not Path(config["tokenizer"]).is_dir():
        raise ValueError("agent_host_tokenizer_missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command, manifest = prepare_worker(config["retrieval"], ROOT, output / "retrieval")
    result = subprocess.run(command + ["--check-config"], capture_output=True, text=True, timeout=120)
    if result.returncode:
        # Worker diagnostics remain in process output; avoid echoing possible credentials.
        raise RuntimeError("retrieval_worker_preflight_failed")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("retrieval_worker_protocol_invalid")
    report = json.loads(lines[0])
    if report.get("status") != "config_checked":
        raise ValueError("retrieval_worker_preflight_invalid")
    required = ("NCBI_API_KEY", "S2_API_KEY", "SERPER_API_KEY", "MINERU_API_TOKEN")
    missing = [name for name in required if not report.get("presence", {}).get(name)]
    print(json.dumps({"preflight": "passed" if not missing else "missing_credentials",
                      "source_files": len(manifest["files"]), "worker_kind": config["retrieval"]["worker"]["kind"],
                      "missing_credential_names": missing, "model_loaded": False,
                      "retrieval_api_calls": 0, "training": False}))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
