"""One boundary for host paths and native/WSL retrieval worker launch."""

import hashlib
import json
import os
from pathlib import Path
import subprocess


def retrieval_timeouts(retrieval):
    """Outer transport must outlive the worker deadline and receipt cleanup."""
    import math
    values = retrieval.get("worker", {}).get("timeouts", {})
    if not isinstance(values, dict) or set(values) - {"startup", "operation", "bridge"}:
        raise ValueError("invalid_retrieval_timeout_config")
    result = dict(startup=300.0, operation=360.0, bridge=390.0)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 3600:
            raise ValueError("invalid_retrieval_timeout_config")
        result[key] = float(value)
    if result["bridge"] < result["operation"] + 10:
        raise ValueError("retrieval_deadlines_need_outer_margin")
    return result


def verified_snapshot(retrieval, repo_root):
    """Resolve the bundled snapshot on the Agent host and verify exact bytes."""
    repo_root = Path(repo_root).resolve()
    value = retrieval.get("snapshot", "retrieval_snapshot_v2")
    if not isinstance(value, str) or not value:
        raise ValueError("invalid_retrieval_snapshot")
    snapshot = Path(value)
    if not snapshot.is_absolute():
        snapshot = repo_root / snapshot
    snapshot = snapshot.resolve()
    if not snapshot.is_relative_to(repo_root) or not snapshot.is_dir():
        raise ValueError("snapshot_must_be_in_repository")
    manifest = json.loads((snapshot / "MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or "worker.py" not in files:
        raise ValueError("invalid_retrieval_manifest")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("unsafe_retrieval_manifest_path")
        path = snapshot / relative
        if path.is_symlink() or not path.resolve().is_relative_to(snapshot) or not path.is_file():
            raise ValueError("unsafe_retrieval_snapshot")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("retrieval_snapshot_changed")
    for name, expected in manifest.get("shared_context_files", {}).items():
        path = repo_root / name
        if (not path.resolve().is_relative_to(repo_root / "src/trainable_agent/context_processing")
                or path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("shared_context_source_changed")
    for name, expected in manifest.get("shared_runtime_files", {}).items():
        if name not in {"src/trainable_agent/retrieval_deadlines.py"}:
            raise ValueError("unsafe_shared_runtime_path")
        path = repo_root / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(repo_root / "src/trainable_agent")
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("shared_runtime_source_changed")
    return snapshot, manifest


def _wsl_path(path, distro, runner=subprocess.run):
    if os.name != "nt":
        raise ValueError("wsl_worker_requires_windows_host")
    path = Path(path).resolve()
    if not path.drive or str(path).startswith("\\\\"):
        raise ValueError("wsl_shared_path_must_be_windows_drive")
    # WSL interprets backslashes in unquoted Windows arguments as escapes.
    # Forward slashes preserve the drive path across this boundary.
    result = runner(["wsl.exe", "--distribution", distro, "--", "wslpath", "-a", "-u", path.as_posix()],
                    check=True, capture_output=True, text=True, timeout=20)
    translated = result.stdout.strip()
    if not translated.startswith("/") or "\n" in translated:
        raise ValueError("invalid_wsl_path_translation")
    return translated


def worker_command(retrieval, snapshot, output, *, runner=subprocess.run):
    """Build a command without shell interpolation or caller-supplied output path."""
    worker = retrieval.get("worker")
    if not isinstance(worker, dict):
        raise ValueError("retrieval_worker_config_required")
    retrieval_timeouts(retrieval)
    kind = worker.get("kind")
    python = worker.get("python")
    models = worker.get("models")
    if kind not in ("native", "wsl") or not isinstance(python, str) or not python or not isinstance(models, dict):
        raise ValueError("invalid_retrieval_worker_config")
    model_names = ("tokenizer", "embedding", "reranker", "paper_reranker")
    if any(not isinstance(models.get(name), str) or not models[name] for name in model_names):
        raise ValueError("missing_retrieval_model_path")
    secrets = worker.get("secrets")
    if secrets is not None and (not isinstance(secrets, str) or not secrets):
        raise ValueError("invalid_retrieval_secrets_path")
    if kind == "wsl":
        distro = worker.get("distribution")
        if not isinstance(distro, str) or not distro or distro.startswith("-"):
            raise ValueError("wsl_distribution_required")
        for value in (python, *models.values(), secrets):
            if value is not None and (not isinstance(value, str) or not value.startswith("/")):
                raise ValueError("wsl_worker_paths_must_be_linux_absolute")
        prefix = ["wsl.exe", "--distribution", distro, "--", python, "-B"]
        worker_path = _wsl_path(snapshot / "worker.py", distro, runner)
        output_path = _wsl_path(output, distro, runner)
    else:
        if worker.get("distribution") is not None:
            raise ValueError("native_worker_has_distribution")
        if not Path(python).is_file():
            raise ValueError("native_retrieval_python_missing")
        prefix = [python, "-B"]
        worker_path = str(snapshot / "worker.py")
        output_path = str(Path(output).resolve())
        for value in models.values():
            if not Path(value).is_dir():
                raise ValueError("native_retrieval_model_missing")
        if secrets is not None and not Path(secrets).is_file():
            raise ValueError("native_retrieval_secrets_missing")
    args = ["--output", output_path,
            "--tokenizer", models["tokenizer"],
            "--embedding-model", models["embedding"],
            "--reranker-model", models["reranker"],
            "--paper-reranker-model", models["paper_reranker"]]
    if secrets is not None:
        args += ["--secrets", secrets]
    return prefix + [worker_path] + args


def prepare_worker(retrieval, repo_root, output, *, runner=subprocess.run):
    snapshot, manifest = verified_snapshot(retrieval, repo_root)
    return worker_command(retrieval, snapshot, output, runner=runner), manifest
