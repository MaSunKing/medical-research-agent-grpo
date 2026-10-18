"""Offline source identity check; no imports of retrieval dependencies."""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
snapshot = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT / "retrieval_snapshot_v2"
manifest = json.loads((snapshot / "MANIFEST.json").read_text(encoding="utf8"))
for name, expected in manifest["files"].items():
    path = snapshot / name
    if path.is_symlink() or not path.resolve().is_relative_to(snapshot.resolve()):
        raise ValueError("unsafe_source")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("source_identity_mismatch:" + name)
    if name.endswith('.py'):
        ast.parse(data.decode("utf-8-sig"), filename=name)
for name, expected in manifest.get("shared_context_files", {}).items():
    path = ROOT / name
    if (not path.resolve().is_relative_to(ROOT / "src/trainable_agent/context_processing")
            or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise ValueError("shared_context_source_changed:" + name)
    ast.parse(path.read_text(encoding="utf8"), filename=name)
for name, expected in manifest.get("shared_runtime_files", {}).items():
    if name not in {"src/trainable_agent/retrieval_deadlines.py"}:
        raise ValueError("unsafe_shared_runtime_path")
    path = ROOT / name
    if (path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(ROOT / "src/trainable_agent")
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise ValueError("shared_runtime_source_changed:" + name)
    ast.parse(path.read_text(encoding="utf8"), filename=name)
common = snapshot / "backend/common.py"
namespace = {}
exec(compile(common.read_text(encoding="utf8"), str(common), "exec"), namespace)
expected = {"registered_callable", "run_callable", "_call_search_unchecked", "call_search", "call_browse"}
if not expected <= namespace.keys():
    raise ValueError("missing_execution_functions")
if any(n in namespace for n in ("policy_prompt", "ChatClient", "score_evidence")):
    raise ValueError("legacy_policy_leaked_into_tool_boundary")
print("SOURCE_IDENTITY_AND_EXECUTION_IMPORT=passed; files=" + str(len(manifest["files"])) + "; API=0; MODEL=0")
