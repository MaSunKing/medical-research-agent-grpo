"""Restore process-local source mappings from verified local Search receipts.

This module does not search, fetch documents, or accept model-provided source
records. Worker callers pass actual Search results or verified completed receipts.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

_HASH = re.compile(r"[a-f0-9]{64}\Z")
_WEB = re.compile(r"WEB:[a-f0-9]{16}\Z")
_PAPER = re.compile(r"(?:PMID:[0-9]+|S2:[a-f0-9]{40})\Z")


def _digest(value):
    # Match OperationJournal.receipt_digest exactly.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def restore_search_result(value, *, register_web, register_paper):
    """Register already-returned Search data without changing its source IDs."""
    if not isinstance(value, dict):
        raise ValueError("source_registry_invalid_search_result")
    if value.get("failed") or value.get("error"):
        return {"web": 0, "paper": 0}
    rows = value.get("data", [])
    if not isinstance(rows, list):
        raise ValueError("source_registry_invalid_search_rows")
    web, paper = [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source_registry_invalid_search_row")
        sid = row.get("source_id")
        if isinstance(sid, str) and _WEB.fullmatch(sid):
            web.append(copy.deepcopy(row))
        elif isinstance(sid, str) and _PAPER.fullmatch(sid):
            paper.append(copy.deepcopy(row))
        else:
            raise ValueError("source_registry_invalid_source_id")
    if web:
        register_web(web)
    if paper:
        register_paper(paper)
    return {"web": len(web), "paper": len(paper)}


def restore_web_search_rows(rows, *, lookup, register):
    """Keep fresh Search provenance when re-registering its public projection."""
    merged = []
    for row in rows:
        try:
            previous = lookup(row['source_id'])
        except ValueError:
            previous = {}
        if not isinstance(previous, dict):
            raise ValueError('source_registry_invalid_existing_web_metadata')
        merged.append({**copy.deepcopy(previous), **copy.deepcopy(row)})
    # The original registration function still validates every ID/hash and URL.
    register(merged)


def _regular_json(path, root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("source_registry_unsafe_receipt_path")
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, UnicodeError, ValueError):
        raise ValueError("source_registry_invalid_receipt_json") from None
    if not isinstance(value, dict):
        raise ValueError("source_registry_invalid_receipt_json")
    return value


def restore_completed_searches(output, restore):
    """Restore only complete, hash-verified Search journals under this output.

    Incomplete operations are left unresolved; this function never re-executes
    them. All journals are checked before any mapping callback is invoked.
    """
    root = Path(output).resolve()
    folder = root / "worker_operations"
    if folder.is_symlink():
        raise ValueError("source_registry_unsafe_receipt_path")
    if not folder.exists():
        return 0
    if not folder.is_dir() or not folder.resolve().is_relative_to(root):
        raise ValueError("source_registry_unsafe_receipt_path")
    completed = []
    for operation in sorted(folder.iterdir()):
        if operation.is_symlink() or not operation.is_dir() or not _HASH.fullmatch(operation.name):
            raise ValueError("source_registry_invalid_operation_path")
        state_path = operation / "state.json"
        if state_path.is_symlink():
            raise ValueError("source_registry_unsafe_receipt_path")
        if not state_path.exists():
            # A process may have stopped between directory creation and the
            # atomic write of its first state. No completion is inferred.
            continue
        state = _regular_json(state_path, folder)
        identity = state.get("identity")
        if (state.get("operation_id") != operation.name or not isinstance(identity, dict)
                or not isinstance(identity.get("arguments_sha256"), str)
                or not _HASH.fullmatch(identity["arguments_sha256"])):
            raise ValueError("source_registry_journal_identity_mismatch")
        if state.get("state") != "complete" or identity.get("operation") != "search":
            continue
        expected = state.get("result_sha256")
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise ValueError("source_registry_journal_result_hash_mismatch")
        value = _regular_json(operation / "result.json", folder)
        if _digest(value) != expected:
            raise ValueError("source_registry_journal_result_hash_mismatch")
        # Restore the actual completion order, not the random operation-hash
        # order. WEB registration intentionally refreshes discovery metadata.
        completed.append((state_path.stat().st_mtime_ns, operation.name, value))
    for _, _, value in sorted(completed, key=lambda item: item[:2]):
        restore(value)
    return len(completed)
