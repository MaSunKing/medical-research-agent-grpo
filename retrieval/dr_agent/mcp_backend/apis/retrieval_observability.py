# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Content-addressed, policy-invisible retrieval observability artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "medgap_retrieval_observability_v1"
FAILURE_SCHEMA_VERSION = "medgap_retrieval_failure_v1"
ENV_STORE_DIR = "MEDGAP_RETRIEVAL_OBSERVABILITY_DIR"
REDACTED = "[REDACTED]"

_SENSITIVE_KEY = re.compile(
    r"^(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer|password|passwd|secret|token)$",
    re.IGNORECASE,
)
_QUERY_SECRET = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|token)=)"
    r"([^&#\s]+)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|token)"
    r"(\s*[:=]\s*)(?!\[REDACTED\])([^\s,;}\]]+)",
    re.IGNORECASE,
)
_BEARER_SECRET = re.compile(r"\b(Bearer\s+)(?!\[REDACTED\])([^\s,;]+)", re.IGNORECASE)


def _redact_text(value: str) -> str:
    """Remove common credential forms while preserving useful failure context."""
    redacted = _QUERY_SECRET.sub(lambda match: match.group(1) + REDACTED, value)
    redacted = _ASSIGNMENT_SECRET.sub(
        lambda match: match.group(1) + match.group(2) + REDACTED,
        redacted,
    )
    return _BEARER_SECRET.sub(lambda match: match.group(1) + REDACTED, redacted)


def redact_secrets(value: Any) -> Any:
    """Recursively redact credentials before diagnostics cross the disk boundary.

    The operation is deliberately schema-agnostic because exception text and
    third-party metadata may place a secret inside a nested string rather than
    under a predictable field name.
    """
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _SENSITIVE_KEY.fullmatch(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8", errors="replace"))


def _canonical_document_content(document: dict[str, Any]) -> dict[str, Any]:
    sections = []
    for section in document.get("sections") or []:
        if not isinstance(section, dict):
            continue
        sections.append(
            {
                "heading": str(section.get("heading") or ""),
                "text": str(section.get("text") or ""),
            }
        )
    return {
        "title": str(document.get("title") or ""),
        "text": str(document.get("text") or ""),
        "sections": sections,
    }


def _atomic_write(path: Path, payload: bytes, *, gzip_payload: bool) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            if gzip_payload:
                with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
                    compressed.write(payload)
            else:
                handle.write(payload)
        try:
            os.replace(temporary, path)
        except OSError:
            if not path.exists():
                raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def persist_retrieval_observation(
    *,
    document: dict[str, Any],
    focused_query: str,
    original_question: str | None = None,
    tool_name: str,
    retrieval: dict[str, Any],
    document_metadata: dict[str, Any] | None = None,
    fetch_method: str | None = None,
    store_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Persist one canonical document and one deterministic retrieval manifest.

    The returned pointer is backend-only diagnostics.  Callers must not insert
    it into the model-visible policy observation.
    """
    configured = str(store_dir or os.getenv(ENV_STORE_DIR) or "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    content = _canonical_document_content(document)
    content_bytes = canonical_json_bytes(content)
    raw_document_sha256 = sha256_bytes(content_bytes)
    document_relpath = Path("documents") / raw_document_sha256[:2] / (
        raw_document_sha256 + ".json.gz"
    )
    _atomic_write(root / document_relpath, content_bytes, gzip_payload=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool_name": str(tool_name),
        "source_id": str(retrieval.get("source_id") or ""),
        "focused_query": str(focused_query),
        "focused_query_sha256": sha256_text(str(focused_query)),
        # The current MCP Browse schema does not expose the original question.
        # Keep this nullable until a policy-invisible, rollout-safe side channel exists.
        "original_question_sha256": (
            sha256_text(str(original_question)) if original_question else None
        ),
        "raw_document_sha256": raw_document_sha256,
        "raw_document_path": document_relpath.as_posix(),
        "document_metadata": redact_secrets(
            dict(document_metadata or document.get("metadata") or {})
        ),
        "fetch_method": fetch_method,
        "chunker": retrieval.get("chunker"),
        "reranker": retrieval.get("reranker"),
        "budgets": retrieval.get("budgets"),
        "all_chunks": retrieval.get("all_chunks") or [],
        "returned_chunks": retrieval.get("returned_chunks") or [],
        "handoff_audit": retrieval.get("handoff_audit") or [],
        "browse_preprocessing": retrieval.get("browse_preprocessing") or {},
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    manifest_relpath = Path("manifests") / manifest_sha256[:2] / (manifest_sha256 + ".json")
    _atomic_write(root / manifest_relpath, manifest_bytes, gzip_payload=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "manifest_path": manifest_relpath.as_posix(),
        "raw_document_sha256": raw_document_sha256,
        "raw_document_path": document_relpath.as_posix(),
    }


def persist_retrieval_failure(
    *,
    source_id: str,
    focused_query: str,
    tool_name: str,
    error_code: str,
    error_type: str | None = None,
    fetch_attempts: list[dict[str, Any]] | None = None,
    document_metadata: dict[str, Any] | None = None,
    store_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Persist a policy-invisible failed retrieval funnel record.

    Failure artifacts deliberately contain no raw exception traceback and no
    credentials. They make pre-document failures auditable without pretending
    a canonical document or chunk ranking existed.
    """
    configured = str(store_dir or os.getenv(ENV_STORE_DIR) or "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    manifest = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "tool_name": str(tool_name),
        "source_id": str(source_id),
        "focused_query_sha256": sha256_text(str(focused_query)),
        "error_code": str(error_code),
        "error_type": str(error_type or ""),
        "fetch_attempts": redact_secrets(list(fetch_attempts or [])),
        "document_metadata": redact_secrets(dict(document_metadata or {})),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    relative = Path("failures") / manifest_sha256[:2] / (manifest_sha256 + ".json")
    _atomic_write(root / relative, manifest_bytes, gzip_payload=False)
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "manifest_path": relative.as_posix(),
    }


def load_canonical_document(root: Path, relative_path: str) -> dict[str, Any]:
    with gzip.open(root / relative_path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))
