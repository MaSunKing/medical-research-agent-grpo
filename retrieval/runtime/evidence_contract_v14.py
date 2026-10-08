"""Lossless Browse-to-Runtime evidence projection.

This module only transports retrieval facts.  It does not infer table shape from
prose and it does not assign semantic support labels.
"""
from __future__ import annotations

import copy


STRUCTURE_FIELDS = (
    "structure_kind",
    "boundary_incomplete",
    "table_integrity_verified",
)

PROVENANCE_FIELDS = (
    "source_id",
    "title",
    "text",
    "url",
    "heading",
    "section_index",
    "chunk_index",
    "start_char",
    "end_char",
    "content_span_start_char",
    "content_span_end_char",
    "text_sha256",
    "returned_text_sha256",
    "window_identity_sha256",
    "boundary_policy",
    "structure_audit",
)


def validate_structure_chunk(chunk: dict) -> dict:
    if not isinstance(chunk, dict):
        raise ValueError("opened evidence chunk must be an object")
    missing = [name for name in STRUCTURE_FIELDS if name not in chunk]
    if missing:
        raise ValueError("missing opened evidence structure fields: " + ",".join(missing))
    if chunk["structure_kind"] not in {"prose", "table_like"}:
        raise ValueError("invalid opened evidence structure_kind")
    if type(chunk["boundary_incomplete"]) is not bool:
        raise ValueError("boundary_incomplete must be boolean")
    if type(chunk["table_integrity_verified"]) is not bool:
        raise ValueError("table_integrity_verified must be boolean")
    if chunk["structure_kind"] == "table_like":
        if chunk["boundary_incomplete"] and chunk["table_integrity_verified"]:
            raise ValueError("incomplete table cannot be integrity verified")
    return chunk


def evidence_payload(output: dict) -> dict:
    rows = []
    for item in output.get("data") or []:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        validate_structure_chunk(item)
        row = {
            key: copy.deepcopy(item[key])
            for key in PROVENANCE_FIELDS + STRUCTURE_FIELDS
            if key in item
        }
        row["text"] = str(row["text"])[:5000]
        # Retrieval must already have returned a safe complete table window.
        # A local text cap is never allowed to create a new partial table.
        if item["structure_kind"] == "table_like" and row["text"] != item["text"]:
            raise ValueError("table evidence exceeds Runtime handoff limit")
        rows.append(row)
    return {
        "failed": bool(output.get("failed") or output.get("error")),
        "error": str(output.get("error") or "")[:500],
        "document_metadata": copy.deepcopy(output.get("document_metadata") or {}),
        "chunks": rows,
        "structure_contract": "lossless_browse_runtime_v14",
    }
