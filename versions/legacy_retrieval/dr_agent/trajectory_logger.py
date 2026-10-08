# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Append-only trajectory records for teacher-data generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Optional


_WRITE_LOCK = Lock()
VALID_EVENT_TYPES = {"tool_call", "tool_output", "assistant_message"}


def _validate_events(events: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    normalized = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError(f"trajectory[{index}] must be an object")
        event_type = event.get("type")
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"trajectory[{index}].type must be one of {sorted(VALID_EVENT_TYPES)}"
            )
        if event_type in {"tool_call", "tool_output"} and not event.get("tool"):
            raise ValueError(f"trajectory[{index}].tool is required")
        if event_type == "tool_call" and not isinstance(event.get("arguments"), dict):
            raise ValueError(f"trajectory[{index}].arguments must be an object")
        normalized.append(dict(event))
    return normalized


def append_trajectory(
    output_path: str | Path,
    *,
    question: str,
    model: str,
    prompt_version: str,
    tool_schema_version: str,
    trajectory: Iterable[Dict[str, Any]],
    final_answer: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and append one UTF-8 JSONL teacher trajectory."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not final_answer.strip():
        raise ValueError("final_answer must not be empty")

    record = {
        "schema_version": "medical_teacher_trajectory_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "model": model,
        "prompt_version": prompt_version,
        "tool_schema_version": tool_schema_version,
        "trajectory": _validate_events(trajectory),
        "final_answer": final_answer,
        "metadata": metadata or {},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
    return record

