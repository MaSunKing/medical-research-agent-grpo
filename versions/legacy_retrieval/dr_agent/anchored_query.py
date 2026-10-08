# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Private original-question anchoring for the unchanged medical tool protocol.

The Agent still emits the four V1 tools with their historical arguments.  The
runtime wraps only the query value sent to the MCP process, and the backend
immediately unwraps it.  The envelope is therefore never shown to the model or
sent to PubMed/Serper.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any


QUERY_CONTEXT_PREFIX = "__MEDGAP_QUERY_CONTEXT_V1__:"
MEDICAL_QUERY_TOOLS = frozenset(
    {"pubmed_search", "browse_document", "medical_web_search", "browse_webpage"}
)
_CHATML_USER_RE = re.compile(
    r"<\|im_start\|>user\s*\n(.*?)(?=<\|im_end\|>|<\|im_start\|>)",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class AnchoredQuery:
    original_question: str
    focused_query: str
    encoded: bool = False


def _compact(value: Any, *, max_bytes: int) -> str:
    normalized = " ".join(str(value or "").split())
    encoded = normalized.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return normalized
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def extract_original_user_question(transcript: str) -> str:
    """Extract the real user question, never a later tool query/observation."""

    matches = [match.strip() for match in _CHATML_USER_RE.findall(str(transcript or ""))]
    matches = [match for match in matches if match]
    if matches:
        return _compact(matches[-1], max_bytes=640)
    return ""


def encode_anchored_query(original_question: str, focused_query: str) -> str:
    original = _compact(original_question, max_bytes=640)
    focused = _compact(focused_query, max_bytes=320)
    if not original or not focused:
        return focused
    payload = json.dumps(
        {"o": original, "f": focused},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return QUERY_CONTEXT_PREFIX + token


def decode_anchored_query(value: str) -> AnchoredQuery:
    raw = str(value or "").strip()
    if not raw.startswith(QUERY_CONTEXT_PREFIX):
        focused = _compact(raw, max_bytes=2048)
        return AnchoredQuery("", focused, False)
    token = raw[len(QUERY_CONTEXT_PREFIX) :]
    try:
        token += "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
        original = _compact(payload.get("o"), max_bytes=640)
        focused = _compact(payload.get("f"), max_bytes=320)
        if not focused:
            raise ValueError("focused query is empty")
        return AnchoredQuery(original, focused, True)
    except Exception:
        # A malformed envelope must not become an external search query.
        return AnchoredQuery("", "", False)


def anchored_execution_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    original_question: str,
) -> dict[str, Any]:
    """Return private execution arguments without mutating policy-visible data."""

    copied = dict(arguments or {})
    enabled = os.getenv("MEDGAP_ANCHORED_QUERY_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return copied
    if tool_name not in MEDICAL_QUERY_TOOLS:
        return copied
    focused = str(copied.get("query") or "").strip()
    if focused:
        copied["query"] = encode_anchored_query(original_question, focused)
    return copied


def anchored_relevance_query(original_question: str, focused_query: str) -> str:
    """Bounded joint text for passage models; focused intent remains dominant."""

    original = _compact(original_question, max_bytes=640)
    focused = _compact(focused_query, max_bytes=320)
    if not original or original.casefold() == focused.casefold():
        return focused or original
    # Repeating the focus approximates the configured 0.4/0.6 weighting for
    # retrievers that accept one query string rather than two score vectors.
    return f"Original question: {original}\nCurrent focus: {focused}\nFocus: {focused}"
