# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Canonical schemas and conservative aliases for the four medical tools."""

from __future__ import annotations

import re
from typing import Any, Iterable


CANONICAL_TOOLS = {
    "pubmed_search", "browse_document", "medical_web_search", "browse_webpage"
}

SOURCE_TYPE_ALIASES = {
    "guideline": "guideline",
    "guidelines": "guideline",
    "clinical_guideline": "guideline",
    "clinical_guidelines": "guideline",
    "practice_guideline": "guideline",
    "practice_guidelines": "guideline",
    "professional_guideline": "guideline",
    "professional_guidelines": "guideline",
    "consensus_guideline": "guideline",
    "consensus_statement": "guideline",
    "evidence_review": "evidence_review",
    "evidence_reviews": "evidence_review",
    "systematic_review": "evidence_review",
    "systematic_reviews": "evidence_review",
    "meta_analysis": "evidence_review",
    "evidence_synthesis": "evidence_review",
    "health_technology_assessment": "evidence_review",
    "public_health": "public_health",
    "public_health_guidance": "public_health",
    "government_health_guidance": "public_health",
    "health_authority": "public_health",
    "official_health_guidance": "public_health",
    "regulatory": "regulatory",
    "regulator": "regulatory",
    "regulatory_guidance": "regulatory",
    "drug_label": "regulatory",
    "medication_label": "regulatory",
    "safety_warning": "regulatory",
    "safety_communication": "regulatory",
    "approval_status": "regulatory",
}

ARGUMENT_ALIASES = {
    "query": {"q", "search_query", "focused_query", "focus_query", "question", "keywords"},
    "source_id": {"id", "document_id", "paper_id", "article_id", "web_id", "result_id"},
    "source_types": {"source_type", "evidence_type", "evidence_types", "category", "categories"},
    "limit": {"top_n", "max_results", "num_results", "result_limit"},
    "top_k": {"num_chunks", "max_chunks", "chunk_limit"},
    "max_chars": {"char_limit", "max_characters"},
    "max_output_tokens": {"max_tokens", "output_tokens", "token_limit"},
    "gl": {"country_code", "geo"},
    "hl": {"language_code", "lang"},
}

TOOL_ARGUMENTS = {
    "pubmed_search": {"query", "limit", "offset"},
    "medical_web_search": {"query", "source_types", "limit", "gl", "hl"},
    "browse_document": {"source_id", "query", "top_k", "max_chars", "max_output_tokens"},
    "browse_webpage": {"source_id", "query", "top_k", "max_chars", "max_output_tokens"},
}

INTEGER_BOUNDS = {
    "limit": (1, 10),
    "offset": (0, 1_000_000),
    "top_k": (1, 5),
    "max_chars": (200, 12_000),
    "max_output_tokens": (1, 4_096),
}


def _token(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def normalize_source_types(source_types: str | Iterable[str]) -> list[str]:
    """Map unambiguous evidence-type aliases to four canonical categories."""
    raw = re.split(r"[,;|/]", source_types) if isinstance(source_types, str) else list(source_types)
    values = [value for value in (_token(item) for item in raw) if value]
    if not values:
        return ["guideline"]
    unknown = sorted({value for value in values if value not in SOURCE_TYPE_ALIASES})
    if unknown:
        raise ValueError(
            f"Unsupported medical source types: {unknown}. Supported canonical values: "
            "['evidence_review', 'guideline', 'public_health', 'regulatory']"
        )
    return list(dict.fromkeys(SOURCE_TYPE_ALIASES[value] for value in values))


def _canonical_key(key: Any) -> str:
    normalized = _token(key)
    for canonical, aliases in ARGUMENT_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def _bounded_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, not a boolean")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    lower, upper = INTEGER_BOUNDS[key]
    return max(lower, min(parsed, upper))


def normalize_medical_tool_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize safe aliases and reject arguments outside the tool schema."""
    if tool not in CANONICAL_TOOLS:
        raise ValueError(f"Unsupported medical tool: {tool}")
    canonical: dict[str, Any] = {}
    for raw_key, value in dict(arguments or {}).items():
        key = _canonical_key(raw_key)
        if key in canonical and canonical[key] not in (None, "", [], {}):
            continue
        canonical[key] = value
    unknown = sorted(set(canonical) - TOOL_ARGUMENTS[tool])
    if unknown:
        raise ValueError(f"Unsupported arguments for {tool}: {unknown}")
    for key in set(canonical) & set(INTEGER_BOUNDS):
        canonical[key] = _bounded_int(canonical[key], key)
    if "source_types" in canonical:
        canonical["source_types"] = ",".join(normalize_source_types(canonical["source_types"]))
    for key in ("query", "source_id", "gl", "hl"):
        if key in canonical:
            canonical[key] = str(canonical[key]).strip()
    if "gl" in canonical:
        canonical["gl"] = canonical["gl"].lower()
    if "hl" in canonical:
        canonical["hl"] = canonical["hl"].lower().split("-", 1)[0]
    return canonical
