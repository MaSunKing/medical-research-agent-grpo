# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Conservative structural checks for flattened statistical tables.

The check never rewrites source text.  It only decides whether a window needs
whole-structure provenance before it may be used as cited evidence.
"""
import re


_HEADER_PATTERNS = (
    r"\btable\s+\d+[a-z]?\b",
    r"\b(?:sample\s+size|participants?|patients?)\b",
    r"\b(?:events?|cases?|deaths?|hospitali[sz]ations?)\b",
    r"\b(?:hr|rr|or|md|smd)\s*(?:\(|$)",
    r"\b95\s*%\s*(?:ci|confidence\s+interval)\b",
    r"\b(?:adjusted|unadjusted)\b",
    r"\b(?:heterogeneity|egger(?:'s)?\s+test|i\s*[²2])\b",
    r"\b(?:p\s*(?:value|=)|row\s*%|col(?:umn)?\s*%)\b",
)


def table_signature(text, declared_kind=None):
    """Return an auditable conservative table classification.

    Statistical vocabulary alone is not table structure.  A normal sentence
    may legitimately contain patients, events, RR and a confidence interval.
    Explicit parser structure and cell separators remain authoritative; the
    flattened-text heuristic is used only when the fragment has no complete
    prose sentence boundary.
    """
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    header_hits = [pattern for pattern in _HEADER_PATTERNS if re.search(pattern, value, re.I)]
    strong_hits = [pattern for index, pattern in enumerate(_HEADER_PATTERNS)
                   if index not in {1, 2} and re.search(pattern, value, re.I)]
    numeric_cells = re.findall(r"(?<![\w.])[<>]?\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?%?\*?(?![\w.])", value)
    pipe_cells = value.count("|")
    declared = str(declared_kind or "").lower() in {"table", "table_like"}
    sentence_boundaries = len(re.findall(r"(?:[.!?](?=\s|$)|[。！？｡])", value))
    structural_table = declared or pipe_cells >= 2
    flattened_table = (
        len(strong_hits) >= 2
        and len(header_hits) >= 3
        and len(numeric_cells) >= 3
        and len(value) <= 1600
    )
    prose_override = not structural_table and sentence_boundaries >= 1
    inferred = structural_table or (flattened_table and not prose_override)
    return {
        "structure_kind": "table_like" if inferred else "prose",
        "declared_table": declared,
        "header_signal_count": len(header_hits),
        "strong_header_signal_count": len(strong_hits),
        "numeric_cell_count": len(numeric_cells),
        "pipe_separator_count": pipe_cells,
        "sentence_boundary_count": sentence_boundaries,
        "structural_table_signal": structural_table,
        "flattened_table_signal": flattened_table,
        "prose_sentence_override": prose_override,
        "classifier": "flattened_statistical_table_v2",
    }


def incomplete_table_window(text, *, start, end, span_start, span_end, declared_kind=None):
    signature = table_signature(text, declared_kind)
    incomplete = signature["structure_kind"] == "table_like" and (
        start > span_start
        or end < span_end
        or bool(re.search(r"(?:[,;:|]|\b(?:and|or|vs))\s*$", str(text or ""), re.I))
    )
    return signature, incomplete
