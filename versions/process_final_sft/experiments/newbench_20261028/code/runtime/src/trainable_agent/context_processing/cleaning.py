"""Conservative post-Browse cleanup; no model, relevance scoring or medical rewriting."""
import copy
import re
from ..compression import text_sha
from ..storage import digest
from .browser import high_confidence_nonbody_shape

VERSION = "browse_rule_cleanup_shared_nonbody_v6"

# Match whole standalone UI lines, NOT occurrences of these words in medical prose.
TEMPLATES = (
    ("navigation_control", r"(?:Open (?:table|figure|image) in a new tab|View full size|Download (?:figure|image))[.!]?"),
    ("cookie_control", r"(?:Accept all cookies|Reject all cookies|Manage cookie preferences|Cookie settings)[.!]?"),
    ("cookie_notice", r"We use cookies (?:to improve your (?:browsing )?experience|on this website)[.!]?"),
    ("access_barrier", r"(?:Please )?(?:enable JavaScript(?: and cookies)? to continue|sign in to continue|log in to continue)[.!]?"),
    ("access_barrier", r"(?:Access denied|403 Forbidden|Verify you are human|Checking your browser|Subscribe to read (?:this|the) (?:article|full text))[.!]?"),
    ("access_notice", r"(?:404 Not Found|502 Bad Gateway)[.!]?"),
    ("navigation_menu", r"Home\s*[|›>]\s*(?:About(?: us)?|Products|Services)\s*[|›>]\s*(?:Contact(?: us)?|Help)"),
    ("advertisement_label", r"(?:Advertisement|ADVERTISEMENT)"),
)


def clean_chunk(chunk, *, classification_heading=None):
    original = chunk["text"]
    removed, kept = [], []
    offset = 0
    metadata = chunk.get("metadata") or {}
    heading = classification_heading if classification_heading is not None else chunk.get("heading", metadata.get("heading", ""))
    # Structured table projections already carry source-window mappings. Even
    # a cell containing a known UI label must not change those mappings here.
    table_projection = bool(chunk.get("table_window") or metadata.get("table_window")
                            or chunk.get("structure_kind") in {"table", "table_like"}
                            or metadata.get("structure_kind") in {"table", "table_like"}
                            or re.search(r'<table\b|\|\s*:?-{3,}:?\s*\|', original, re.I))
    # Recognize consecutive dotted page-number entries only at the chunk start.
    # Remove the prefix, never a following clinical paragraph.
    toc = re.match(r"(?:[^.\n]{1,120}?\.{3,}\s*\d+(?:\s*[-–]\s*\d+)?\s*){3,}", original)
    if toc and not table_projection:
        offset = toc.end()
        removed.append(dict(start=0, end=offset, reason="table_of_contents_prefix"))
    tail = original[offset:]
    editorial = (
        tail.lstrip().startswith("Suggested citation:") and
        all(re.search(marker, tail, re.I) for marker in
            (r"Editor in Chief", r"Managing Editor", r"Editorial Board")) and
        not re.search(r"\b(?:patients?|participants?|trial results?|adverse events?|"
                      r"risk ratio|odds ratio|contraindicated|we recommend)\b", tail, re.I)
    )
    if editorial and not table_projection:
        removed.append(dict(start=offset, end=len(original), reason="publication_editorial_frontmatter"))
        tail = ""
    shape = None if table_projection else high_confidence_nonbody_shape(heading, tail)
    if shape:
        removed.append(dict(start=offset, end=len(original), reason=shape[1]))
        tail = ""
    for line in tail.splitlines(keepends=True):
        body = line.strip()
        shape = None if table_projection else high_confidence_nonbody_shape(heading, body)
        reason = None if table_projection else next((name for name, pattern in TEMPLATES
                       if re.fullmatch(pattern, body, re.I)), shape[1] if shape else None)
        if reason:
            removed.append(dict(start=offset, end=offset+len(line), reason=reason))
        else:
            if kept and kept[-1][1] == offset:
                kept[-1] = (kept[-1][0], offset+len(line))
            else:
                kept.append((offset, offset+len(line)))
        offset += len(line)
    # Preserve whitespace and every retained character; no substitutions or dedup of medical text.
    text = "".join(original[a:b] for a, b in kept)
    result = copy.deepcopy(chunk)
    result["text"] = text
    result["cleaning"] = dict(version=VERSION, original_text_sha256=text_sha(original),
                              cleaned_text_sha256=text_sha(text), kept_spans=[list(x) for x in kept],
                              removed_spans=removed, semantic_verified=False)
    # Advisory only: do not "correct" symbols without a validated PDF glyph map.
    url=str(chunk.get("metadata",{}).get("url") or "").split("?",1)[0].lower()
    if url.endswith(".pdf") and re.search(r"‡\s*\d|\\\$\s*\d", text):
        result.setdefault("metadata",{}).setdefault("source_quality_warnings",[]).append(
            "PDF numeric comparator may be misencoded; do not infer its inequality. Verify against a reliable extraction or alternate source before using that threshold.")
    for key in ("source_id", "chunk_id", "title"):
        if result.get(key) != chunk.get(key):
            raise ValueError("cleanup_source_identity_changed")
    return result


def clean_observation(observation):
    """Input must be validated. Failure/empty observations are not reinterpreted."""
    original = copy.deepcopy(observation)
    result = copy.deepcopy(observation)
    audit = dict(version=VERSION, original_observation_sha256=digest(original), chunks=[],
                 usable_body=False, medical_relevance_assessed=False)
    if observation["status"] != "ok":
        return result, audit
    cleaned = []
    for chunk in observation["chunks"]:
        # Heading is parser context, not the webpage title. Normalization may
        # omit it from the model-visible chunk; recover it only from the exact
        # matching saved receipt without adding/changing evidence metadata.
        heading = None
        backend = observation.get("backend_return") or {}
        for raw in backend.get("data", []) if isinstance(backend, dict) else []:
            if (isinstance(raw, dict)
                    and chunk.get("chunk_id") in {raw.get("chunk_id"), raw.get("source_id")}
                    and raw.get("text") == chunk["text"]):
                heading = raw.get("heading")
                break
        value = clean_chunk(chunk, classification_heading=heading)
        audit["chunks"].append(dict(source_id=chunk["source_id"], chunk_id=chunk["chunk_id"],
                                    **value["cleaning"], usable_body=bool(value["text"].strip())))
        if value["text"].strip():
            cleaned.append(value)
    result["chunks"] = cleaned
    audit["usable_body"] = bool(cleaned)
    if not cleaned:
        barrier = any(span["reason"] == "access_barrier" for row in audit["chunks"] for span in row["removed_spans"])
        result.update(status="access_blocked" if barrier else "source_unreadable",
                      error_code="confirmed_source_access_barrier" if barrier else "no_usable_body_after_rule_cleanup",
                      retryable=False)
    return result, audit
