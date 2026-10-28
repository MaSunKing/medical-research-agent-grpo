"""Lossless deterministic model-visible projection; originals stay in storage."""
import copy

VERSION = "deterministic_context_projection_v3"
NOTICE = (
    "Engineering cleaning/window hashes and offsets are archived outside this input. "
    "An evidence item with context_reference has identical text at the given "
    "character span of the referenced visible evidence item; it is not absent "
    "evidence or a semantic summary. All source identities and evidence IDs remain."
)

def visible_metadata(metadata):
    """Project program-owned receipts, preserving clinical fields and coverage.

    Unknown receipt schemas remain visible rather than guessing their meaning.
    The original maps/hashes remain in tool records, evidence and Pre-Final.
    """
    result = copy.deepcopy(metadata)
    window = result.get("table_window")
    identity = result.get("window_identity")
    known_identity = isinstance(identity, dict) and identity.get("version") in {
        "returned_window_identity_v1", "returned_table_window_identity_v2"}
    if known_identity:
        result.pop("window_identity", None)
        result.pop("window_identity_sha256", None)
    for key in ("table_window", "novelty_projection"):
        if key in result and result[key] is None:
            result.pop(key)
    if not isinstance(window, dict) or window.get("version") != "exact_table_rows_v1":
        return result
    result.pop("window_identity", None)
    result.pop("window_identity_sha256", None)
    result["table_window"] = {
        "scope": "selected_table_rows",
        "selected_row_count": len(window.get("row_indices", [])),
        "complete_table": window.get("complete_table") is True,
        "complete_row_groups": window.get("complete_row_groups") is True,
        "all_columns_retained": window.get("all_columns_retained") is True,
        "all_footnotes_retained": window.get("all_footnotes_retained") is True,
        "notice": "Selected original table rows with their headers and dependencies. "
                  "Other rows may be omitted; this does not establish the requested clinical relationship.",
    }
    return result


def visible_table_metadata(metadata):
    """Compatibility entry point; all evidence uses the same projection."""
    return visible_metadata(metadata)


def visible_evidence_chunk(chunk):
    result = copy.deepcopy(chunk)
    result.pop("cleaning", None)
    if "metadata" in result:
        result["metadata"] = visible_metadata(result["metadata"])
    return result


def project_evidence(original):
    result = copy.deepcopy(original)
    omitted = []
    table_omitted = []
    window_omitted = []
    for eid, chunk in result.items():
        if "cleaning" in chunk:
            chunk.pop("cleaning")
            omitted.append(eid)
        if "metadata" in chunk:
            visible = visible_metadata(chunk["metadata"])
            if visible != chunk["metadata"]:
                if isinstance(chunk["metadata"].get("table_window"), dict) and chunk["metadata"]["table_window"].get("version") == "exact_table_rows_v1":
                    table_omitted.append(eid)
                window_omitted.append(eid)
                chunk["metadata"] = visible
    order = sorted(result, key=lambda eid: -len(result[eid]["text"]))
    witnesses = []
    references = {}
    for eid in order:
        chunk = result[eid]
        text = chunk["text"]
        match = None
        # Short headings or coincidental phrases are not deduplicated.
        if len(text) >= 100:
            for target in witnesses:
                larger = result[target]
                if larger["source_id"] != chunk["source_id"]:
                    continue
                start = larger["text"].find(text)
                if start >= 0:
                    match = dict(evidence_id=target, start=start, end=start+len(text))
                    break
        if match is None:
            witnesses.append(eid)
        else:
            references[eid] = match
            chunk["text"] = ""
            chunk["context_reference"] = match
    validate_projection(result, original)
    return result, dict(version=VERSION, provenance_omitted_ids=omitted,
                        table_provenance_omitted_ids=table_omitted,
                        window_provenance_omitted_ids=window_omitted,
                        exact_duplicate_references=references)

def validate_projection(projected, original):
    if set(projected) != set(original):
        raise ValueError("projection_evidence_ids_changed")
    for eid, chunk in projected.items():
        saved = original[eid]
        for key, value in saved.items():
            if key == "metadata":
                value = visible_metadata(value)
            if key not in ("text", "cleaning") and chunk.get(key) != value:
                raise ValueError("projection_identity_changed")
        ref = chunk.get("context_reference")
        if ref is None:
            if chunk["text"] != saved["text"]:
                raise ValueError("projection_text_changed")
        else:
            target = projected.get(ref["evidence_id"])
            if target is None or target.get("context_reference") or target["source_id"] != chunk["source_id"]:
                raise ValueError("projection_invalid_reference")
            a, b = ref["start"], ref["end"]
            if not (0 <= a < b <= len(target["text"])) or target["text"][a:b] != saved["text"]:
                raise ValueError("projection_reference_text_mismatch")
