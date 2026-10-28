"""Run-owned aliases: never derived from rank or recycled. Raw storage keeps real IDs."""
import copy
import re


def label(index):
    value = ""
    while index:
        index, digit = divmod(index - 1, 26)
        value = chr(65 + digit) + value
    return value


def register(state):
    mapping = state.setdefault("source_aliases", {})  # real ID -> alias, persisted in checkpoint
    # Canonical JSON sorts source keys on disk; alias identity cannot depend on
    # dictionary insertion order after checkpoint restore.
    if (any(not isinstance(value, str) for value in mapping.values())
            or set(mapping.values()) != {label(i) for i in range(1, len(mapping)+1)}):
        raise ValueError("invalid_source_alias_map")
    ids = list(state.get("candidate_history", {}))
    ids += [c["source_id"] for c in state.get("candidates", [])]
    ids += [c["source_id"] for c in state.get("evidence", {}).values()]
    ids += list(state.get("source_headers", {})) + list(state.get("blocked_sources", {}))
    ids += [h.get("action", {}).get("source_id") for h in state.get("history", [])]
    for sid in ids:
        if sid and sid not in mapping:
            mapping[sid] = label(len(mapping) + 1)
    return mapping


def project(payload, state):
    scratch = copy.deepcopy(state)
    mapping = register(scratch)
    # Replace source fields and source-keyed tables only; prose and chunk/E IDs stay untouched.
    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, dict):
            return {mapping.get(k, k): (mapping.get(v, v) if k == "source_id" and isinstance(v, str)
                                       else visit(v)) for k, v in value.items()}
        return value
    result = visit(payload)
    result.pop("source_aliases", None)
    result["source_alias_notice"] = "source_id letters are stable per-question source handles, not evidence citations. Browse a visible handle; State/Final cite E IDs. Original identities are archived."
    return result


def resolve_action(value, state):
    result = copy.deepcopy(value)
    if isinstance(result, dict) and result.get("action") == "browse":
        sid = result.get("source_id")
        if isinstance(sid, str) and re.fullmatch(r"[A-Z]+", sid):
            reverse = {a: s for s, a in state.get("source_aliases", {}).items()}
            if sid not in reverse:
                raise ValueError("unknown_source_alias")
            result["source_id"] = reverse[sid]
    return result
