"""Per-run exact-query results, with lossless source-scoped availability filtering."""
import copy
from .contracts import search_signature, browse_availability_error
from .storage import digest


VERSION = "exact_search_cache_unread_first_v1"


def key(action, state):
    return digest([state["run_id"], state["question"], state["identities"]["tools"], search_signature(action)])


def normal_search_seen(action, state):
    return any(row.get("action", {}).get("action") == "search"
               and search_signature(row["action"]) == search_signature(action)
               and row.get("feedback", {}).get("status") in {"ok", "empty"}
               for row in state.get("history", []))


def latest_search(state):
    return next((row["action"] for row in reversed(state.get("history", []))
                 if row.get("action", {}).get("action") == "search"
                 and row.get("feedback", {}).get("status") in {"ok", "empty"}), None)


def source_scope(state, candidate):
    # Presentation-only read status must not change the underlying route identity.
    original = {k: v for k, v in candidate.items() if k != "read_status"}
    return digest([state["identities"]["tools"], original,
                   {eid: e for eid, e in state.get("evidence", {}).items()
                    if e["source_id"] == candidate["source_id"]}])


def visible_candidates(state, candidates, search=None):
    """Stable unread-first; no semantic reranking, invented sources or refills."""
    visible, removed = [], []
    for candidate in candidates:
        sid = candidate["source_id"]
        if sid in state.get("blocked_sources", {}):
            removed.append(dict(source_id=sid, reason="access_blocked"))
            continue
        history = [r for r in state.get("history", [])
                   if r.get("action", {}).get("action") == "browse"
                   and r["action"].get("source_id") == sid]
        if history:
            last = history[-1]
            probe = dict(last["action"])
            if search:
                probe["requirement_ids"] = search.get("requirement_ids", [])
            scratch = dict(state, candidates=[candidate])
            reason = browse_availability_error(probe, scratch)
            content_failure = last.get("feedback", {}).get("status") == "source_unreadable"
            same_search = (search and last.get("feedback", {}).get("failed_search_signature")
                           == list(search_signature(search)))
            if reason and (not content_failure or same_search):
                removed.append(dict(source_id=sid, reason=reason.split(":", 1)[0]))
                continue
        exhausted = False
        if search:
            for row in reversed(history):
                feedback = row.get("feedback", {})
                if (feedback.get("content_status") == "no_new_passages"
                        and feedback.get("exhausted_search_signature") == list(search_signature(search))
                        and set(search.get("requirement_ids", [])) <= set(row["action"].get("requirement_ids", []))
                        and feedback.get("exhausted_source_scope") == source_scope(state, candidate)):
                    exhausted = True
                    break
        if exhausted:
            removed.append(dict(source_id=sid, reason="no_new_passages_current_query_route"))
            continue
        read = any(e["source_id"] == sid for e in state.get("evidence", {}).values())
        item = copy.deepcopy(candidate)
        item["read_status"] = "partially_read" if read else "unread"
        visible.append(item)
    visible.sort(key=lambda c: c["read_status"] == "partially_read")
    return visible, removed


def lookup(state, action, store):
    """Old checkpoints recover exact results from persisted receipts, not current mix."""
    cache = state.setdefault("search_cache", {})
    identity = key(action, state)
    if identity in cache:
        return copy.deepcopy(cache[identity])
    if not normal_search_seen(action, state):
        return None
    saved = store.load(state["run_id"])
    if saved and saved["identities"]["tools"] != state["identities"]["tools"]:
        raise ValueError("search_cache_tool_identity_mismatch")
    for event in reversed(store.events(state["run_id"], "tool")):
        data = event["data"]
        prior = data.get("request", {}).get("action", {})
        observation = data.get("observation", {})
        if (prior.get("action") == "search" and search_signature(prior) == search_signature(action)
                and observation.get("status") in {"ok", "empty"}
                and not data.get("cache_hit")):
            cache[identity] = dict(observation=copy.deepcopy(observation),
                                   original_turn=None, receipt_key=data.get("receipt_key"))
            return copy.deepcopy(cache[identity])
    raise ValueError("prior_search_cache_receipt_missing: do not silently requery or substitute mixed candidates")
