"""Decision views derived from committed history, never model-written summaries."""
import copy


def decision_observations(state):
    history = state.get("history", [])
    latest = history[-1] if history else {}
    action = latest.get("action", {})
    ids = action.get("requirement_ids", [])
    target = dict(previous_action=copy.deepcopy(action), turn=latest.get("turn"),
                  requirements={rid: state["requirements"][rid] for rid in ids},
                  current_assessments={rid: copy.deepcopy(state["assessments"][rid]) for rid in ids},
                  previous_status=latest.get("feedback", {}).get("status"))
    search = next((row for row in reversed(history) if row.get("action", {}).get("action") == "search"), None)
    target["latest_search"] = (dict(turn=search.get("turn"), action=copy.deepcopy(search["action"]),
                                    status=search.get("feedback", {}).get("status")) if search else None)
    # Keep the most recent Browse that actually added evidence, even after a
    # later Search/failed/duplicate Browse. Its original turn makes age explicit.
    browse = next((row for row in reversed(history)
                   if row.get("action", {}).get("action") == "browse"
                   and row.get("feedback", {}).get("new_evidence_ids")), None)
    if browse is None:
        return target, None, {}
    eids = browse["feedback"]["new_evidence_ids"]
    if len(eids) != len(set(eids)) or any(eid not in state["evidence"] for eid in eids):
        raise ValueError("latest_browse_evidence_history_mismatch")
    if any(state["evidence"][eid]["source_id"] != browse["action"]["source_id"] for eid in eids):
        raise ValueError("latest_browse_source_history_mismatch")
    observation = dict(turn=browse.get("turn"), action=copy.deepcopy(browse["action"]), evidence_ids=list(eids),
                       status=browse["feedback"].get("status"), scope="most_recent_browse_with_new_evidence")
    return target, observation, {eid: copy.deepcopy(state["evidence"][eid]) for eid in eids}
