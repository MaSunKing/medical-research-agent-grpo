"""Explanatory feedback only: no action selection, execution, or rule changes."""
import json
from .contracts import search_signature, browse_context_signature


def _failures(history, action, state=None):
    """Mirror existing per-source / per-query transport accounting for display."""
    if action.get("action") == "browse":
        same = [h for h in history if h.get("action", {}).get("action") == "browse"
                and h["action"].get("source_id") == action.get("source_id")]
        if state is not None:
            scope = browse_context_signature(action, state)
            same = [h for h in same if h.get("feedback", {}).get("browse_context_sha256", scope) == scope]
        failures = []
        for h in reversed(same):
            if h.get("feedback", {}).get("status") == "ok":
                break
            if h.get("feedback", {}).get("status") == "environment_failure":
                failures.append(h)
        return same, failures
    same = [h for h in history if h.get("action", {}).get("action") == "search"
            and search_signature(h["action"]) == search_signature(action)]
    return same, [h for h in same if h.get("feedback", {}).get("status") == "environment_failure"]


def decision_repair_feedback(base, error, output):
    """Use ONLY the actual model-visible request; never inject hidden candidates."""
    visible = {}
    for message in base.get("messages", []):
        if message.get("role") != "user":
            continue
        try:
            value = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "question" in value:
            visible = value
            break
    try:
        action = json.loads(output)
    except (ValueError, TypeError):
        action = {}
    if not isinstance(action, dict):
        action = {}
    code = str(error).splitlines()[0].split(":")[0] if error else "invalid_decision"
    # The full original validation error stays in the generation event.
    lines = [code + ". No tool call executed or charged."]
    candidates = visible.get("candidates", [])
    ids = [c["source_id"] for c in candidates]
    if "source" in code or "browse" in code or "retry_limit" in code:
        if isinstance(action.get("source_id"), str):
            lines.append("Rejected source_id=" + json.dumps(action["source_id"][:40]) + ".")
        lines.append("Currently visible source_ids=" + json.dumps(ids, separators=(",", ":")) + ".")
        eligible = []
        for sid in ids:
            same, failures = _failures(visible.get("history", []), dict(action="browse", source_id=sid))
            last = same[-1].get("feedback", {}) if same else {}
            if (sid not in visible.get("blocked_sources", {}) and len(failures) < 2
                    and not (last.get("status") == "environment_failure" and last.get("retryable") is False)):
                eligible.append(sid)
        lines.append("Visible IDs passing access/retry checks=" + json.dumps(eligible, separators=(",", ":"))
                     + "; relevance not assessed.")
    if code.startswith("repeated_search"):
        matched = [h for h in visible.get("history", []) if h.get("action", {}).get("action") == "search"
                   and search_signature(h["action"]) == search_signature(action)
                   and h.get("feedback", {}).get("status") in {"ok", "empty"}]
        last = matched[-1] if matched else {}
        lines.append("This channel/query already returned " + str(last.get("feedback", {}).get("status", "normally"))
                     + "; do not repeat it. Choose a meaningful different query/channel or a relevant visible Browse.")
    elif "retry_limit" in code or code.startswith("nonretryable_"):
        _, failures = _failures(visible.get("history", []), action)
        count = str(len(failures)) if failures else "not resolved from this visible ID"
        lines.append(f"Same-source/query transport failures={count}; remaining retries=0. Choose another retrieval path.")
    elif code.startswith("source_access_blocked"):
        lines.append("This source is blocked for this run; same-source retry is not permitted. Choose another source/query.")
    elif code in {"source_not_in_visible_candidates", "unknown_source_alias"}:
        lines.append("Use a currently visible source_id, or Search for alternatives; history IDs may not be visible now.")
    else:
        lines.append("Return ONE JSON action: search, browse, or stop. Search uses channel (paper/web) and query; Browse uses source_id and reading_focus.")
        lines.append("Use 1-2 distinct existing requirement_ids=" + json.dumps(list(visible.get("checklist", {}))) + ".")
    lines.append("Choose the next action yourself from the current gap and budget; no new fields or evidence.")
    return " ".join(lines)


def transport_retry_feedback(state, action, observation):
    """Called after the current failed tool call, BEFORE it is appended to history."""
    _, previous = _failures(state.get("history", []), action, state)
    failures = len(previous) + 1
    remaining = max(0, 2 - failures) if observation.retryable else 0
    unknown_timeout = (observation.error_code == "retrieval_operation_timeout"
                       and observation.execution.get("resume_mode") != "resume_same_task")
    if unknown_timeout:
        remaining = 0
    status = dict(transport_failures=failures, remaining_retries=remaining,
                  scope="same source" if action["action"] == "browse" else "same channel/query")
    if action["action"] == "browse":
        status["source_id"] = action["source_id"]
    message = (f"Retrieval failed without new evidence; prior evidence and State remain. "
               f"Same-source/query transport failures={failures}; remaining retries={remaining}. ")
    if unknown_timeout:
        message += "The remote outcome is unknown; do not replay this request. Choose another source/query or retain the gap."
    elif remaining:
        message += ("One explicit retry may resume the verified existing PDF task: resume the same task; do not create a new task. "
                    if observation.execution.get("resume_mode") == "resume_same_task" else
                    "One explicit retry is permitted, not required; another source/query is also possible. ")
    else:
        message += "No same-source/query transport retry is permitted now; choose another source/query or retain the gap."
    return status, message
