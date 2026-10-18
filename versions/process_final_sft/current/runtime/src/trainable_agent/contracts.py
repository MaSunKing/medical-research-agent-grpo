from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, field_validator
from .model_json import parse_model_json

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]


SourceID = Annotated[str, StringConstraints(pattern=r"^(PMID:[0-9]+|S2:[a-zA-Z0-9_-]+|WEB:[a-zA-Z0-9_-]+)$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Checklist(Contract):
    # IDs are assigned by the runtime, never copied from untrusted model output.
    requirements: list[Text] = Field(min_length=1, max_length=8)


class Search(Contract):
    action: Literal["search"]
    requirement_ids: list[Text] = Field(min_length=1, max_length=2)
    channel: Literal["paper", "web"]
    query: Text


class Browse(Contract):
    action: Literal["browse"]
    requirement_ids: list[Text] = Field(min_length=1, max_length=2)
    source_id: SourceID
    reading_focus: Text


class Stop(Contract):
    action: Literal["stop"]
    reason: Literal["sufficient", "diminishing_returns", "no_readable_sources"]


Action = Annotated[Search | Browse | Stop, Field(discriminator="action")]
ACTION = TypeAdapter(Action)


class Update(Contract):
    requirement_id: Text
    support: Literal["missing", "partial", "direct"]
    evidence_ids: list[Text] = Field(max_length=24)
    gap: str = Field(max_length=1200)
    conflict: str = Field(max_length=1200)


class Delta(Contract):
    updates: list[Update] = Field(max_length=8)


class Candidate(Contract):
    source_id: SourceID
    channel: Literal["paper", "web"]
    title: str
    preview: str
    url: str = ""


class Chunk(Contract):
    source_id: SourceID
    chunk_id: Text
    # Source text is immutable data: stripping here would invalidate character-span provenance.
    text: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1, max_length=10000)]
    title: str = ""
    metadata: dict = Field(default_factory=dict)
    cleaning: dict = Field(default_factory=dict)


class ToolExecution(Contract):
    outcome: Literal["unknown"]
    phase: Literal["worker_startup", "worker_transport", "worker_operation"]
    receipt_id: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    resume_mode: Literal["unknown", "resume_same_task"]


class Observation(Contract):
    status: Literal["ok", "empty", "environment_failure", "access_blocked", "source_unreadable", "invalid_request"]
    error_code: str = ""
    retryable: bool = False
    retry_after_seconds: float = Field(default=0.0, ge=0, le=300)
    candidates: list[Candidate] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list, max_length=3)
    backend_return: dict | None = None
    execution: dict = Field(default_factory=dict)

    @field_validator("execution")
    @classmethod
    def execution_contract(cls, value):
        return ToolExecution.model_validate(value).model_dump() if value else {}



def source_script_hint(candidate):
    """Conservative script hint, not a claim of exact language detection."""
    import re
    text=str(candidate.get("title", ""))+" "+str(candidate.get("preview", ""))
    latin=len(re.findall(r"[A-Za-z]",text));han=len(re.findall(r"[\u3400-\u9fff]",text))
    if latin+han<80:return "unknown"
    if latin/(latin+han)>=0.9:return "latin"
    if han/(latin+han)>=0.7:return "cjk"
    return "mixed_or_unknown"

def validate_browse_script(focus,candidate):
    import re
    hint=source_script_hint(candidate)
    latin=len(re.findall(r"[A-Za-z]",focus));han=len(re.findall(r"[\u3400-\u9fff]",focus))
    if (hint=="latin" and han>latin) or (hint=="cjk" and han==0 and latin>30):
        raise ValueError("browse_query_script_mismatch: reading_focus is the passage-ranking query. Write a concise query in the selected source's language, preserving entities and the current gap; do not translate source text or change the original question.")


def search_signature(action):
    return (action.get("channel"), " ".join(action.get("query", "").casefold().split()))


def browse_context_signature(action, state):
    """Identity of available source/seen windows, not fallible semantic State."""
    import hashlib
    import json
    source = action.get("source_id")
    candidate = next((c for c in state.get("candidates", []) if c.get("source_id") == source), None)
    if candidate is None:
        candidate = state.get("candidate_history", {}).get(source)
    if candidate is not None:
        candidate = {k: v for k, v in candidate.items() if k != "read_status"}
    windows = {eid: {k: e.get(k) for k in ("source_id", "chunk_id", "text", "metadata")}
               for eid, e in state.get("evidence", {}).items() if e.get("source_id") == source}
    value = dict(candidate=candidate, opened=windows, tools=state.get("identities", {}).get("tools"))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def browse_no_progress_warning(action, state):
    if action.get("action") != "browse":
        return None
    focus = " ".join(action.get("reading_focus", "").casefold().split())
    signature = browse_context_signature(action, state)
    for row in reversed(state.get("history", [])):
        previous, feedback = row.get("action", {}), row.get("feedback", {})
        if (previous.get("action") != "browse" or previous.get("source_id") != action["source_id"]
                or " ".join(previous.get("reading_focus", "").casefold().split()) != focus):
            continue
        if (feedback.get("browse_context_sha256") == signature
                and feedback.get("status") in {"source_unreadable", "empty", "ok"}
                and not feedback.get("new_evidence_ids")
                and not feedback.get("retryable")
                and feedback.get("execution", {}).get("resume_mode") != "resume_same_task"):
            return ("repeated_browse_no_progress: the same source/focus previously added no evidence "
                    "with unchanged preview and opened windows. Consider another visible source, "
                    "a useful different focus/query, or Stop with the gap. This is advisory, not an "
                    "action rejection; an executed content-only/no-new-evidence retry costs normal "
                    "tool budget. This does not prove the entire source is unreadable.")


def near_duplicate_search_warning(action, state):
    """Lexical advisory only; never infer semantic equivalence or veto a refinement."""
    from difflib import SequenceMatcher
    if action.get("action") != "search":
        return None
    channel, query = search_signature(action)
    if len(query) < 24:
        return None
    best = None
    for row in state.get("history", []):
        previous = row.get("action", {})
        if previous.get("action") != "search" or row.get("feedback", {}).get("status") not in {"ok", "empty"}:
            continue
        prior_channel, prior_query = search_signature(previous)
        if channel != prior_channel or query == prior_query or len(prior_query) < 24:
            continue
        score = SequenceMatcher(None, query, prior_query, autojunk=False).ratio()
        if score >= .90 and (best is None or score > best["lexical_similarity"]):
            best = dict(previous_turn=row.get("turn"), lexical_similarity=score)
    if best is not None:
        best["lexical_similarity"] = round(best["lexical_similarity"], 3)
        best["message"] = "Query is lexically similar to a prior normal Search, not proven semantically identical. Ensure a meaningful gap refinement; prefer existing candidates when results repeat. This is advisory, not a veto."
    return best


def validate_action(value, state):
    action = ACTION.validate_python(value)
    if isinstance(action, Stop):
        return action.model_dump()
    ids = action.requirement_ids
    if len(ids) != len(set(ids)) or not set(ids) <= set(state["requirements"]):
        raise ValueError("unknown_or_duplicate_requirement_id")
    if isinstance(action, Browse) and action.source_id in state.get("blocked_sources", {}):
        raise ValueError("source_access_blocked_choose_another")
    if isinstance(action, Browse) and action.source_id not in {c["source_id"] for c in state["candidates"]}:
        raise ValueError("source_not_in_visible_candidates")
    if isinstance(action, Browse):
        selected=next(c for c in state["candidates"] if c["source_id"]==action.source_id)
        validate_browse_script(action.reading_focus,selected)
        unavailable = browse_availability_error(action.model_dump(), state)
        if unavailable:
            raise ValueError(unavailable)
    if isinstance(action, Search):
        same = [row for row in state.get("history", [])
                if row.get("action", {}).get("action") == "search"
                and search_signature(row["action"]) == search_signature(action.model_dump())]
        normal = any(row.get("feedback", {}).get("status") in {"ok", "empty"} for row in same)
        # A normal exact-query result is re-presented from the per-run cache.
        # Transport retry limits only apply when no normal cached result exists.
        if same and not normal:
            last = same[-1].get("feedback", {})
            if last.get("status") != "environment_failure" or last.get("retryable") is not True:
                raise ValueError("nonretryable_search_failure: choose another query/channel")
            if sum(row.get("feedback", {}).get("status") == "environment_failure" for row in same) >= 2:
                raise ValueError("temporary_failure_retry_limit: same query permits only one extra transport retry")
    return action.model_dump()


def browse_availability_error(action, state):
    """Shared execution/display rules; failures do not erase source history."""
    signature = browse_context_signature(action, state)
    same = []
    for row in state.get("history", []):
        previous, feedback = row.get("action", {}), row.get("feedback", {})
        if previous.get("action") != "browse" or previous.get("source_id") != action.get("source_id"):
            continue
        # Legacy transport records have no scope; retain their existing guard.
        if feedback.get("browse_context_sha256", signature) == signature:
            same.append(row)
    if not same:
        return None
    last = same[-1]["feedback"]
    if last.get("status") == "environment_failure":
        unknown = (last.get("error_code") == "retrieval_operation_timeout"
                   and last.get("execution", {}).get("resume_mode") != "resume_same_task")
        if last.get("retryable") is False or unknown:
            return "nonretryable_source_failure_choose_another"
    failures = 0
    for row in reversed(same):
        if row["feedback"].get("status") == "ok":
            break
        failures += row["feedback"].get("status") == "environment_failure"
    if failures >= 2:
        return "temporary_failure_retry_limit: choose another source"
    previous = same[-1]["action"]
    if (last.get("status") == "source_unreadable" and last.get("retryable") is False
            and last.get("browse_context_sha256") == signature
            and last.get("content_status") != "no_new_passages"
            and last.get("execution", {}).get("resume_mode") != "resume_same_task"
            and " ".join(previous.get("reading_focus", "").casefold().split())
                == " ".join(action.get("reading_focus", "").casefold().split())
            and set(action.get("requirement_ids", [])) <= set(previous.get("requirement_ids", []))):
        return "nonretryable_source_failure_choose_another"
    return None


def validate_delta(value, state):
    delta = Delta.model_validate(value)
    seen = set()
    for update in delta.updates:
        if update.requirement_id in seen or update.requirement_id not in state["requirements"]:
            raise ValueError("unknown_or_duplicate_requirement_id")
        seen.add(update.requirement_id)
        if len(update.evidence_ids) != len(set(update.evidence_ids)):
            raise ValueError("duplicate_evidence_id")
        if not set(update.evidence_ids) <= set(state["evidence"]):
            raise ValueError("evidence_not_opened")
        if update.support != "missing" and not update.evidence_ids:
            raise ValueError("support_requires_opened_evidence")
        if update.support != "direct" and not update.gap.strip():
            raise ValueError("missing_or_partial_requires_gap")
        # No monotonicity rule: later evidence can invalidate an earlier assessment.
    return delta.model_dump()
