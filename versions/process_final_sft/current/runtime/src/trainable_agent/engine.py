from __future__ import annotations

import copy
import json
import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .contracts import Checklist, Observation, validate_action, validate_delta, near_duplicate_search_warning, browse_context_signature, browse_no_progress_warning
from .storage import digest
from .compression import SemanticSelector
from .context_processing.aliases import register, resolve_action
from .context_processing.fallback import ScopedFallback
from .cleaning import clean_observation
from .context import ContextOverflow
from .failure_feedback import decision_repair_feedback, transport_retry_feedback
from . import search_cache
from .contracts import parse_model_json
from .model_json import JSON_STAGES, normalize_model_json


# Only deterministic action-availability failures may return to Decision.
# JSON/schema, requirement, State and citation errors still fail closed.
RECOVERABLE_DECISION_ERRORS = frozenset({
    "repeated_search_already_executed", "source_access_blocked_choose_another",
    "temporary_failure_retry_limit", "source_not_in_visible_candidates",
    "unknown_source_alias", "nonretryable_source_failure_choose_another",
    "nonretryable_search_failure",
})


def recoverable_decision_error(error):
    return error.partition("\n")[0].split(":", 1)[0] in RECOVERABLE_DECISION_ERRORS


class GraphState(TypedDict):
    # One JSON-serializable value: reducers never implicitly append model messages.
    session: dict


def initial_state(run_id, question, policy_identity, final_identity, tool_identity, process_only=False,
                  max_charged=6, max_exemptions=5, max_invalid_decisions=6):
    if not question.strip():
        raise ValueError("empty_question")
    if (type(max_charged) is not int or type(max_exemptions) is not int or max_charged < 1 or
        max_exemptions < 0 or max_charged + max_exemptions > 16):
        raise ValueError("invalid_tool_budget")
    if type(max_invalid_decisions) is not int or not 2 <= max_invalid_decisions <= 12:
        raise ValueError("invalid_decision_budget")
    return dict(run_id=run_id, question=question, next_node="checklist", process_only=process_only,
                identities={"process": policy_identity, "final": final_identity, "tools": tool_identity},
                requirements={}, assessments={}, evidence={}, source_headers={}, history=[], candidates=[],
                candidate_history={}, search_cache={}, source_aliases={}, blocked_sources={}, feedback={}, new_evidence_ids=[], turn=0, generation=0,
                budget={"charged": 0, "actual": 0, "exemptions": 0,
                        "max_charged": max_charged, "max_actual": max_charged + max_exemptions,
                        "max_exemptions": max_exemptions},
                decision_budget={"rejected": 0, "max_rejected": max_invalid_decisions},
                termination=None, prefinal=None, final=None)


class Engine:
    def __init__(self, store, process, final, tools, context, history_ranker, compressor=None, compression_context_tokens=10240):
        self.store, self.process, self.final = store, process, final
        self.tools, self.context, self.history_ranker = tools, context, history_ranker
        self.compressor, self.compression_limit = compressor, compression_context_tokens

    async def generate(self, stage, state, validator):
        selector = SemanticSelector(self.compressor, self.context.codec, self.store, state["run_id"],
                                    self.compression_limit) if self.compressor is not None else None
        register(state)
        fallback = ScopedFallback(self.compressor, self.context.codec, self.store, state["run_id"],
                                  self.compression_limit,max_calls=self.context.compression_max_calls) if self.compressor is not None else None
        try:
            base = await self.context.prepare(stage, state, selector, fallback=fallback)
        except ContextOverflow as exc:
            self.store.event(state["run_id"], "context_failure", dict(
                stage=stage, error_code=str(exc), state_sha256=digest(state),
                input_tokens=getattr(exc, "input_tokens", None),
                output_reserve=getattr(exc, "output_reserve", None),
                safety_margin=self.context.safety_margin,
                repair_reserve_tokens=self.context.repair_reserve_tokens,
                context_limit=self.context.limit_for(stage)))
            state.update(termination="context_budget_exhausted", next_node="prefinal")
            state["feedback"] = {"error_code": "context_budget_exhausted", "stage": stage,
                                 "action_executed": False}
            return None
        raw_context = base.pop("raw_context_payload", None)
        if raw_context is not None:
            self.store.event(state["run_id"], "context_archive", dict(stage=stage, payload=raw_context,
                            sha256=digest(raw_context), projection=base["context_audit"]))
        rejected_errors = []
        for attempt in range(2):
            request = copy.deepcopy(base)
            if attempt:
                try:
                    request = self.context.repair(stage, base, repair_detail if stage == "decision" else error)
                except ContextOverflow as exc:
                    self.store.event(state["run_id"], "context_failure", dict(stage=stage,
                        error_code="repair_context_overflow", action_executed=False,
                        input_tokens=exc.input_tokens,output_reserve=exc.output_reserve,
                        context_limit=self.context.limit_for(stage),safety_margin=self.context.safety_margin))
                    state.update(termination="context_budget_exhausted", next_node="prefinal")
                    state["feedback"] = dict(error_code="repair_context_overflow", stage=stage, action_executed=False)
                    return None
            request.update(stage=stage, role="final" if stage == "final" else "process")
            provider = self.final if stage == "final" else self.process
            key = digest([state["run_id"], state["identities"], stage, state["generation"], attempt, request])
            self.store.event(state["run_id"], "model_request", {"stage": stage, "request": request, "receipt_key": key})
            response = await self.store.once(key, lambda: provider.generate(request))
            json_repair = None
            try:
                effective_text = response["text"]
                if stage in JSON_STAGES:
                    try:
                        effective_text, json_repair = normalize_model_json(effective_text)
                    except json.JSONDecodeError:
                        # Unrepairable syntax stays with the original validator/error
                        # path (also preserves generic test/custom validators).
                        pass
                parsed = validator(effective_text)
                accepted, error = True, ""
            except (ValueError, TypeError, KeyError) as exc:
                # No raw provider exception or secret is copied into the model context.
                error = str(exc)[:1500]
                accepted, parsed = False, None
            record = dict(stage=stage, request=request, response=response, accepted=accepted,
                          error=error, receipt_key=key, attempt=attempt,
                          policy_identity=state["identities"][request["role"]],
                          state_before_sha256=digest(state))
            if json_repair is not None:
                record["json_repair"] = {**json_repair, "schema_accepted": accepted}
                self.store.event(state["run_id"], "model_json_repair", dict(
                    stage=stage, attempt=attempt, receipt_key=key, **record["json_repair"]))
            self.store.event(state["run_id"], "generation", record)
            if accepted:
                state["generation"] += 1
                return parsed
            rejected_errors.append(error)
            if stage == "decision":
                state["decision_budget"]["rejected"] += 1
                self.store.event(state["run_id"], "decision_rejection_budget", dict(
                    **state["decision_budget"], error=error,
                    action_executed=False, tool_budget_charged=False))
                if (recoverable_decision_error(error) and
                        state["decision_budget"]["rejected"] >= state["decision_budget"]["max_rejected"]):
                    break
            if stage == "decision" and attempt == 0:
                repair_detail = decision_repair_feedback(base, error, response["text"])
                self.store.event(state["run_id"], "decision_repair_feedback", dict(
                    error=error, feedback=repair_detail, action_executed=False, tool_budget_charged=False))
        if stage == "decision" and all(recoverable_decision_error(e) for e in rejected_errors):
            # A new Decision must get a new receipt identity, even if no tool ran.
            state["generation"] += 1
            exhausted = state["decision_budget"]["rejected"] >= state["decision_budget"]["max_rejected"]
            state["feedback"] = dict(error_code=error.split(":", 1)[0], stage=stage,
                executed=False, tool_budget_charged=False,
                decision_budget=copy.deepcopy(state["decision_budget"]),
                message=decision_repair_feedback(base, error, response["text"]))
            if exhausted:
                state.update(termination="decision_failure_budget_exhausted", next_node="prefinal",
                             process_status="degraded", completion_mode="degraded")
            else:
                state["next_node"] = "decision"
            self.store.event(state["run_id"], "decision_recovery", dict(
                feedback=state["feedback"], exhausted=exhausted,
                action_executed=False, tool_budget_charged=False))
            return None
        # Fail closed. Bad State must not be silently skipped or treated as correct.
        state.update(termination="model_contract_failure", next_node="prefinal")
        state["feedback"] = {"error_code": "model_contract_failure", "stage": stage}
        return None

    def commit(self, state, node):
        self.store.transition(state, node)
        return {"session": state}

    async def checklist(self, graph_state):
        s = copy.deepcopy(graph_state["session"])
        self.active_state = s
        def validate(text):
            obj = Checklist.model_validate(parse_model_json(text))
            if len(set(obj.requirements)) != len(obj.requirements):
                raise ValueError("duplicate_requirements")
            return obj.requirements
        items = await self.generate("checklist", s, validate)
        if items:
            s["requirements"] = {f"R{i}": x for i, x in enumerate(items, 1)}
            s["assessments"] = {rid: dict(support="missing", evidence_ids=[], gap=text, conflict="")
                                for rid, text in s["requirements"].items()}
            s["next_node"] = "decision"
        return self.commit(s, "checklist")

    async def decision(self, graph_state):
        s = copy.deepcopy(graph_state["session"])
        self.active_state = s
        b = s["budget"]
        if s["decision_budget"]["rejected"] >= s["decision_budget"]["max_rejected"]:
            s.update(termination="decision_failure_budget_exhausted", next_node="prefinal",
                     process_status="degraded", completion_mode="degraded")
        elif b["charged"] >= b["max_charged"] or b["actual"] >= b["max_actual"] or s["turn"] >= 16:
            s.update(termination="runtime_budget_exhausted", next_node="prefinal")
        else:
            action = await self.generate("decision", s, lambda t: validate_action(resolve_action(parse_model_json(t), s), s))
            if action:
                s["action"] = action
                if action["action"] == "stop":
                    s.update(termination="policy_stop:" + action["reason"], next_node="prefinal")
                else:
                    s["next_node"] = "execute"
        return self.commit(s, "decision")

    async def execute(self, graph_state):
        s = copy.deepcopy(graph_state["session"])
        self.active_state = s
        register(s)  # Reserve handles before filtering, including resumed legacy views.
        action = s["action"]
        cached = search_cache.lookup(s, action, self.store) if action["action"] == "search" else None
        browse_signature = browse_context_signature(action, s) if action["action"] == "browse" else None
        search_warning = near_duplicate_search_warning(action, s)
        browse_warning = browse_no_progress_warning(action, s)
        if browse_warning:
            self.store.event(s["run_id"], "browse_no_progress_warning", {"message": browse_warning})
        if search_warning:
            self.store.event(s["run_id"], "search_query_similarity_warning", search_warning)
        args = {"question": s["question"], "action": action}
        if action["action"] == "browse":
            args["channel"] = next(c["channel"] for c in s["candidates"] if c["source_id"] == action["source_id"])
            args["opened_windows"] = [dict(evidence_id=eid, source_id=e["source_id"], chunk_id=e["chunk_id"], text=e["text"],
                                          table_window=copy.deepcopy(e.get("metadata", {}).get("table_window")))
                                      for eid,e in s["evidence"].items() if e["source_id"] == action["source_id"]]
        key = digest([s["run_id"], s["identities"]["tools"], s["turn"], args])
        self.store.event(s["run_id"], "tool_request", {"request": args, "receipt_key": key,
                         "cache_hit": cached is not None, "network_requested": cached is None})
        async def invoke():
            # Only an explicit new action may retry. Backoff applies to repeated temporary failures.
            prior = []
            for row in reversed(s["history"]):
                previous = row.get("action", {})
                same = previous.get("action") == action["action"] and (
                    previous.get("source_id") == action.get("source_id") if action["action"] == "browse" else
                    previous.get("channel") == action.get("channel") and
                    " ".join(previous.get("query", "").casefold().split()) == " ".join(action.get("query", "").casefold().split()))
                if same:
                    if row.get("feedback", {}).get("status") == "environment_failure" and row["feedback"].get("retryable"):
                        prior = [row]
                    break
            if prior:
                import asyncio
                wait = max(1.0, float(prior[-1]["feedback"].get("retry_after_seconds", 0)))
                self.store.event(s["run_id"], "retry_wait", {"receipt_key": key, "seconds": wait})
                await asyncio.sleep(wait)
            self.store.event(s["run_id"], "tool_attempt_started", {"receipt_key": key, "action": action["action"]})
            return await self.tools.execute(args)
        raw = (copy.deepcopy(cached["observation"]) if cached is not None
               else await self.store.once(key, invoke))
        # Persist before normalization/cleaning validation; source originals are never overwritten.
        self.store.event(s["run_id"], "tool_raw", dict(request=args, observation=raw, receipt_key=key,
                         cache_hit=cached is not None, network_executed=cached is None))
        obs = Observation.model_validate(raw)
        if action["action"] == "search" and obs.chunks:
            raise ValueError("search_cannot_create_opened_evidence")
        if action["action"] == "browse" and any(c.source_id != action["source_id"] for c in obs.chunks):
            raise ValueError("browse_source_identity_mismatch")
        if obs.status != "ok" and (obs.candidates or obs.chunks):
            raise ValueError("failure_observation_contains_results")
        if action["action"] == "search" and obs.status in {"ok", "empty"} and cached is None:
            s.setdefault("search_cache", {})[search_cache.key(action, s)] = dict(
                observation=copy.deepcopy(obs.model_dump()), original_turn=s["turn"] + 1, receipt_key=key)
        cleanup_audit = None
        previous_candidates = set(s["candidate_history"])
        if action["action"] == "browse":
            cleaned, cleanup_audit = clean_observation(obs.model_dump())
            obs = Observation.model_validate(cleaned)
            self.store.event(s["run_id"], "browse_cleanup", dict(receipt_key=key, audit=cleanup_audit,
                            cleaned_observation=cleaned))
        s["turn"] += 1
        s["budget"]["actual"] += 1
        exempt = obs.status in {"environment_failure", "access_blocked"} and s["budget"]["exemptions"] < s["budget"]["max_exemptions"]
        s["budget"]["exemptions" if exempt else "charged"] += 1
        if action["action"] == "browse" and (obs.status == "access_blocked" or obs.error_code == "archive_redirect_limit"):
            s.setdefault("blocked_sources", {})[action["source_id"]] = obs.error_code
        s["new_evidence_ids"] = []
        cache_removed = []
        if action["action"] == "search" and cached is not None:
            s["candidates"], cache_removed = search_cache.visible_candidates(
                s, [c.model_dump() for c in obs.candidates][:8], action)
            register(s)
        elif action["action"] == "search" and obs.status == "ok":
            await self.update_candidates(s, [c.model_dump() for c in obs.candidates])
        if action["action"] == "browse" and obs.status == "ok":
            known = {digest([e["source_id"], e["text"]]) for e in s["evidence"].values()}
            for chunk in obs.chunks:
                sig = digest([chunk.source_id, chunk.text])
                if sig in known:
                    continue
                known.add(sig)
                eid = f"E{len(s['evidence']) + 1}"
                s["evidence"][eid] = chunk.model_dump()
                s["new_evidence_ids"].append(eid)
                source = next(c for c in s["candidates"] if c["source_id"] == chunk.source_id)
                s["source_headers"][chunk.source_id] = {k: source[k] for k in ("title", "url", "channel")}
        s["feedback"] = dict(status=obs.status, error_code=obs.error_code, retryable=obs.retryable,
                             new_evidence_ids=s["new_evidence_ids"], executed=True, budget_exempt=exempt,
                             retry_after_seconds=obs.retry_after_seconds)
        if browse_signature is not None:
            s["feedback"]["browse_context_sha256"] = browse_signature
            current_search = search_cache.latest_search(s)
            if current_search:
                s["feedback"]["failed_search_signature"] = list(search_cache.search_signature(current_search))
        if obs.execution:
            s["feedback"]["execution"] = obs.execution
        if action["action"] == "browse":
            omissions = (obs.backend_return or {}).get("metadata", {}).get("table_window_omissions", [])
            if omissions:
                s["feedback"]["table_window_omissions"] = [
                    {k: row[k] for k in ("chunk_id", "rank", "reason", "status", "coverage_verified") if k in row}
                    for row in omissions if isinstance(row, dict)]
                s["feedback"]["table_coverage_message"] = (
                    "Some ranked tables could not be returned as safe complete row windows. "
                    "Inspect the omission ranks and reasons; other returned passages do not establish coverage "
                    "of the omitted rows. Try another reading focus/source or retain the gap.")
        if action["action"] == "browse" and obs.status == "ok":
            quality = [dict(evidence_id=eid, warnings=s["evidence"][eid]["metadata"]["source_quality_warnings"])
                       for eid in s["new_evidence_ids"] if s["evidence"][eid]["metadata"].get("source_quality_warnings")]
            if quality:
                s["feedback"].update(source_quality_warnings=quality,
                    message="Some opened text has a PDF numeric-comparator encoding warning. This is not evidence about the threshold; do not guess its direction. Use unaffected facts, verify another readable source, or retain the ambiguity.")
        if action["action"] == "browse" and obs.status == "ok":
            abstract_ids=[eid for eid in s["new_evidence_ids"] if s["evidence"][eid]["metadata"].get("abstract_only") is True]
            if abstract_ids:
                s["feedback"].update(abstract_only_evidence_ids=abstract_ids,
                    content_scope_message="This retrieval returned abstract-only content, not article body. A different focus cannot recover missing body from the same cached content. Read unseen abstract passages if useful, choose another source/route or retain the gap; this does not prove full text is unavailable everywhere.")
        if action["action"] == "browse" and obs.status == "ok":
            scopes=[dict(evidence_id=eid,scope=s["evidence"][eid]["metadata"]["content_scope"])
                    for eid in s["new_evidence_ids"] if s["evidence"][eid]["metadata"].get("content_scope")]
            if scopes:
                s["feedback"].update(retrieved_content_scopes=scopes,
                    retrieved_scope_message="These are selected parsed passages, not verified complete webpage/article coverage. Tables may contain useful evidence but can also be baseline/eligibility data; inspect their actual content against the gap. Successful access alone does not resolve a requirement. Use another focus/source or retain missing information when needed.")
        if obs.status != "ok":
            s["feedback"]["message"] = (
                "上次工具调用未获得新增证据，已有证据和State保留。"
                + ("当前获取路径被阻断或已不可用，请选择其他可见来源或搜索替代来源，不要重复同一路径。"
                   if obs.status == "access_blocked" or obs.error_code == "archive_redirect_limit" else
                   "这是可能恢复的暂时性故障；同一请求最多额外重试一次，或改选其他来源。"
                   if obs.retryable else "当前来源或内容未提供有效证据；请选择其他来源、调整阅读重点或诚实停止。"))
        if action["action"] == "browse" and obs.status == "source_unreadable":
            s["feedback"]["message"] = (
                "This Browse returned no new usable evidence. The receipt does not establish that the entire "
                "source is permanently unreadable or that no full text exists elsewhere. "
                "Consider another visible source, a useful different focus/Search, or Stop with the gap. "
                "The unchanged failed source/focus is unavailable in the current candidate view; "
                "a genuinely different focus or changed retrieval content is not a permanent source ban. "
                "An abstract-only cache cannot supply missing article body merely by changing focus.")
        if obs.error_code == "retrieval_operation_timeout":
            s["feedback"]["message"] = (
                "Retrieval timed out without new evidence. Prior evidence and State remain. "
                + ("An existing PDF task identity and document hash were verified; an explicit retry resumes that same task, at most once. "
                   if obs.execution.get("resume_mode") == "resume_same_task" else
                   "The remote outcome is unknown; do not replay this same request. Choose another visible source or query, or stop with the gap. "))
        if obs.status == "environment_failure":
            retry_status, message = transport_retry_feedback(s, action, obs)
            s["feedback"].update(transport_retry_status=retry_status, message=message)
        if action["action"] == "browse" and not s["new_evidence_ids"] and obs.status in {"ok", "empty"}:
            s["feedback"].update(progress_warning="browse_added_no_new_evidence",
                message="No new evidence was returned. Same-source exact/contained opened windows are skipped; remaining candidates may be exhausted or unable to fit safely. Change focus/source or stop honestly; existing evidence and State remain available.")
        if action["action"] == "search":
            added = set(s["candidate_history"]) - previous_candidates
            s["feedback"].update(new_candidate_count=len(added), opened_evidence_added=False,
                next_step_hint="Search previews are not evidence. Browse a relevant visible source before assessing it; "
                               "if none fits, change the gap/channel/query or stop with an honest limitation.")
            if not added:
                s["feedback"]["progress_warning"] = "search_added_no_new_candidates"
        if search_warning:
            s["feedback"]["query_similarity_warning"] = search_warning
        if browse_warning:
            s["feedback"]["browse_repeat_warning"] = browse_warning
        if action["action"] == "browse" and not s["new_evidence_ids"]:
            backend = obs.backend_return or {}
            metadata = backend.get("metadata") or {}
            if metadata.get("novelty_status") == "no_new_passages":
                s["feedback"].update(content_status="no_new_passages", message=(
                    "The backend reports no new passages in the currently parsed retrieval route. "
                    "This does not establish that the whole webpage/article has been read or that "
                    "full text is unavailable elsewhere. Consider another source/route, a useful "
                    "different focus/Search, or Stop with the gap. A repeated content-only Browse "
                    "is allowed but costs normal tool budget; existing evidence and State remain."))
                current_search = search_cache.latest_search(s)
                candidate = next((c for c in s["candidates"] if c["source_id"] == action["source_id"]), None)
                if current_search and candidate:
                    s["feedback"].update(exhausted_search_signature=list(search_cache.search_signature(current_search)),
                        exhausted_source_scope=search_cache.source_scope(s, candidate))
                document = backend.get("document_metadata") or {}
                if document.get("alternate_fetch_method") == "pubmed_abstract":
                    s["feedback"]["content_scope_message"] = (
                        "This route provided a PubMed abstract, not article full text. "
                        "Changing focus alone cannot create missing article body from this cached content.")
        s["history"].append({"turn": s["turn"], "action": action, "feedback": s["feedback"]})
        if action["action"] == "browse":
            availability_search = search_cache.latest_search(s)
            if availability_search:
                availability_search = dict(availability_search, requirement_ids=action["requirement_ids"])
            s["candidates"], cache_removed = search_cache.visible_candidates(
                s, s["candidates"], availability_search)
        if cached is not None:
            s["feedback"].update(cache_hit=True, network_executed=False,
                original_search_turn=cached["original_turn"], original_search_receipt=cached["receipt_key"],
                cache_notice="Exact channel/query repeated: cached previews, not a new network search or evidence. Unread sources first, partially read sources last; explicitly exhausted current-query routes and blocked sources omitted. One normal action budget charged.",
                removed_candidates=cache_removed)
            self.store.event(s["run_id"], "search_cache_hit", dict(query_signature=list(search_cache.search_signature(action)),
                returned_source_ids=[c["source_id"] for c in s["candidates"]], removed_candidates=cache_removed,
                original_receipt=cached["receipt_key"], network_executed=False, budget_charged=True))
        if cache_removed:
            s["feedback"]["removed_candidates"] = cache_removed
            s["feedback"]["candidate_availability_notice"] = (
                "Unavailable current retrieval conditions were omitted from selectable candidates. "
                "Choose another visible source/query or stop with the gap; history and evidence are preserved.")
            self.store.event(s["run_id"], "candidate_availability_projection", dict(removed=cache_removed,
                scope="current query/requirements/parsed route; not whole-document coverage"))
        s["next_node"] = "state" if s["new_evidence_ids"] else "decision"
        self.store.event(s["run_id"], "tool", dict(request=args, observation=obs.model_dump(), receipt_key=key,
                         cleanup_audit=cleanup_audit, cache_hit=cached is not None))
        return self.commit(s, "execute")

    async def update_candidates(self, s, additions):
        prior = list(s["candidate_history"])
        latest = list(dict.fromkeys(c["source_id"] for c in additions if c["preview"].strip()))
        s["candidate_history"].update({c["source_id"]: c for c in additions if c["preview"].strip()})
        latest = [sid for sid in latest if sid not in s.get("blocked_sources", {})]
        valid_prior = [sid for sid in prior if sid not in latest and sid not in s.get("blocked_sources", {})]
        quota = 4 if prior else 8
        ids = latest[:quota]
        if valid_prior and len(ids) < 8:
            rows = [s["candidate_history"][sid] for sid in valid_prior]
            key = digest([s["run_id"], "history_ranking", s["identities"]["tools"], s["question"], rows])
            scores = await self.store.once(key, lambda: self.history_ranker(s["question"], rows))
            import math
            if len(scores) != len(rows) or not all(math.isfinite(x) for x in scores):
                raise ValueError("invalid_history_ranking_scores")
            ids += [sid for _, sid in sorted(zip(scores, valid_prior), key=lambda x: -x[0])][:8-len(ids)]
        ids += [sid for sid in latest if sid not in ids][:8-len(ids)]
        s["candidates"], _ = search_cache.visible_candidates(s, [s["candidate_history"][sid] for sid in ids])
        register(s)

    async def state(self, graph_state):
        s = copy.deepcopy(graph_state["session"])
        self.active_state = s
        delta = await self.generate("state", s, lambda t: validate_delta(parse_model_json(t), s))
        if delta:
            before = copy.deepcopy(s["assessments"])
            for update in delta["updates"]:
                s["assessments"][update["requirement_id"]] = {k: v for k, v in update.items() if k != "requirement_id"}
            self.store.event(s["run_id"], "state_delta", dict(before=before, delta=delta, after=s["assessments"]))
            s["next_node"] = "decision"
        return self.commit(s, "state")

    async def prefinal(self, graph_state):
        s = copy.deepcopy(graph_state["session"])
        self.active_state = s
        s["prefinal"] = dict(question=s["question"], checklist=s["requirements"], state=s["assessments"],
                             opened_evidence=s["evidence"], source_headers=s["source_headers"],
                             unresolved={k: v for k, v in s["assessments"].items()
                                         if v["support"] != "direct" or v["conflict"] or v.get("gap", "").strip()},
                             stop_reason=s.get("process_termination") or s["termination"], compact_history=copy.deepcopy(s["history"]),
                             source_aliases=copy.deepcopy(register(s)),
                             evidence_snapshot_sha256=digest(s["evidence"]), semantic_review="PENDING")
        if s.get("completion_mode") == "degraded":
            s["prefinal"]["completion_mode"] = "degraded"
            s["prefinal"]["decision_failure"] = copy.deepcopy(s["feedback"])
        s.setdefault("process_termination", s["termination"])
        s["run_status"] = "failed" if s["termination"] in {"engineering_failure", "model_contract_failure", "context_budget_exhausted"} else "process_complete"
        if s.get("completion_mode") == "degraded":
            s["run_status"] = "degraded_process_complete"
        s["final_status"] = "not_requested" if s["process_only"] else "not_started"
        s["next_node"] = "end" if s["process_only"] or s["run_status"] == "failed" else "final"
        return self.commit(s, "prefinal")

    async def final_node(self, graph_state):
        s = copy.deepcopy(graph_state["session"])
        self.active_state = s
        def validate(text):
            if not re.fullmatch(r"\s*<answer>\s*\S[\s\S]*?</answer>\s*", text) or text.count("<answer>") != 1 or text.count("</answer>") != 1:
                raise ValueError("expected_one_nonempty_answer_block")
            for group in re.findall(r"\[([^\]]+)\]", text):
                for eid in re.findall(r"\bE[0-9]+\b", group):
                    if eid not in s["evidence"]:
                        raise ValueError("unknown_citation_id:" + eid)
            return text
        s["final"] = await self.generate("final", s, validate)
        s["final_status"] = "complete" if s["final"] is not None else "failed"
        s["run_status"] = "complete" if s["final"] is not None else "failed"
        if s["final"] is not None and s.get("completion_mode") == "degraded":
            s["run_status"] = "degraded_complete"
        s["next_node"] = "end"
        return self.commit(s, "final")

    def graph(self):
        builder = StateGraph(GraphState)
        nodes = {"checklist": self.checklist, "decision": self.decision, "execute": self.execute,
                 "state": self.state, "prefinal": self.prefinal, "final": self.final_node}
        route = lambda value: value["session"]["next_node"]
        paths = {**{k: k for k in nodes}, "end": END}
        builder.add_conditional_edges(START, route, paths)
        for name, fn in nodes.items():
            async def guarded(value, fn=fn, name=name):
                before = copy.deepcopy(value["session"])
                self.active_state = before
                try:
                    return await fn(value)
                except Exception as exc:
                    s = copy.deepcopy(self.active_state)
                    failure = {"stage": name, "exception_type": type(exc).__name__,
                               "error_code": getattr(exc, "error_code", "engineering_exception"),
                               "outcome": "failed" if getattr(exc, "execution_outcome", None) == "failed" else "unknown",
                               "resume_state": before}
                    self.store.event(s["run_id"], "engineering_failure", failure)
                    s["failure"] = {k: v for k, v in failure.items() if k != "resume_state"}
                    s.update(termination="engineering_failure", run_status="failed")
                    if name == "final":
                        s.update(final_status="failed", next_node="end")
                    else:
                        s["next_node"] = "prefinal"
                    return self.commit(s, name + "_failure")
            builder.add_node(name, guarded)
            builder.add_conditional_edges(name, route, paths)
        return builder.compile()

    async def run(self, state):
        owner = self.store.acquire(state["run_id"])
        try:
            return await self._run(state)
        finally:
            self.store.release(state["run_id"], owner)

    async def _run(self, state):
        saved = self.store.load(state["run_id"])
        if saved:
            if any(saved[k] != state[k] for k in ("question", "identities", "process_only")):
                raise ValueError("resume_identity_mismatch")
            state = saved
        else:
            self.store.checkpoint(state)
        return (await self.graph().ainvoke({"session": state}, config={"recursion_limit": 100}))["session"]
