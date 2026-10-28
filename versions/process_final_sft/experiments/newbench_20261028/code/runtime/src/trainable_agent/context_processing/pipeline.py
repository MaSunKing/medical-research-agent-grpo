import copy
from ..prompts import PROMPTS
from ..storage import canonical, digest
from ..compression import VERSION, NOTICE, validate_card, structured_table_window
from .aliases import project as project_aliases, register as register_aliases
from .deterministic_context import project_evidence, visible_evidence_chunk, NOTICE as DETERMINISTIC_NOTICE
from .decision import decision_observations
from .extraction import excerpt_target, text_tokens
from .final_pack import pack_final, pack_state


class ContextOverflow(ValueError):
    pass


class CardSelectionRequired(Exception):
    def __init__(self, key, chunk, focus):
        self.key, self.chunk, self.focus = key, chunk, focus


class ChatCodec:
    """Exact tokenizer only. Same renderer is used at runtime and training export."""
    exact = True

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, messages):
        # Transformers 5 may default to BatchEncoding; list(mapping) is NOT token IDs.
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, return_dict=False)
        if not isinstance(ids, (list, tuple)) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('chat_template_must_return_flat_integer_token_ids')
        return list(ids)

    def target(self, text):
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if self.tokenizer.eos_token_id is None:
            raise ValueError("missing_eos_token")
        return ids + [self.tokenizer.eos_token_id]


class OfflineCodec:
    """Byte encoding for fixtures ONLY; cannot produce trainable token supervision."""
    exact = False

    def encode(self, messages):
        return list(canonical(messages).encode())

    def target(self, text):
        return list(text.encode())


class ContextBuilder:
    def __init__(self, codec, context_tokens=10240, evidence_cards=True, safety_margin=0, fallback_rounds=3, compression_max_calls=12, final_context_tokens=None, state_context_tokens=None, repair_reserve_tokens=0, soft_context_tokens=None, hard_context_tokens=None):
        self.codec = codec
        self.limit = context_tokens
        if final_context_tokens is not None and (type(final_context_tokens) is not int or final_context_tokens < 1):
            raise ValueError('invalid_final_context_limit')
        self.final_limit = final_context_tokens
        if state_context_tokens is not None and (type(state_context_tokens) is not int or state_context_tokens < 1):
            raise ValueError('invalid_state_context_limit')
        if type(repair_reserve_tokens) is not int or not 0 <= repair_reserve_tokens <= 2048:
            raise ValueError('invalid_repair_reserve')
        self.state_limit = state_context_tokens
        if (soft_context_tokens is None) != (hard_context_tokens is None):
            raise ValueError('both_soft_and_hard_context_limits_required')
        if soft_context_tokens is not None and (
                type(soft_context_tokens) is not int or type(hard_context_tokens) is not int
                or not 0 < soft_context_tokens <= hard_context_tokens):
            raise ValueError('invalid_soft_hard_context_limits')
        self.soft_limit, self.hard_limit = soft_context_tokens, hard_context_tokens
        self._hard_candidates = None
        self.repair_reserve_tokens = repair_reserve_tokens
        self.evidence_cards = evidence_cards
        if type(safety_margin) is not int or safety_margin < 0 or type(fallback_rounds) is not int or not 1 <= fallback_rounds <= 5:
            raise ValueError("invalid_context_fallback_settings")
        self.safety_margin, self.fallback_rounds = safety_margin, fallback_rounds
        if type(compression_max_calls) is not int or not 1 <= compression_max_calls <= 64:
            raise ValueError('invalid_compression_call_limit')
        self.compression_max_calls=compression_max_calls
        self.reserves = {"checklist": 1200, "decision": 400, "state": 1200, "final": 2400}

    def limit_for(self, stage):
        if self.hard_limit is not None: return self.hard_limit
        if stage == 'state' and self.state_limit is not None: return self.state_limit
        return self.final_limit if stage == 'final' and self.final_limit is not None else self.limit

    @property
    def input_padding(self):
        return self.safety_margin + self.repair_reserve_tokens

    def repair(self, stage, base, error):
        """Bound only program validation feedback; never trim evidence or model output."""
        prefix = 'Previous output rejected; no action executed. Correct this error: '
        detail = str(error)[:1500]
        initial = len(base['prompt_token_ids'])
        allowance = self.limit_for(stage) - initial - base['max_new_tokens'] - self.safety_margin
        if self.repair_reserve_tokens:
            allowance = min(allowance, self.repair_reserve_tokens)
        while True:
            request = copy.deepcopy(base)
            request['messages'].append({'role':'user', 'content':prefix+detail})
            ids = self.codec.encode(request['messages'])
            if len(ids) - initial <= allowance:
                request.update(prompt_token_ids=ids, input_sha256=digest(request['messages']))
                request['context_audit'] = {**request.get('context_audit', {}),
                    'repair_feedback_tokens':len(ids)-initial,
                    'repair_feedback_bounded':detail != str(error)[:1500]}
                return request
            if not self.repair_reserve_tokens or len(detail) <= 16:
                exc = ContextOverflow('repair_context_overflow')
                exc.input_tokens, exc.output_reserve = len(ids), base['max_new_tokens']
                raise exc
            detail = detail[:max(16, len(detail)//2)]

    async def prepare(self, stage, state, selector=None, *, fallback=None):
        if self.hard_limit is None:
            return await self._prepare(stage, state, selector, fallback=fallback)
        # Isolated per-request copy: never mutate shared builder limits during
        # async compression, and never count floor/empty-text probes as inputs.
        soft = copy.copy(self)
        soft.limit = soft.final_limit = soft.state_limit = self.soft_limit
        soft.soft_limit = soft.hard_limit = None
        soft._hard_candidates = dict(limit=self.hard_limit, best=None)
        reason = None
        try:
            request = await soft._prepare(stage, state, selector, fallback=fallback)
        except ContextOverflow as exc:
            request = soft._hard_candidates['best']
            if request is None:
                exc.context_limit = self.hard_limit
                exc.soft_context_limit = self.soft_limit
                raise
            reason = str(exc)
        request['context_audit'] = {**request.get('context_audit', {}), 'budget_policy':dict(
            version='soft_hard_context_v1',soft_limit=self.soft_limit,hard_limit=self.hard_limit,
            soft_target_met=len(request['prompt_token_ids'])+request['max_new_tokens']+self.input_padding<=self.soft_limit,
            soft_failure_reason=reason, compression_calls=getattr(getattr(fallback,'budget',None),'calls',0))}
        assert len(request['prompt_token_ids'])+request['max_new_tokens']+self.input_padding<=self.hard_limit
        return request

    async def _prepare(self, stage, state, selector=None, *, fallback=None):
        limit = self.limit_for(stage)
        if stage == 'checklist':
            return self.build(stage,state,allow_cards=False)
        if stage == "decision":
            return await self.prepare_search(state, fallback)
        if fallback is not None:
            overrides = {}
            attempts = []
            originals = copy.deepcopy((state.get("prefinal") or {}).get("opened_evidence", {}) if stage == "final" else state.get("evidence", {}))
            focus = dict(question=state["question"], checklist=state["requirements"])
            last_error = None
            attempted = set()
            # Deterministic projection runs in build; model extraction is last resort.
            for round_index in range(self.fallback_rounds):
                mode,ratio='compress',.85**round_index
                try:
                    return self.build(stage, state, overrides=overrides, allow_cards=False, fallback_audit=attempts)
                except ContextOverflow as exc:
                    last_error = exc
                # Structured windows cannot be sentence-extracted. Their text
                # belongs to the mandatory floor, not the compressor allowance.
                empty = {eid: {**copy.deepcopy(chunk), "text": chunk["text"] if structured_table_window(chunk) else ""}
                         for eid,chunk in originals.items()}
                fixed = self._measure(stage, state, empty, [])
                available = max(0, limit - self.reserves[stage] - self.input_padding - fixed)
                if not available or not originals:
                    raise last_error
                protected={eid for a in state['assessments'].values() for eid in a.get('evidence_ids',[])}
                rows = sorted(originals, key=lambda eid:(eid in protected,
                              -text_tokens(self.codec,overrides.get(eid,originals[eid])["text"])))
                for eid in rows:
                    if structured_table_window(originals[eid]):
                        continue
                    if getattr(getattr(fallback,'budget',None),'exhausted',False):raise last_error
                    # Always return to original text, never compress a previous excerpt.
                    current_input = self._measure(stage, state, overrides, attempts)
                    deficit = max(1, current_input + self.reserves[stage] + self.input_padding - limit)
                    # Text and full JSON rendering do not have additive token
                    # costs. Reserve additional shrinkage, not less safety.
                    deficit += min(96, max(1,available//8))
                    visible_chunk = overrides.get(eid, originals[eid])
                    target=excerpt_target(self.codec,originals[eid]['text'],current_text=visible_chunk['text'],
                                          available=available,deficit=deficit,round_index=round_index,dependency_aware=True)
                    if target is None:continue
                    attempt_key = (eid,target,digest(visible_chunk['text']))
                    if attempt_key in attempted:continue
                    attempted.add(attempt_key)
                    projected = await fallback.transform(originals[eid], focus, target, mode)
                    attempts.append(dict(evidence_id=eid, mode=mode, ratio=ratio, proposed=projected is not None))
                    if projected is None:
                        continue
                    trial = {**overrides, eid: projected}
                    # Accept only a decrease in the full rendered request, including provenance.
                    before = self._measure(stage, state, overrides, attempts)
                    after = self._measure(stage, state, trial, attempts)
                    if after < before:
                        overrides = trial
                    try:
                        return self.build(stage, state, overrides=overrides, allow_cards=False, fallback_audit=attempts)
                    except ContextOverflow as exc:
                        last_error = exc
            raise last_error
        cards = {}
        while True:
            try:
                return self.build(stage, state, cards=cards)
            except CardSelectionRequired as need:
                if selector is None:
                    raise ContextOverflow("semantic_compression_provider_required") from None
                try:
                    cards[need.key] = await selector.select(need.chunk, need.focus)
                except ValueError:
                    # Preserve raw on extraction/format failure; try other old chunks. No lexical fallback.
                    cards[need.key] = None

    async def prepare_search(self, state, fallback):
        from .search import clean_candidates
        try:
            return self.build("decision", state, allow_cards=False)
        except ContextOverflow as exc:
            last_error=exc
        originals=copy.deepcopy(state.get("candidates", []))
        projected,audit=clean_candidates(originals)
        if fallback is not None:
            fallback.store.event(fallback.run_id,"search_cleanup",dict(original_candidates=originals,projected=projected,audit=audit))
        try:
            return self.build("decision",state,allow_cards=False,candidate_overrides=projected)
        except ContextOverflow as exc:
            last_error=exc
        evidence = decision_observations(state)[2]
        empty={c["source_id"]:{"preview":""} for c in originals}
        empty_evidence={eid:{**copy.deepcopy(chunk),"text":chunk["text"] if structured_table_window(chunk) else ""}
                        for eid,chunk in evidence.items()}
        floor=self.search_measure(state,empty,empty_evidence)
        available=self.limit-self.reserves["decision"]-self.input_padding-floor-192
        def fail(reason, candidates, chunks, attempts):
            measured=self.search_measure(state,candidates,chunks,attempts)
            detail=dict(stage='decision',reason=reason,input_tokens=measured,
                        output_reserve=self.reserves['decision'],safety_margin=self.safety_margin,
                        context_limit=self.limit,fixed_floor_input_tokens=floor,
                        repair_reserve_tokens=self.repair_reserve_tokens,
                        remaining_deficit=max(0,measured+self.reserves['decision']+self.input_padding-self.limit),
                        compression_calls=getattr(getattr(fallback,'budget',None),'calls',0),
                        compression_max_calls=getattr(getattr(fallback,'budget',None),'max_calls',0),
                        candidate_preview_tokens=sum(text_tokens(self.codec,candidates.get(c['source_id'],c)['preview']) for c in originals),
                        latest_browse_text_tokens=sum(text_tokens(self.codec,chunks.get(eid,c)['text']) for eid,c in evidence.items()))
            if fallback is not None:
                fallback.store.event(fallback.run_id,'context_soft_capacity_failure' if self._hard_candidates is not None else 'context_capacity_failure',detail)
            last_error.input_tokens=measured
            last_error.capacity_detail=detail
            last_error.args=('context_budget_exceeded:'+reason,)
            raise last_error
        if available<=0 or not (originals or evidence) or fallback is None:
            fail('fixed_floor_or_provider_unavailable',projected,{},[])
        overrides,attempts={},[]
        focus=dict(question=state["question"],checklist=state["requirements"],
                   current_gaps={rid:a["gap"] for rid,a in state["assessments"].items() if a["support"]!="direct" or a.get("gap") or a.get("conflict")})
        resources=[("candidate",c["source_id"],c) for c in originals]+[("evidence",eid,c) for eid,c in evidence.items()]
        resources.sort(key=lambda row:-len(row[2]["preview" if row[0]=="candidate" else "text"]))
        attempted=set()
        for round_index in range(self.fallback_rounds):
            mode,ratio='compress',.85**round_index
            def visible_cost(row):
                kind,key,original=row
                visible=projected.get(key,original) if kind=='candidate' else overrides.get(key,original)
                return text_tokens(self.codec,visible['preview' if kind=='candidate' else 'text'])
            # Recompute from the current view, not the original lengths. Empty
            # previews are finished, and Search/Browse share one text allowance.
            if round_index:
                resources.sort(key=lambda row:-visible_cost(row))
            for kind,key,original in resources:
                if kind == 'evidence' and structured_table_window(original):
                    continue
                if getattr(getattr(fallback,'budget',None),'exhausted',False):
                    fail('compression_call_budget_exhausted',projected,overrides,attempts)
                before=self.search_measure(state,projected,overrides,attempts)
                deficit=max(1,before+self.reserves["decision"]+self.input_padding-self.limit)
                visible=projected.get(key,original) if kind=="candidate" else overrides.get(key,original)
                field="preview" if kind=="candidate" else "text"
                target=excerpt_target(self.codec,original[field],current_text=visible[field],available=available,
                                      deficit=deficit,round_index=round_index,dependency_aware=kind=='evidence')
                if target is None:continue
                if not visible[field]:continue
                if round_index:
                    # Tighten against the allowance for ALL bodies before using
                    # remaining calls; the first pass stays gentle. Never pick
                    # arbitrary text to meet a quota: the selector may decline.
                    total=sum(visible_cost(row) for row in resources)
                    share=max(1,int(available*text_tokens(self.codec,visible[field])/max(1,total)*.8))
                    target=min(target,share)
                attempt_key=(kind,key,target)
                if attempt_key in attempted:continue
                attempted.add(attempt_key)
                if kind=="candidate":
                    value=(await fallback.project_candidates([original],focus,{key:target})).get(key)
                    trial_candidates={**projected,key:value} if value is not None else projected
                    trial_evidence=overrides
                else:
                    value=await fallback.transform(original,focus,target,mode)
                    trial_candidates=projected
                    trial_evidence={**overrides,key:value} if value is not None else overrides
                attempts.append(dict(kind=kind,id=key,mode=mode,ratio=ratio,proposed=value is not None))
                # Both previews and latest Browse use the same full-request budget.
                if value is not None and self.search_measure(state,trial_candidates,trial_evidence,attempts)<self.search_measure(state,projected,overrides,attempts):
                    projected,overrides=trial_candidates,trial_evidence
                try:
                    return self.build("decision",state,allow_cards=False,candidate_overrides=projected,
                                      overrides=overrides,fallback_audit=attempts)
                except ContextOverflow as exc:
                    last_error=exc
        fail('no_safe_reduction_within_rounds',projected,overrides,attempts)

    def search_measure(self,state,overrides,evidence_overrides=None,attempts=None):
        try:return len(self.build("decision",state,allow_cards=False,candidate_overrides=overrides,
                                  overrides=evidence_overrides,fallback_audit=attempts,capture_hard=False)["prompt_token_ids"])
        except ContextOverflow as exc:return exc.input_tokens

    def _measure(self, stage, state, overrides, attempts):
        try:
            return len(self.build(stage, state, overrides=overrides, allow_cards=False,
                                  fallback_audit=attempts,capture_hard=False)["prompt_token_ids"])
        except ContextOverflow as exc:
            return exc.input_tokens

    def build(self, stage, state, *, cards=None, overrides=None, allow_cards=True, fallback_audit=None, candidate_overrides=None, capture_hard=True):
        limit = self.limit_for(stage)
        if stage == "checklist":
            payload = {"question": state["question"]}
        elif stage == "final":
            payload = copy.deepcopy(state["prefinal"])
        else:
            payload = {"question": state["question"], "checklist": state["requirements"],
                       "state": state["assessments"], "opened_evidence": copy.deepcopy(state["evidence"]),
                       "history": copy.deepcopy(state["history"]), "feedback": copy.deepcopy(state["feedback"]),
                       "budget": state["budget"], "blocked_sources": state.get("blocked_sources", {})}
            if stage == "decision":
                target,latest_browse,latest_evidence = decision_observations(state)
                payload["opened_evidence"] = latest_evidence
                payload["current_target"] = target
                payload["latest_browse"] = latest_browse
                payload["context_role"] = "next_action_with_latest_browse"
                payload["opened_evidence_index"] = {eid:{k:chunk[k] for k in ("source_id","chunk_id","title") if k in chunk}
                                                     for eid,chunk in state["evidence"].items()}
                payload["candidates"] = copy.deepcopy(state["candidates"])

                payload["source_headers"] = copy.deepcopy(state.get("source_headers",{}))
            if stage == "state":
                payload["new_evidence_ids"] = state["new_evidence_ids"]
        removed = 0
        raw_payload = copy.deepcopy(payload)
        # Rejection counts bound execution, not the model's semantic decision.
        # Preserve them in raw audit/session/checkpoints, never in policy input.
        if stage == "decision":
            raw_payload["decision_budget"] = copy.deepcopy(state.get("decision_budget", {}))
        for feedback_key in ("feedback", "decision_failure"):
            if isinstance(payload.get(feedback_key), dict):
                payload[feedback_key].pop("decision_budget", None)
        raw_payload["source_aliases"] = register_aliases(copy.deepcopy(state))
        # Action-replay signatures are program identity, not medical evidence.
        # Keep the exact hashes in raw_context/audit, not repeated in LLM history.
        internal_feedback = {}
        for index, row in enumerate(payload.get('history', payload.get('compact_history', []))):
            feedback = row.get('feedback', {}) if isinstance(row, dict) else {}
            if 'browse_context_sha256' in feedback:
                internal_feedback[str(index)] = feedback.pop('browse_context_sha256')
        if 'browse_context_sha256' in payload.get('feedback', {}):
            internal_feedback['current'] = payload['feedback'].pop('browse_context_sha256')
        final_pack_audit = None
        if stage in {'final', 'state'}:
            packed, final_pack_audit = (pack_final if stage == 'final' else pack_state)(payload)
            # Reference factoring must reduce exact rendered tokens, not only characters.
            def cost(value):
                return len(self.codec.encode([{'role':'system','content':PROMPTS[stage]},
                                              {'role':'user','content':canonical(value)}]))
            if final_pack_audit.get('applied') and cost(packed) < cost(payload):
                payload = packed
            else:
                final_pack_audit = {'version':stage+'_pack_v1', 'applied':False}
        if stage == "decision":
            from ..contracts import source_script_hint
            for row,original in zip(payload["candidates"],state["candidates"]):
                row["source_script_hint"] = source_script_hint(original)
        if candidate_overrides:
            if stage != "decision" or set(candidate_overrides)-{c["source_id"] for c in state["candidates"]}:
                raise ValueError("invalid_candidate_projection_scope")
            for row in payload["candidates"]:
                projection=candidate_overrides.get(row["source_id"])
                if projection is not None:
                    if set(projection)-{"preview","preview_projection"} or not isinstance(projection.get("preview"),str):
                        raise ValueError("candidate_projection_identity_changed")
                    row.update(copy.deepcopy(projection))
            payload["candidate_preview_notice"] = "Some previews are cleaned or incomplete original excerpts; they are source-selection aids, not evidence. Omission does not establish absence. Full original Search receipts are archived."

        if "opened_evidence" in payload:
            payload["opened_evidence"], deterministic_audit = project_evidence(payload["opened_evidence"])
        else:
            deterministic_audit = {}
        if overrides:
            if set(overrides)-set(raw_payload.get("opened_evidence", {})):
                raise ValueError("invalid_evidence_projection_scope")
            # A referenced witness cannot be changed while its dependents still point to raw offsets.
            for eid, chunk in payload.get("opened_evidence", {}).items():
                if chunk.get("context_reference", {}).get("evidence_id") in overrides:
                    chunk.pop("context_reference")
                    chunk["text"] = raw_payload["opened_evidence"][eid]["text"]
                    deterministic_audit["exact_duplicate_references"].pop(eid, None)
            for eid, value in overrides.items():
                original = raw_payload["opened_evidence"][eid]
                if any(value.get(k) != v for k,v in original.items() if k != "text"):
                    raise ValueError("scoped_projection_identity_changed")
                if "evidence_card" in value:
                    validate_card(value, original)
                # Apply the same receipt projection as normal requests and
                # floor measurements. Overrides must never restore raw maps.
                item = visible_evidence_chunk(value)
                # Full spans/hashes remain in scoped_compression events, not repeated in policy input.
                if "evidence_card" in item:
                    item["evidence_card"] = {"coverage_complete": False, "semantic_verified": False}
                if "model_cleanup" in item:
                    item["model_cleanup"] = {"semantic_verified": False, "mode": "program_validated_line_deletion"}
                payload["opened_evidence"][eid] = item
            payload["scoped_compression_notice"] = "Only designated evidence text is projected. Cards are incomplete original excerpts, not full source or absence claims. History, State and IDs are unchanged."
        payload = project_aliases(payload, state)
        projected = bool(deterministic_audit.get("provenance_omitted_ids") or
                         deterministic_audit.get("exact_duplicate_references"))
        if projected:
            payload["deterministic_projection_notice"] = DETERMINISTIC_NOTICE
        evidence = payload.get("opened_evidence", {})
        card_originals = copy.deepcopy(evidence)
        protected = set(state.get("new_evidence_ids", []))
        cited = {eid for assessment in state["assessments"].values() for eid in assessment["evidence_ids"]}
        conflict = {eid for assessment in state["assessments"].values() if assessment.get("conflict")
                    for eid in assessment["evidence_ids"]}
        # Uncited old evidence first; support/conflict evidence last. Latest Browse stays raw.
        order = sorted((eid for eid in evidence if eid not in protected and not evidence[eid].get("context_reference") and eid not in {c.get("context_reference", {}).get("evidence_id") for c in evidence.values()}),
                       key=lambda eid: (eid in conflict, eid in cited, -len(evidence[eid]["text"]), eid))
        focus = dict(question=state["question"], checklist=state["requirements"], state=state["assessments"])
        compressed = list(overrides or {})
        while True:
            payload["context_projection"] = {"history_events_omitted": removed,
                                              "evidence_compressed": bool(compressed), "version": VERSION,
                                              "compressed_evidence_ids": compressed,
                                              "raw_payload_sha256": digest(raw_payload),
                                              "deterministic": deterministic_audit,
                                              "scoped_fallback": {"attempts": len(fallback_audit or [])},
                                              "safety_margin": self.safety_margin, "source_aliases": True,
                                              "source_selection_isolated": False,
                                              "decision_latest_browse_only": stage == "decision",
                                              "candidate_projected_ids": list(candidate_overrides or {})}
            messages = [{"role": "system", "content": PROMPTS[stage]},
                        {"role": "user", "content": canonical(payload)}]
            ids = self.codec.encode(messages)
            # Deterministic final packing BEFORE model extraction: the verbose
            # program-owned audit is kept in context_audit/events, not repeated
            # in a model request that otherwise exceeds capacity. No question,
            # history, State, evidence, IDs or clinical metadata are shortened.
            model_projection = copy.deepcopy(payload["context_projection"])
            if internal_feedback:
                model_projection['internal_feedback_hashes_archived'] = internal_feedback
            if final_pack_audit is not None:
                model_projection[stage+'_pack'] = final_pack_audit
            if self.repair_reserve_tokens:
                model_projection['repair_reserve_tokens'] = self.repair_reserve_tokens
            if len(ids) + self.reserves[stage] + self.input_padding > limit:
                compact = copy.deepcopy(payload)
                compact["context_projection"] = {
                    k: model_projection[k] for k in (
                        "version", "history_events_omitted", "evidence_compressed",
                        "compressed_evidence_ids", "decision_latest_browse_only")}
                compact["context_projection"]["audit_archived"] = True
                compact_messages = [{"role":"system","content":PROMPTS[stage]},
                                    {"role":"user","content":canonical(compact)}]
                compact_ids = self.codec.encode(compact_messages)
                if len(compact_ids) < len(ids):
                    messages,ids = compact_messages,compact_ids
                    model_projection["model_visible_audit_compacted"] = True
            request = {"messages": messages, "prompt_token_ids": ids, "exact_tokenizer": self.codec.exact,
                        "max_new_tokens": self.reserves[stage], "input_sha256": digest(messages),
                        "context_audit": model_projection,
                        "raw_context_payload": raw_payload}
            if capture_hard and self._hard_candidates is not None:
                target = self._hard_candidates
                best = target['best']
                if len(ids)+self.reserves[stage]+self.input_padding<=target['limit'] and (
                        best is None or len(ids)<len(best['prompt_token_ids'])):
                    target['best'] = copy.deepcopy(request)
            if len(ids) + self.reserves[stage] + self.input_padding <= limit:
                return request
            if allow_cards and self.evidence_cards and order:
                eid = order.pop(0)
                key = digest([VERSION, evidence[eid], focus])
                if cards is None or key not in cards:
                    raise CardSelectionRequired(key, evidence[eid], focus)
                proposed = cards[key]
                if proposed is None:
                    continue
                validate_card(proposed, card_originals[eid])
                # Accept only a reduction in the exact full rendered prompt, including metadata.
                original = evidence[eid]
                old_notice = payload.get("evidence_card_notice")
                evidence[eid] = proposed
                payload["evidence_card_notice"] = NOTICE
                compressed.append(eid)
                payload["context_projection"].update(evidence_compressed=True, compressed_evidence_ids=compressed)
                projected = [{"role": "system", "content": PROMPTS[stage]},
                             {"role": "user", "content": canonical(payload)}]
                if len(self.codec.encode(projected)) >= len(ids):
                    evidence[eid] = original
                    compressed.pop()
                    if old_notice is None:
                        payload.pop("evidence_card_notice", None)
                    else:
                        payload["evidence_card_notice"] = old_notice
            else:
                # IDs, latest Browse, State, candidate order and recent history are never silently dropped.
                error = ContextOverflow("context_budget_exceeded_without_safe_evidence_compression")
                error.input_tokens = len(ids)
                error.output_reserve = self.reserves[stage]
                error.context_limit = limit
                raise error
