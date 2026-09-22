"""Budget the actual shared stage, using explicit structured evidence paths.

No changes to runtime source registry, original question, or frozen checklist.
Only over-budget model-visible copies are compressed; originals are archived.
"""
import copy, hashlib, json, re
from pathlib import Path
from context_compat_v24 import render_input
from evidence_cards_v32 import SPAN_PROMPT, CARD_VERSION, passages, resolve, validate, ordered_quotes

WARNING=('Evidence cards contain selected verbatim excerpts from Browse output, not full documents. '
         'Omitted information is not absent information. Do not infer complete coverage from a card.')
CARD_VISIBLE_TOKEN_LIMIT = 240
SEPARATOR = '\n[Separate original excerpt; intervening text may be omitted]\n'
DECISION_RECEIPT_VERSION = 'decision_evidence_receipt_v1'
FINAL_RECEIPT_VERSION = 'final_evidence_receipt_v1'


def sentence_ends(text):
    """Conservative sentence ends for verbatim card budgeting."""
    ends = []
    for index, char in enumerate(text):
        if char not in '.!?。！？':
            continue
        before = text[index - 1] if index else ''
        after = text[index + 1] if index + 1 < len(text) else ''
        if char == '.' and before.isdigit() and after.isdigit():
            continue
        if char == '.' and re.search(
                r'\b(?:e\.g|i\.e|Dr|Mr|Ms|Prof|vs|et al)\.$', text[:index + 1], re.I):
            continue
        if after and not after.isspace():
            continue
        ends.append(index + 1)
    return ends


def compact_rejected_feedback(feedback):
    """Remove rejected model text from an over-budget model-visible copy.

    Rejected public output is correction/audit data, not an executed action or
    evidence.  Keep its identity, size, validation reason and guidance while
    the exact original remains in the archived raw stage.
    """
    result = copy.deepcopy(feedback)
    compacted = 0
    for row in result if isinstance(result, list) else []:
        if not isinstance(row, dict):
            continue
        value = row.get('runtime_feedback')
        prefix = 'VALIDATION_FEEDBACK='
        if not isinstance(value, str) or not value.startswith(prefix):
            continue
        try:
            payload = json.loads(value[len(prefix):])
        except json.JSONDecodeError:
            continue
        for rejection in payload.get('rejections', []):
            raw = rejection.get('rejected_public_output')
            if not isinstance(raw, str) or not raw:
                continue
            rejection['rejected_public_output'] = (
                '[omitted_from_budgeted_copy '
                f'sha256={hashlib.sha256(raw.encode()).hexdigest()} chars={len(raw)}]'
            )
            rejection['public_output_omitted_from_budgeted_copy'] = True
            compacted += 1
        row['runtime_feedback'] = prefix + json.dumps(
            payload, ensure_ascii=False, separators=(',', ':'))
    return result, compacted


def bounded_card_quotes(card, source_text, tokenizer, max_tokens=CARD_VISIBLE_TOKEN_LIMIT):
    """Select whole, ordered sentence spans from grounded card quotes."""
    quotes = sorted(set(f['quote'] for f in card['facts']), key=source_text.index)
    kept = []
    for quote in quotes:
        ends = sentence_ends(quote)
        starts = [0] + ends[:-1]
        spans = [quote[start:end] for start, end in zip(starts, ends)]
        if not ends or ends[-1] < len(quote):
            spans.append(quote[ends[-1] if ends else 0:])
        for span in spans:
            if not span:
                continue
            candidate = SEPARATOR.join(kept + [span])
            if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens:
                kept.append(span)
    return SEPARATOR.join(kept)


def bounded_existing_card_text(text, tokenizer, max_tokens=CARD_VISIBLE_TOKEN_LIMIT):
    """Bound an already-grounded card without model re-summarization."""
    if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
        return text
    ends = sentence_ends(text)
    starts = [0] + ends[:-1]
    spans = [text[start:end] for start, end in zip(starts, ends)]
    if not ends or ends[-1] < len(text):
        spans.append(text[ends[-1] if ends else 0:])
    kept = []
    for span in spans:
        if not span:
            continue
        candidate = SEPARATOR.join(kept + [span])
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens:
            kept.append(span)
    return SEPARATOR.join(kept)


def decision_evidence_receipt(chunk):
    """Retain provenance while omitting older text from an over-budget Decision.

    This is a model-visible Decision copy only.  The immutable runtime state,
    State input and Final input keep the complete Browse chunk.  A receipt is
    used only after ordinary verbatim cards still do not fit, and records the
    exact omitted text identity instead of pretending that omitted content is
    negative evidence.
    """
    text = chunk.get('text', '')
    if not isinstance(text, str):
        raise ValueError('Browse chunk text must be text')
    digest = hashlib.sha256(text.encode()).hexdigest()
    # Do not deepcopy the full parser/ranking payload: that defeats the purpose
    # of a bounded Decision receipt.  Keep only model-useful identity and the
    # structure gates; all other metadata remains in the canonical ledger.
    keep = (
        'source_id', 'title', 'heading', 'url', 'source_type',
        'structure_kind', 'boundary_incomplete', 'table_integrity_verified',
    )
    result = {key: copy.deepcopy(chunk[key]) for key in keep if key in chunk}
    structure_audit = chunk.get('structure_audit')
    structure_audit_digest = None
    if structure_audit is not None:
        structure_audit_digest = hashlib.sha256(json.dumps(
            structure_audit, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
    result['text'] = (
        '[Decision-only evidence receipt; full Browse text remains available '
        f'to State/Final. sha256={digest} chars={len(text)}]'
    )
    result['decision_budget_receipt'] = {
        'version': DECISION_RECEIPT_VERSION,
        'content_sha256': digest,
        'original_chars': len(text),
        'scope': 'decision_model_visible_copy_only',
        'full_text_retained_for_state_and_final': True,
        'coverage_complete': False,
        'omission_is_not_absence': True,
    }
    if structure_audit_digest is not None:
        result['decision_budget_receipt'][
            'structure_audit_sha256'
        ] = structure_audit_digest
    return result


def final_evidence_receipt(chunk):
    """Bound a Final-visible copy without losing the immutable evidence ledger.

    This fallback is reached only after ordinary verbatim evidence cards still
    exceed the shared window.  The receipt is provenance, not evidence text:
    it is explicitly non-citable and must not be treated as negative evidence.
    Exact Browse text remains in the archived stage and canonical trajectory.
    """
    text = chunk.get('text', '')
    if not isinstance(text, str):
        raise ValueError('Browse chunk text must be text')
    digest = hashlib.sha256(text.encode()).hexdigest()
    # Keep this receipt deliberately small.  Human-facing title/URL/heading and
    # the exact text remain in the canonical trajectory; repeating them in a
    # dozen Final receipts would crowd out the State-bound evidence that the
    # model actually needs to answer.
    keep = (
        'source_id', 'structure_kind', 'boundary_incomplete',
        'table_integrity_verified',
    )
    result = {key: copy.deepcopy(chunk[key]) for key in keep if key in chunk}
    structure_audit = chunk.get('structure_audit')
    structure_audit_digest = None
    if structure_audit is not None:
        structure_audit_digest = hashlib.sha256(json.dumps(
            structure_audit, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
    result['text'] = (
        '[Final-budget evidence receipt; exact Browse text is omitted from '
        f'this model-visible copy. sha256={digest} chars={len(text)}. '
        'This receipt is not citable evidence and omission is not absence.]'
    )
    result['final_budget_receipt'] = {
        'version': FINAL_RECEIPT_VERSION,
        'content_sha256': digest,
        'original_chars': len(text),
        'scope': 'final_model_visible_copy_only',
        'exact_text_retained_in_canonical_trajectory': True,
        'citable': False,
        'coverage_complete': False,
        'omission_is_not_absence': True,
    }
    if structure_audit_digest is not None:
        result['final_budget_receipt'][
            'structure_audit_sha256'
        ] = structure_audit_digest
    return result

def card_for(chunk,question,*,tokenizer,cache,generate,context_limit):
    text=chunk.get('text','')
    if not isinstance(text,str):raise ValueError('Browse chunk text must be text')
    # A complete table must remain whole or be omitted by the existing table
    # budget gate. Never turn rows into a prose-like partial evidence card.
    if chunk.get('structure_kind') == 'table_like':
        return copy.deepcopy(chunk)
    if chunk.get('model_extraction'):
        # Never summarize a summary with another model call. Cards cached by an
        # earlier turn can predate the visible-token cap, so deterministically
        # retain only ordered whole sentences in the Decision copy.
        result = copy.deepcopy(chunk)
        bounded = bounded_existing_card_text(text, tokenizer)
        if bounded:
            result['text'] = bounded
            result['model_extraction'] = dict(result['model_extraction'])
            result['model_extraction'].update(
                verbatim_card_token_limit=CARD_VISIBLE_TOKEN_LIMIT,
                whole_sentence_spans_only=True,
                recursively_summarized=False,
            )
        return result
    source={'question':question,'source_id':chunk['source_id'],'text':text}
    key=hashlib.sha256((CARD_VERSION+SPAN_PROMPT+json.dumps(source,sort_keys=True)).encode()).hexdigest()
    path=cache/(key+'.card.json')
    if path.exists():
        card=validate(json.dumps(json.loads(path.read_text(encoding='utf8'))['card']),text)
    else:
        spans=passages(text)
        actual={'question':question,'source_id':chunk['source_id'],'browse_passages':spans}
        message=SPAN_PROMPT+'\nActual input:\n'+json.dumps(actual,ensure_ascii=False)
        attempts=[]
        for attempt in range(2):
            request=[{'role':'user','content':message}]
            n=len(tokenizer.encode(render_input(tokenizer,request,None,{'enable_thinking':False}),add_special_tokens=False))
            if n+600>context_limit:
                raise ValueError('single Browse chunk exceeds card extraction budget; original archived')
            response=generate(request,max_tokens=600,temperature=.1,json_mode=True,seed=20260905+attempt)
            try:
                card=resolve(response['completion'],spans,text)
                break
            except (ValueError,TypeError,KeyError) as exc:
                attempts.append({'error':str(exc),'raw_completion':response['completion']})
                (cache/(key+'.rejected.json')).write_text(json.dumps(attempts,ensure_ascii=False),encoding='utf8')
                message=SPAN_PROMPT+'\nActual input:\n'+json.dumps(actual,ensure_ascii=False)+'\nFormat feedback: '+str(exc)
        else:
            raise ValueError('card extraction invalid after bounded retry; original archived')
        path.write_text(json.dumps({'card':card,'source_id':chunk['source_id'],'semantic_verified':False},ensure_ascii=False),encoding='utf8')
    result=copy.deepcopy(chunk)
    # Avoid retaining a second copy of the same raw text under preview aliases.
    for key in ('snippet','abstract','content','search_preview'):
        result.pop(key,None)
    result['text']=bounded_card_quotes(card,text,tokenizer)
    if not result['text']:
        return copy.deepcopy(chunk)
    result['model_extraction']={'card_version':CARD_VERSION,'summary_withheld':'semantic_unverified_draft_in_cache',
                               'scope':'Browse_returned_content_only','coverage_complete':False,
                               'verbatim_card_token_limit':CARD_VISIBLE_TOKEN_LIMIT,
                               'whole_sentence_spans_only':True}
    return result

def prepare(context,*,render,tokenizer,max_tokens,cache,generate,context_limit=8192):
    """render(context) returns (stage_object, exact_rendered_text)."""
    value=copy.deepcopy(context)
    def measured():
        obj,text=render(value)
        return obj,text,len(tokenizer.encode(text,add_special_tokens=False))
    obj,text,n=measured()
    audit={'mode':'raw','input_tokens':n,'reserved_output_tokens':max_tokens,'context_limit':context_limit,
           'counted_actual_shared_input':True,'overflow_tokens':max(0,n+max_tokens-context_limit)}
    if not audit['overflow_tokens']:return obj,text,audit
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    original=json.dumps(context,ensure_ascii=False,sort_keys=True)
    digest=hashlib.sha256(original.encode()).hexdigest()
    archive=cache/(digest+'.raw_stage.json');archive.write_text(original,encoding='utf8')
    audit['raw_archive']=str(archive)
    kwargs=value['kwargs'];stage=value['stage']
    if stage in ('decision','decision_stop'):
        payload=kwargs['state']['current_view']
        # Search previews have already been budgeted at Search time. Preserve
        # candidates, their order, and feedback; only Browse text may use cards.
        audit['search_previews_preserved']=True
        kwargs['feedback'], compacted = compact_rejected_feedback(kwargs.get('feedback'))
        audit['rejected_public_outputs_compacted'] = compacted
    else:
        payload=kwargs.get('payload',{})
    chunks=payload.get('opened_evidence',[])
    if not isinstance(chunks,list):raise ValueError('structured opened_evidence must be a list')
    if not chunks:
        raise ValueError('fixed question/checklist/context exceeds budget with no compressible evidence; original archived')
    payload['opened_evidence']=[]
    _,_,fixed_tokens=measured()
    audit['fixed_tokens_after_feedback_compaction'] = fixed_tokens
    payload['opened_evidence']=chunks
    if fixed_tokens+max_tokens>context_limit:
        raise ValueError('fixed question/checklist exceeds budget even without evidence; originals preserved')
    payload['evidence_card_notice']=WARNING
    failures=[]
    # Compress longest chunks first and stop as soon as actual input fits.
    order=sorted(range(len(chunks)),key=lambda i:len(chunks[i].get('text','')),reverse=True)
    for i in order:
        try:
            proposed=card_for(chunks[i],value['question'],tokenizer=tokenizer,cache=cache,generate=generate,context_limit=context_limit)
        except ValueError as exc:
            # Retain the exact original on failure; try other chunks, never invent a card.
            failures.append({'source_id':chunks[i]['source_id'],'error':str(exc)})
            continue
        if len(tokenizer.encode(json.dumps(proposed),add_special_tokens=False)) >= len(tokenizer.encode(json.dumps(chunks[i]),add_special_tokens=False)):
            continue
        chunks[i]=proposed
        obj,text,n=measured()
        if n+max_tokens<=context_limit:break
    obj,text,n=measured()
    audit.update(mode='cards',final_input_tokens=n,card_failures=failures,
                 cards=sum('model_extraction' in c for c in chunks),semantic_verified=False)
    # A long trajectory can accumulate enough individually bounded cards that
    # their total still exceeds the Decision window.  Older evidence has
    # already been exposed to the State update, so progressively replace only
    # the Decision-visible copies with immutable identity receipts.  Prefer
    # retaining complete tables and the newest evidence text.  The canonical
    # opened_evidence ledger used by State/Final is never modified because
    # ``value`` is a deep copy of ``context``.
    receipts = []
    if n + max_tokens > context_limit and stage in ('decision', 'decision_stop'):
        receipt_order = (
            [i for i, chunk in enumerate(chunks)
             if chunk.get('structure_kind') != 'table_like']
            + [i for i, chunk in enumerate(chunks)
               if chunk.get('structure_kind') == 'table_like']
        )
        for i in receipt_order:
            proposed = decision_evidence_receipt(chunks[i])
            if len(tokenizer.encode(json.dumps(proposed), add_special_tokens=False)) >= len(
                    tokenizer.encode(json.dumps(chunks[i]), add_special_tokens=False)):
                continue
            chunks[i] = proposed
            receipts.append({
                'source_id': proposed.get('source_id'),
                'content_sha256': proposed['decision_budget_receipt']['content_sha256'],
                'original_chars': proposed['decision_budget_receipt']['original_chars'],
            })
            obj, text, n = measured()
            if n + max_tokens <= context_limit:
                break
        audit.update(
            mode='cards_and_decision_receipts',
            final_input_tokens=n,
            decision_receipts=receipts,
            decision_receipt_count=len(receipts),
            newest_evidence_preferred=True,
            state_final_full_evidence_unchanged=True,
        )
    # Final has a much larger output reserve than Decision.  With many opened
    # chunks, even individually bounded verbatim cards can exceed that input
    # allowance.  Keep the newest evidence cards verbatim and replace older
    # Final-visible copies with non-citable receipts until the measured request
    # fits.  Never receipt every chunk: at least the newest evidence text must
    # remain available to the model.  ``context`` is unchanged because all
    # edits above are made to its deep copy ``value``.
    final_receipts = []
    if n + max_tokens > context_limit and stage == 'final':
        state = payload.get('policy_evidence_state', {})
        bound_ids = {
            evidence_id
            for group in ('requirements', 'optional_requirements')
            for row in state.get(group, []) if isinstance(row, dict)
            for evidence_id in row.get('evidence_ids', [])
            if isinstance(evidence_id, str)
        }
        bound_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk.get('source_id') in bound_ids
        ]
        unbound_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk.get('source_id') not in bound_ids
        ]
        # Evidence explicitly bound by State is the strongest Final input.
        # Demote all unbound prose first, then unbound complete tables, and
        # only then older bound evidence while preserving at least one bound
        # chunk verbatim.  If State has no bindings, preserve the newest chunk.
        receipt_order = (
            [i for i in unbound_indices
             if chunks[i].get('structure_kind') != 'table_like']
            + [i for i in unbound_indices
               if chunks[i].get('structure_kind') == 'table_like']
            + [i for i in bound_indices[:-1]
               if chunks[i].get('structure_kind') != 'table_like']
            + [i for i in bound_indices[:-1]
               if chunks[i].get('structure_kind') == 'table_like']
        )
        if not bound_indices:
            receipt_order = [i for i in receipt_order if i != len(chunks) - 1]
        for i in receipt_order:
            proposed = final_evidence_receipt(chunks[i])
            if len(tokenizer.encode(json.dumps(proposed), add_special_tokens=False)) >= len(
                    tokenizer.encode(json.dumps(chunks[i]), add_special_tokens=False)):
                continue
            chunks[i] = proposed
            final_receipts.append({
                'source_id': proposed.get('source_id'),
                'content_sha256': proposed['final_budget_receipt']['content_sha256'],
                'original_chars': proposed['final_budget_receipt']['original_chars'],
                'citable': False,
            })
            obj, text, n = measured()
            if n + max_tokens <= context_limit:
                break
        audit.update(
            mode='cards_and_final_receipts',
            final_input_tokens=n,
            final_receipts=final_receipts,
            final_receipt_count=len(final_receipts),
            newest_evidence_preferred=True,
            state_bound_evidence_preferred=True,
            state_bound_evidence_ids=sorted(bound_ids),
            state_bound_full_evidence_retained=any(
                chunk.get('source_id') in bound_ids
                and 'final_budget_receipt' not in chunk
                for chunk in chunks
            ) if bound_ids else None,
            final_full_evidence_ids=[
                chunk.get('source_id') for chunk in chunks
                if 'final_budget_receipt' not in chunk
            ],
            at_least_one_full_evidence_retained=any(
                'final_budget_receipt' not in chunk for chunk in chunks
            ),
            canonical_evidence_ledger_unchanged=True,
        )
    (cache/(digest+'.budget.json')).write_text(json.dumps(audit,indent=2),encoding='utf8')
    if n+max_tokens>context_limit:
        raise ValueError('shared context still over budget after bounded compression; originals preserved: '+str(archive))
    return obj,text,audit
