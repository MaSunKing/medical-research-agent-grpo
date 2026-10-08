"""Sentence-context repair, retaining the trained anchors/State contract."""
import json,re
from checklist_examples_v56 import example,repair_example

INIT_PROMPT=(
    'Extract 1-8 requests whose evidence coverage can be assessed separately. '
    'Return {"anchors":[["exact source quote"]]}. '
    'Split independently answerable requests even within one sentence: one may be satisfied '
    'while another remains unresolved. Separate efficacy, safety, guideline recommendations, '
    'screening items or monitoring intervals only when the question actually requests them. '
    'Merge synonymous repetition; keep a genuinely single request intact. Do not split merely '
    'to increase the item count, or merge several independent requests into one whole-question item. '
    'Each item must express a complete, independently understandable request, not isolated keywords. '
    'Prefer a complete request sentence from the question; when its subject or scope is shared, '
    'combine exact source quotes to retain the necessary context without inventing wording. '
    'Attach shared population, intervention, comparator, negation, time, source and output '
    'constraints to each relevant item; do not make these disconnected tasks. '
    'Every string must be a contiguous, exact, whole-word quote from the question. '
    'Multiple quotes in one item must follow source order; shared quotes may repeat across items. '
    'Cover all question wording across items, preserving drug combinations and alternatives. '
    'Only conjunctions separating tasks may be omitted. Add no conditions, answers or extra tasks. '
    'Runtime assigns frozen IDs and descriptions. Output JSON only, without thinking or Markdown.'
)

def feedback(question,raw,error):
    covered=set()
    try:
        groups=json.loads(raw)['anchors']
        for fragments in groups:
            end=0
            for fragment in fragments:
                start=question.find(fragment,end)
                if start<0:continue
                end=start+len(fragment);covered.update(range(start,end))
    except (ValueError,KeyError,TypeError):pass
    sentences=[];start=0
    for match in re.finditer(r'[.!?。！？](?=\s|$)|$',question):
        end=match.end()
        if end>start and any(question[i].isalnum() and i not in covered for i in range(start,end)):
            sentences.append(question[start:end].strip())
        start=end
    return ('\nChecklist repair feedback (not evidence): '+json.dumps({
        'validation_error':str(error),'previous_output':raw,
        'original_sentences_containing_uncovered_text':sentences},ensure_ascii=False)+
        '\nReview these sentences in the full original question above. Preserve missing requirements '
        'with the related requests, or as a separate request when appropriate. Do not turn isolated '
        'missing words into tasks. Return the complete corrected anchors object, not a patch.'+repair_example())

def install(anchor,build):
    build.INIT_PROMPT=INIT_PROMPT+example()
    anchor.INSTRUCTION=build.INIT_PROMPT
    def initialize(policy,question,seed):
        payload={'question':question,'anchor_initialization':True}
        base=anchor.INSTRUCTION+'\n\n'+json.dumps(payload,ensure_ascii=False)
        request=base
        for attempt in range(2):
            raw=policy.choices(messages=[{'role':'user','content':request}],n=1,temperature=.1,
                               max_tokens=1200,json_mode=True,seed=(seed or 0)+attempt)[0]
            try:state=anchor.assemble(question,json.loads(raw))
            except (ValueError,TypeError,KeyError) as exc:
                print('V55_ANCHOR_REJECT='+str(exc),flush=True)
                request=base+feedback(question,raw,exc)
            else:
                print('V55_CHECKLIST_FROZEN='+json.dumps(state,ensure_ascii=False),flush=True)
                return [(raw,state)]
        state=anchor.assemble(question,{'anchors':[[question]]})
        print('V55_CHECKLIST_FALLBACK=full_original_question',flush=True)
        return [(raw,state)]
    anchor.initialize=initialize
