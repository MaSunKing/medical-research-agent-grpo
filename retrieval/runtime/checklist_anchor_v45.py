"""Extractive task definitions; coverage checks are not semantic validation."""
import json

INSTRUCTION=('Select 1-8 answerable task items from the original question. Default to one item '
 'containing the whole question, especially for short or single questions. Split only explicitly '
 'distinct requests. Return {"anchors":[["exact contiguous quote from question"]]}. '
 'An item may combine multiple exact quotes, in original order; shared context may repeat. '
 'Cover all question wording across items, keeping shared scope. Do not paraphrase, add '
 'conditions, answers or optional tasks. Code assigns IDs and freezes the text.')

def assemble(question,value):
    if not isinstance(value,dict) or set(value)!={'anchors'}:raise ValueError('only anchors allowed')
    groups=value['anchors']
    if not isinstance(groups,list) or not 1<=len(groups)<=8:raise ValueError('1-8 items')
    covered=set();seen=set();items=[];provenance={}
    for quotes in groups:
        if not isinstance(quotes,list) or not quotes:raise ValueError('nonempty quotes')
        end=0;spans=[]
        for quote in quotes:
            if not isinstance(quote,str) or not quote.strip():raise ValueError('nonempty exact quote')
            start=question.find(quote,end)
            if start<0:raise ValueError('quote absent or out of original order')
            end=start+len(quote);spans.append([start,end]);covered.update(range(start,end))
        text=' … '.join(quotes)
        if text in seen:raise ValueError('duplicate item')
        seen.add(text)
        items.append(dict(id=f'R{len(items)+1}',description=text,status='unknown',evidence_ids=[]))
        provenance[items[-1]['id']]=spans
    if any(c.isalnum() and i not in covered for i,c in enumerate(question)):
        raise ValueError('question wording omitted; use full question if unsure')
    return dict(schema_version='medgap_v71_policy_evidence_state_v1',requirements=items,optional_requirements=[],
                partition_schema='public_core_optional_v1',core_source_spans=provenance,
                core_wording_verified=True,wording_diagnostics=[],semantic_completeness_verified=False)

def initialize(policy,question,seed):
    payload=dict(question=question,anchor_initialization=True)
    request=INSTRUCTION+'\n\n'+json.dumps(payload,ensure_ascii=False)
    for attempt in range(2):
        raw=policy.choices(messages=[dict(role='user',content=request)],n=1,temperature=.1,max_tokens=1200,json_mode=True,seed=(seed or 0)+attempt)[0]
        try:state=assemble(question,json.loads(raw))
        except (ValueError,TypeError,KeyError) as exc:
            print('V45_ANCHOR_REJECT='+str(exc),flush=True)
            request=INSTRUCTION+'\n\n'+json.dumps(payload,ensure_ascii=False)+'\nFormat feedback: '+str(exc)
        else:
            print('V45_CHECKLIST_FROZEN='+json.dumps(state,ensure_ascii=False),flush=True)
            return [(raw,state)]
    state=assemble(question,{'anchors':[[question]]})
    print('V45_CHECKLIST_FALLBACK=full_original_question',flush=True)
    return [(raw,state)]

def install(common):
    previous=common.sample_state_updates
    def sample(policy,prompt,*,k,v71,seed=None,**kwargs):
        payload=json.JSONDecoder().raw_decode(prompt.split('\n\n',1)[1])[0]
        if 'policy_evidence_state' not in payload and 'opened_evidence' not in payload:
            if k!=1:raise ValueError('one checklist')
            return initialize(policy,payload['question'],seed)
        return previous(policy,prompt,k=k,v71=v71,seed=seed,**kwargs)
    common.sample_state_updates=sample
