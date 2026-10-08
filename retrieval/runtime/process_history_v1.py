"""Shared offline/Runtime causal process history. No current target is accepted."""
import copy
import collections
import html
import json
import re

VERSION = 'causal_process_history_v4'
MAX_HISTORY_TOKENS = 512
TOOL = re.compile(r'<call_tool\s+name=[\"\']([^\"\']+)[\"\']([^>]*)>(.*?)</call_tool>',re.S)
OUTPUT = re.compile(r'<tool_output\b[^>]*>(.*?)</tool_output>',re.S)

def dumps(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))

def tools_visible_in(record):
    """Only inspect already-visible, structured history blocks, not source text."""
    result=[]
    for m in OUTPUT.finditer(record.get('assistant_prefix') or ''):
        try:value=json.loads(m[1])
        except ValueError:continue
        rows=value.get('history',[]) if isinstance(value,dict) else value if isinstance(value,list) else []
        for row in rows:
            if not isinstance(row,dict):continue
            public={k:copy.deepcopy(row[k]) for k in ('tool','argument','query','source_id','executed',
                'call_id','requested_tool','executed_tool','call_id_origin',
                'failed','environment_failure','failure_type','result_count','result_observability',
                'returned_ids','historical_returned_ids','historical_new_source_ids','observation') if k in row}
            # Observation summaries may carry identities/counts, never nested article bodies.
            if isinstance(public.get('observation'),dict):
                public['observation']={k:v for k,v in public['observation'].items() if k in
                    ('source_id','new_text_chunks','unchanged_text_chunks','no_new_text')}
            if public.get('tool'):
                result.append({'kind':'tool_observation',**public})
    return result

def summarize(past,current):
    """Past completion records only; current contains inputs but NEVER completion."""
    if 'completion' in current:
        raise ValueError('current target supplied to history generator')
    events=[];previous={};observed=[];seen_counts={}
    # Only real, already-visible validator feedback can establish rejection.
    # Absence of rejection is NOT an acceptance receipt.
    rejected={}
    marker='Checklist repair feedback (not evidence): '
    for visible in [*past,current]:
        for message in visible.get('messages',[]):
            if message.get('role')!='user':continue
            content=message.get('content','')
            if not isinstance(content,str):continue
            for match in re.finditer(re.escape(marker),content):
                try:feedback,_=json.JSONDecoder().raw_decode(content[match.end():])
                except ValueError:continue
                if (isinstance(feedback,dict) and isinstance(feedback.get('previous_output'),str)
                    and feedback.get('validation_error')):
                    rejected[feedback['previous_output']]=str(feedback['validation_error'])
    def observe(record):
        counts=collections.Counter()
        for event in tools_visible_in(record):
            key=dumps(event);counts[key]+=1
            if counts[key]>seen_counts.get(key,0):observed.append(event)
        for key,n in counts.items():seen_counts[key]=max(n,seen_counts.get(key,0))
    for index,row in enumerate(past):
        observe(row)
        kind=row['stage'];text=row.get('completion','')
        if not isinstance(text,str) or not text:continue
        if row.get('generation_error'):
            events.append({'kind':'stage_validation_failed','stage_index':index,
                'stage':kind,'validation_error':row['generation_error'],
                'supervision_eligible':False})
            continue
        if kind in ('decision','decision_stop'):
            public=re.sub(r'^\s*<think>.*?</think>\s*','',text,flags=re.S).strip()
            match=TOOL.fullmatch(public)
            if match:
                attrs=dict((k,html.unescape(v)) for k,_,v in re.findall(r'(\w+)\s*=\s*([\"\'])(.*?)\2',match[2],re.S))
                events.append({'kind':'policy_action','stage_index':index,'tool':match[1],
                    'argument':html.unescape(match[3]).strip(),
                    **({'query':attrs['query']} if 'query' in attrs else {})})
            elif public=='FINAL_READY':
                events.append({'kind':'policy_stop','stage_index':index})
            else:
                events.append({'kind':'unparsed_policy_attempt','stage_index':index,'attempt':public})
        elif kind=='state_update':
            try:updates=json.loads(re.sub(r'^\s*<think>.*?</think>\s*','',text,flags=re.S))['updates']
            except (ValueError,KeyError,TypeError):
                events.append({'kind':'unparsed_state_attempt','stage_index':index});continue
            changes=[]
            for item in updates:
                now={k:copy.deepcopy(item[k]) for k in ('id','status','evidence_ids')}
                if previous.get(item['id'])!=now:
                    old=previous.get(item['id'])
                    changes.append({**now,**({'previous_status':old['status']} if old else {})})
                previous[item['id']]=now
            if changes:events.append({'kind':'state_transition','stage_index':index,'changes':changes})
        elif kind=='checklist_init':
            # Full original question and frozen requirement wording remain in the stage input.
            error=rejected.get(text)
            events.append({'kind':'checklist_validation_failed' if error else 'checklist_init',
                'stage_index':index,**({'validation_error':error} if error else {})})
        elif kind=='final':
            # A previous rejected Final is not copied into a new target's prefix.
            events.append({'kind':'previous_final_attempt','stage_index':index})
    observe(current)
    # Link only receipts actually witnessed in a later input. The requested
    # action and the receipt's reported execution name remain separate facts.
    actions=[e for e in events if e['kind']=='policy_action']
    for receipt in observed:
        if receipt.get('call_id'):continue
        candidates=[a for a in actions if
            (receipt.get('source_id') or receipt.get('argument') or receipt.get('query'))==a['argument']
            and (not receipt.get('query') or receipt['query']==a.get('query',a['argument']))]
        if len(candidates)==1:
            a=candidates[0]
            # Ordinary same-name receipts are already present in native
            # Decision history. Only a real name conversion needs extra
            # routing metadata; otherwise we would defeat deduplication.
            if a['tool']==receipt['tool']:continue
            receipt.update(call_id='policy-stage-'+str(a['stage_index']),
                call_id_origin='derived_unique_action_receipt_link',requested_tool=a['tool'])
            if receipt.get('executed') is not False:receipt['executed_tool']=receipt['tool']
    # Preserve older receipts even if a later native history window omits them.
    # Do not duplicate observations already present in this Decision's native input.
    if current['stage'] in ('decision','decision_stop'):
        native=collections.Counter(dumps(e) for e in tools_visible_in(current))
        missing=[]
        for e in observed:
            key=dumps(e)
            if native[key]:native[key]-=1
            else:missing.append(e)
        events=missing+events
    else:events=observed+events
    if any(e.get('stage_index',-1)>=len(past) for e in events):
        raise ValueError('future history index')
    return events

def state_safe_indices(events,indices):
    """Never retain an older State alone when its later revision is omitted.

    Keep whole source events (not paraphrases). A multi-requirement event is
    conservatively omitted if any of its changed requirements has an omitted
    successor. Iterate because removing one event can invalidate another.
    Full histories retain all real transitions, including downgrades.
    """
    latest={}
    for i,e in enumerate(events):
        for c in e.get('changes',[]):latest[c['id']]=i
    selected=set(indices)
    while True:
        unsafe={i for i in selected if any(latest[c['id']] not in selected
            for c in events[i].get('changes',[]))}
        if not unsafe:break
        selected-=unsafe
    return sorted(selected)

def rendered(tok,row):
    return tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,
        **row['template_options'])+(row.get('assistant_prefix') or '')

def compact(events, indices):
    """Lossless representation of selected events; no medical paraphrasing."""
    selected=[events[i] for i in indices]
    counts={}
    for event in selected:
        for change in event.get('changes',[]):
            for ident in change['evidence_ids']:counts[ident]=counts.get(ident,0)+1
    aliases={ident:'E'+str(i) for i,ident in enumerate(sorted(
        ident for ident,n in counts.items() if n>1 and len(ident)>20))}
    packed=[]
    for event in selected:
        kind=event['kind'];index=event.get('stage_index')
        if kind=='policy_action':
            item={'at':index,'call':[event['tool'],event['argument']]}
            if event.get('query') and event['query']!=event['argument']:item['query']=event['query']
        elif kind=='state_transition':
            item={'at':index,'state':[dict(
                requirement=c['id'],status=c['status'],
                evidence=[('@'+aliases[x]) if x in aliases else x for x in c['evidence_ids']],
                **({'previous_status':c['previous_status']} if 'previous_status' in c else {})) for c in event['changes']]}
        elif kind=='tool_observation':
            facts={k:copy.deepcopy(v) for k,v in event.items() if k!='kind'}
            if facts.get('query')==facts.get('argument'):facts.pop('query',None)
            if facts.get('historical_returned_ids')==facts.get('returned_ids'):facts.pop('historical_returned_ids',None)
            if isinstance(facts.get('observation'),dict) and facts['observation'].get('source_id')==facts.get('source_id'):
                facts['observation'].pop('source_id',None)
            item={'result':facts}
        else:
            item={'at':index,'event':kind}
            if kind=='checklist_validation_failed':item['validation_error']=event['validation_error']
            if kind=='unparsed_policy_attempt':item['attempt']=event.get('attempt','')
        packed.append(item)
    payload={'schema':VERSION,'events':packed}
    if aliases:payload['evidence_id_aliases']={'@'+alias:ident for ident,alias in aliases.items()}
    if len(indices)<len(events):payload['omitted_event_count']=len(events)-len(indices)
    return payload

def selection_groups(events):
    """Keep a witnessed failed read and its same-argument follow-up indivisible.

    This does not claim the follow-up succeeded: the original receipt flags stay.
    Different-query replanning is not falsely labelled a retry.
    """
    groups=[{i} for i in range(len(events))]
    def identity(e):
        return (e.get('tool'),e.get('argument') or e.get('source_id') or e.get('query'))
    for i,e in enumerate(events):
        if e['kind']!='tool_observation' or not (e.get('failed') or e.get('environment_failure') or e.get('executed') is False):continue
        key=identity(e)
        if not key[1]:continue
        matches=[j for j in range(i+1,len(events)) if events[j]['kind']=='tool_observation' and identity(events[j])==key]
        if matches:
            j=matches[0]
            merged=set().union(*(g for g in groups if i in g or j in g))
            groups=[g for g in groups if not(g & merged)]+[merged]
    def priority(g):
        es=[events[i] for i in g]
        failure=any(e.get('failed') or e.get('environment_failure') or e.get('executed') is False or e['kind'].startswith('unparsed_') for e in es)
        state=any(e['kind'] in ('state_transition','policy_stop') for e in es)
        return (0 if failure else 1 if state else 2,-max(g))
    return sorted(groups,key=priority)

def retention(events,chosen):
    groups=selection_groups(events);chosen=set(chosen)
    kinds={}
    for i,e in enumerate(events):
        name=e['kind'];item=kinds.setdefault(name,{'available':0,'kept':0})
        item['available']+=1;item['kept']+=int(i in chosen)
    failure_groups=[g for g in groups if any(events[i].get('failed') or events[i].get('environment_failure') or events[i].get('executed') is False for i in g)]
    return {'event_retention':kinds,'failure_groups_available':len(failure_groups),
            'failure_groups_kept':sum(g<=chosen for g in failure_groups),
            'split_failure_groups':sum(bool(g & chosen) and not g<=chosen for g in failure_groups)}

def attach(current,past,tok,reserve,context_limit=10240):
    """Budget-independent of target. Never removes or rewrites existing evidence.

    Additional history is an entire preceding user message; the existing last
    stage payload and assistant continuation prefix remain byte-identical.
    """
    if 'completion' in current:raise ValueError('target-dependent history construction prohibited')
    row=copy.deepcopy(current)
    base_text=rendered(tok,row)
    base=len(tok.encode(base_text,add_special_tokens=False))
    if base+reserve>context_limit:
        # Online shared_budget must retain its existing evidence-card path.
        return row,{'mode':'defer_to_existing_evidence_budget','history_tokens':0,'omitted_events':0}
    if current['stage']=='checklist_init':
        return row,{'mode':'initial_no_history','history_tokens':0,'omitted_events':0}
    events=summarize(past,current)
    if not events:return row,{'mode':'no_past_events','history_tokens':0,'omitted_events':0}
    def trial_for(indices):
        indices=state_safe_indices(events,indices)
        trial=copy.deepcopy(row);trial['messages'].insert(len(trial['messages'])-1,
            {'role':'user','content':dumps({'process_history':compact(events,indices)})})
        text=rendered(tok,trial)
        # Qwen chat messages have special-token seams. If this exact inserted
        # user block removes back to the untouched base string, tokenize the
        # added block only. Validate additivity once per request; generic
        # templates retain full-render counting. Final encoded datasets and
        # Runtime prompt accounting independently check every accepted input.
        start=text.find('<|im_start|>user\n'+trial['messages'][-2]['content']+'<|im_end|>\n')
        block='<|im_start|>user\n'+trial['messages'][-2]['content']+'<|im_end|>\n'
        if start>=0 and text[:start]+text[start+len(block):]==base_text:
            extra=len(tok.encode(block,add_special_tokens=False));n=base+extra
            if not trial_for.additivity_verified:
                exact=len(tok.encode(text,add_special_tokens=False))
                if n!=exact:raise ValueError('chat-block token additivity failed')
                trial_for.additivity_verified=True
        else:
            n=len(tok.encode(text,add_special_tokens=False));extra=n-base
        return trial,extra,bool(indices) and n+reserve<=context_limit and extra<=MAX_HISTORY_TOKENS
    trial_for.additivity_verified=False
    all_indices=list(range(len(events)))
    best,added,fits=trial_for(all_indices)
    chosen=all_indices if fits else []
    if not fits:
        best=None;added=0
        for group in selection_groups(events):
            indices=state_safe_indices(events,sorted(set(chosen)|group))
            trial,extra,fits=trial_for(indices)
            if fits:chosen=indices;best=trial;added=extra
    if best is None:
        return row,{'mode':'history_omitted_for_budget','history_tokens':0,'omitted_events':len(events),**retention(events,[])}
    return best,{'mode':'history_added' if len(chosen)==len(events) else 'history_bounded',
        'history_tokens':added,'omitted_events':len(events)-len(chosen),
        'kept_events':len(chosen),'past_stage_count':len(past),**retention(events,chosen)}
