"""Inference-only history/window separation and source-exact excerpts."""
import copy
import hashlib
import json
import math
import re
from collections import Counter

VERSION='candidate_history_window_v2_audited_projection'
STOP=set('a an the of in on and or to for with without is are does do how whether people patients study studies trial trials randomized controlled effect effects'.split())

def terms(text):
    return [t.rstrip('s') if len(t)>4 and t.endswith('s') else t for t in re.findall(r'\w+',str(text).casefold()) if t not in STOP]

def excerpt(text, query, cap):
    text=re.sub(r'\s+',' ',str(text or '')).strip()
    spans=[]; start=0
    for m in re.finditer(r'[.!?。！？](?=\s|$)',text):
        if re.search(r'\b(?:e\.g|i\.e|vs|al|Dr|Fig)\.$',text[:m.end()],re.I):continue
        spans.append((start,m.end()));start=m.end()
    if text[start:].strip():spans.append((start,len(text)))
    q=set(terms(query))
    ranked=sorted(spans,key=lambda se:(-len(q & set(terms(text[se[0]:se[1]]))),se[0]))
    chosen=[];used=0
    for a,b in ranked:
        while a<b and text[a].isspace():a+=1
        if b-a+bool(chosen)<=cap-used:
            chosen.append((a,b));used+=b-a+bool(len(chosen)>1)
        elif not chosen:
            # Do not discard the most relevant long sentence in favour of a
            # short generic design sentence. Preserve a labelled native prefix.
            end=min(b,a+cap)
            boundary=text.rfind(' ',a,end)
            if boundary>a:end=boundary
            chosen.append((a,end));break
    chosen.sort()
    shown=' '.join(text[a:b] for a,b in chosen)
    return shown,chosen,shown!=text

def native_preview(item, sid, clean):
    paper=sid.startswith(('PMID:','S2:'))
    field=next((f for f in (('abstract','text') if paper else ('snippet',)) if clean(item.get(f))),None)
    raw=clean(item.get(field)) if field else ''
    if paper:shown,spans,truncated=excerpt(raw,item.get('_preview_query',''),240)
    else:
        shown=raw[:160];spans=[(0,len(shown))] if shown else [];truncated=shown!=raw
    return dict(search_preview=shown,preview_kind='paper_abstract' if paper else 'web_search_snippet',
        preview_source_field=field,preview_available=bool(raw),preview_truncated=truncated,
        preview_status='search_preview_not_opened_evidence',preview_version=VERSION,
        preview_selected_spans=spans,preview_selection='native_excerpt_not_rewritten',
        _native_preview_text=raw)

def bound_previews(rows):
    result=copy.deepcopy(rows);remaining=1200
    share=remaining//max(1,len(result))
    for row in result:
        cap=min(240 if str(row.get('source_id','')).startswith(('PMID:','S2:')) else 160,share,remaining)
        raw=str(row.get('search_preview') or '')
        end=cap
        if len(raw)>cap and raw.rfind(' ',0,cap)>0:end=raw.rfind(' ',0,cap)
        row['search_preview']=raw[:end]
        row['preview_truncated']=bool(row.get('preview_truncated')) or len(raw)>end
        remaining-=len(row['search_preview'])
    return result

class CandidateHistory:
    def __init__(self):
        self.rows={}
        self.queries={}
        self.projection_history=[]

    def add(self, rows):
        for row in rows:
            sid=row['source_id'];old=self.rows.get(sid,{})
            merged={**old,**copy.deepcopy(row)}
            if not merged.get('_native_preview_text') and old.get('_native_preview_text'):
                merged['_native_preview_text']=old['_native_preview_text']
            self.rows[sid]=merged
            query=row.get('query')
            if query and query not in self.queries.setdefault(sid,[]):self.queries[sid].append(query)

    def window(self, query, original_question, limit, reread_registry):
        rows=list(self.rows.values())
        docs=[terms(r.get('title',''))*2+terms(r.get('_native_preview_text','')) for r in rows]
        df=Counter(t for d in docs for t in set(d));avg=sum(map(len,docs))/max(1,len(docs))
        def score(d,q):
            tf=Counter(d);value=0
            for t in set(terms(q)):
                f=tf[t]
                if f:value+=math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5))*f*2.2/(f+1.2*(.25+.75*len(d)/max(1,avg)))
            return value
        scored=[]
        for row,doc in zip(rows,docs):
            focus_score=score(doc,query)
            question_score=score(doc,original_question)
            combined=.8*focus_score+.2*question_score
            scored.append((row,doc,focus_score,question_score,combined))
        ranked=sorted(scored,key=lambda rd:(-rd[4],rd[0]['source_id']))
        visible_ids={row['source_id'] for row,_,_,_,_ in ranked[:max(0,limit)]}
        projection=[]
        for rank,(row,_,focus_score,question_score,combined) in enumerate(ranked,1):
            sid=row['source_id']
            visible=sid in visible_ids
            projection.append(dict(
                source_id=sid, rank=rank, visible=visible,
                disposition='visible' if visible else 'dropped',
                reason='within_top_k' if visible else 'outside_top_k',
                candidate_limit=limit,
                focus_score=round(focus_score,12),
                original_question_score=round(question_score,12),
                combined_score=round(combined,12),
                previously_browsed=bool(reread_registry.get(sid)),
            ))
        receipt=dict(version=VERSION,focus_query=str(query),
            original_question=str(original_question),candidate_limit=limit,
            candidates=projection)
        receipt['projection_receipt_id']=hashlib.sha256(
            json.dumps(receipt,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        ).hexdigest()
        self.projection_history.append(receipt)
        selected=[]
        for row,_,_,_,_ in ranked[:limit]:
            r=copy.deepcopy(row);sid=r['source_id']
            raw=r.get('_native_preview_text','')
            if sid.startswith(('PMID:','S2:')):
                r['search_preview'],r['preview_selected_spans'],r['preview_truncated']=excerpt(raw,query,240)
            r['previous_browse_queries']=list(reread_registry.get(sid,{}).get('previous_browse_queries',[]))
            r['requires_new_query']=False
            selected.append(r)
        return selected

def patch_sources(base,changes,here):
    def once(s,old,new):
        if s.count(old)!=1:raise ValueError('candidate window anchor: '+old[:70])
        return s.replace(old,new)
    for arm in ('title_strict','preview_strict','preview_relaxed'):
        prefix=f'variants/{arm}/script/'
        changes[prefix+'candidate_window.py']=(here/'candidate_window.py').read_text(encoding='utf-8')
        p=prefix+'collect_medgap_v71_decision_groups.py';s=changes.get(p,(base/p).read_text(encoding='utf-8'))
        s+='\nfrom candidate_window import native_preview as _window_native, bound_previews as bound_public_previews\ndef native_search_preview(item, source_id):\n    return _window_native(item, source_id, clean_search_preview)\n'
        changes[p]=s
        p=prefix+'run_medgap_v71_on_policy_retrieval.py';s=(base/p).read_text(encoding='utf-8')
        s=once(s,'        candidate_pool: list[dict] = []','        from candidate_window import CandidateHistory\n        candidate_history=CandidateHistory()\n        current_focus=question\n        candidate_pool: list[dict] = []')
        s=once(s,'            candidate_pool = collection.with_reopen_candidates(candidate_pool, reread_registry)',
            '            candidate_pool = candidate_history.window(current_focus, question, args.browse_candidates, reread_registry)')
        s=once(s,'collection.candidates_from_search(search_output, v71, args.browse_candidates)',
            'collection.candidates_from_search(search_output, v71, len(search_output.get("data") or []))')
        a=s.index('                from priority_v5 import stable_candidate_pool\n');b=s.index('                seen_candidate_actions.update(',a)
        s=s[:a]+'                candidate_history.add(additions)\n                current_focus=action["query"]\n                candidate_pool=candidate_history.window(current_focus, question, args.browse_candidates, reread_registry)\n'+s[b:]
        s=once(s,'                    "query_signature": signature,\n                "environment_failure":',
            '                    "query_signature": signature, "repeated_query": repeated,\n                "environment_failure":')
        s=s.replace('"candidate_pool_size": len(candidate_pool),','"candidate_pool_size": len(candidate_pool),\n                    "historical_candidate_count": len(candidate_history.rows),')
        s=once(s,'"attempted_source_ids":', '"candidate_history": list(candidate_history.rows.values()),\n                "candidate_history_queries": candidate_history.queries,\n                "candidate_projection_audit": candidate_history.projection_history,\n                "attempted_source_ids":')
        changes[p]=s
        p=prefix+'priority_v5.py';s=changes.get(p,(base/p).read_text(encoding='utf-8'))
        s=s.replace("('tool','query','environment_failure')", "('tool','query','environment_failure','repeated_query')")
        changes[p]=s
