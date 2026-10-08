"""Reserve gap candidates, then fill from history; never rerank merged slots."""
import copy
import hashlib
import json
import math
from candidate_semantic_window_v11 import local_scores

VERSION='candidate_split_window_v12'

def split_window(self, query, original_question, limit, reread_registry, scorer=None):
    from candidate_window import excerpt
    rows=list(self.rows.values()); limit=max(0,int(limit))
    if not rows:return []
    gaps=str(query.get('checklist_gaps') or '') if isinstance(query,dict) else ''
    current=str(query.get('current_query') or original_question) if isinstance(query,dict) else str(query)
    texts=['\n'.join((str(r.get('title') or ''),str(r.get('_native_preview_text') or ''))) for r in rows]
    score=scorer or local_scores
    history_scores=[float(v) for v in score(str(original_question),texts)]
    gap_scores=[float(v) for v in score(gaps,texts)] if gaps.strip() else history_scores
    query_scores=[float(v) for v in score(current,texts)] if gaps.strip() else history_scores
    for values in (history_scores,gap_scores,query_scores):
        if len(values)!=len(rows) or not all(math.isfinite(v) for v in values):
            raise ValueError('invalid split-window scores')
    # Current focus reinforces unresolved requirements, not the whole question.
    focused=[.5*g+.5*q for g,q in zip(gap_scores,query_scores)]
    gap_order=sorted(range(len(rows)),key=lambda i:(-focused[i],i))
    history_order=sorted(range(len(rows)),key=lambda i:(-history_scores[i],i))
    gap_quota=min(4,(limit+1)//2) if gaps.strip() else 0
    gap_ids=gap_order[:gap_quota]
    gap_set=set(gap_ids)
    history_ids=[i for i in history_order if i not in gap_set][:max(0,limit-len(gap_ids))]
    order=gap_ids+history_ids
    visible=set(order)
    positions={i:p for p,i in enumerate(order,1)}
    receipt=dict(version=VERSION,original_question=str(original_question),checklist_gaps=gaps,
        current_query=current,candidate_limit=limit,gap_quota=gap_quota,merge_policy='gap_first_no_global_rerank',
        scorer='shared_MiniLM_cross_encoder',
        candidates=[dict(source_id=r['source_id'],visible=i in visible,display_rank=positions.get(i),
            group='gap' if i in gap_set else 'history' if i in visible else 'outside_window',
            gap_score=gap_scores[i],query_score=query_scores[i],focused_score=focused[i],
            history_score=history_scores[i],previously_browsed=bool(reread_registry.get(r['source_id'])))
            for i,r in enumerate(rows)])
    receipt['projection_receipt_id']=hashlib.sha256(json.dumps(receipt,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    self.projection_history.append(receipt)
    selected=[]
    for i in order:
        row=copy.deepcopy(rows[i]);sid=row['source_id']
        if sid.startswith(('PMID:','S2:')):
            row['search_preview'],row['preview_selected_spans'],row['preview_truncated']=excerpt(row.get('_native_preview_text',''),current,240)
        row['previous_browse_queries']=list(reread_registry.get(sid,{}).get('previous_browse_queries',[]))
        row['requires_new_query']=False
        selected.append(row)
    return selected
