"""Search-order latest slots plus historical slots, without a merged rerank."""
import copy
import hashlib
import json
import math
from candidate_semantic_window_v11 import local_scores

VERSION = 'candidate_latest_window_v13'

def install(cls):
    original_add = cls.add
    def add(self, rows):
        rows = list(rows)
        self._v13_prior_ids = list(self.rows)
        original_add(self, rows)
        self._v13_latest_ids = list(dict.fromkeys(r['source_id'] for r in rows))
    cls.add = add
    cls.window = latest_window

def latest_window(self, query, original_question, limit, reread_registry, scorer=None):
    from candidate_window import excerpt
    limit = max(0, int(limit))
    current = str(query.get('current_query') or original_question) if isinstance(query, dict) else str(query)
    valid = {sid:r for sid,r in self.rows.items() if str(r.get('_native_preview_text') or '').strip()}
    latest = [sid for sid in getattr(self, '_v13_latest_ids', []) if sid in valid]
    prior = [sid for sid in getattr(self, '_v13_prior_ids', []) if sid in valid and sid not in latest]
    first = not any(sid in valid for sid in getattr(self, '_v13_prior_ids', []))
    quota = limit if first else min(4, limit)
    order = latest[:quota]
    groups = {sid:'latest' for sid in order}
    scores = {}
    if len(order) < limit and prior:
        # Preserve the previous history relevance criterion; only fresh slots change.
        texts = ['\n'.join((str(valid[s].get('title') or ''), valid[s]['_native_preview_text'])) for s in prior]
        values = [float(v) for v in (scorer or local_scores)(str(original_question), texts)]
        if len(values) != len(prior) or not all(math.isfinite(v) for v in values):
            raise ValueError('invalid historical candidate scores')
        scores = dict(zip(prior, values))
        for sid in sorted(prior, key=lambda s:-scores[s])[:limit-len(order)]:
            order.append(sid); groups[sid]='history'
    # If historical slots are unavailable, fill from the remaining fresh results.
    for sid in latest:
        if len(order) >= limit: break
        if sid not in groups:
            order.append(sid); groups[sid]='latest_backfill'
    receipt = dict(version=VERSION, original_question=str(original_question), current_query=current,
        candidate_limit=limit, latest_quota=quota, first_search=first,
        merge_policy='latest_first_no_global_rerank', latest_returned_ids=getattr(self,'_v13_latest_ids',[]),
        scorer='Search_backend_order_for_latest; MiniLM_original_question_for_history',
        candidates=[dict(source_id=s, visible=s in groups, display_rank=order.index(s)+1 if s in groups else None,
            group=groups.get(s, 'outside_window' if s in valid else 'no_readable_preview'),
            history_score=scores.get(s), previously_browsed=bool(reread_registry.get(s))) for s in self.rows])
    receipt['projection_receipt_id']=hashlib.sha256(json.dumps(receipt,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    self.projection_history.append(receipt)
    selected=[]
    for sid in order:
        row=copy.deepcopy(valid[sid])
        if sid.startswith(('PMID:','S2:')):
            row['search_preview'],row['preview_selected_spans'],row['preview_truncated']=excerpt(row['_native_preview_text'],current,240)
        row['previous_browse_queries']=list(reread_registry.get(sid,{}).get('previous_browse_queries',[]))
        row['requires_new_query']=False
        selected.append(row)
    return selected
