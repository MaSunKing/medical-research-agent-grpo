"""Unified, local-only mixed-source candidate window; no evidence rewriting."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

VERSION = 'candidate_semantic_window_v11'
_MODEL = None

def local_scores(query, texts):
    global _MODEL
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if _MODEL is None:
        config = json.loads(Path(os.environ['SCHOOL_AGENT_CONFIG']).read_text())
        path = os.environ.get('MEDGAP_CANDIDATE_RERANKER_MODEL') or config['retrieval_models']['reranker_model']
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).cpu().eval()
        _MODEL = tokenizer, model
    tokenizer, model = _MODEL
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = texts[start:start+16]
            inputs = tokenizer([query]*len(batch), batch, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')
            values.extend(model(**inputs).logits.float().reshape(-1).tolist())
    return values

def semantic_window(self, query, original_question, limit, reread_registry, scorer=None):
    from candidate_window import excerpt
    rows = list(self.rows.values())
    if not rows:
        return []
    gaps = str(query.get('checklist_gaps') or '') if isinstance(query, dict) else ''
    texts = ['\n'.join((str(r.get('title') or ''), str(r.get('_native_preview_text') or ''))) for r in rows]
    score = scorer or local_scores
    question_scores = score(str(original_question), texts)
    gap_scores = score(gaps, texts) if gaps.strip() else question_scores
    if len(question_scores) != len(rows) or len(gap_scores) != len(rows):
        raise ValueError('candidate semantic score count mismatch')
    values = [.7*float(q)+.3*float(g) if gaps.strip() else float(q)
              for q, g in zip(question_scores, gap_scores)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('candidate semantic scores nonfinite')
    # Stable ties retain accumulated Search order; no lexical override.
    order = sorted(range(len(rows)), key=lambda i: (-values[i], i))
    receipt = dict(version=VERSION, original_question=str(original_question), checklist_gaps=gaps,
                   candidate_limit=limit, scorer='shared_MiniLM_cross_encoder',
                   weights={'question':.7 if gaps.strip() else 1., 'gaps':.3 if gaps.strip() else 0.},
                   candidates=[dict(source_id=rows[i]['source_id'], rank=rank, visible=rank<=max(0,limit),
                                    combined_score=values[i], question_score=float(question_scores[i]),
                                    gap_score=float(gap_scores[i]), previously_browsed=bool(reread_registry.get(rows[i]['source_id'])))
                               for rank,i in enumerate(order,1)])
    receipt['projection_receipt_id'] = hashlib.sha256(json.dumps(receipt, sort_keys=True,
        ensure_ascii=False, separators=(',',':')).encode()).hexdigest()
    self.projection_history.append(receipt)
    selected = []
    focus = str(query.get('current_query') or original_question) if isinstance(query, dict) else str(query)
    for i in order[:max(0,limit)]:
        row = copy.deepcopy(rows[i]); sid = row['source_id']
        if sid.startswith(('PMID:', 'S2:')):
            row['search_preview'], row['preview_selected_spans'], row['preview_truncated'] = excerpt(row.get('_native_preview_text',''),focus,240)
        row['previous_browse_queries'] = list(reread_registry.get(sid,{}).get('previous_browse_queries',[]))
        row['requires_new_query'] = False
        selected.append(row)
    return selected
