# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""MedCPT ranking on fused real title/abstract candidates; no generated evidence."""
import logging
import math
import os
LOG=logging.getLogger(__name__)

def rank_paper_search_candidates(query, rows, *, original_question=None, backend=None):
    if len(rows)<2 or os.getenv('MEDGAP_PAPER_SEARCH_RERANK','semantic')=='rrf': return rows
    try:
        if backend is None:
            from .medical_passage_retriever_v28 import get_v28_backend
            backend=get_v28_backend()
        texts=['\n'.join(str(r.get(k) or '').strip() for k in ('title','abstract') if str(r.get(k) or '').strip()) for r in rows]
        question=str(original_question or query).strip()
        qs,_=backend.cross_scores(question,texts)
        fs=qs if question==query else backend.cross_scores(query,texts)[0]
        if len(qs)!=len(rows) or len(fs)!=len(rows): raise ValueError('score count mismatch')
        scores=[0.7*float(q)+0.3*float(f) for q,f in zip(qs,fs)]
        if not all(math.isfinite(s) for s in scores): raise ValueError('nonfinite score')
        # Known abstract availability breaks exact semantic ties only. No claim
        # that a missing abstract proves full text is unavailable.
        order=sorted(range(len(rows)),key=lambda i:(-scores[i],-bool(str(rows[i].get('abstract') or '').strip()),i))
        LOG.info('PAPER_SEARCH_RERANK=medcpt candidates=%d',len(rows))
        return [rows[i] for i in order]
    except Exception as exc:
        LOG.warning('PAPER_SEARCH_RERANK=rrf_fallback reason=%s',type(exc).__name__)
        return rows
