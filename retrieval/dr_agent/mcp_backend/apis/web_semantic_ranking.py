# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Rank existing web previews without modifying candidates or tool contracts."""
from __future__ import annotations
import logging
import math
import os
from ...runtime_policy import rank_web_candidates

LOG = logging.getLogger(__name__)

def rank_web_search_candidates(query, candidates, *, original_question=None, backend=None):
    lexical = rank_web_candidates(query, candidates, original_question=original_question)
    if len(lexical) < 2 or os.environ.get('MEDGAP_WEB_SEARCH_RERANK', 'semantic') == 'lexical':
        return lexical
    try:
        if backend is None:
            from .hybrid_passage_retriever import get_hybrid_backend
            backend = get_hybrid_backend()
        texts = ['\n'.join(dict.fromkeys(str(row.get(k) or '').strip()
                  for k in ('title', 'snippet') if str(row.get(k) or '').strip()))
                 for row in lexical]
        question = str(original_question or query).strip()
        q_scores, _ = backend.rerank_scores(question, texts)
        f_scores = q_scores if question == query else backend.rerank_scores(query, texts)[0]
        if len(q_scores) != len(lexical) or len(f_scores) != len(lexical):
            raise ValueError('reranker score count mismatch')
        scores = [0.7 * float(q) + 0.3 * float(f) for q, f in zip(q_scores, f_scores)]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError('nonfinite reranker score')
        # Authority/diversity remains a tie-breaker, not a relevance override.
        order = sorted(range(len(lexical)), key=lambda i: (-scores[i], i))
        LOG.info('WEB_SEARCH_RERANK=semantic candidates=%d', len(lexical))
        return [lexical[i] for i in order]
    except Exception as exc:
        # Preserve usable search results if the local model is unavailable.
        # Do not log queries, credentials or exception payloads.
        LOG.warning('WEB_SEARCH_RERANK=lexical_fallback reason=%s', type(exc).__name__)
        return lexical
