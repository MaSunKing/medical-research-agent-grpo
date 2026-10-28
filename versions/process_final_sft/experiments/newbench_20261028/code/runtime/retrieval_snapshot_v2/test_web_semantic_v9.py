import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent / 'backend/agent'))
from dr_agent.mcp_backend.apis.web_semantic_ranking import rank_web_search_candidates
from dr_agent.runtime_policy import rank_web_candidates

class Backend:
    def __init__(self, scores): self.scores = scores
    def rerank_scores(self, query, texts): return self.scores, {}

class SemanticRankingTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{'source_id': 'WEB:a', 'title': 'renal monitoring', 'snippet': 'clinical renal function'},
                     {'source_id': 'WEB:b', 'title': 'other topic', 'snippet': 'unrelated'}]
    def test_reorder_without_mutation(self):
        before = copy.deepcopy(self.pool)
        lexical = rank_web_candidates('renal', self.pool)
        ranked = rank_web_search_candidates('renal', self.pool, backend=Backend([0, 1]))
        self.assertEqual(ranked, lexical[::-1])
        self.assertEqual(self.pool, before)
        self.assertEqual(set(ranked[0]), set(before[0]))
    def test_ties_preserve_lexical(self):
        self.assertEqual(rank_web_search_candidates('renal', self.pool, backend=Backend([1, 1])), rank_web_candidates('renal', self.pool))
    def test_bad_scores_fallback(self):
        for scores in ([1], [float('nan'), 0]):
            self.assertEqual(rank_web_search_candidates('renal', self.pool, backend=Backend(scores)), rank_web_candidates('renal', self.pool))
    def test_rollback(self):
        with patch.dict('os.environ', {'MEDGAP_WEB_SEARCH_RERANK': 'lexical'}):
            self.assertEqual(rank_web_search_candidates('renal', self.pool, backend=Backend([0, 1])), rank_web_candidates('renal', self.pool))
    def test_empty(self):
        self.assertEqual(rank_web_search_candidates('renal', []), [])

if __name__ == '__main__': unittest.main()
