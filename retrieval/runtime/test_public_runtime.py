"""Offline public-component regression; no network/model calls."""
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from candidate_window import CandidateHistory
from environment_failure_budget_v1 import ToolBudget
from sft_interface_v20 import HELP

def rows(start, end):
    return [dict(source_id=f'PMID:{i}', title=str(i), _native_preview_text='native preview')
            for i in range(start, end)]

class PublicRuntimeTests(unittest.TestCase):
    def test_latest_then_history(self):
        h=CandidateHistory(); h.add(rows(0,12)); h.add(rows(12,20))
        visible=h.window({},'original question',8,{},lambda q,t:list(range(len(t))))
        self.assertEqual([r['source_id'] for r in visible],
                         [f'PMID:{i}' for i in (12,13,14,15,11,10,9,8)])

    def test_first_search_fills_eight(self):
        h=CandidateHistory();h.add(rows(0,12))
        visible=h.window({},'question',8,{},lambda q,t:[0]*len(t))
        self.assertEqual(len(visible),8)

    def test_three_exemptions_do_not_force_stop(self):
        b=ToolBudget()
        for _ in range(3): self.assertTrue(b.settle('search',{'error_code':'timeout'})['exempted'])
        self.assertEqual(b.remaining,6)
        self.assertFalse(b.settle('search',{'error_code':'timeout'})['exempted'])
        self.assertEqual(b.remaining,5)

    def test_empty_search_is_charged(self):
        b=ToolBudget()
        self.assertFalse(b.settle('search',{'error_code':'search_no_results'})['exempted'])
        self.assertEqual(b.remaining,5)

    def test_tool_contract_is_not_extended(self):
        self.assertIn('Do not add a requirement-ID field',HELP)
        self.assertIn('Boolean',HELP)
        self.assertIn('Navigation menus',HELP)

    def test_python_sources_parse(self):
        for p in HERE.glob('*.py'):ast.parse(p.read_bytes())

    def test_search_semantic_weight_and_fallback(self):
        # Execute only the pure ranking function with explicit mocked dependencies.
        api=HERE.parent/'dr_agent/mcp_backend/apis'
        import logging
        import math
        import os
        from unittest.mock import patch
        for filename,name,backend_fn in (
            ('web_semantic_ranking.py','rank_web_search_candidates','rerank_scores'),
            ('paper_semantic_ranking.py','rank_paper_search_candidates','cross_scores')):
            source=ast.parse((api/filename).read_text())
            node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name)
            namespace=dict(logging=logging,math=math,os=os,LOG=logging.getLogger('offline-test'),
                rank_web_candidates=lambda q,rows,**kw:rows)
            exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),namespace)
            class Backend:
                def rerank_scores(self,q,texts):
                    return ([0.0,2.0] if q=='original' else [3.0,0.0]),None
                cross_scores=rerank_scores
            candidates=[dict(title='second',snippet='body',abstract='body'),
                        dict(title='first',snippet='body',abstract='body')]
            with patch.dict(os.environ,{'MEDGAP_WEB_SEARCH_RERANK':'semantic','MEDGAP_PAPER_SEARCH_RERANK':'semantic'}):
                result=namespace[name]('current',candidates,original_question='original',backend=Backend())
            self.assertEqual(result[0]['title'],'first')

if __name__=='__main__':unittest.main()
