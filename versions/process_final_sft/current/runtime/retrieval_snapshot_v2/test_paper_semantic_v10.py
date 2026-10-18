import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'backend/agent'))
from dr_agent.mcp_backend.apis.paper_semantic_ranking import rank_paper_search_candidates
from dr_agent.mcp_backend.apis.dual_paper import Registry,parallel_search

class Backend:
    def __init__(self,scores): self.scores=scores
    def cross_scores(self,query,texts): return self.scores,{}
class PaperTests(unittest.TestCase):
    def test_raw_preserved(self):
        rows=[dict(source_id='PMID:1',title='A',abstract='raw A'),dict(source_id='PMID:2',title='B',abstract='raw B')]
        before=copy.deepcopy(rows)
        self.assertEqual(rank_paper_search_candidates('q',rows,backend=Backend([0,1])),rows[::-1])
        self.assertEqual(rows,before)
    def test_invalid_fallback(self):
        rows=[dict(title='A'),dict(title='B')]
        for scores in ([1],[float('nan'),0]):
            self.assertEqual(rank_paper_search_candidates('q',rows,backend=Backend(scores)),rows)
    def test_both_channels_rank_before_limit(self):
        calls={'pubmed':lambda:dict(data=[dict(pmid='1',title='A',abstract='A')]),
               'semantic_scholar':lambda:dict(data=[dict(paperId='a'*40,title='B',abstract='B')])}
        result=parallel_search(calls,Registry(),'q',limit=1,rerank_backend=Backend([0,1]))
        self.assertEqual(result['data'][0]['source_id'],'S2:'+'a'*40)
    def test_missing_abstract_not_deleted(self):
        rows=[dict(title='A'),dict(title='B',abstract='readable')]
        ranked=rank_paper_search_candidates('q',rows,backend=Backend([1,1]))
        self.assertEqual(len(ranked),2)
        self.assertEqual(ranked[0]['title'],'B')
if __name__=='__main__':unittest.main()
