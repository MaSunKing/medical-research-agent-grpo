"""Synthetic UNIT fixtures only; never export these as training trajectories."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from trainable_agent.context import ContextBuilder, OfflineCodec
from trainable_agent.contracts import Observation, validate_action
from trainable_agent.engine import Engine, initial_state
from trainable_agent.failure_feedback import decision_repair_feedback, transport_retry_feedback
from trainable_agent.storage import Store


def fixture():
    s=initial_state('feedback-fixture','Compare intervention A with placebo.','p','f','t',process_only=True)
    s['requirements']={'R1':'Compare effectiveness.'}
    s['assessments']={'R1':dict(support='missing',evidence_ids=[],gap='Effectiveness unknown.',conflict='')}
    s['candidates']=[dict(source_id='PMID:1',title='Study one',preview='Trial abstract.',channel='paper',url=''),
                     dict(source_id='PMID:2',title='Study two',preview='Trial abstract.',channel='paper',url='')]
    return s


class Model:
    """Deterministic test callback, NOT an LLM or new sampling run."""
    def __init__(self,outputs):self.outputs=list(outputs);self.requests=[]
    async def generate(self,request):
        self.requests.append(copy.deepcopy(request))
        return dict(text=json.dumps(self.outputs.pop(0)))


class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def assert_repair(self,s,bad,expected):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'trajectory.sqlite')
            try:
                good=dict(action='search',requirement_ids=['R1'],channel='web',query='intervention A placebo effectiveness trial alternative')
                model=Model([bad,good]);before=copy.deepcopy(s['budget'])
                engine=Engine(store,model,model,None,ContextBuilder(OfflineCodec(),100000),None)
                result=(await engine.decision(dict(session=s)))['session']
                self.assertEqual(result['action'],good)
                self.assertEqual(result['next_node'],'execute')
                self.assertEqual(result['budget'],before)
                self.assertEqual(len(model.requests),2)
                first,repair=model.requests
                self.assertEqual(first['messages'],repair['messages'][:-1])
                feedback=repair['messages'][-1]['content']
                for text in expected:self.assertIn(text,feedback)
                events=store.events(s['run_id'],'generation')
                self.assertFalse(events[0]['data']['accepted'])
                self.assertTrue(events[1]['data']['accepted'])
                self.assertIn(events[0]['data']['error'].splitlines()[0].split(':')[0],feedback)
                self.assertEqual(len(store.events(s['run_id'],'decision_repair_feedback')),1)
            finally:store.db.close()

    async def test_invisible_source_lists_actual_visible_ids(self):
        s=fixture()
        await self.assert_repair(s,dict(action='browse',requirement_ids=['R1'],source_id='Z',reading_focus='effect'),
            ['unknown_source_alias','source_id="Z"','visible source_ids=["A","B"]','No tool call executed or charged'])

    async def test_blocked_visible_source_not_advertised_as_access_eligible(self):
        s=fixture();s['blocked_sources']={'PMID:1':'confirmed barrier'}
        await self.assert_repair(s,dict(action='browse',requirement_ids=['R1'],source_id='A',reading_focus='effect'),
            ['source_access_blocked_choose_another','passing access/retry checks=["B"]','blocked for this run'])

    async def test_repeat_empty_search_is_accepted_without_repair(self):
        s=fixture();search=dict(action='search',requirement_ids=['R1'],channel='paper',query='existing query')
        s['history']=[dict(turn=1,action=search,feedback=dict(status='empty',executed=True))]
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'trajectory.sqlite')
            try:
                model=Model([search])
                engine=Engine(store,model,model,None,ContextBuilder(OfflineCodec(),100000),None)
                result=(await engine.decision(dict(session=s)))['session']
                self.assertEqual(result['next_node'],'execute')
                self.assertEqual(result['decision_budget']['rejected'],0)
                self.assertEqual(len(model.requests),1)
            finally:
                store.db.close()

    async def test_exhausted_retry_details(self):
        s=fixture();a=dict(action='browse',requirement_ids=['R1'],source_id='PMID:1',reading_focus='effect')
        s['history']=[dict(action=a,feedback=dict(status='environment_failure',retryable=True)) for _ in range(2)]
        await self.assert_repair(s,dict(a,source_id='A'),['temporary_failure_retry_limit','failures=2','remaining retries=0','passing access/retry checks=["B"]'])

    async def test_wrong_action_enum_gets_minimal_schema(self):
        await self.assert_repair(fixture(),dict(action='research',requirement_ids=['R1'],query='effect'),
            ['search, browse, or stop','Search uses channel','existing requirement_ids=["R1"]'])

    async def test_nonretryable_search_has_no_retry_offer(self):
        s=fixture();a=dict(action='search',requirement_ids=['R1'],channel='paper',query='existing query')
        s['history']=[dict(action=a,feedback=dict(status='environment_failure',retryable=False))]
        await self.assert_repair(s,a,['nonretryable_search_failure','remaining retries=0','another retrieval path'])

    async def test_same_bad_output_still_fails_after_one_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'trajectory.sqlite')
            try:
                s=fixture();bad=dict(action='research');model=Model([bad,bad])
                result=(await Engine(store,model,model,None,ContextBuilder(OfflineCodec(),100000),None).decision(dict(session=s)))['session']
                self.assertEqual(result['termination'],'model_contract_failure')
                self.assertEqual(len(model.requests),2)
                self.assertEqual(result['budget'],s['budget'])
            finally:store.db.close()

    async def test_retry_status_enters_next_decision_and_raw_retryable_is_preserved(self):
        class Tools:
            async def execute(self,args):return dict(status='environment_failure',retryable=True)
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'trajectory.sqlite')
            try:
                s=fixture();s['action']=dict(action='browse',requirement_ids=['R1'],source_id='PMID:1',reading_focus='effect')
                model=Model([dict(action='search',requirement_ids=['R1'],channel='web',query='new intervention trial')])
                engine=Engine(store,model,model,Tools(),ContextBuilder(OfflineCodec(),100000),None)
                one=(await engine.execute(dict(session=s)))['session']
                self.assertEqual(one['feedback']['transport_retry_status']['remaining_retries'],1)
                self.assertEqual(validate_action(one['action'],one),one['action'])
                two=(await engine.execute(dict(session=one)))['session']
                self.assertEqual(two['feedback']['transport_retry_status']['remaining_retries'],0)
                self.assertTrue(two['feedback']['retryable'])
                self.assertEqual(two['next_node'],'decision')
                self.assertEqual(two['assessments'],s['assessments'])
                self.assertEqual(two['budget']['actual'],2)
                self.assertEqual(two['budget']['charged'],0)
                self.assertIn('No same-source/query transport retry',two['feedback']['message'])
                self.assertNotIn('PMID:1', [c['source_id'] for c in two['candidates']])
                with self.assertRaisesRegex(ValueError,'source_not_in_visible_candidates'):validate_action(two['action'],two)
                await engine.decision(dict(session=two))
                visible=json.loads(model.requests[0]['messages'][1]['content'])
                self.assertEqual(visible['feedback']['transport_retry_status']['source_id'],'A')
                self.assertEqual(visible['feedback']['transport_retry_status']['remaining_retries'],0)
            finally:store.db.close()


class PureFeedbackTests(unittest.TestCase):
    def test_no_hidden_candidates_are_injected(self):
        base=dict(messages=[dict(role='user',content=json.dumps(dict(question='q',checklist={'R1':'need'},candidates=[])))])
        text=decision_repair_feedback(base,'source_not_in_visible_candidates',json.dumps(dict(action='browse',source_id='Z')))
        self.assertIn('visible source_ids=[]',text)
        self.assertIn('passing access/retry checks=[]',text)

    def test_search_retry_scope_and_success_reset_for_browse(self):
        s=fixture();a=dict(action='search',channel='paper',query='X')
        s['history']=[dict(action=a,feedback=dict(status='environment_failure',retryable=True))]
        stat,_=transport_retry_feedback(s,dict(a,query=' x '),Observation(status='environment_failure',retryable=True))
        self.assertEqual(stat['remaining_retries'],0)
        b=dict(action='browse',source_id='PMID:1')
        s['history']=[dict(action=b,feedback=dict(status='environment_failure')),
                      dict(action=b,feedback=dict(status='ok'))]
        stat,_=transport_retry_feedback(s,b,Observation(status='environment_failure',retryable=True))
        self.assertEqual(stat['remaining_retries'],1)

    def test_unknown_timeout_does_not_grant_replay(self):
        stat,text=transport_retry_feedback(fixture(),dict(action='browse',source_id='PMID:1'),
            Observation(status='environment_failure',retryable=True,error_code='retrieval_operation_timeout'))
        self.assertEqual(stat['remaining_retries'],0)
        self.assertIn('remote outcome is unknown',text)


if __name__=='__main__':unittest.main()
