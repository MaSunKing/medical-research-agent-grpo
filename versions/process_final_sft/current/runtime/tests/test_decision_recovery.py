"""Synthetic offline callbacks only; no LLM, retrieval, or training calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from trainable_agent.context import ContextBuilder, OfflineCodec
from trainable_agent.engine import Engine, initial_state
from trainable_agent.storage import Store
from test_agent import ScriptModel, ScriptTools, CHECK, SEARCH, STOP, FOUND, BROWSE, OPENED, UPDATE, rank
BAD_SOURCE = {**BROWSE, 'source_id': 'PMID:999'}


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'trajectory.sqlite')
        self.ctx = ContextBuilder(OfflineCodec(), 100000)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def engine(self, policy, tools, final=None):
        return Engine(self.store, policy, final or ScriptModel(['<answer>Evidence is insufficient.</answer>']),
                      tools, self.ctx, rank)

    def state(self, **kw):
        return initial_state('recovery', 'Does treatment work?', 'p', 'f', 't', **kw)

    async def test_source_recovery_then_model_chooses_browse(self):
        p = ScriptModel([CHECK, SEARCH, BAD_SOURCE, BAD_SOURCE, BROWSE, UPDATE, STOP])
        tools = ScriptTools([FOUND, OPENED])
        result = await self.engine(p, tools).run(self.state())
        self.assertEqual(len(tools.calls), 2)
        self.assertEqual(result['budget']['actual'], 2)
        self.assertEqual(result['decision_budget']['rejected'], 2)
        self.assertEqual(result['run_status'], 'complete')
        self.assertEqual(result['evidence']['E1']['text'], 'Observed outcome.')
        next_input = json.loads(p.requests[4]['messages'][1]['content'])
        self.assertFalse(next_input['feedback']['executed'])
        self.assertIn('source_not_in_visible_candidates', next_input['feedback']['message'])
        self.assertNotIn('decision_budget', next_input)
        self.assertNotIn('decision_budget', next_input['feedback'])
        self.assertEqual(result['decision_budget']['rejected'], 2)
        self.assertEqual(result['budget']['actual'], next_input['budget']['actual'] + 1)
        for request in p.requests:
            text = json.dumps(request['messages'])
            self.assertNotIn('"decision_budget"', text)
            self.assertNotIn('Rejected-decision allowance remaining=', text)
        self.assertEqual(len(self.store.events('recovery', 'decision_rejection_budget')), 2)
        keys = [x['data']['receipt_key'] for x in self.store.events('recovery', 'generation')]
        self.assertEqual(len(keys), len(set(keys)))

    async def test_permanent_source_rejection_bounded_then_final_without_fake_evidence(self):
        p = ScriptModel([CHECK, SEARCH] + [BAD_SOURCE] * 6)
        tools = ScriptTools([FOUND])
        result = await self.engine(p, tools).run(self.state())
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(result['budget']['charged'], 1)
        self.assertEqual(len(result['history']), 1)
        self.assertEqual(result['decision_budget']['rejected'], 6)
        self.assertEqual(result['termination'], 'decision_failure_budget_exhausted')
        self.assertEqual(result['process_status'], 'degraded')
        self.assertEqual(result['run_status'], 'degraded_complete')
        self.assertEqual(result['final_status'], 'complete')
        self.assertEqual(result['prefinal']['completion_mode'], 'degraded')
        self.assertEqual(result['prefinal']['opened_evidence'], {})
        self.assertIn('R1', result['prefinal']['unresolved'])
        self.assertEqual(result['prefinal']['decision_failure']['decision_budget']['rejected'], 6)
        # The stored Pre-Final remains complete, but the model-facing projection
        # omits the numeric internal rejection allowance.
        visible_final = json.loads(self.ctx.build('final', result)['messages'][1]['content'])
        self.assertNotIn('decision_budget', visible_final['decision_failure'])

    def test_rejection_budget_hidden_without_mutating_session_or_raw_audit(self):
        s = self.state()
        s['requirements'] = {'R1': 'Assess effectiveness.'}
        s['assessments'] = {'R1': dict(support='missing', evidence_ids=[], gap='Unknown.', conflict='')}
        s['decision_budget']['rejected'] = 2
        s['feedback'] = dict(error_code='repeated_search_already_executed',
            decision_budget=copy.deepcopy(s['decision_budget']), message='Choose a meaningful different query or source.')
        before = copy.deepcopy(s)
        for stage in ('decision', 'state'):
            request = self.ctx.build(stage, s)
            visible = json.loads(request['messages'][1]['content'])
            self.assertNotIn('decision_budget', visible)
            self.assertNotIn('decision_budget', visible['feedback'])
            self.assertEqual(visible['budget'], before['budget'])
            self.assertEqual(visible['question'], before['question'])
            self.assertEqual(visible['state'], before['assessments'])
            self.assertEqual(request['raw_context_payload']['feedback']['decision_budget'], before['decision_budget'])
        self.assertEqual(s, before)

    async def test_degraded_final_preserves_existing_evidence_and_state(self):
        p = ScriptModel([CHECK, SEARCH, BROWSE, UPDATE] + [BAD_SOURCE] * 2)
        tools = ScriptTools([FOUND, OPENED])
        final = ScriptModel(['<answer>Limited observation [E1].</answer>'])
        result = await self.engine(p, tools, final).run(self.state(max_invalid_decisions=2))
        self.assertEqual(result['prefinal']['opened_evidence'], result['evidence'])
        self.assertEqual(result['prefinal']['state'], result['assessments'])
        self.assertEqual(result['assessments']['R1']['support'], 'partial')
        self.assertEqual(result['run_status'], 'degraded_complete')

    async def test_final_unknown_citation_still_fails(self):
        p = ScriptModel([CHECK, SEARCH] + [BAD_SOURCE] * 2)
        f = ScriptModel(['<answer>Unsupported [E1].</answer>'] * 2)
        result = await self.engine(p, ScriptTools([FOUND]), f).run(self.state(max_invalid_decisions=2))
        self.assertEqual(result['run_status'], 'failed')
        self.assertEqual(result['final_status'], 'failed')
        self.assertEqual(result['process_termination'], 'decision_failure_budget_exhausted')

    async def test_process_only_degraded_does_not_call_final(self):
        f = ScriptModel([])
        result = await self.engine(ScriptModel([CHECK, SEARCH] + [BAD_SOURCE] * 2), ScriptTools([FOUND]), f).run(
            self.state(process_only=True, max_invalid_decisions=2))
        self.assertEqual(f.requests, [])
        self.assertEqual(result['run_status'], 'degraded_process_complete')
        self.assertEqual(result['final_status'], 'not_requested')

    async def test_successful_tool_does_not_reset_rejected_output_allowance(self):
        p = ScriptModel([CHECK, SEARCH, BAD_SOURCE, BROWSE, UPDATE] + [BAD_SOURCE] * 5)
        result = await self.engine(p, ScriptTools([FOUND, OPENED])).run(self.state())
        self.assertEqual(result['decision_budget']['rejected'], 6)
        self.assertEqual(result['budget']['actual'], 2)
        self.assertEqual(result['run_status'], 'degraded_complete')
        self.assertEqual(len(p.outputs), 0)  # sixth rejection ends without a seventh request

    async def test_schema_error_still_fails_closed(self):
        f = ScriptModel([])
        result = await self.engine(ScriptModel([CHECK, {'action': 'research'}, {'action': 'research'}]),
                                   ScriptTools([]), f).run(self.state())
        self.assertEqual(result['termination'], 'model_contract_failure')
        self.assertEqual(f.requests, [])

    async def test_source_rejection_categories_return_to_decision(self):
        for category in ('blocked', 'invisible', 'alias', 'retry_limit', 'nonretryable'):
            with self.subTest(category=category):
                s = self.state()
                s['run_id'] = category
                s['next_node'] = 'decision'
                s['requirements'] = {'R1': 'Assess effectiveness.'}
                s['assessments'] = {'R1': dict(support='missing', evidence_ids=[], gap='Unknown.', conflict='')}
                s['candidates'] = FOUND['candidates']
                bad = copy.deepcopy(BROWSE)
                if category == 'blocked': s['blocked_sources'] = {'PMID:1': 'blocked'}
                if category == 'invisible': bad['source_id'] = 'PMID:999'
                if category == 'alias': bad['source_id'] = 'ZZ'
                if category in ('retry_limit', 'nonretryable'):
                    s['history'] = [dict(action=BROWSE, feedback=dict(status='environment_failure',
                        retryable=category != 'nonretryable'))] * (2 if category == 'retry_limit' else 1)
                p = ScriptModel([bad, bad, STOP]); tools = ScriptTools([])
                result = await self.engine(p, tools).run(s)
                self.assertEqual(result['run_status'], 'complete')
                self.assertEqual(result['decision_budget']['rejected'], 2)
                self.assertEqual(tools.calls, [])

    async def test_checkpoint_resume_keeps_rejection_budget(self):
        s = self.state()
        s['requirements'] = {'R1': 'Assess effectiveness.'}
        s['assessments'] = {'R1': dict(support='missing', evidence_ids=[], gap='Unknown.', conflict='')}
        s['history'] = [dict(action=SEARCH, feedback=dict(status='ok'))]
        s['next_node'] = 'decision'
        p = ScriptModel([BAD_SOURCE] * 2)
        engine = self.engine(p, ScriptTools([]))
        saved = (await engine.decision({'session': s}))['session']
        result = await self.engine(ScriptModel([BAD_SOURCE] * 4), ScriptTools([])).run(self.state())
        self.assertEqual(result['decision_budget']['rejected'], 6)
        # Two more rejected Decision rounds plus one accepted Final generation.
        self.assertEqual(result['generation'], saved['generation'] + 3)

    def test_invalid_configuration_rejected(self):
        for value in (True, 0, 1, 13, 2.5):
            with self.assertRaisesRegex(ValueError, 'invalid_decision_budget'):
                self.state(max_invalid_decisions=value)


if __name__ == '__main__': unittest.main()
