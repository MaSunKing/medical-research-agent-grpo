import copy
import json
import tempfile
import unittest
from pathlib import Path

from trainable_agent.contracts import validate_action, near_duplicate_search_warning
from trainable_agent.engine import Engine, initial_state
from trainable_agent.context import ContextBuilder, OfflineCodec
from trainable_agent.storage import Store
from test_agent import ScriptModel, ScriptTools, CHECK, SEARCH, BROWSE, UPDATE, STOP, FOUND, OPENED, rank


class RepeatSearch(unittest.TestCase):
    def setUp(self):
        self.state = initial_state('repeat', 'Question', 'p', 'f', 't')
        self.state['requirements'] = {'R1': 'Cure', 'R2': 'Recurrence'}

    def row(self, status='ok', **feedback):
        return dict(turn=1, action=copy.deepcopy(SEARCH), feedback=dict(status=status, **feedback))

    def test_one_normal_or_empty_result_allows_cached_search_after_browse(self):
        for status in ['ok', 'empty']:
            with self.subTest(status=status):
                self.state['history'] = [self.row(status), dict(action=BROWSE, feedback={'new_evidence_ids': ['E1']})]
                before = copy.deepcopy(self.state)
                validate_action(SEARCH, self.state)
                self.assertEqual(self.state, before)

    def test_case_whitespace_and_requirement_changes_do_not_bypass(self):
        self.state['history'] = [self.row()]
        validate_action({**SEARCH, 'query': '  TREATMENT\t effectiveness  ', 'requirement_ids': ['R2']}, self.state)

    def test_other_channel_and_real_refinement_are_allowed(self):
        self.state['history'] = [self.row()]
        validate_action({**SEARCH, 'channel': 'web'}, self.state)
        validate_action({**SEARCH, 'query': 'treatment recurrence adverse effects'}, self.state)

    def test_retryable_failure_only_one_retry_and_no_browse_reset(self):
        self.state['history'] = [self.row('environment_failure', retryable=True)]
        validate_action(SEARCH, self.state)
        self.state['history'].extend([dict(action=BROWSE, feedback={'new_evidence_ids': ['E1']}), self.row('environment_failure', retryable=True)])
        with self.assertRaisesRegex(ValueError, 'retry_limit'):
            validate_action(SEARCH, self.state)

    def test_retry_success_allows_cache_and_nonretryable_or_unclear_failures_block(self):
        self.state['history'] = [self.row('environment_failure', retryable=True), self.row()]
        validate_action(SEARCH, self.state)
        for feedback in [dict(status='environment_failure', retryable=False), dict(status='environment_failure'), dict(status='invalid_request')]:
            self.state['history'] = [dict(action=SEARCH, feedback=feedback)]
            with self.assertRaisesRegex(ValueError, 'nonretryable_search_failure'):
                validate_action(SEARCH, self.state)

    def test_checkpoint_history_survives_roundtrip(self):
        self.state['history'] = [self.row()]
        restored = json.loads(json.dumps(self.state))
        validate_action(SEARCH, restored)

    def test_lexical_warning_does_not_block_refinement_or_other_channel(self):
        original = {**SEARCH, 'query': 'adult treatment versus placebo mortality at 30 days'}
        refined = {**original, 'query': 'adult treatment versus placebo mortality at 90 days'}
        self.state['history'] = [dict(turn=3, action=original, feedback={'status': 'ok'})]
        validate_action(refined, self.state)
        warning = near_duplicate_search_warning(refined, self.state)
        self.assertEqual(warning['previous_turn'], 3)
        self.assertIn('not proven semantically identical', warning['message'])
        self.assertIsNone(near_duplicate_search_warning({**refined, 'channel': 'web'}, self.state))
        self.assertIsNone(near_duplicate_search_warning(original, self.state))


class RepeatIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'r.sqlite')
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.store.db.close)

    def engine(self, policy, tools):
        return Engine(self.store, policy, ScriptModel([]), tools, ContextBuilder(OfflineCodec(), 100000), rank)

    async def test_duplicate_after_new_evidence_cached_with_normal_action_cost(self):
        s = initial_state('r', 'Question', 'p', 'f', 't', process_only=True)
        p = ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, SEARCH, STOP])
        tools = ScriptTools([FOUND, OPENED])
        result = await self.engine(p, tools).run(s)
        self.assertEqual(result['budget']['actual'], 3)
        self.assertEqual(result['budget']['charged'], 3)
        self.assertEqual(len(tools.calls), 2)
        rejected = [e for e in self.store.events('r', 'generation') if not e['data']['accepted']]
        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(self.store.events('r', 'search_cache_hit')), 1)
        self.assertEqual(result['prefinal']['opened_evidence'], result['evidence'])

    async def test_similarity_advisory_is_archived_and_visible_without_extra_calls(self):
        s = initial_state('r', 'Question', 'p', 'f', 't', process_only=True)
        first = {**SEARCH, 'query': 'adult treatment versus placebo mortality at 30 days'}
        second = {**first, 'query': 'adult treatment versus placebo mortality at 90 days'}
        p = ScriptModel([CHECK, first, second, STOP])
        tools = ScriptTools([FOUND, {'status': 'empty'}])
        result = await self.engine(p, tools).run(s)
        self.assertEqual(result['budget']['actual'], 2)
        self.assertEqual(len(p.requests), 4)
        self.assertEqual(len(self.store.events('r', 'search_query_similarity_warning')), 1)
        self.assertIn('query_similarity_warning', result['history'][-1]['feedback'])
        last = json.loads(p.requests[-1]['messages'][1]['content'])
        self.assertIn('query_similarity_warning', last['feedback'])

    async def test_retry_wait_applies_and_success_caches_third_search(self):
        from unittest.mock import patch, AsyncMock
        s = initial_state('r', 'Question', 'p', 'f', 't', process_only=True)
        p = ScriptModel([CHECK, SEARCH, SEARCH, SEARCH, STOP])
        tools = ScriptTools([{'status': 'environment_failure', 'retryable': True, 'retry_after_seconds': 7}, FOUND])
        with patch('asyncio.sleep', new_callable=AsyncMock) as wait:
            result = await self.engine(p, tools).run(s)
            wait.assert_awaited_once_with(7)
        self.assertEqual(result['budget']['actual'], 3)
        self.assertEqual(result['budget']['exemptions'], 1)
        self.assertEqual(len(tools.calls), 2)
