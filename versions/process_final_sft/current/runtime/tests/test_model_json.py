import unittest
import json
from trainable_agent.contracts import parse_model_json
from trainable_agent.model_json import normalize_model_json


class ModelJSONTests(unittest.TestCase):
    def test_nested_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_json_key:requirement_id'):
            parse_model_json('{"updates":[{"requirement_id":"R1","requirement_id":"R3"}]}')

    def test_top_level_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_json_key:action'):
            parse_model_json('{"action":"search","action":"browse"}')

    def test_separate_updates_preserved(self):
        value = parse_model_json('{"updates":[{"requirement_id":"R1"},{"requirement_id":"R2"}]}')
        self.assertEqual(len(value['updates']), 2)

    def test_valid_json_unchanged(self):
        text = '  {"gap":"1,} and 2 mg; not 20 mg\\nnext", "updates":[]}\n'
        normalized, audit = normalize_model_json(text)
        self.assertEqual(normalized, text)
        self.assertIsNone(audit)

    def test_only_known_complete_wrappers(self):
        for text in ('```json\n{"updates":[]}\n```', '({"updates":[]})',
                     '({"updates":[]}', '\ufeff{"updates":[]}', '{"updates":[]}”',
                     '{"updates":[]}"', '{"updates":[]}’'):
            with self.subTest(text=text):
                normalized, audit = normalize_model_json(text)
                self.assertEqual(json.loads(normalized), {"updates": []})
                self.assertTrue(audit['applied'])
                self.assertFalse(audit['model_generated'])

    def test_nested_trailing_commas_and_untouched_string(self):
        value = parse_model_json('{"requirements":["Keep 1 mg, not 10 mg; literal ,} stays",],}')
        self.assertEqual(value, {'requirements': ['Keep 1 mg, not 10 mg; literal ,} stays']})

    def test_literal_newline_cr_tab_preserve_string_contents(self):
        normalized, audit = normalize_model_json('{"requirements":["line1\nline2\r\t5 mg"]}')
        self.assertEqual(json.loads(normalized)['requirements'], ['line1\nline2\r\t5 mg'])
        self.assertIn('escape_literal_string_whitespace', audit['changes'])

    def test_ambiguous_or_incomplete_text_rejected(self):
        for text in ('Here is JSON: {"updates":[]}', '{"updates":[]} explanation',
                     '{"updates":[]} {"updates":[1]}', '{"requirements":["truncated',
                     '{"updates":[]', '{"gap":"a" "conflict":""}',
                     "{'updates':[]}", '{“updates”:[]}', '{,"updates":[]}',
                     '{"updates":[],,}', '{"gap":"x\x00y"}',
                     '{"gap":"x\\\ny"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_json(text)

    def test_repair_does_not_silently_deduplicate(self):
        for text in ('```json\n{"action":"search","action":"browse"}\n```',
                     '{"updates":[{"gap":"a","gap":"b",}],}'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'duplicate_json_key:'):
                parse_model_json(text)

    def test_schema_and_id_limits_not_repaired(self):
        from trainable_agent.contracts import validate_action, validate_delta
        state = {'requirements': {'R1': 'one', 'R2': 'two', 'R3': 'three'},
                 'evidence': {}, 'candidates': [], 'history': []}
        invalid = '{"action":"search","requirement_ids":["R1","R2","R3"],"channel":"paper","query":"evidence",}'
        with self.assertRaises(ValueError):
            validate_action(parse_model_json(invalid), state)
        invalid = '{"updates":[{"requirement_id":"R1","support":"partial","evidence_ids":["E1"],"gap":"unknown","conflict":"",}],}'
        with self.assertRaisesRegex(ValueError, 'evidence_not_opened'):
            validate_delta(parse_model_json(invalid), state)


class JSONRepairGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_process_nodes_use_repair_without_new_calls(self):
        import tempfile
        from pathlib import Path
        from test_agent import ScriptModel, ScriptTools, CHECK, SEARCH, BROWSE, UPDATE, STOP, FOUND, OPENED, rank
        from trainable_agent.context import ContextBuilder, OfflineCodec
        from trainable_agent.engine import Engine, initial_state
        from trainable_agent.storage import Store
        from trainable_agent.training import sft_samples
        class ExactFixture(OfflineCodec):
            exact = True  # Synthetic codec, not a real model tokenization claim.
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/'db.sqlite')
            codec = ExactFixture()
            outputs = ['```json\n'+json.dumps(CHECK)+'\n```',
                       '('+json.dumps(SEARCH), json.dumps(BROWSE),
                       json.dumps(UPDATE) + '”', STOP]
            model, tools = ScriptModel(outputs), ScriptTools([FOUND, OPENED])
            state = initial_state('repair', 'Does treatment work?', 'p', 'f', 't', process_only=True)
            try:
                result = await Engine(store, model, ScriptModel([]), tools,
                                      ContextBuilder(codec, 100000), rank).run(state)
                self.assertEqual(result['run_status'], 'process_complete')
                self.assertEqual(len(model.requests), 5)
                self.assertEqual(len(tools.calls), 2)
                self.assertEqual(result['decision_budget']['rejected'], 0)
                self.assertEqual(result['assessments']['R1']['support'], 'partial')
                self.assertEqual(result['evidence']['E1']['text'], 'Observed outcome.')
                rows = [r['data'] for r in store.events('repair', 'generation')]
                self.assertEqual(rows[0]['response']['text'], outputs[0])
                self.assertEqual(rows[3]['response']['text'], outputs[3])
                repairs = [r['data'] for r in store.events('repair', 'model_json_repair')]
                self.assertEqual([r['stage'] for r in repairs], ['checklist', 'decision', 'state'])
                self.assertTrue(all(r['schema_accepted'] for r in repairs))
                exported = list(sft_samples(store, 'repair', codec))
                self.assertEqual([r['stage'] for r in exported], ['decision', 'decision'])
                self.assertTrue(all('```' not in r['target'] and not r['target'].endswith('”') for r in exported))
            finally:
                store.db.close()

    async def test_bad_action_and_evidence_still_rejected_after_syntax_repair(self):
        import tempfile
        from pathlib import Path
        from test_agent import ScriptModel, ScriptTools, CHECK, STOP, rank
        from trainable_agent.context import ContextBuilder, OfflineCodec
        from trainable_agent.engine import Engine, initial_state
        from trainable_agent.storage import Store
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/'db.sqlite')
            invalid = '{"action":"browse","requirement_ids":["R1"],"source_id":"E8","reading_focus":"read",}'
            policy = ScriptModel([CHECK, invalid, invalid])
            tools = ScriptTools([])
            try:
                result = await Engine(store, policy, ScriptModel([]), tools,
                    ContextBuilder(OfflineCodec(), 100000), rank).run(
                    initial_state('bad', 'Question?', 'p','f','t',process_only=True))
                self.assertEqual(result['termination'], 'model_contract_failure')
                self.assertEqual(len(policy.requests), 3)
                self.assertEqual(tools.calls, [])
                self.assertEqual(result['decision_budget']['rejected'], 2)
                repairs = [r['data'] for r in store.events('bad', 'model_json_repair')]
                self.assertEqual(len(repairs), 2)
                self.assertTrue(all(not r['schema_accepted'] for r in repairs))
            finally:
                store.db.close()

    async def test_final_not_repaired(self):
        import tempfile
        from pathlib import Path
        from test_agent import ScriptModel, ScriptTools, CHECK, STOP, rank
        from trainable_agent.context import ContextBuilder, OfflineCodec
        from trainable_agent.engine import Engine, initial_state
        from trainable_agent.storage import Store
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/'db.sqlite')
            final = ScriptModel(['```json\n{"answer":"text"}\n```']*2)
            try:
                result = await Engine(store, ScriptModel([CHECK, STOP]), final, ScriptTools([]),
                    ContextBuilder(OfflineCodec(),100000),rank).run(initial_state('final','Question?','p','f','t'))
                self.assertEqual(result['final_status'], 'failed')
                self.assertEqual(len(final.requests), 2)
                self.assertEqual(store.events('final', 'model_json_repair'), [])
            finally:
                store.db.close()
