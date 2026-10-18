"""Offline capacity/repair/dependency checks; no model or retrieval calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from trainable_agent.context import ContextBuilder, ContextOverflow, OfflineCodec
from trainable_agent.context_processing.final_pack import pack_state, unpack_state
from trainable_agent.context_processing.fallback import ScopedFallback
from trainable_agent.context_processing.extraction import SelectionRunner, text_tokens
from trainable_agent.engine import Engine, initial_state
from trainable_agent.storage import Store


def state():
    s = initial_state('capacity-fixture', 'Compare intervention A with placebo.', 'p', 'f', 't')
    s['requirements'] = {'R1':'Determine the effect and uncertainty.'}
    s['assessments'] = {'R1':dict(support='partial', evidence_ids=['E1'], gap='Long-term effects unknown.', conflict='')}
    s['evidence'] = {'E1':dict(source_id='WEB:one', chunk_id='c1', text='No benefit was established. Long-term effects remain uncertain.')}
    s['new_evidence_ids'] = ['E1']
    return s


class BudgetTests(unittest.TestCase):
    def test_independent_stage_windows(self):
        b = ContextBuilder(OfflineCodec(), 10240, state_context_tokens=15000, final_context_tokens=12288, repair_reserve_tokens=384)
        self.assertEqual([b.limit_for(x) for x in ('checklist','decision','state','final')], [10240,10240,15000,12288])
        self.assertEqual(b.reserves['state'], 1200)
        self.assertEqual(b.input_padding, 384)

    def test_bad_budget_configs_rejected(self):
        for value in (True, 0, -1, 1.5, '15000'):
            with self.assertRaises(ValueError): ContextBuilder(OfflineCodec(), state_context_tokens=value)
        for value in (True, -1, 2049, 1.5):
            with self.assertRaises(ValueError): ContextBuilder(OfflineCodec(), repair_reserve_tokens=value)

    def test_fixed_floor_still_fails_closed(self):
        s = state(); s['question'] = 'Original protected question. '*1000
        with self.assertRaises(ContextOverflow):
            ContextBuilder(OfflineCodec(), state_context_tokens=15000, repair_reserve_tokens=384).build('state',s,allow_cards=False)

    def test_exact_repair_space_reserved_and_full_request_remeasured(self):
        s = state(); b = ContextBuilder(OfflineCodec(), 100000, safety_margin=128, repair_reserve_tokens=384)
        req = b.build('decision',s,allow_cards=False)
        b.limit = len(req['prompt_token_ids'])+400+128+384
        base = b.build('decision',s,allow_cards=False)
        repaired = b.repair('decision',base,'source_not_in_visible_candidates: '+'x'*1400)
        self.assertLessEqual(len(repaired['prompt_token_ids'])+400+128,b.limit)
        self.assertLessEqual(len(repaired['prompt_token_ids'])-len(base['prompt_token_ids']),384)
        self.assertEqual(repaired['messages'][:-1],base['messages'])
        self.assertIn('source_not_in_visible_candidates',repaired['messages'][-1]['content'])

    def test_repair_without_reserved_space_preserves_legacy_failure(self):
        b = ContextBuilder(OfflineCodec(), 100000, safety_margin=128)
        req = b.build('decision',state(),allow_cards=False)
        b.limit = len(req['prompt_token_ids'])+400+128
        with self.assertRaisesRegex(ContextOverflow,'repair_context_overflow'):
            b.repair('decision',req,'source_not_in_visible_candidates')

    def test_state_pack_is_exactly_reversible_and_no_evidence_changed(self):
        s = state(); warning = 'These are selected parsed passages, not verified complete coverage. '*8
        scope = dict(full_document_verified=False, scope='selected passages only '*10)
        s['history'] = [dict(turn=i,action=dict(action='browse',source_id='WEB:one',reading_focus='effect'),
            feedback=dict(retrieved_scope_message=warning, content_scope_message=warning,
                retrieved_content_scopes=[dict(evidence_id='E1',scope=scope)], custom='keep')) for i in range(6)]
        payload = dict(history=s['history'], question=s['question'], state=s['assessments'], opened_evidence=s['evidence'])
        packed,audit = pack_state(payload)
        self.assertTrue(audit['applied']); self.assertEqual(unpack_state(packed),payload)
        self.assertEqual(packed['opened_evidence'],s['evidence'])
        before = copy.deepcopy(s)
        req = ContextBuilder(OfflineCodec(),100000).build('state',s,allow_cards=False)
        self.assertTrue(req['context_audit']['state_pack']['applied'])
        self.assertEqual(req['raw_context_payload']['history'],s['history'])
        self.assertEqual(s,before)

    def test_internal_browse_hash_archived_not_sent_as_evidence(self):
        s = state(); s['feedback'] = dict(browse_context_sha256='a'*64,status='source_unreadable')
        s['history'] = [dict(action=dict(action='browse'),feedback=copy.deepcopy(s['feedback']))]
        req = ContextBuilder(OfflineCodec(),100000).build('state',s,allow_cards=False)
        self.assertNotIn('a'*64,req['messages'][1]['content'])
        self.assertEqual(req['raw_context_payload']['feedback']['browse_context_sha256'],'a'*64)
        self.assertEqual(req['context_audit']['internal_feedback_hashes_archived']['current'],'a'*64)


class Model:
    identity = 'offline-fixture-not-a-real-model'
    def __init__(self): self.calls = []
    async def generate(self, request):
        self.calls.append(request)
        return dict(text='{"passage_ids":["S0"]}')


class DependencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_call_for_infeasible_parent_child_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder)/'audit.sqlite')
            try:
                model = Model(); fallback = ScopedFallback(model,OfflineCodec(),store,'fixture',10000)
                chunk = dict(source_id='WEB:one',chunk_id='c1',text='Important original qualification '+'q'*180+':\n\nShort outcome.')
                self.assertIsNone(await fallback.compress(chunk,dict(question='q',checklist={}),120))
                self.assertEqual(model.calls,[])
            finally: store.db.close()

    async def test_feasible_choices_include_parent_and_count_separator_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder)/'audit.sqlite')
            try:
                model = Model(); fallback = ScopedFallback(model,OfflineCodec(),store,'fixture',10000)
                chunk = dict(source_id='WEB:one',chunk_id='c1',text='Required context:\n\nA negative result.\n\nAnother result '+ 'detail '*60+'.')
                await fallback.compress(chunk,dict(question='q',checklist={}),300)
                payload = json.loads(model.calls[0]['messages'][1]['content'])
                self.assertEqual(payload['feasible_original_selections']['S1']['passage_ids'],['S0','S1'])
                self.assertGreater(payload['feasible_original_selections']['S1']['total_text_tokens'],len('A negative result.'))
            finally: store.db.close()

    async def test_dependency_is_present_in_each_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder)/'audit.sqlite')
            try:
                model = Model(); runner = SelectionRunner(model,OfflineCodec(),store,'fixture',350)
                parts = {'S0':'Intro.', 'S1':'x'*100, 'S2':'y'*100}
                def messages(visible): return [dict(role='user',content=json.dumps(visible))]
                await runner.select(parts,messages,lambda x:x['passage_ids'],lambda ns:' '.join(parts[n] for n in ns),
                    target=200,max_ids=3,reserve=10,kind='fixture_compression',dependencies={'S1':['S0'],'S2':['S0']})
                self.assertGreaterEqual(len(model.calls),2)
                for req in model.calls:
                    p=json.loads(req['messages'][0]['content'])
                    if 'S1' in p or 'S2' in p: self.assertIn('S0',p)
            finally: store.db.close()
