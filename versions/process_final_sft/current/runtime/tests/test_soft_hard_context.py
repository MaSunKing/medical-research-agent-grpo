"""Soft budgets are compression targets, never mislabeled as hard failures."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

from trainable_agent.context import ContextBuilder, ContextOverflow, OfflineCodec
from trainable_agent.compression import card
from trainable_agent.engine import Engine, initial_state
from trainable_agent.storage import Store


def fixture(size=1):
    s=initial_state('soft-fixture','Assess benefits and harms.','p','f','t')
    s['requirements']={'R1':'Assess benefits and harms.'}
    s['assessments']={'R1':dict(support='partial',evidence_ids=['E1'],gap='Long-term uncertainty.',conflict='')}
    text='No benefit was established. '+('Independent archive sentence. '*size)+'Long-term safety is unknown.'
    s['evidence']={'E1':dict(source_id='WEB:one',chunk_id='c1',text=text)}
    s['new_evidence_ids']=['E1']
    s['prefinal']=dict(question=s['question'],checklist=s['requirements'],state=s['assessments'],
                       opened_evidence=copy.deepcopy(s['evidence']),compact_history=[],unresolved=s['assessments'])
    return s


def total(b,req): return len(req['prompt_token_ids'])+req['max_new_tokens']+b.input_padding


class Decline:
    def __init__(self):
        from trainable_agent.context_processing.extraction import CompressionBudget
        self.budget=CompressionBudget(1)
    async def transform(self,*args):
        self.budget.calls+=1
        return None


class SoftHardTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_four_stages_soft_success_and_no_limit_mutation(self):
        b=ContextBuilder(OfflineCodec(),soft_context_tokens=12288,hard_context_tokens=15000,repair_reserve_tokens=384,safety_margin=128)
        s=fixture(); original=copy.deepcopy(s)
        for stage in ('checklist','decision','state','final'):
            req=await b.prepare(stage,s)
            self.assertTrue(req['context_audit']['budget_policy']['soft_target_met'])
            self.assertLessEqual(total(b,req),12288)
            self.assertEqual(b.limit_for(stage),15000)
        self.assertIsNone(b._hard_candidates)
        self.assertEqual(s,original)

    async def test_each_stage_between_soft_and_hard_is_allowed(self):
        s=fixture(80)
        for stage in ('checklist','decision','state','final'):
            with self.subTest(stage=stage):
                b=ContextBuilder(OfflineCodec(),soft_context_tokens=100000,hard_context_tokens=100000,safety_margin=128,repair_reserve_tokens=384)
                initial=b.build(stage,s,allow_cards=False)
                cost=total(b,initial)
                b.soft_limit=cost-500; b.hard_limit=cost+500
                req=await b.prepare(stage,s)
                self.assertLessEqual(total(b,req),b.hard_limit)
                self.assertFalse(req['context_audit']['budget_policy']['soft_target_met'])
                self.assertEqual(req['raw_context_payload']['question'],s['question'])

    async def test_declined_compression_returns_safe_original_under_hard(self):
        s=fixture(120); b=ContextBuilder(OfflineCodec(),soft_context_tokens=5000,hard_context_tokens=15000)
        before=copy.deepcopy(s); fallback=Decline()
        req=await b.prepare('state',s,fallback=fallback)
        self.assertGreater(fallback.budget.calls,0)
        self.assertLessEqual(total(b,req),15000)
        self.assertEqual(s,before)
        self.assertIn('No benefit was established',req['messages'][1]['content'])

    async def test_partial_valid_projection_retained_when_soft_still_unreachable(self):
        s=fixture(); s['evidence']['E1']['text']=' '.join('Original observation '+str(i)+' '+('x'*400)+'.' for i in range(7))
        b=ContextBuilder(OfflineCodec(),soft_context_tokens=100000,hard_context_tokens=100000)
        raw=b.build('state',s,allow_cards=False)
        short=card(s['evidence']['E1'],['S0','S1','S2'])
        projected=b.build('state',s,overrides={'E1':short},allow_cards=False)
        rawcost,smallcost=total(b,raw),total(b,projected)
        b.soft_limit=smallcost-800; b.hard_limit=(smallcost+rawcost)//2
        class Partial(Decline):
            async def transform(self,*args):
                self.budget.calls+=1
                return short
        req=await b.prepare('state',s,fallback=Partial())
        self.assertLessEqual(total(b,req),b.hard_limit)
        self.assertGreater(total(b,req),b.soft_limit)
        self.assertTrue(req['context_audit']['evidence_compressed'])
        self.assertEqual(req['raw_context_payload']['opened_evidence'],s['evidence'])

    async def test_floor_probe_never_leaks_blank_evidence_as_accepted_input(self):
        s=fixture(1000); b=ContextBuilder(OfflineCodec(),soft_context_tokens=2500,hard_context_tokens=4000)
        with self.assertRaises(ContextOverflow): await b.prepare('state',s,fallback=Decline())

    async def test_checklist_fixed_question_above_hard_still_fails(self):
        s=fixture(); s['question']='Original protected input. '*1000
        b=ContextBuilder(OfflineCodec(),soft_context_tokens=12288,hard_context_tokens=15000)
        with self.assertRaises(ContextOverflow): await b.prepare('checklist',s)

    async def test_provider_program_errors_are_not_hidden_by_hard_fallback(self):
        class Broken(Decline):
            async def transform(self,*args): raise RuntimeError('fixture backend error')
        b=ContextBuilder(OfflineCodec(),soft_context_tokens=5000,hard_context_tokens=15000)
        with self.assertRaisesRegex(RuntimeError,'fixture backend error'):
            await b.prepare('state',fixture(120),fallback=Broken())

    async def test_correction_above_soft_still_obeys_hard_and_one_retry(self):
        class Model:
            calls=0
            async def generate(self,req):
                self.calls+=1
                return dict(text='bad' if self.calls==1 else 'good')
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/'audit.sqlite')
            try:
                s=fixture(120); b=ContextBuilder(OfflineCodec(),soft_context_tokens=3000,hard_context_tokens=15000,safety_margin=128,repair_reserve_tokens=384)
                p=Model(); e=Engine(store,p,p,None,b,None)
                def check(text):
                    if text=='bad':raise ValueError('source_not_in_visible_candidates')
                    return text
                self.assertEqual(await e.generate('state',s,check),'good')
                self.assertEqual(p.calls,2)
            finally: store.db.close()

    async def test_two_simultaneous_preparations_do_not_share_candidate_sinks(self):
        b=ContextBuilder(OfflineCodec(),soft_context_tokens=3000,hard_context_tokens=15000)
        a,c=fixture(100),fixture(150);c['question']='A different immutable question.'
        r1,r2=await asyncio.gather(b.prepare('state',a,fallback=Decline()),b.prepare('state',c,fallback=Decline()))
        self.assertEqual(r1['raw_context_payload']['question'],a['question'])
        self.assertEqual(r2['raw_context_payload']['question'],c['question'])
        self.assertIsNone(b._hard_candidates)

    async def test_equal_limits_are_supported(self):
        b=ContextBuilder(OfflineCodec(),soft_context_tokens=15000,hard_context_tokens=15000)
        self.assertLessEqual(total(b,await b.prepare('state',fixture())),15000)

    async def test_missing_or_invalid_soft_hard_pair_rejected(self):
        for kwargs in (dict(soft_context_tokens=12288),dict(hard_context_tokens=15000),
                       dict(soft_context_tokens=16000,hard_context_tokens=15000),
                       dict(soft_context_tokens=True,hard_context_tokens=15000),
                       dict(soft_context_tokens=0,hard_context_tokens=15000)):
            with self.assertRaises(ValueError):ContextBuilder(OfflineCodec(),**kwargs)
