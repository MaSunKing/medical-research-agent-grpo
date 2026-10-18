import copy
import json
import tempfile
import unittest
from pathlib import Path
from trainable_agent.context import ContextBuilder, OfflineCodec, ContextOverflow
from trainable_agent.context_processing.aliases import project, register
from trainable_agent.context_processing.fallback import ScopedFallback
from trainable_agent.engine import initial_state, Engine
from trainable_agent.storage import Store, canonical
from trainable_agent.training import sft_samples
from test_agent import ScriptModel, ScriptTools, CHECK, SEARCH, BROWSE, STOP, UPDATE, FOUND, OPENED, rank


def fixture():
    s=initial_state('latest','Compare benefits and harms of exercise in adults.','p','f','t',process_only=True)
    s['requirements']={'R1':'Establish benefits in adults.','R2':'Establish harms in adults.'}
    s['assessments']={rid:dict(support='missing',evidence_ids=[],gap=text,conflict='') for rid,text in s['requirements'].items()}
    s['evidence']={'E1':dict(source_id='PMID:1',chunk_id='old',title='Earlier',text='Earlier evidence body only.'),
                   'E2':dict(source_id='PMID:2',chunk_id='new',title='Recent',text='No benefit in adults. Risk was 12% versus 8%.')}
    s['history']=[dict(turn=1,action={**BROWSE,'source_id':'PMID:1'},feedback=dict(status='ok',new_evidence_ids=['E1'])),
                  dict(turn=2,action={**SEARCH,'query':'exercise harms adults'},feedback=dict(status='ok')),
                  dict(turn=3,action={**BROWSE,'source_id':'PMID:2','reading_focus':'exercise harms adults','requirement_ids':['R2']},
                       feedback=dict(status='ok',new_evidence_ids=['E2']))]
    s['candidates']=[dict(source_id='PMID:2',channel='paper',title='Recent',preview='Harms in adults.',url='')]
    s['new_evidence_ids']=['E2']
    return s


class ObservationTests(unittest.TestCase):
    def test_sorted_checkpoint_keys_do_not_change_or_reject_source_aliases(self):
        s=initial_state('r','Q','p','f','t')
        s['candidates']=[{'source_id':'PMID:9'},{'source_id':'PMID:1'}]
        self.assertEqual(register(s),{'PMID:9':'A','PMID:1':'B'})
        restored=json.loads(canonical(s))
        restored['candidates'].append({'source_id':'PMID:5'})
        self.assertEqual(register(restored),{'PMID:9':'A','PMID:1':'B','PMID:5':'C'})

    def test_current_target_latest_raw_and_all_history_without_earlier_body(self):
        s=fixture();before=copy.deepcopy(s)
        req=ContextBuilder(OfflineCodec(),100000).build('decision',s)
        p=json.loads(req['messages'][1]['content'])
        self.assertEqual(set(p['opened_evidence']),{'E2'})
        self.assertEqual(p['opened_evidence']['E2']['text'],s['evidence']['E2']['text'])
        self.assertEqual(set(p['opened_evidence_index']),{'E1','E2'})
        self.assertNotIn('Earlier evidence body only',json.dumps(p))
        self.assertEqual(p['history'],project({'history':s['history']},s)['history'])
        self.assertEqual(p['current_target']['requirements'],{'R2':s['requirements']['R2']})
        self.assertEqual(p['current_target']['latest_search']['action']['query'],'exercise harms adults')
        self.assertEqual(p['latest_browse']['turn'],3)
        self.assertEqual(req['raw_context_payload']['opened_evidence']['E2'],s['evidence']['E2'])
        self.assertFalse(req['context_audit']['source_selection_isolated'])
        self.assertEqual(s,before)

    def test_failed_empty_or_later_search_keeps_previous_browse_with_explicit_age(self):
        for action,status in [(BROWSE,'environment_failure'),(BROWSE,'empty'),(SEARCH,'ok')]:
            s=fixture();s['history'].append(dict(turn=4,action=action,feedback=dict(status=status,new_evidence_ids=[])))
            s['new_evidence_ids']=[]
            restored=json.loads(json.dumps(s))
            p=json.loads(ContextBuilder(OfflineCodec(),100000).build('decision',restored)['messages'][1]['content'])
            self.assertEqual(p['latest_browse']['turn'],3)
            self.assertEqual(p['current_target']['turn'],4)
            self.assertEqual(p['current_target']['previous_status'],status)
            self.assertEqual(set(p['opened_evidence']),{'E2'})

    def test_before_first_tool_has_no_fixed_target_or_fabricated_evidence(self):
        s=initial_state('r','Question','p','f','t')
        p=json.loads(ContextBuilder(OfflineCodec(),100000).build('decision',s)['messages'][1]['content'])
        self.assertEqual(p['current_target']['requirements'],{})
        self.assertIsNone(p['latest_browse'])
        self.assertEqual(p['opened_evidence'],{})

    def test_mismatched_browse_history_fails_closed(self):
        for bad in ['E999','E1']:
            s=fixture();s['history'][-1]['feedback']['new_evidence_ids']=[bad]
            with self.assertRaisesRegex(ValueError,'history_mismatch'):
                ContextBuilder(OfflineCodec(),100000).build('decision',s)

    def test_state_still_reads_all_evidence_and_old_evidence_cannot_be_injected_into_decision(self):
        s=fixture();b=ContextBuilder(OfflineCodec(),100000)
        self.assertEqual(set(json.loads(b.build('state',s)['messages'][1]['content'])['opened_evidence']),{'E1','E2'})
        with self.assertRaisesRegex(ValueError,'projection_scope'):
            b.build('decision',s,overrides={'E1':s['evidence']['E1']})

    def test_latest_browse_containment_reference_has_visible_original_witness(self):
        s=fixture();text='No reduction was demonstrated in adults compared with control. '*3
        s['evidence']['E2']['text']=text
        s['evidence']['E3']={**s['evidence']['E2'],'chunk_id':'new3','text':text+'Additional context.'}
        s['history'][-1]['feedback']['new_evidence_ids']=['E2','E3']
        p=json.loads(ContextBuilder(OfflineCodec(),100000).build('decision',s)['messages'][1]['content'])
        ref=p['opened_evidence']['E2']['context_reference']
        self.assertEqual(p['opened_evidence'][ref['evidence_id']]['text'][ref['start']:ref['end']],text)
        self.assertEqual(set(p['opened_evidence']),{'E2','E3'})


class ExtractProvider:
    identity='offline-extraction-fixture'
    def __init__(self):self.requests=[]
    async def generate(self,request):
        self.requests.append(copy.deepcopy(request))
        p=json.loads(request['messages'][1]['content'])
        if 'candidates' in p:
            return {'text':json.dumps({'selections':{r['candidate_id']:[next(iter(r['passages']))] for r in p['candidates']}})}
        return {'text':json.dumps({'remove_line_ids':[]} if p['mode']=='clean' else {'passage_ids':['S0']})}


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_browse_can_be_extracted_without_history_or_source_rewrite(self):
        s=fixture()
        limit=ContextBuilder(OfflineCodec(),100000).search_measure(s,{})+400+128+1000
        s['evidence']['E2']['text']+=' '+ ' '.join(f'Background observation {i} was recorded separately.' for i in range(180))
        before=copy.deepcopy(s);provider=ExtractProvider()
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/'r.sqlite')
            try:
                f=ScopedFallback(provider,OfflineCodec(),store,'latest',50000)
                req=await ContextBuilder(OfflineCodec(),limit,safety_margin=128).prepare('decision',s,fallback=f)
                p=json.loads(req['messages'][1]['content'])
                self.assertLessEqual(len(req['prompt_token_ids'])+400+128,limit)
                self.assertTrue(req['context_audit']['evidence_compressed'])
                self.assertIn('No benefit in adults.',p['opened_evidence']['E2']['text'])
                self.assertFalse(p['opened_evidence']['E2']['evidence_card']['coverage_complete'])
                self.assertEqual(req['raw_context_payload']['opened_evidence']['E2'],before['evidence']['E2'])
                self.assertEqual(p['history'],project({'history':s['history']},s)['history'])
                self.assertEqual(s,before)
                for r in provider.requests:
                    payload=json.loads(r['messages'][1]['content'])
                    self.assertNotIn('history',payload)
                    self.assertNotIn('state',payload)
            finally:store.db.close()

    async def test_fixed_overflow_never_calls_compressor_or_drops_history(self):
        s=fixture();s['history'][0]['feedback']['message']='history '*5000;before=copy.deepcopy(s)
        provider=ExtractProvider()
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/'r.sqlite')
            try:
                f=ScopedFallback(provider,OfflineCodec(),store,'latest',50000)
                with self.assertRaises(ContextOverflow):
                    await ContextBuilder(OfflineCodec(),8000).prepare('decision',s,fallback=f)
                self.assertFalse(provider.requests)
                self.assertEqual(s,before)
            finally:store.db.close()

    async def test_existing_nodes_record_latest_browse_and_mask_entire_prompt_for_sft(self):
        class ExactFixtureCodec(OfflineCodec):
            exact=True  # A deterministic test codec, not a production tokenizer.
        codec=ExactFixtureCodec();policy=ScriptModel([CHECK,SEARCH,BROWSE,UPDATE,STOP]);tools=ScriptTools([FOUND,OPENED])
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/'r.sqlite')
            try:
                s=initial_state('r','Does treatment work?','p','f','t',process_only=True)
                result=await Engine(store,policy,ScriptModel([]),tools,ContextBuilder(codec,100000),rank).run(s)
                p=json.loads(policy.requests[-1]['messages'][1]['content'])
                self.assertEqual(p['opened_evidence']['E1']['text'],'Observed outcome.')
                self.assertEqual(p['current_target']['requirements'],{'R1':'Assess effectiveness.'})
                self.assertEqual(result['prefinal']['opened_evidence'],result['evidence'])
                rows=list(sft_samples(store,'r',codec))
                row=rows[-1];n=len(codec.encode(row['messages']))
                self.assertTrue(all(v==-100 for v in row['labels'][:n]))
                self.assertEqual(row['messages'],policy.requests[-1]['messages'])
                self.assertEqual(len(tools.calls),2)
            finally:store.db.close()
