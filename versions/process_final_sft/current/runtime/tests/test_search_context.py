import copy,json,tempfile,unittest
from pathlib import Path
from trainable_agent.context import ContextBuilder,ContextOverflow,OfflineCodec
from trainable_agent.context_processing.fallback import ScopedFallback
from trainable_agent.context_processing.search import SearchFallback
from trainable_agent.engine import initial_state
from trainable_agent.storage import Store

class ExtractProvider:
    identity='offline-qwen-interface-fixture'
    def __init__(self,bad=False):self.requests=[];self.bad=bad
    async def generate(self,request):
        self.requests.append(copy.deepcopy(request))
        rows=json.loads(request['messages'][1]['content'])['candidates']
        choices={r['candidate_id']:[next(iter(r['passages']))] for r in rows}
        if self.bad:choices={'INVENTED':['P999']}
        return {'text':json.dumps({'selections':choices})}

def fixture(long=False):
    s=initial_state('search','Benefits and harms in adults?','p','f','t')
    s['requirements']={'R1':'Benefits and harms'}
    s['assessments']={'R1':dict(support='partial',evidence_ids=['E1'],gap='Comparator',conflict='')}
    s['evidence']={'E1':dict(source_id='PMID:1',chunk_id='c1',title='Opened',text='Private opened witness. '*1000)}
    s['history']=[dict(turn=1,action={'action':'search','query':'prior query'},feedback={'status':'ok'})]
    preview='No benefit in adults versus placebo; risk was 12% versus 8%. '
    if long:preview+=' '.join(f'Background observation {i} records separate conditions.' for i in range(22))
    s['candidates']=[dict(source_id=f'PMID:{i+1}',channel='paper',title=f'Trial {i}',preview=preview,url='') for i in range(8)]
    return s

class SearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(Path(self.tmp.name)/'r.sqlite')
        self.addCleanup(self.tmp.cleanup);self.addCleanup(self.store.db.close)
    def fallback(self,p,limit=6000):return ScopedFallback(p,OfflineCodec(),self.store,'search',limit)
    async def test_full_preview_when_fits_and_old_browse_body_not_in_decision(self):
        s=fixture();before=copy.deepcopy(s);p=ExtractProvider()
        req=await ContextBuilder(OfflineCodec(),100000).prepare('decision',s,fallback=self.fallback(p))
        visible=json.loads(req['messages'][1]['content'])
        self.assertEqual(visible['opened_evidence'],{});self.assertNotIn('Private opened witness',json.dumps(visible))
        self.assertEqual([r['preview'] for r in visible['candidates']],[r['preview'] for r in s['candidates']])
        self.assertEqual(visible['history'],s['history']);self.assertEqual(s,before);self.assertFalse(p.requests)
        state=json.loads(ContextBuilder(OfflineCodec(),100000).build('state',s)['messages'][1]['content'])
        self.assertNotIn('candidates',state);self.assertIn('Private opened witness',state['opened_evidence']['E1']['text'])
    async def test_overflow_batch_projection_raw_archive_and_identity(self):
        s=fixture(True);before=copy.deepcopy(s);p=ExtractProvider()
        limit=ContextBuilder(OfflineCodec(),100000).search_measure(fixture(),{})+400+128+2000
        builder=ContextBuilder(OfflineCodec(),limit,safety_margin=128)
        req=await builder.prepare('decision',s,fallback=self.fallback(p))
        self.assertLessEqual(len(req['prompt_token_ids'])+400+128,limit)
        v=json.loads(req['messages'][1]['content']);self.assertEqual(len(v['candidates']),8)
        self.assertEqual([r['title'] for r in v['candidates']],[r['title'] for r in s['candidates']])
        self.assertTrue(all('No benefit' in r['preview'] and '12%' in r['preview'] for r in v['candidates']))
        self.assertTrue(any('preview_projection' in r for r in v['candidates']))
        self.assertTrue(all(r['preview_projection']['coverage_complete'] is False for r in v['candidates'] if 'preview_projection' in r))
        self.assertEqual(req['raw_context_payload']['candidates'],s['candidates']);self.assertEqual(s,before)
        self.assertGreater(len(p.requests),1)
        for request in p.requests:
            self.assertLessEqual(len(request['prompt_token_ids'])+800+128,6000)
            text=request['messages'][1]['content'];self.assertNotIn('Private opened witness',text);self.assertNotIn('prior query',text)
    async def test_invalid_model_projection_bounded_fail_closed(self):
        p=ExtractProvider(True);s=fixture(True)
        limit=ContextBuilder(OfflineCodec(),100000).search_measure(fixture(),{})+400+128+2000
        b=ContextBuilder(OfflineCodec(),limit,fallback_rounds=2)
        with self.assertRaises(ContextOverflow):await b.prepare('decision',s,fallback=self.fallback(p))
        self.assertTrue(p.requests);self.assertEqual(s['candidates'],fixture(True)['candidates'])
        self.assertLess(len(p.requests),20)
    async def test_fixed_overflow_no_compressor_call(self):
        s=fixture(True);s['history']=[{'query':'historical action '*1000}];p=ExtractProvider()
        with self.assertRaises(ContextOverflow):await ContextBuilder(OfflineCodec(),6000).prepare('decision',s,fallback=self.fallback(p))
        self.assertFalse(p.requests)
    async def test_cleaning_before_model(self):
        s=fixture();s['candidates'][0]['preview']='  '+s['candidates'][0]['preview']+' '*20000
        p=ExtractProvider();limit=ContextBuilder(OfflineCodec(),100000).search_measure(fixture(),{})+400+1500
        req=await ContextBuilder(OfflineCodec(),limit).prepare('decision',s,fallback=self.fallback(p))
        self.assertFalse(p.requests);self.assertTrue(req['context_audit']['candidate_projected_ids'])
    async def test_unknown_ids_cannot_modify_title_or_order(self):
        s=fixture();b=ContextBuilder(OfflineCodec(),100000)
        with self.assertRaises(ValueError):b.build('decision',s,candidate_overrides={'PMID:999':{'preview':'x'}})
        with self.assertRaises(ValueError):b.build('decision',s,candidate_overrides={'PMID:1':{'preview':'x','title':'invented'}})
    async def test_unfittable_complete_passage_not_sent(self):
        p=ExtractProvider();rows=[{'source_id':'PMID:1','title':'T','preview':'A'*20000}]
        result=await SearchFallback(p,OfflineCodec(),self.store,'search',3000).project(rows,{'question':'Q'}, {'PMID:1':500})
        self.assertEqual(result,{});self.assertFalse(p.requests)
