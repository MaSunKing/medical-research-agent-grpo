import copy, json, unittest
from test_search_context import fixture, ExtractProvider
from trainable_agent.context import ContextBuilder, ContextOverflow, OfflineCodec
from trainable_agent.context_processing.search import SearchFallback, clean_candidates
from trainable_agent.context_processing.fallback import ScopedFallback


class Memory:
    def __init__(self): self.cache={}; self.events=[]
    async def once(self,key,fn):
        if key not in self.cache: self.cache[key]=await fn()
        return self.cache[key]
    def event(self,run,kind,data): self.events.append((kind,copy.deepcopy(data)))


class EmptyProvider(ExtractProvider):
    async def generate(self,request):
        self.requests.append(copy.deepcopy(request))
        rows=json.loads(request['messages'][1]['content'])['candidates']
        return {'text':json.dumps({'selections':{r['candidate_id']:[] for r in rows}})}


class EmptyProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_empty_keeps_identity_history_state_and_raw_receipt(self):
        s=fixture(True); before=copy.deepcopy(s); p=EmptyProvider(); store=Memory()
        b=ContextBuilder(OfflineCodec(),100000,safety_margin=128)
        b.limit=b.search_measure(s,{c['source_id']:{'preview':''} for c in s['candidates']})+528+3000
        request=await b.prepare('decision',s,fallback=ScopedFallback(p,OfflineCodec(),store,'empty',6000))
        payload=json.loads(request['messages'][-1]['content'])
        self.assertEqual(payload['history'],s['history'])
        self.assertEqual(payload['state'],s['assessments'])
        self.assertEqual([c['title'] for c in payload['candidates']],[c['title'] for c in s['candidates']])
        self.assertEqual(len(payload['candidates']),8)
        self.assertTrue(any(c['preview']=='' and c['preview_projection']['mode']=='no_safe_relevant_excerpt' for c in payload['candidates']))
        self.assertEqual(request['raw_context_payload']['candidates'],s['candidates'])
        self.assertEqual(s,before)
        self.assertLessEqual(len(request['prompt_token_ids'])+528,b.limit)

    async def test_invalid_selection_never_means_empty(self):
        s=fixture(True); p=ExtractProvider(True); store=Memory()
        result=await SearchFallback(p,OfflineCodec(),store,'invalid',6000).project(s['candidates'][:1],{}, {'PMID:1':500})
        self.assertEqual(result,{})
        self.assertTrue(all(not d['accepted'] for k,d in store.events if k=='search_compression' and 'request' in d))

    async def test_partial_batch_failure_never_blanks_whole_preview(self):
        s=fixture(True); row=s['candidates'][0]; row['preview']=row['preview']*5
        p=EmptyProvider(); store=Memory()
        from trainable_agent.context_processing.extraction import CompressionBudget
        result=await SearchFallback(p,OfflineCodec(),store,'partial',4000,budget=CompressionBudget(1)).project([row],{}, {'PMID:1':500})
        self.assertEqual(result,{})
        self.assertEqual(len(p.requests),1)

    async def test_empty_browse_is_not_deleted(self):
        class EmptyEvidence:
            identity='empty-evidence'
            async def generate(self,request): return {'text':'{"passage_ids":[]}'}
        chunk=dict(source_id='PMID:1',chunk_id='c',title='T',text='No clinical benefit. Risk was 12 percent. '*10)
        original=copy.deepcopy(chunk)
        result=await ScopedFallback(EmptyEvidence(),OfflineCodec(),Memory(),'browse',6000).compress(chunk,{'question':'Q','checklist':{}},500)
        self.assertIsNone(result); self.assertEqual(chunk,original)

    async def test_budget_exhaustion_has_component_diagnostics(self):
        s=fixture(True); before=copy.deepcopy(s); store=Memory(); p=ExtractProvider(True)
        b=ContextBuilder(OfflineCodec(),100000,safety_margin=128)
        empty={c['source_id']:{'preview':''} for c in s['candidates']}
        b.limit=b.search_measure(s,empty)+528+700
        with self.assertRaises(ContextOverflow) as caught:
            await b.prepare('decision',s,fallback=ScopedFallback(p,OfflineCodec(),store,'bad',6000,max_calls=2))
        detail=caught.exception.capacity_detail
        self.assertEqual(detail['reason'],'compression_call_budget_exhausted')
        self.assertEqual(detail['compression_calls'],2)
        self.assertGreater(detail['remaining_deficit'],0)
        self.assertGreater(detail['candidate_preview_tokens'],0)
        self.assertEqual(s,before)

    async def test_catalogue_shape_does_not_erase_or_reorder_candidates(self):
        s=fixture(); s['candidates'][0]['title']='International Symposium Proceedings'
        s['candidates'][0]['preview']='\n'.join(f'P{i:03d} Title and contributor names.' for i in range(900))
        before=copy.deepcopy(s['candidates']); projected,audit=clean_candidates(s['candidates'])
        self.assertTrue(audit[0]['proceedings_catalogue_suspected'])
        self.assertEqual(s['candidates'],before)
        self.assertTrue(projected.get('PMID:1',before[0])['preview'])
