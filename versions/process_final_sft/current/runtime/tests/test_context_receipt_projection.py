import copy
import json
import unittest
from trainable_agent.context import ContextBuilder, ContextOverflow, OfflineCodec
from trainable_agent.context_processing.deterministic_context import project_evidence, validate_projection
from trainable_agent.engine import initial_state
from trainable_agent.compression import card


def evidence(text, table=False):
    metadata = dict(year=2024, abstract_only=False, url='https://example.org/study',
        content_scope=dict(full_document_verified=False, retrieved_scope='selected_parsed_passages'),
        source_quality_warnings=['Do not infer an uncertain comparator.'],
        window_identity=dict(version='returned_window_identity_v1',text_sha256='a'*64,
            start_char=0,end_char=len(text)),window_identity_sha256='b'*64,
        table_window=None,novelty_projection=None)
    if table:
        metadata['table_window']=dict(version='exact_table_rows_v1',row_indices=[1,2],
            complete_table=False,complete_row_groups=True,all_columns_retained=True,
            all_footnotes_retained=True,source_spans=[dict(start_char=i,end_char=i+1,text_sha256='c'*64) for i in range(300)])
    return dict(source_id='PMID:1',chunk_id='PMID:1#c'+str(table),title='Clinical study',text=text,metadata=metadata)


class ReceiptProjectionTests(unittest.TestCase):
    def state(self):
        s=initial_state('fixture','Question','p','f','t')
        s['requirements']={'R1':'Establish the comparative outcome.'}
        s['assessments']={'R1':dict(support='partial',evidence_ids=['E1'],gap='Comparison unresolved.',conflict='')}
        s['evidence']={'E1':evidence('No benefit was established. Additional complete observation.'),
                       'E2':evidence('Group A | Group B | 12% | 8% | * Adults.',True)}
        s['history']=[dict(turn=1,action=dict(action='browse',source_id='PMID:1'),feedback=dict(status='ok'))]
        return s

    def test_prose_receipts_omitted_without_changing_claims_or_clinical_metadata(self):
        s=self.state();saved=copy.deepcopy(s)
        projected,audit=project_evidence(s['evidence'])
        self.assertEqual(s,saved)
        for eid,item in projected.items():
            raw=s['evidence'][eid]
            self.assertEqual(item['text'],raw['text'])
            for key in ('source_id','chunk_id','title'):self.assertEqual(item[key],raw[key])
            for key in ('year','abstract_only','url','content_scope','source_quality_warnings'):
                self.assertEqual(item['metadata'][key],raw['metadata'][key])
            self.assertNotIn('window_identity',item['metadata'])
            self.assertNotIn('window_identity_sha256',item['metadata'])
        self.assertEqual(audit['window_provenance_omitted_ids'],['E1','E2'])
        validate_projection(projected,s['evidence'])
        projected['E1']['metadata']['year']=2025
        with self.assertRaisesRegex(ValueError,'identity_changed'):validate_projection(projected,s['evidence'])

    def test_normal_floor_and_excerpt_override_use_identical_metadata(self):
        s=self.state();b=ContextBuilder(OfflineCodec(),100000,evidence_cards=False)
        s['evidence']['E1']['text'] += ' Further complete background observation.'*15
        normal=b.build('state',s,allow_cards=False)
        empty={eid:{**copy.deepcopy(c),'text':''} for eid,c in s['evidence'].items()}
        floor=b.build('state',s,overrides=empty,allow_cards=False)
        excerpt=b.build('state',s,overrides={'E1':card(s['evidence']['E1'],['S0'])},allow_cards=False)
        payloads=[json.loads(r['messages'][-1]['content']) for r in (normal,floor,excerpt)]
        for eid in s['evidence']:
            self.assertEqual(payloads[0]['opened_evidence'][eid]['metadata'],payloads[1]['opened_evidence'][eid]['metadata'])
            self.assertEqual(payloads[0]['opened_evidence'][eid]['metadata'],payloads[2]['opened_evidence'][eid]['metadata'])
        self.assertLess(len(floor['prompt_token_ids']),len(normal['prompt_token_ids']))
        self.assertEqual(excerpt['raw_context_payload']['opened_evidence'],s['evidence'])
        self.assertEqual(payloads[2]['history'],payloads[0]['history'])
        self.assertEqual(excerpt['raw_context_payload']['history'],s['history'])
        self.assertEqual(payloads[2]['state'],s['assessments'])
        bad=copy.deepcopy(empty);bad['E1']['metadata']['year']=2025
        with self.assertRaisesRegex(ValueError,'identity_changed'):b.build('state',s,overrides=bad)

    def test_unknown_window_schema_is_not_silently_erased(self):
        s=self.state();s['evidence']['E1']['metadata']['window_identity']['version']='unknown_receipt'
        projected,_=project_evidence(s['evidence'])
        self.assertEqual(projected['E1']['metadata']['window_identity'],s['evidence']['E1']['metadata']['window_identity'])
        self.assertEqual(projected['E1']['metadata']['window_identity_sha256'],'b'*64)


class CapacityFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_table_is_mandatory_floor_and_only_prose_is_extracted(self):
        s=ReceiptProjectionTests().state()
        s['evidence']['E1']['text']='No benefit was established. '+('Additional background observation. '*200)
        saved=copy.deepcopy(s)
        roomy=ContextBuilder(OfflineCodec(),100000,evidence_cards=False,safety_margin=128)
        floor={'E1':{**copy.deepcopy(s['evidence']['E1']),'text':''},'E2':copy.deepcopy(s['evidence']['E2'])}
        limit=roomy._measure('state',s,floor,[])+1200+128+900
        builder=ContextBuilder(OfflineCodec(),limit,evidence_cards=False,safety_margin=128)
        with self.assertRaises(ContextOverflow):builder.build('state',s,allow_cards=False)
        class OriginalOnlyFallback:
            calls=[]
            async def transform(self,chunk,focus,target,mode):
                self.calls.append(copy.deepcopy(chunk))
                if chunk['metadata'].get('table_window'):raise AssertionError('protected table reached sentence extractor')
                return card(chunk,['S0'])
        fallback=OriginalOnlyFallback()
        request=await builder.prepare('state',s,fallback=fallback)
        self.assertEqual(s,saved)
        self.assertTrue(fallback.calls)
        self.assertLessEqual(len(request['prompt_token_ids'])+request['max_new_tokens']+128,limit)
        visible=json.loads(request['messages'][-1]['content'])
        self.assertEqual(visible['opened_evidence']['E2']['text'],s['evidence']['E2']['text'])
        self.assertTrue(visible['opened_evidence']['E2']['metadata']['table_window']['all_columns_retained'])
        self.assertEqual(request['raw_context_payload']['history'],s['history'])
        restored=copy.deepcopy(visible['history'])
        reverse={v:k for k,v in request['raw_context_payload']['source_aliases'].items()}
        for row in restored:row['action']['source_id']=reverse[row['action']['source_id']]
        self.assertEqual(restored,s['history'])
        self.assertEqual(request['raw_context_payload']['opened_evidence'],s['evidence'])
