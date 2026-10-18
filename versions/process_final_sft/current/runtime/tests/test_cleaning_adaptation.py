import copy
import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from trainable_agent.context_processing.browser import preprocess_document, classify_section

# Load the lightweight tool modules without importing the legacy agent client.
APIS=Path(__file__).resolve().parents[1]/'retrieval_snapshot_v2/backend/agent/dr_agent/mcp_backend/apis'
PKG='offline_cleaning_apis'
if PKG not in sys.modules:
    pkg=types.ModuleType(PKG);pkg.__path__=[str(APIS)];sys.modules[PKG]=pkg
parser=importlib.import_module(PKG+'.medical_document_parser')
handoff=importlib.import_module(PKG+'.evidence_handoff_v3')

class CleaningRegression(unittest.TestCase):
    def test_controls_and_keyword_lists_not_clinical_short_claims(self):
        for heading,text in [('Title','- Review -'),('Keywords','renal treatment inhibitors'),('Title','Open table in a new tab')]:
            self.assertNotEqual(classify_section(heading,text)[0],'content')
        for heading,text in [('Review','No benefit'),('Keywords','Treatment is not recommended.'),('Results','12% versus 8%'),('Methods','Randomized placebo controlled trial')]:
            self.assertEqual(classify_section(heading,text)[0],'content')

    def test_page_catalog_not_clinical_table(self):
        toc='S118 | Tables, figures | S124 | Foreword | S125 | Chapter 1 | S140 | Chapter 2'
        self.assertNotEqual(classify_section('Document',toc)[0],'content')
        table='Table 1 | Patients | Control | Treatment | 120 | 100 | Events | 12 | 8'
        self.assertEqual(classify_section('Results',table)[0],'content')

    def test_repetition_keeps_first_context_and_original_offsets(self):
        text='Treatment did not improve mortality compared with placebo; 12% experienced adverse events.'
        d={'sections':[{'heading':'Results','text':text},{'heading':'Results','text':text},{'heading':'Methods','text':text}]}
        saved=copy.deepcopy(d);out=preprocess_document(d)
        self.assertEqual(d,saved);self.assertEqual(len(out['sections']),3)
        self.assertTrue(out['sections'][0]['retrieval_spans']);self.assertFalse(out['sections'][1]['retrieval_spans'])
        self.assertTrue(out['sections'][2]['retrieval_spans'])

    def test_contributor_span_does_not_eat_appendix(self):
        text='Statistical expertise: A; Administrative, technical, or logistic support: B. Appendix Figure. Patients should receive monitoring; risk was 12% versus 8%.'
        out=preprocess_document({'sections':[{'heading':'Document','text':text}]})
        s=out['sections'][0];retained=' '.join(s['text'][x['start_char']:x['end_char']] for x in s['retrieval_spans'])
        self.assertNotIn('Statistical expertise',retained);self.assertIn('12% versus 8%',retained)

    def test_real_html_library_with_table_and_negation_guards(self):
        html='<html><head><title>Clinical report</title></head><body><aside><p>Subscribe to our newsletter</p></aside><article><h1>Trial</h1><p>Treatment did not reduce mortality; 53% versus 41% reported adverse events.</p><p>Patients should stop treatment if severe symptoms occur.</p><table><tr><th>Group</th><th>Events</th></tr><tr><td>Placebo</td><td>12</td></tr></table></article></body></html>'
        d=parser.parse_medical_html(html)
        self.assertEqual(d['metadata']['body_extraction']['library'],'trafilatura')
        self.assertEqual(d['metadata']['body_extraction']['raw_source_sha256'],hashlib.sha256(html.encode()).hexdigest())
        self.assertIn('53% versus 41%',d['text']);self.assertIn('did not reduce',d['text'])
        self.assertTrue(any(s.get('structure_kind')=='table' and s['table_structure_complete'] for s in d['sections']))
        self.assertNotIn('Subscribe',d['text']);self.assertEqual(__import__('base64').b64decode(d['raw_source']['content']).decode(),html)

    def test_html_without_article_preserves_short_recommendation(self):
        html='<html><body><p>Patients should avoid treatment.</p></body></html>'
        d=parser.parse_medical_html(html);self.assertIn('should avoid',d['text'])

    def test_packing_refills_after_control_and_restored_duplicate(self):
        body='First sentence. Outcome did not improve.'
        d={'sections':[{'heading':'Title','text':'- Review -'},{'heading':'Results','text':body},{'heading':'Results','text':'An independent study used placebo.'}]}
        chunks=[]
        for i,(si,a,b) in enumerate([(0,0,10),(1,0,len(body)),(1,6,len(body)),(2,0,len(d['sections'][2]['text']))]):
            chunks.append(dict(chunk_id=str(i),source_id='WEB:1',section_index=si,heading=d['sections'][si]['heading'],start_char=a,end_char=b,text=d['sections'][si]['text'][a:b],structure_kind='prose'))
        out,audit=handoff.pack_browse(chunks,d,tokenizer=None,max_tokens=1000,top_k=2)
        self.assertEqual([c.get('canonical_chunk_id',c['chunk_id']) for c in out],['1','3'])
        self.assertTrue(any(x['reason']=='contained_aligned_window' for x in audit))
        for c in out:self.assertEqual(d['sections'][c['section_index']]['text'][c['start_char']:c['end_char']],c['text'])

    def test_archive_preserves_html_bytes_and_table_receipts(self):
        import base64
        obs=importlib.import_module(PKG+'.retrieval_observability')
        raw=b'<html>\xff</html>'
        d={'title':'Trial','text':'Group | 12','sections':[{'heading':'Results','text':'Group | 12','structure_kind':'table','table_structure_complete':True}],
           'raw_source':{'encoding':'base64','content':base64.b64encode(raw).decode()},'parser_audit':{'source_format':'html'}}
        saved=obs._canonical_document_content(d)
        self.assertEqual(base64.b64decode(saved['raw_source']['content']),raw)
        self.assertTrue(saved['sections'][0]['table_structure_complete'])

    def test_block_controls_do_not_join_or_change_clinical_claim(self):
        text='Treatment did not reduce mortality; 53% versus 41% reported adverse events.'
        for d in [parser.parse_medical_markdown('## Results\n- Review -\n'+text),
                  parser.parse_medical_html('<html><body><article><h2>Results</h2><p>- Review -</p><p>'+text+'</p></article></body></html>')]:
            out=preprocess_document(d)
            retained=' '.join(s['text'][x['start_char']:x['end_char']] for s in out['sections'] for x in s['retrieval_spans'])
            self.assertNotIn('- Review -',retained);self.assertIn(text,retained)

    def test_page_catalog_with_unseparated_labels(self):
        text='Guideline title S118 | Tables, figures S124 | Foreword S125 | Chapter 1 S140 | Chapter 2'
        self.assertNotEqual(classify_section('Table of Contents',text)[0],'content')

    def test_repeated_paragraph_keeps_original_source_for_audit(self):
        text='Treatment did not reduce mortality compared with placebo; 12% reported adverse events.'
        d=parser.parse_medical_markdown('## Results\n'+text+'\n'+text)
        out=preprocess_document(d);s=out['sections'][0]
        self.assertEqual(s['text'],text+' '+text)
        self.assertEqual(len(s['retrieval_spans']),1)
        self.assertTrue(any(x['reason']=='exact_repeated_paragraph' for x in out['browse_preprocessing']['sections'][0]['excluded_spans']))

    def test_unfit_whole_sentence_does_not_block_later_evidence(self):
        long='Participants received treatment with '+('extended context '*100)+'.'
        short='Outcome did not improve.'
        d={'sections':[{'heading':'Results','text':long},{'heading':'Results','text':short}]}
        chunks=[dict(chunk_id=str(i),source_id='WEB:1',section_index=i,heading='Results',start_char=0,end_char=len(s['text']),text=s['text'],structure_kind='prose') for i,s in enumerate(d['sections'])]
        out,audit=handoff.pack_browse(chunks,d,tokenizer=None,max_tokens=100,max_chars=100,top_k=1)
        self.assertEqual(out[0]['text'],short)
        self.assertTrue(any(x['reason']=='no_safe_window_in_total_budget' for x in audit))
