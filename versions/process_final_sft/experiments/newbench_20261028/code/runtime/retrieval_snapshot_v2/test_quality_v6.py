"""Shared paper/web cleaning and candidate ranking, offline only."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend/agent'))
from dr_agent.mcp_backend.apis.browse_preprocess import preprocess_document, retrieval_spans, classify_section
from dr_agent.mcp_backend.apis.medical_document_parser import parse_medical_html, parse_medical_xml, parse_medical_markdown, chunk_medical_document, select_relevant_chunks
from dr_agent.runtime_policy import rank_web_candidates

BODY = 'The trial enrolled 200 patients. Treatment reduced adverse events (RR 0.80; 95% CI 0.65–0.99).'

class Characters:
    def encode(self, text, add_special_tokens=False):
        return list(text)

class SharedQuality(unittest.TestCase):
    def test_publication_shell_generic(self):
        for title in ('Example guideline', 'Unrelated topic', 'Document'):
            self.assertEqual(classify_section(title, 'Example Society Publication Date: May 3, 2023 Page Last Updated: Sep 2, 2026')[0], 'publication_metadata')

    def test_citation_overview_and_mesh(self):
        self.assertEqual(classify_section('Document Overview', 'Example Society. Guidelines title. J Med 2023. doi: 10.1/example. PMID: 123456.')[0], 'bibliography')
        self.assertEqual(classify_section('MeSH Terms', 'Humans Aged Kidney Function Tests')[0], 'bibliography')

    def test_mixed_dates_preserve_contiguous_original_spans(self):
        text = 'Publication Date: May 3, 2023 ' + BODY + ' Page Last Updated: Sep 2, 2026'
        spans, excluded = retrieval_spans('Results', text)
        self.assertEqual([text[s['start_char']:s['end_char']] for s in spans], [BODY])
        self.assertEqual(len(excluded), 2)

    def test_plain_clinical_dates_not_removed(self):
        text = 'Patients were followed from May 3, 2023 to Sep 2, 2026. Testing was recommended every 6 months.'
        spans, excluded = retrieval_spans('Results', text)
        self.assertEqual(len(spans), 1)
        self.assertFalse(excluded)

    def test_prose_bibliographic_mentions_not_censored(self):
        text = BODY + ' Publication Date: May 3, 2023. doi: 10.1/example. PMID: 123456.'
        self.assertEqual(classify_section('Document Overview', text)[0], 'content')

    def test_directory_and_actual_recommendation_distinguished(self):
        self.assertEqual(classify_section('Clinical guidelines', 'Access our resources. These include guidelines on acne and psoriasis. Learn more.')[0], 'navigation_index')
        self.assertEqual(classify_section('Clinical guidelines', 'Patients should receive monitoring every 6 months. Clinicians recommend testing if symptoms worsen.')[0], 'content')

    def test_generic_directory_heading_list_and_schedule(self):
        self.assertEqual(classify_section('Society guidelines', 'Topic A Topic B Topic C Topic D')[0], 'navigation_index')
        self.assertEqual(classify_section('Society pediatric guidelines', 'Guidelines in development. Sections will be released next year. Read the guideline for details.')[0], 'navigation_index')
        self.assertEqual(classify_section('Society guidelines', 'Patients should receive annual testing and follow-up')[0], 'content')

    def test_unpunctuated_recommendation_not_a_directory(self):
        for text in ('Annual monitoring is recommended for these patients', 'Avoid treatment if the patient is pregnant', 'Consider testing before starting treatment'):
            self.assertEqual(classify_section('Society guidelines',text)[0],'content')
        self.assertEqual(classify_section('Updated guidelines for management of a disease', 'Monitoring every six months')[0],'content')

    def test_publication_records_have_metadata_not_navigation_labels(self):
        self.assertEqual(classify_section('New guidelines', 'PUBLISHED 12/28/18 BY Example Author')[0], 'publication_metadata')
        self.assertEqual(classify_section('Updated guidelines', 'Example Author Research output : Contribution to journal › Article › Academic › peer-review')[0], 'publication_metadata')

    def test_actual_dev5_directory_chunks_excluded(self):
        for chunk in json.loads((ROOT/'fixtures/dev5_q5_directory_chunks.json').read_text(encoding='utf-8')):
            self.assertEqual(classify_section(chunk['heading'],chunk['text'])[0], 'navigation_index')

    def test_table_under_metadata_heading_not_reenabled(self):
        doc = {'sections': [{'heading': 'MeSH terms', 'text': 'Terms | Categories Humans | Aged', 'structure_kind':'table','table_structure_complete':True}]}
        self.assertFalse(preprocess_document(doc)['sections'][0]['retrieval_spans'])

    def test_complete_guideline_table_not_mistaken_for_topic_list(self):
        doc = parse_medical_html('<main><h1>Society guidelines</h1><table><tr><th>Age</th><th>Interval</th></tr><tr><td>65 years</td><td>6 months</td></tr></table></main>')
        prepared = preprocess_document(doc)
        self.assertTrue(prepared['sections'][0]['retrieval_spans'])
        chunks = chunk_medical_document(prepared,source_id='WEB:table')
        self.assertTrue(chunks)
        self.assertTrue(all(classify_section(c['heading'],c['text'])[0]=='content' for c in chunks))

    def test_paper_jats_web_html_and_pdf_markdown_shared_gate(self):
        docs = [parse_medical_html('<main><h1>Results</h1><p>'+BODY+'</p><h2>MeSH Terms</h2><p>Humans Aged</p></main>'),
                parse_medical_xml('<article><body><sec><title>Results</title><p>'+BODY+'</p></sec><sec><title>MeSH Terms</title><p>Humans Aged</p></sec></body></article>'),
                parse_medical_markdown('# Results\n'+BODY+'\n## MeSH Terms\nHumans Aged')]
        for doc in docs:
            for source_id in ('PMID:123', 'WEB:abc'):
                prepared = preprocess_document(doc)
                chunks = chunk_medical_document(prepared, source_id=source_id)
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0]['text'], BODY)
                self.assertEqual(prepared['sections'][1]['text'], 'Humans Aged')
                self.assertFalse(prepared['sections'][1]['retrieval_spans'])
                self.assertEqual(chunks[0]['chunk_id'], source_id+'#s0-c0')

    def test_select_relevant_chunks_gate_for_both_tools(self):
        doc = {'title':'Test','sections':[{'heading':'Results','text':BODY},{'heading':'MeSH Terms','text':'Humans Aged'}], 'metadata':{}}
        for source_id in ('PMID:123','WEB:abc'):
            result = select_relevant_chunks(doc,'treatment adverse events',source_id=source_id,retrieval_mode='bm25',tokenizer=Characters(),max_output_tokens=700)
            self.assertTrue(result['chunks'])
            self.assertTrue(all('Humans Aged' not in c['text'] for c in result['chunks']))

    def test_actual_dev5_publication_chunks_excluded(self):
        path = ROOT / 'fixtures/dev5_q1_publication_chunks.json'
        for chunk in json.loads(path.read_text(encoding='utf-8')):
            self.assertNotEqual(classify_section(chunk['heading'],chunk['text'])[0], 'content')

    def test_topical_source_outranks_authoritative_directory(self):
        items = [{'title':'Clinical guidelines','url':'https://www.aad.org/member/clinical-quality/guidelines','snippet':'Access guidelines for acne psoriasis and other dermatologic conditions.'},
                 {'title':'Alopecia evaluation in adults','url':'https://example.org/alopecia','snippet':'Persistent alopecia evaluation in adults: dermoscopy biopsy and diagnostic testing.'}]
        ranked = rank_web_candidates('persistent alopecia adults evaluation biopsy',items,original_question='How to evaluate persistent alopecia in adults?')
        self.assertEqual(ranked[0]['title'],items[1]['title'])
        self.assertEqual(len(ranked),2)  # ranking, not a hard block

    def test_opposing_findings_not_filtered(self):
        for text in ('The trial enrolled 200 patients. Treatment increased adverse events.', 'The trial enrolled 200 patients. Treatment did not increase adverse events.'):
            self.assertEqual(classify_section('Results',text)[0],'content')

if __name__ == '__main__':
    unittest.main()
