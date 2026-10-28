"""Actual exported text plus source-structure regressions; no model/API."""
import json
import sys
import unittest
from pathlib import Path
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend/agent'))
from dr_agent.mcp_backend.apis.browse_preprocess import preprocess_document, strip_html_templates
from dr_agent.mcp_backend.apis.medical_document_parser import parse_medical_html, parse_medical_xml, chunk_medical_document
from dr_agent.mcp_backend.apis.evidence_handoff_v3 import pack_browse
from dr_agent.mcp_backend.apis.table_integrity import table_signature

class Characters:
    def encode(self,text,add_special_tokens=False):return list(text)

def run_handoff(document,budget=10000):
    prepared=preprocess_document(document)
    chunks=chunk_medical_document(prepared,source_id='TEST:1',chunk_chars=200)
    returned,audit=pack_browse(chunks,prepared,tokenizer=Characters(),max_tokens=budget,max_chars=budget,top_k=20)
    return prepared,chunks,returned,audit

class MetadataStructures(unittest.TestCase):
    def test_license_trademark_footer_not_evidence(self):
        text='ISSN: 2168-8184 Public user content licensed CC-BY 4.0 EXAMPLE is a registered trademark of Example, Inc. in the United States.'
        prepared,chunks,returned,_=run_handoff({'sections':[{'heading':'Newsletter','text':text}]})
        self.assertEqual(prepared['sections'][0]['text'],text)
        self.assertFalse(chunks)
        self.assertFalse(returned)
        self.assertEqual(prepared['browse_preprocessing']['document_readability_status'],'no_readable_body')

    def test_backmatter_excluded_body_preserved(self):
        doc={'sections':[{'heading':'Results','text':'The trial enrolled 200 patients. Treatment reduced adverse events (RR 0.80; 95% CI 0.65–0.99).'},
                         {'heading':'Funding','text':'The work was supported by NIH grants AI093315 and AI093220.'},
                         {'heading':'Author information','text':'Correspondence to Louis M. Weiss .'},
                         {'heading':'Additional information','text':'Reprints and permissions'}]}
        prepared,chunks,returned,_=run_handoff(doc)
        self.assertEqual(len(prepared['sections']),4)
        self.assertEqual(len(returned),1)
        self.assertIn('RR 0.80',returned[0]['text'])
        self.assertTrue(all(not s['retrieval_spans'] for s in prepared['sections'][1:]))

    def test_medical_prose_mentioning_licenses_and_funding_preserved(self):
        text=('The trial enrolled 200 patients and was funded by a public grant. '
              'Treatment reduced adverse events (RR 0.80; 95% CI 0.65–0.99). '
              'The questionnaire is licensed CC-BY, and the instrument name is a registered trademark.')
        _,_,returned,_=run_handoff({'sections':[{'heading':'Results','text':text}]})
        self.assertEqual(returned[0]['text'],text)

    def test_decomposed_descendant_is_not_read_as_live_template(self):
        soup=BeautifulSoup('<main><aside class="related"><div>Links</div></aside>'
                           '<article><p>Treatment reduced symptoms in the trial.</p></article></main>','html.parser')
        descendant=soup.aside.div
        descendant.attrs=None  # BeautifulSoup can leave a named tag with cleared attrs after decompose().
        original_find_all=soup.find_all
        soup.find_all=lambda *args,**kwargs: [descendant]+original_find_all(*args,**kwargs)
        strip_html_templates(soup)
        self.assertIn('Treatment reduced symptoms',soup.get_text(' ',strip=True))

    def test_nested_template_removed_without_losing_article(self):
        html=('<html><body><main><aside class="related"><div class="related">'
              '<a href="/other">Other article</a></div></aside><article><h1>Results</h1>'
              '<p>Treatment reduced symptoms in the trial.</p></article></main></body></html>')
        doc=parse_medical_html(html)
        text=' '.join(section['text'] for section in doc['sections'])
        self.assertIn('Treatment reduced symptoms',text)
        self.assertNotIn('Other article',text)

    def test_actual_racgp_export_excluded_and_source_preserved(self):
        export=ROOT/'fixtures/racgp_metadata_export.json'
        data=json.loads(export.read_text(encoding='utf-8'))
        actual=data['text']
        p,chunks,returned,audit=run_handoff({'sections':[{'heading':'RACGP','text':actual}]})
        self.assertEqual(p['sections'][0]['text'],actual)
        self.assertEqual(chunks,[])
        reasons={e['reason'] for e in p['browse_preprocessing']['sections'][0]['excluded_spans']}
        self.assertIn('serialized_head_metadata',reasons)
        self.assertIn('tracking_iframe_link',reasons)

    def test_mixed_metadata_keeps_body_and_offsets(self):
        body='Treatment reduced adverse events (RR 0.80; 95% CI 0.65–0.99).'
        text='--- meta-og:title: Example meta-description: '+body+' --- '+body
        p,chunks,returned,_=run_handoff({'sections':[{'heading':'Results','text':text}]})
        self.assertEqual([c['text'] for c in returned],[body])
        self.assertEqual(p['sections'][0]['text'][chunks[0]['start_char']:chunks[0]['end_char']],body)

    def test_statistical_prose_remains_prose(self):
        text='The trial enrolled 200 patients. Treatment reduced adverse events (RR 0.80; 95% CI 0.65–0.99).'
        self.assertEqual(table_signature(text)['structure_kind'],'prose')
        _,_,returned,_=run_handoff({'sections':[{'heading':'Results','text':text}]})
        self.assertEqual(returned[0]['text'],text)

    def test_table_mention_does_not_reclassify_prose(self):
        _,chunks,_,_=run_handoff({'sections':[{'heading':'Results','text':'Table 4 presents the data. Treatment reduced risk in 200 patients.'}]})
        self.assertTrue(all(c['structure_kind']=='prose' for c in chunks))

    def table(self):
        return '<table><tr><th>Statement</th><th>Percentage consensus</th></tr><tr><td>Initiate long-term ERT after confirmed diagnosis. Evidence Grade B.</td><td>89%</td></tr></table>'

    def test_html_complete_table_atomic_and_sentence_not_override(self):
        doc=parse_medical_html('<html><body><main><h1>Guidance</h1><p>Before.</p>'+self.table()+'<p>After.</p></main></body></html>')
        _,chunks,returned,_=run_handoff(doc)
        tables=[c for c in returned if c['structure_kind']=='table_like']
        self.assertEqual(len(tables),1)
        self.assertIn('Statement',tables[0]['text'])
        self.assertIn('89%',tables[0]['text'])
        self.assertTrue(tables[0]['table_integrity_verified'])
        self.assertFalse(tables[0]['structure_audit']['prose_sentence_override'])

    def test_jats_table_inside_paragraph_not_flattened_or_duplicated(self):
        xml='<article><front><article-title>Guideline</article-title></front><body><sec><title>ERT</title><p>Before.<table-wrap><label>Table 4</label><caption><p>Guidance statement for galsulfase</p></caption>'+self.table()+'</table-wrap>After.</p></sec></body></article>'
        doc=parse_medical_xml(xml)
        tables=[s for s in doc['sections'] if s.get('structure_kind')=='table']
        self.assertEqual(len(tables),1)
        prose=' '.join(s['text'] for s in doc['sections'] if s.get('structure_kind')!='table')
        self.assertNotIn('89%',prose)
        self.assertIn('After.',prose)
        _,_,returned,_=run_handoff(doc)
        self.assertEqual(sum(c['structure_kind']=='table_like' for c in returned),1)

    def test_table_over_budget_not_returned_as_prose(self):
        doc=parse_medical_html('<main>'+self.table()+'</main>')
        p,chunks,returned,_=run_handoff(doc,30)
        self.assertTrue(chunks)
        self.assertFalse(returned)
        self.assertIn('89%',p['sections'][0]['text'])

    def test_missing_header_or_data_preserved_but_not_evidence(self):
        for table in ('<table><tr><td>56</td><td>14974</td></tr></table>',
                      '<table><tr><th>Outcome</th><th>Events</th></tr></table>'):
            p,chunks,returned,_=run_handoff(parse_medical_html('<main>'+table+'</main>'))
            self.assertFalse(chunks)
            self.assertFalse(returned)
            self.assertTrue(p['sections'][0]['text'])
            self.assertEqual(p['browse_preprocessing']['sections'][0]['excluded_spans'][0]['reason'],'source_table_header_or_data_unverified')

if __name__=='__main__':unittest.main()
