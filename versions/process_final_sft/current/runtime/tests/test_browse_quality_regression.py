import copy,unittest
from test_browse_novelty import handoff,fixture,Chars
from trainable_agent.context_processing.browser import classify_section,preprocess_document
from trainable_agent.context_processing.cleaning import clean_chunk

class QualityRegression(unittest.TestCase):
    def test_resource_pointer_excluded_but_recommendation_preserved(self):
        text="Additional information on treatment and follow-up testing can be found online at: Clinical Guidelines Screening Recommendations These guidelines will be updated as data become available."
        self.assertEqual(classify_section('Document',text)[0],'navigation_index')
        self.assertEqual(classify_section('Document',text+' Patients should receive monitoring.')[0],'content')
        self.assertEqual(classify_section('Results','The guideline recommends treatment for adults.')[0],'content')
    def test_access_location_not_guideline_claim(self):
        t='The proposed guidelines are available under the Supporting and Related Materials tab in the docket for this notice on https://example.org.'
        self.assertEqual(classify_section('Document',t)[0],'navigation_index')
        self.assertEqual(classify_section('Document','The trial found increased risk; additional information can be found online at https://example.org.')[0],'content')
    def test_verbatim_long_prefix_trimmed_and_offsets_preserved(self):
        old='Adult participants received treatment and underwent scheduled monitoring for adverse events.'
        new='The risk was 13% versus 8%, with no benefit observed.'
        doc,ranked=fixture([old+' '+new]);saved=copy.deepcopy((doc,ranked))
        rows=[dict(source_id='WEB:one',evidence_id='E1',text=old)]
        out,audit=handoff.pack_browse(ranked,doc,tokenizer=Chars(),max_tokens=1000,max_chars=1000,opened_windows=rows)
        self.assertEqual((doc,ranked),saved);self.assertEqual(out[0]['text'],new)
        self.assertEqual(out[0]['novelty_projection']['prior_evidence_ids'],['E1'])
        self.assertEqual(doc['sections'][0]['text'][out[0]['start_char']:out[0]['end_char']],new)
        self.assertTrue(any(a['reason']=='already_opened_sentence_edges_trimmed' for a in audit))
    def test_changed_number_and_other_source_not_trimmed(self):
        old='Adult participants received treatment and underwent scheduled monitoring; risk was 12%.'
        new=old.replace('12%','13%')+' No benefit was observed.'
        doc,ranked=fixture([new]);rows=[dict(source_id='WEB:one',evidence_id='E1',text=old)]
        out,_=handoff.pack_browse(ranked,doc,tokenizer=Chars(),max_tokens=1000,max_chars=1000,opened_windows=rows)
        self.assertEqual(out[0]['text'],new)
        other=[dict(source_id='PMID:2',evidence_id='E1',text=new)]
        out,_=handoff.pack_browse(ranked,doc,tokenizer=Chars(),max_tokens=1000,max_chars=1000,opened_windows=other)
        self.assertEqual(out[0]['text'],new)
    def test_suffix_trimmed_without_stitching(self):
        old='Adult participants received treatment and underwent scheduled monitoring for adverse events.'
        new='The risk was 13% versus 8%.'
        doc,ranked=fixture([new+' '+old])
        out,_=handoff.pack_browse(ranked,doc,tokenizer=Chars(),max_tokens=1000,max_chars=1000,opened_windows=[dict(source_id='WEB:one',evidence_id='E1',text=old)])
        self.assertEqual(out[0]['text'],new)
    def test_pdf_warning_preserves_raw_symbols_and_metadata(self):
        c=dict(source_id='WEB:one',chunk_id='one',text=r'Adults with eGFR ‡20 or eGFR \$20 qualify.',metadata=dict(url='https://example.org/a.pdf',year=2024))
        saved=copy.deepcopy(c);out=clean_chunk(c)
        self.assertEqual(c,saved);self.assertEqual(out['text'],c['text']);self.assertEqual(out['metadata']['year'],2024)
        self.assertTrue(out['metadata']['source_quality_warnings'])
        c['metadata']['url']='https://example.org/fees.html';self.assertNotIn('source_quality_warnings',clean_chunk(c)['metadata'])
