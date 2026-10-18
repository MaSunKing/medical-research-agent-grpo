import unittest
from trainable_agent.context_processing.browser import preprocess_document, classify_section, repeated_window

class BoundedNoiseTests(unittest.TestCase):
    def kept(self, text, heading='Document'):
        result=preprocess_document({'sections':[{'heading':heading,'text':text}]})
        s=result['sections'][0]
        self.assertEqual(s['text'],text)
        return [s['text'][x['start_char']:x['end_char']] for x in s['retrieval_spans']],result
    def test_bare_date_excluded_without_losing_source(self):
        spans,result=self.kept('09.25.2024')
        self.assertFalse(spans)
        self.assertEqual(result['browse_preprocessing']['sections'][0]['reason'],'standalone_date_not_clinical_evidence')
    def test_dates_and_short_clinical_claims_preserved(self):
        for heading,text in [('Results','09.25.2024'),('Timeline','2024-09-25'),('Document','Follow-up ended on 09.25.2024.'),('Results','12% versus 8%'),('Results','No benefit'),('Document','The guideline was updated on 2024-09-25.')]:
            self.assertEqual(self.kept(text,heading)[0],[text])
    def test_mixed_paywall_preserves_claim_and_original_offsets(self):
        claim='Further work is needed to monitor adverse effects.'
        notice='This is a preview of subscription content, log in via an institution to check access.'
        text=claim+' '+notice+' Patients should receive monitoring.'
        spans,result=self.kept(text)
        self.assertEqual(spans,[claim,'Patients should receive monitoring.'])
        self.assertTrue(any(e['reason']=='bounded_subscription_access_notice' for e in result['browse_preprocessing']['sections'][0]['excluded_spans']))
    def test_access_prose_not_censored(self):
        text='The study asked participants to log in via an institution to check access.'
        self.assertEqual(self.kept(text)[0],[text])
    def test_access_control_standalone_rejected(self):
        self.assertNotEqual(classify_section('Document','This is a preview of subscription content, log in via an institution to check access.')[0],'content')
    def test_new_numbers_negations_and_populations_not_deduplicated(self):
        old=[{'source_id':'WEB:1','text':'Adult patients improved; risk was 12% versus 8%.'}]
        for text in ['Adult patients did not improve; risk was 12% versus 8%.','Adult patients improved; risk was 13% versus 8%.','Pediatric patients improved; risk was 12% versus 8%.','Adult patients improved; risk was 12% versus 8%. New outcomes were reported.']:
            self.assertIsNone(repeated_window(text,'WEB:1',old))
