"""Generic bibliography variants and actual clinical prose protection."""
import json
import sys
import unittest
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'backend/agent'))
from dr_agent.mcp_backend.apis.browse_preprocess import classify_section,retrieval_spans

class BibliographyVariants(unittest.TestCase):
    def test_variants_need_bibliographic_record(self):
        for heading in ('Reference','References','Reference list','List of references','Bibliographic references','1. Reference','REFERENCE:'):
            self.assertEqual(classify_section(heading,'Example AB, et al. Trial title. J Med. 2025;12:4-8. doi:10.1234/example')[0],'bibliography')
            self.assertEqual(classify_section(heading,'A heading alone does not establish that this is a publication record.')[0],'content')

    def test_medical_assertions_under_reference_heading(self):
        text='The trial enrolled 200 patients in 2025. Treatment reduced mortality (RR 0.8).'
        for heading in ('Reference','References','Reference list'):
            self.assertEqual(classify_section(heading,text)[0],'content')

    def test_reference_mentions_in_body_preserved(self):
        text='The 2025 reference interval is 20–80. Patients should undergo monitoring.'
        self.assertEqual(classify_section('Results',text)[0],'content')

    def test_actual_saved_teacher_reference(self):
        root=PACKAGE.parents[2]
        cases=json.loads((root/'artifacts/historical_retrieval_v7_replay_20261005_022933/CASES.json').read_text())
        record=cases[0]
        reference=next(c for c in record['chunks'] if c.get('heading')=='Reference')
        spans,excluded=retrieval_spans('Reference',reference['text'])
        self.assertFalse(spans)
        self.assertEqual(excluded[0]['reason'],'bibliographic_records_not_article_body')
        for chunk in record['chunks']:
            if chunk.get('heading')!='Reference':
                self.assertTrue(retrieval_spans(chunk.get('heading',''),chunk['text'])[0])
if __name__=='__main__':unittest.main()
