"""Generic structure rules: no source, disease, identifier or host exceptions."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'backend/agent'))
from dr_agent.mcp_backend.apis.browse_preprocess import classify_section, retrieval_spans
from dr_agent.runtime_policy import rank_web_candidates

class GenericQuality(unittest.TestCase):
    def test_descriptor_syntax_without_heading(self):
        self.assertEqual(classify_section('Document Scope', 'D008508 - Medication Errors D000067561 - Potentially Inappropriate Medication List D054539 - Medication Therapy Management')[0], 'bibliography')
        self.assertEqual(classify_section('Any heading', 'D123456 - Topic Alpha D234567 - Topic Beta D345678 - Topic Gamma')[0], 'bibliography')
        self.assertEqual(classify_section('Results', 'D123456 - Topic Alpha D234567 - Topic Beta D345678 - Topic Gamma Patients should undergo monitoring.')[0], 'content')

    def test_resource_catalog(self):
        for heading in ('Supplemental Implementation Resources', 'Downloads', 'Related Resources'):
            for topic in ('renal monitoring', 'crop production', 'asthma management'):
                self.assertEqual(classify_section(heading, 'Patient Information Quick-Reference Guide Slide Set Checklist Podcast '+topic)[0], 'navigation_index')

    def test_resources_with_recommendations_preserved(self):
        self.assertEqual(classify_section('Resources', 'Patient Information Slide Set Checklist Patients should undergo testing every 6 months.')[0], 'content')
        self.assertEqual(classify_section('Resources', 'Patient Information Slide Set Checklist Treatment reduced mortality by 20%.')[0], 'content')
        self.assertEqual(classify_section('Resources', '| Patient Information | Checklist | Slide Set |\n|---|---|---|\n| 12 | 8 | 4 |')[0], 'content')
        self.assertEqual(classify_section('Results', 'Patient Information Quick-Reference Guide Slide Set Checklist Podcast')[0], 'content')

    def test_site_credit_and_research_protection(self):
        for text in ('Design and created by Example Team in participation with Example Society.', 'Site developed by Example Studio.'):
            self.assertEqual(classify_section('Document', text)[0], 'web_boilerplate')
        self.assertEqual(classify_section('Methods', 'Created by investigators for testing drug efficacy.')[0], 'content')
        self.assertEqual(classify_section('Methods', 'Design by investigators. The trial enrolled 200 patients.')[0], 'content')

    def test_media_contact(self):
        self.assertEqual(classify_section('Document', 'Media Contact: 212-308-1414 info@example.org')[0], 'publication_metadata')
        self.assertEqual(classify_section('Results', 'Patients should contact the clinic at info@example.org.')[0], 'content')

    def test_no_colon_metadata_and_mixed_body(self):
        self.assertEqual(classify_section('Document', 'Document Publication Date May 3, 2023 Page Last Reviewed/Updated Sep 2, 2026')[0], 'publication_metadata')
        body = 'Patients should undergo testing every 6 months.'
        text = 'Publication Date May 3, 2023 '+body+' Page Last Reviewed/Updated Sep 2, 2026'
        spans, excluded = retrieval_spans('Results', text)
        self.assertEqual([text[s['start_char']:s['end_char']] for s in spans], [body])
        self.assertEqual(len(excluded), 2)

    def test_topic_beats_generic_guideline_words(self):
        for topic, other in [('asthma', 'psoriasis'), ('fracture', 'diabetes'), ('alopecia', 'dermatitis')]:
            items = [{'title':f'Current guidelines evaluation adults {other}', 'url':'https://example.org/other'},
                     {'title':f'Persistent {topic} adults diagnosis unknown cause', 'url':'https://example.net/topic'}]
            ranked = rank_web_candidates(f'current guideline evaluation adults {topic}', items,
                                         original_question=f'What is the current guideline for evaluation of persistent {topic} in adults without a known cause?')
            self.assertEqual(ranked[0]['url'], items[1]['url'])

if __name__ == '__main__': unittest.main()
