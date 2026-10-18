import copy
import hashlib
import unittest

from trainable_agent.context_processing.browser import (
    classify_section, high_confidence_nonbody_shape, preprocess_document,
)
from trainable_agent.context_processing.cleaning import clean_chunk, clean_observation


CHECKOUT = ('Tax calculation will be finalised at checkout Purchases are for '
            'personal use only Institutional subscriptions')
CATALOG = ('Teaching Slides Educational slide sets to teach about treatment. '
           'Power point | PDF Guideline Development Infographic Development Process '
           'Resources Resources to better understand the guideline. '
           'Understanding Health Infographic Back to Guidelines')
CAPTION = 'Example Life Cycle. Source: Example Society'


class NonbodyBlockTests(unittest.TestCase):
    def test_checkout_and_catalog_share_pre_rank_and_registration_rule(self):
        for text in (CHECKOUT, CATALOG):
            self.assertIsNotNone(high_confidence_nonbody_shape('Document', text))
            self.assertNotEqual(classify_section('Document', text)[0], 'content')
            self.assertEqual(clean_chunk({'text': text})['text'], '')

    def test_source_blocks_remove_only_catalog_between_medical_claims(self):
        first = 'Treatment did not reduce mortality.'
        last = 'Risk was 12% versus 8%.'
        text = first + ' ' + CATALOG + ' ' + last
        start = len(first) + 1
        doc = {'sections': [{'heading': 'Results', 'text': text,
                'source_blocks': [{'start_char': start, 'end_char': start + len(CATALOG)}]}]}
        original = copy.deepcopy(doc)
        result = preprocess_document(doc)
        section = result['sections'][0]
        self.assertEqual(doc, original)
        self.assertEqual(section['text'], text)
        self.assertEqual([text[s['start_char']:s['end_char']] for s in section['retrieval_spans']], [first, last])
        audit = result['browse_preprocessing']['sections'][0]
        self.assertEqual(audit['original_text_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(text[audit['excluded_spans'][0]['start_char']:audit['excluded_spans'][0]['end_char']], CATALOG)

    def test_registration_records_original_offsets_hash_and_identity(self):
        text = 'No benefit was observed.\n' + CHECKOUT + '\nRisk was 12%.\n'
        chunk = {'source_id': 'WEB:test', 'chunk_id': 'WEB:test#c1', 'title': 'Report', 'text': text}
        original = copy.deepcopy(chunk)
        result = clean_chunk(chunk)
        self.assertEqual(chunk, original)
        self.assertEqual(result['text'], 'No benefit was observed.\nRisk was 12%.\n')
        self.assertEqual(result['cleaning']['original_text_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(''.join(text[a:b] for a,b in result['cleaning']['kept_spans']), result['text'])
        for name in ('source_id', 'chunk_id', 'title'):
            self.assertEqual(result[name], chunk[name])

    def test_clinical_or_uncertain_catalog_text_is_not_removed(self):
        for extra in (' Patients should receive monitoring.', ' No benefit.',
                      ' Risk decreased by 20%.', ' Guidance updated in 2026.',
                      ' Treatment is the preferred option.', ' Infection causes complications.'):
            text = CATALOG + extra
            self.assertIsNone(high_confidence_nonbody_shape('Resources', text))
            self.assertEqual(clean_chunk({'text': text})['text'], text)
            self.assertEqual(classify_section('Document', text)[0], 'content')

    def test_checkout_words_inside_clinical_prose_preserved(self):
        text = 'Tax calculation will be finalised at checkout. Patients should receive monitoring.'
        self.assertEqual(clean_chunk({'text': text})['text'], text)
        self.assertEqual(classify_section('Document', text)[0], 'content')

    def test_bare_caption_requires_figure_context_and_specific_title_shape(self):
        self.assertNotEqual(classify_section('Figure 1', CAPTION)[0], 'content')
        self.assertEqual(classify_section('Document', CAPTION)[0], 'content')
        for text in ('No treatment benefit. Source: Example Society',
                     'Example Life Cycle lasts 12 days. Source: Example Society',
                     'Risk reduced by 20%. Source: Example Society',
                     'The diagram shows progression. Source: Example Society',
                     'Group distribution by severity. Source: Example Society'):
            self.assertEqual(classify_section('Figure 1', text)[0], 'content')

    def test_registration_recovers_heading_only_from_exact_saved_receipt(self):
        chunk = {'source_id': 'WEB:test', 'chunk_id': 'WEB:test#c1', 'title': 'Document', 'text': CAPTION}
        raw = {'source_id': chunk['chunk_id'], 'text': CAPTION, 'heading': 'Figure 1'}
        observation = {'status': 'ok', 'chunks': [chunk], 'backend_return': {'data': [raw]}}
        original = copy.deepcopy(observation)
        result, audit = clean_observation(observation)
        self.assertEqual(observation, original)
        self.assertFalse(result['chunks'])
        self.assertEqual(result['status'], 'source_unreadable')
        self.assertEqual(audit['chunks'][0]['removed_spans'][0]['reason'], 'figure_title_and_attribution_without_readable_figure')
        raw['source_id'] = 'WEB:other#c1'
        self.assertEqual(clean_observation(observation)[0]['chunks'][0]['text'], CAPTION)
        raw['source_id'] = chunk['chunk_id']
        raw['text'] = 'Different source text'
        self.assertEqual(clean_observation(observation)[0]['chunks'][0]['text'], CAPTION)

    def test_annotated_bibliography_and_method_tables_are_protected(self):
        text = ('Smith AB et al. Treatment report. J Med. 2025;12:4-8. PMID:12345. '
                'Comment: Patients should receive monitoring. Treatment is not recommended for children.')
        self.assertEqual(classify_section('References', text)[0], 'content')
        self.assertEqual(clean_chunk({'text': text, 'heading': 'References'})['text'], text)
        table = '| Group | Events |\n|---|---|\n| Control | 12 |'
        self.assertEqual(clean_chunk({'text': table, 'heading': 'Methods'})['text'], table)

    def test_table_projection_text_and_mapping_are_not_mutated(self):
        for metadata in ({'table_window': {'source_spans': [[0, 13], [80, 89]]}},
                         {'structure_kind': 'table'}):
            chunk = {'text': 'Advertisement\n' + CHECKOUT, 'metadata': metadata}
            original = copy.deepcopy(chunk)
            result = clean_chunk(chunk)
            self.assertEqual(result['text'], chunk['text'])
            self.assertEqual(result['metadata'], metadata)
            self.assertEqual(result['cleaning']['removed_spans'], [])
            self.assertEqual(chunk, original)

    def test_pipe_table_cannot_be_mistaken_for_resource_chooser(self):
        text = CATALOG + ' | Group | Resource | Outcome |'
        self.assertIsNone(high_confidence_nonbody_shape('Methods', text))
        self.assertEqual(clean_chunk({'text': text})['text'], text)

    def test_unknown_short_content_and_guideline_version_stay(self):
        for text in ('Guideline version 2026', 'Updated guidance', 'No benefit',
                     'Educational slide sets describe a clinical management pathway.'):
            self.assertEqual(clean_chunk({'text': text})['text'], text)


if __name__ == '__main__':
    unittest.main()
