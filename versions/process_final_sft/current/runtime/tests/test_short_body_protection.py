"""Weak nonbody rules remain enabled while clinical short shapes stay readable."""
import copy
import unittest

from trainable_agent.context_processing.browser import classify_section, preprocess_document


class ShortBodyProtectionTests(unittest.TestCase):
    def assert_retained(self, heading, text):
        self.assertEqual(classify_section(heading, text)[0], 'content')
        source = {'sections': [{'heading': heading, 'text': text}]}
        original = copy.deepcopy(source)
        result = preprocess_document(source)
        self.assertEqual(source, original)
        section = result['sections'][0]
        self.assertTrue(section['retrieval_spans'])
        self.assertEqual(section['text'], text)
        self.assertEqual(''.join(text[s['start_char']:s['end_char']] for s in section['retrieval_spans']), text)

    def test_recommendation_is_independent_of_final_period(self):
        base = 'Oxygen should be administered when oxygen saturation falls below 92%'
        for end in ('', '.', '!', '?', '。', '；'):
            with self.subTest(end=end):
                self.assert_retained('Respiratory support', base + end)

    def test_dose_phrase_under_guidelines_without_assertion_verb(self):
        for end in ('', '.', ';', ':'):
            with self.subTest(end=end):
                self.assert_retained('Clinical guidelines', 'Aspirin 75 mg daily for secondary prevention' + end)

    def test_measurement_and_threshold_shapes(self):
        for text in ('Monitoring threshold 10 mmHg before treatment',
                     'Infusion rate 5 mL per hour',
                     'Dose 10 µg per kg daily',
                     '每日给予75毫克并定期复查',
                     'Target threshold ≥ 92% in adults'):
            with self.subTest(text=text):
                self.assert_retained('Recommended support', text)

    def test_negative_and_comparative_findings(self):
        for text in ('No benefit in adults with severe disease',
                     'Without improvement compared with usual care',
                     'Out-of-pocket costs persist despite community care'):
            with self.subTest(text=text):
                self.assert_retained('Community recommendations', text)

    def test_chinese_instructions_without_final_punctuation(self):
        for text in ('患者需监测血氧并及时随访', '监测血压；记录症状', '不推荐常规使用，未见明显获益'):
            with self.subTest(text=text):
                self.assert_retained('社区 支持 指南', text)

    def test_table_under_support_heading_preserved(self):
        text = '| Group | Events | |---|---| | Control | 12 |'
        self.assert_retained('Community support', text)

    def test_source_blocks_preserve_original_claim_coordinates(self):
        text = 'Aspirin 75 mg daily for secondary prevention'
        source = {'sections': [{'heading': 'Clinical guidelines', 'text': text,
                               'source_blocks': [{'start_char': 0, 'end_char': len(text)}]}]}
        result = preprocess_document(source)
        self.assertEqual(result['sections'][0]['retrieval_spans'],
                         [{'start_char': 0, 'end_char': len(text), 'kind': 'body'}])

    def test_original_weak_rules_still_filter_plain_topic_menus(self):
        self.assertEqual(classify_section('Support', 'Home Resources Downloads Contact'),
                         ('web_boilerplate', 'template_heading_and_short_list'))
        self.assertEqual(classify_section('Clinical guidelines', 'Cardiology Dermatology Neurology Respiratory'),
                         ('navigation_index', 'guideline_heading_and_topic_list'))

    def test_bare_numbers_and_punctuation_are_not_blanket_exemptions(self):
        for text in ('Home 2026 Resources Contact', 'Home, Resources; Downloads: Contact',
                     'Home page 2 Resources Contact'):
            with self.subTest(text=text):
                self.assertEqual(classify_section('Support', text)[0], 'web_boilerplate')

    def test_explicit_guideline_directory_and_ui_still_removed(self):
        for heading, text in (
            ('Clinical guidelines', 'Read the guideline and view all guidelines in development'),
            ('Support', 'Technical support browser resolution 1024'),
            ('Newsletter', 'Subscribe to our newsletter and sign up'),
            ('Support', 'Copyright Example All rights reserved'),
            ('Document', 'Tax calculation will be finalised at checkout Purchases are for personal use only Institutional subscriptions'),
        ):
            with self.subTest(text=text):
                self.assertNotEqual(classify_section(heading, text)[0], 'content')


if __name__ == '__main__':
    unittest.main()
