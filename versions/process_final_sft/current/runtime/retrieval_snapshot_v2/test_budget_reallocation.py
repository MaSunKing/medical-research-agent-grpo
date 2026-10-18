"""Offline contract tests for the shared paper/web Browse budget handoff."""
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend/agent'))
from dr_agent.mcp_backend.apis.evidence_handoff_v3 import pack_browse


class CharacterTokenizer:
    def encode(self, value, add_special_tokens=False):
        return list(value)


def ranked_pair(first, second, source_id):
    body = first + ' ' + second
    document = {'sections': [{'heading': 'Results', 'text': body}]}
    chunks = []
    for index, (start, end) in enumerate(((0, len(first)), (len(first) + 1, len(body)))):
        chunks.append({
            'chunk_id': f'{source_id}#s0-c{index}',
            'section_index': 0,
            'start_char': start,
            'end_char': end,
            'content_span_start_char': 0,
            'content_span_end_char': len(body),
            'text': body[start:end],
            'structure_kind': 'prose',
        })
    return document, chunks


class BrowseBudgetReallocation(unittest.TestCase):
    def test_all_selected_windows_fit_for_paper_and_web(self):
        first = 'Study design was observational. Genital infection was higher: ratio 4.04 (95% CI 3.46 to 4.71).'
        second = 'The conclusion reports increased infection risk.'
        for source_id in ('PMID:38234181', 'WEB:abc123'):
            with self.subTest(source_id=source_id):
                document, ranked = ranked_pair(first, second, source_id)
                budget = len(first) + len(second) + 5
                returned, audit = pack_browse(
                    ranked, document, tokenizer=CharacterTokenizer(),
                    max_tokens=budget, max_chars=budget, top_k=2,
                )
                self.assertEqual(len(returned), 2)
                self.assertIn('ratio 4.04', returned[0]['text'])
                self.assertTrue(all(row['all_selected_windows_fit'] for row in audit))
                self.assertTrue(all(not row['boundary_incomplete'] for row in audit))

    def test_unused_later_share_returns_to_earlier_clipped_window(self):
        first = ('Study design was observational. '
                 'Genital infection was higher: ratio 4.04 (95% CI 3.46 to 4.71). '
                 'Supplemental details are outside the budget.')
        second = 'The conclusion agrees.'
        document, ranked = ranked_pair(first, second, 'PMID:38234181')
        returned, audit = pack_browse(
            ranked, document, tokenizer=CharacterTokenizer(),
            max_tokens=130, max_chars=130, top_k=2,
        )
        self.assertFalse(audit[0]['all_selected_windows_fit'])
        self.assertGreater(audit[0]['reallocated_tokens'], 0)
        self.assertIn('ratio 4.04', returned[0]['text'])
        self.assertLessEqual(sum(len(row['text']) for row in returned), 130)
        self.assertTrue(all(not row['boundary_incomplete'] for row in audit))

    def test_incomplete_table_still_does_not_enter_evidence(self):
        body = ('Introduction sentence. Table 4 Relative effectiveness Total No. Events '
                'Adjusted 95% CI row % col % values continue without a safe complete table boundary')
        start = body.index('Table')
        ranked = [{
            'chunk_id': 'PMID:1#s0-c1', 'section_index': 0,
            'start_char': start, 'end_char': len(body) - 20,
            'content_span_start_char': 0, 'content_span_end_char': len(body),
            'text': body[start:len(body) - 20], 'structure_kind': 'table_like',
        }]
        returned, audit = pack_browse(
            ranked, {'sections': [{'heading': 'Table', 'text': body}]},
            tokenizer=CharacterTokenizer(), max_tokens=1000, max_chars=1000,
        )
        self.assertEqual(returned, [])
        self.assertEqual(audit[0]['reason'], 'incomplete_table_boundary')


if __name__ == '__main__':
    unittest.main(verbosity=2)
