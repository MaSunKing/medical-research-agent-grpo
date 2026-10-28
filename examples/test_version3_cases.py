"""Verify bounded real-case excerpts and existing scores; no models or network."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'examples/version3_cases'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


class RealCaseTests(unittest.TestCase):
    def setUp(self):
        self.cases = [read(CASES / f'Q{i:02d}_grpo25/CASE.json') for i in (45, 98)]

    def test_original_questions_and_existing_ratings(self):
        exp = ROOT / 'versions/process_final_sft/experiments/newbench_20261028'
        scores = read(exp / 'records_400.json')
        for c in self.cases:
            self.assertEqual(digest(c['question']), c['question_sha256'])
            self.assertIn(c['question'], (exp / f"questions/Q{c['index']:02d}.md").read_text(encoding='utf8'))
            self.assertEqual(len(c['comparison']), 4)
            for row in c['comparison']:
                source = next(x for x in scores if x['index'] == c['index'] and x['arm'] == row['arm'])
                for key, value in row.items():
                    self.assertEqual(value, source[key])

    def test_original_model_text_and_source_hashes(self):
        for c in self.cases:
            self.assertEqual(digest(c['checklist_model_text']), c['checklist_model_text_sha256'])
            for item in c['steps'] + c['rejected_generations']:
                self.assertEqual(digest(item['model_text']), item['model_text_sha256'])
            for value in c['provenance']['source_files'].values():
                self.assertRegex(value, r'^[a-f0-9]{64}$')
            self.assertFalse(c['provenance']['newly_generated'])
            self.assertFalse(c['provenance']['rescored'])

    def test_actions_and_state_follow_actual_evidence(self):
        for c in self.cases:
            known = set()
            for step in c['steps']:
                action = json.loads(step['model_text'])
                executed = step['executed_action']
                self.assertEqual(action['action'], executed['action'])
                self.assertEqual(action['requirement_ids'], executed['requirement_ids'])
                self.assertLessEqual(len(action['requirement_ids']), 2)
                self.assertTrue(step['feedback']['executed'])
                known.update(step['feedback']['new_evidence_ids'])
            self.assertEqual(known, {e['evidence_id'] for e in c['evidence']})
            self.assertEqual(len(c['state_updates']), sum(x['executed_action']['action'] == 'browse' for x in c['steps']))
            for change in c['state_updates']:
                for row in change['after'].values():
                    self.assertTrue(set(row['evidence_ids']).issubset(known))

    def test_short_evidence_quotes_have_coordinates_and_hashes(self):
        for c in self.cases:
            per_url = {}
            for item in c['evidence']:
                self.assertRegex(item['text_sha256'], r'^[a-f0-9]{64}$')
                for q in item['excerpts']:
                    self.assertEqual(digest(q['text']), q['sha256'])
                    self.assertEqual(q['end_char'] - q['start_char'], len(q['text']))
                    self.assertLessEqual(q['end_char'], item['text_characters'])
                    per_url[item['url']] = per_url.get(item['url'], 0) + len(q['text'].split())
            self.assertTrue(all(n <= 25 for n in per_url.values()))

    def test_completed_delivery_is_not_claimed_as_perfect_answer(self):
        c = self.cases[0]
        self.assertEqual(c['status'], 'complete')
        self.assertEqual(c['process_termination'], 'runtime_budget_exhausted')
        self.assertEqual(c['budget']['charged'], 6)
        self.assertEqual(len(c['steps']), 6)
        self.assertTrue(c['actual_final_input_present'])
        self.assertTrue(all(e['actual_final_input_contains_evidence'] for e in c['evidence']))
        self.assertEqual(digest(c['answer']), c['answer_text_sha256'])
        canonical = json.dumps(c['answer'], ensure_ascii=False, separators=(',', ':'))
        self.assertEqual(digest(canonical), c['answer_sha256'])
        answer = (CASES / 'Q45_grpo25/final_answer.md').read_text(encoding='utf8')
        self.assertEqual(answer, c['answer'].replace('\r\n', '\n').rstrip() + '\n')
        self.assertEqual(digest(answer), c['final_answer_markdown_sha256'])
        own = next(x for x in c['comparison'] if x['arm'] == 'grpo25')
        self.assertEqual(own['process_score'], 100)
        self.assertLess(own['benchmark_score'], 0.3)

    def test_failure_and_rejected_attempts_are_preserved(self):
        c = self.cases[1]
        self.assertEqual(c['status'], 'process_failure')
        self.assertEqual(c['process_termination'], 'model_contract_failure')
        self.assertEqual(c['final_status'], 'not_started')
        self.assertFalse(c['actual_final_input_present'])
        self.assertIsNone(c['answer'])
        self.assertEqual(len(c['steps']), 5)
        self.assertEqual(len(c['rejected_generations']), 2)
        for attempt in c['rejected_generations']:
            self.assertEqual(len(json.loads(attempt['model_text'])['requirement_ids']), 3)
            self.assertIn('at most 2 items', attempt['error'])
        own = next(x for x in c['comparison'] if x['arm'] == 'grpo25')
        self.assertEqual(own['p1'], 0)
        self.assertEqual(own['quality_value'], 0)
        self.assertIsNone(own['benchmark_score'])

    def test_public_scope_has_no_private_paths_or_credentials(self):
        for path in CASES.rglob('*'):
            if not path.is_file():
                continue
            self.assertIn(path.suffix, {'.md', '.json'})
            text = path.read_text(encoding='utf8')
            self.assertNotRegex(text, r'/home/' + r'(?:msai|users)/|[A-Z]:[/\\]Users[/\\]')
            self.assertNotRegex(text, r'\b(?:10|172)\.(?:\d{1,3}\.){2}\d{1,3}\b')
            self.assertNotRegex(text, r'\bsk-[A-Za-z0-9_-]{20,}|GOCSPX-[A-Za-z0-9_-]{15,}')

    def test_case_file_manifest(self):
        manifest = read(CASES / 'MANIFEST.json')
        expected = {p.relative_to(CASES).as_posix() for p in CASES.rglob('*')
                    if p.is_file() and p.name != 'MANIFEST.json'}
        self.assertEqual(set(manifest['files']), expected)
        for name, expected_hash in manifest['files'].items():
            self.assertEqual(hashlib.sha256((CASES / name).read_bytes()).hexdigest(), expected_hash)
        self.assertFalse(manifest['new_generation'])
        self.assertFalse(manifest['rescored'])


if __name__ == '__main__':
    unittest.main()
