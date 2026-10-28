"""Regression tests for consequential reward and comparison boundaries."""
import copy
import unittest
from verify_results import load, pair, reward


class PublishedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = load('records_400.json')

    def test_final_failure_does_not_erase_process(self):
        rows = [r for r in self.rows if r['status_original'] == 'final_failure']
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['benchmark_score'] is None for r in rows))
        self.assertTrue(all(reward(r) > 0 for r in rows))

    def test_process_failure_remains_in_full_denominator(self):
        rows = [r for r in self.rows if r['status_original'] == 'process_failure']
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(reward(r) == 0 for r in rows))
        self.assertEqual(sum(pair(self.rows, 'base', 'grpo25', 'quality_value')), 100)
        self.assertLess(sum(pair(self.rows, 'base', 'grpo25', 'benchmark_score')), 100)

    def test_major_discount_and_engineering_hold_are_distinct(self):
        row = copy.deepcopy(next(r for r in self.rows if r['major'] and not r['model_task_failure']))
        self.assertAlmostEqual(reward(row), row['process_score'] * 0.0049)
        row['engineering_hold'] = True
        self.assertIsNone(reward(row))

    def test_coverage_and_quality_wins_are_different_metrics(self):
        self.assertEqual(pair(self.rows, 'grpo25', 'sft', 'p1'), (22, 11, 67))
        self.assertEqual(pair(self.rows, 'grpo25', 'sft', 'quality_value'), (41, 38, 21))

    def test_leak_and_nonwinning_checkpoint_remain_visible(self):
        leak = [r for r in self.rows if r['benchmark_source_leak']]
        self.assertEqual([(r['index'], r['arm']) for r in leak], [(58, 'grpo13')])
        self.assertEqual(pair(self.rows, 'grpo13', 'sft', 'quality_value'), (32, 45, 23))


if __name__ == '__main__':
    unittest.main()
