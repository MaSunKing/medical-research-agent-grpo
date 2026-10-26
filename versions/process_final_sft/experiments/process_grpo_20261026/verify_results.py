"""Recalculate public score tables and file hashes; no model, network or medical inference."""
from pathlib import Path
import csv
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[3]
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    with (root / 'scores_200.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200
    for arm, expected in summary['summary'].items():
        group = [r for r in rows if r['arm'] == arm]
        assert len(group) == 50
        assert {int(r['question_index']) for r in group} == set(range(1, 51))
        for r in group:
            total = sum(float(r[k]) for k in ('p1', 'p2', 'p3', 'p4'))
            assert abs(total - float(r['P'])) < 1e-8
            if r['fatal'] == 'True' or r['model_task_failure'] == 'True':
                reward = 0.0
            else:
                reward = total / 100 * (.49 if r['major'] == 'True' else 1)
            assert abs(reward - float(r['R'])) < 1e-8
        for key, column in [('coverage_mean', 'coverage_percent'), ('P_mean', 'P'), ('R_mean', 'R')]:
            assert abs(sum(float(r[column]) for r in group) / 50 - expected[key]) < 1e-8
        hb = [float(r['hb_score']) for r in group if r['hb_score']]
        assert len(hb) == expected['HB_scored']
        assert abs(sum(hb) / len(hb) - expected['HB_raw_mean']) < 1e-8
    final_dir = root.parent / 'final_sft_dev100'
    fs = json.loads((final_dir / 'summary.json').read_text(encoding='utf-8'))
    with (final_dir / 'scores_400.csv').open(encoding='utf-8', newline='') as f:
        final = list(csv.DictReader(f))
    assert len(final) == 400
    weights = fs['weights']
    for row in final:
        active = [k for k in weights if row[k]]
        candidate = round(100 * sum(weights[k] * float(row[k]) / 4 for k in active) / sum(weights[k] for k in active), 4)
        assert abs(candidate - float(row['candidate_diagnostic_score'])) < 1e-4
        if row['status'] == 'complete':
            assert abs(candidate - float(row['accepted_answer_score'])) < 1e-4
        else:
            assert row['accepted_answer_score'] == ''
    for variant, expected in fs['checkpoints'].items():
        accepted = [r for r in final if r['variant'] == variant and r['status'] == 'complete']
        assert len(accepted) == expected['status_counts']['complete']
        assert abs(sum(float(r['accepted_answer_score']) for r in accepted) / len(accepted) - expected['accepted_mean']) < 1e-4
    valid_sets = [{r['case_index'] for r in final if r['variant'] == v and r['status'] == 'complete'} for v in fs['checkpoints']]
    assert len(set.intersection(*valid_sets)) == fs['common_valid_n'] == 97
    receipt = json.loads((root / 'PUBLICATION.json').read_text(encoding='utf-8'))
    for path, expected in receipt['files'].items():
        assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected, path
    print('PUBLIC_TABLES_AND_HASHES=passed; Process=200; Final=400; clinical_quality_not_certified')


if __name__ == '__main__':
    main()
