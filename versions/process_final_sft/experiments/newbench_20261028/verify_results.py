"""Recompute published record arithmetic and paired counts; no private data or API."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ('base', 'sft', 'grpo13', 'grpo25')


def load(name):
    return json.loads((HERE / name).read_text(encoding='utf8'))


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8), (a, b)


def reward(row):
    if row['engineering_hold']:
        return None
    if row['fatal'] or row['model_task_failure']:
        return 0.0
    return row['process_score'] / 100 * (0.49 if row['major'] else 1)


def pair(rows, a, b, field):
    index = {(r['index'], r['arm']): r for r in rows}
    wins = losses = ties = 0
    for q in sorted({r['index'] for r in rows}):
        x, y = index[q, a][field], index[q, b][field]
        if x is None or y is None:
            continue
        if abs(x - y) < 1e-8:
            ties += 1
        elif x > y:
            wins += 1
        else:
            losses += 1
    return wins, losses, ties


def verify():
    rows, points, summary = load('records_400.json'), load('content_points_100.json'), load('summary.json')
    assert len(rows) == 400 and len(points) == 100
    assert {(r['index'], r['arm']) for r in rows} == set(itertools.product(range(1, 101), ARMS))
    cp = {p['index']: p for p in points}
    assert len(cp) == 100
    csv_rows = list(csv.DictReader((HERE / 'scores_400.csv').open(encoding='utf8', newline='')))
    assert len(csv_rows) == 400
    ci = {(int(r['index']), r['arm']): r for r in csv_rows}
    for r in rows:
        assert r['benchmark'] == ('healthbench' if r['index'] <= 50 else 'medbrowsecomp')
        assert r['notes'].strip() and isinstance(r['issues'], list)
        assert [p['id'] for p in r['points']] == [p['id'] for p in cp[r['index']]['points']]
        assert all(p['score'] in (0, 1, 2) and p['reason'].strip() for p in r['points'])
        close(r['p1'], 40 * sum(p['score'] for p in r['points']) / (2 * len(r['points'])))
        close(r['coverage_percent'], r['p1'] * 2.5)
        close(r['process_score'], sum(r[k] for k in ('p1', 'p2', 'p3', 'p4')))
        assert all(0 <= r[k] <= cap for k, cap in zip(('p1', 'p2', 'p3', 'p4'), (40, 30, 20, 10)))
        expected = reward(r)
        if expected is None:
            assert r['quality_value'] is None
        else:
            close(r['quality_value'], expected)
        assert r['model_task_failure'] == (r['status_original'] == 'process_failure')
        assert (r['benchmark_score'] is None) == (r['status_original'] != 'complete')
        for k in ('p1', 'p2', 'p3', 'p4', 'process_score', 'quality_value', 'coverage_percent', 'benchmark_score'):
            value = ci[r['index'], r['arm']][k]
            if r[k] is None:
                assert value == ''
            else:
                close(float(value), r[k])
        for h in r['provenance'].values():
            assert len(h) == 64 and all(c in '0123456789abcdef' for c in h)
    for scope in ('all', 'healthbench', 'medbrowsecomp'):
        subset = [r for r in rows if scope == 'all' or r['benchmark'] == scope]
        for arm in ARMS:
            rs = [r for r in subset if r['arm'] == arm]
            s = summary['local'][scope][arm]
            assert s['records'] == len(rs)
            for field, output in (('process_score', 'process_mean'), ('quality_value', 'quality_mean'), ('coverage_percent', 'delivery_coverage_percent')):
                close(sum(r[field] for r in rs) / len(rs), s[output])
            assert sum(r['major'] for r in rs) == s['major']
            assert sum(r['status_original'] == 'process_failure' for r in rs) == s['process_failures']
            assert sum(r['status_original'] == 'final_failure' for r in rs) == s['final_failures']
            assert sum(r['behavior']['exact_search_extra_actions'] for r in rs) == s['exact_search_extra_actions']
            assert sum(r['behavior']['search_actions'] for r in rs) == s['total_search_actions']
        for a, b in itertools.combinations(ARMS, 2):
            for field in ('process_score', 'p1', 'quality_value'):
                p = summary['paired'][scope][a + '__' + b][field]
                assert pair(subset, a, b, field) == (p['a_wins'], p['b_wins'], p['ties'])
        if scope != 'all':
            for arm in ARMS:
                rs = [r for r in subset if r['arm'] == arm and r['benchmark_score'] is not None]
                s = summary['benchmark'][scope]['arms'][arm]
                assert s['scored'] == len(rs)
                close(sum(r['benchmark_score'] for r in rs) / len(rs), s['raw_mean'])
            for a, b in itertools.combinations(ARMS, 2):
                s = summary['benchmark'][scope]['paired'][a + '__' + b]['benchmark_score']
                assert pair(subset, a, b, 'benchmark_score') == (s['a_wins'], s['b_wins'], s['ties'])
    questions = load('questions_100.json')
    selection = {q['index']: q for q in load('selection_manifest.json')['questions']}
    assert len(questions) == 100 and {q['index'] for q in questions} == set(range(1, 101))
    for q in questions:
        assert q['question'] and q['question_id'] == selection[q['index']]['question_id']
        assert q['source_id'] == selection[q['index']]['source_id']
        assert hashlib.sha256(q['question'].encode('utf8')).hexdigest() == q['question_sha256']
        page = (HERE / 'questions' / f"Q{q['index']:02d}.md").read_text(encoding='utf8')
        assert q['question'] in page and q['question_sha256'] in page
        assert page.index('## 原题（英文原文）') < page.index('## 共同内容点')
    pages = sorted((HERE / 'questions').glob('Q*.md'))
    assert len(pages) == 100
    for i in range(1, 101):
        text = (HERE / 'questions' / f'Q{i:02d}.md').read_text(encoding='utf8')
        for arm in ('Base', 'Process-SFT', 'GRPO13', 'GRPO25'):
            assert f'## {arm}\n' in text
        for r in (r for r in rows if r['index'] == i):
            assert r['notes'] in text
    print('PUBLIC_RESULTS=passed; 400 records, 100 pages, arithmetic, failure handling, all paired counts; API=0; MODEL=0; TRAINING=0')
    from verify_accuracy import verify as verify_accuracy
    verify_accuracy()


if __name__ == '__main__':
    verify()
