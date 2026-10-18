"""Count State-call exposure from private dossiers; never export source text."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def build(folder):
    rows = []
    for q in range(1, 51):
        for arm in ('base', 'sft'):
            path = folder / f'Q{q:02d}_{arm}.json'
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            assert data['id'] == f'Q{q:02d}_{arm}' and data['arm'] == arm
            state = [g for g in data['generations'] if g.get('stage') == 'state']
            rows.append(dict(id=data['id'], arm=arm,
                accepted_state_calls=sum(g.get('accepted') is True for g in state),
                rejected_state_generations=sum(g.get('accepted') is False for g in state),
                dossier_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    groups = {}
    for arm in ('base', 'sft'):
        counts = [r['accepted_state_calls'] for r in rows if r['arm'] == arm]
        groups[arm] = dict(trajectories=len(counts), accepted_state_calls=sum(counts),
            accepted_calls_per_trajectory_mean=statistics.mean(counts),
            accepted_calls_per_trajectory_median=statistics.median(counts),
            trajectories_with_accepted_state=sum(n > 0 for n in counts))
    return dict(scope='All 50 questions per arm, including failed trajectories',
        unit='Accepted State model calls, including accepted empty updates; not individual requirement updates',
        interpretation='Exposure only, not a quality score or an error-rate denominator',
        state_semantic_repetition_rate=None,
        unresolved='Per-claim correctness and new-evidence update opportunities have not been re-annotated',
        groups=groups, trajectories=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dossiers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.dossiers), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
