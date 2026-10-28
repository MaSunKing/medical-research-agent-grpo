"""CPU-only package, reward, capture and provenance verification."""
import argparse
from pathlib import Path
from contract import load_data,read,require,sha,verify_files,validate_config,batches,acceptance_groups


def run(package):
    package=Path(package)
    verify_files(package,read(package/'FILES.sha256.json'))
    c=read(package/'training/process_grpo_update/CONFIG.json');validate_config(c)
    groups,m,h=load_data(package/'data')
    qs=read(package/'data/QUESTIONS.json');reserved=read(package/'data/RESERVED_EVALUATION.json')
    import re
    norm=lambda x:' '.join(re.findall(r'\w+',x.casefold()))
    banned_ids={q['question_id'] for q in reserved};banned_text={norm(q['question']) for q in reserved}
    require(len(qs)==100 and len({q['question_id'] for q in qs})==100 and
            len({norm(q['question']) for q in qs})==100,'duplicate_training_question')
    require([q['question_id'] for q in qs]==[g['question_id'] for g in groups],'question_order_changed')
    for q in qs:
        require(q['split']=='train' and q['question_id'] not in banned_ids and
                norm(q['question']) not in banned_text and
                'healthbench' not in str(q['provenance']).casefold(),'reserved_training_overlap')
    scores=read(package/'data/SCORES.json')
    require(sha(package/'data/SCORES.json')==m['review_records_sha256'],'score_snapshot_changed')
    by={(r['question_id'],r['rollout']):r for r in scores['records']}
    for g in groups:
        for r in g['rollouts']:
            original=by[g['question_id'],r['rollout']];s=original['score'];v=original['review']
            require(s['R']==r['R'] and s['A']==r['A'] and
                    original['source_status']==r['source_status'],'score_binding_changed')
            active=[i for i in v['issues'] if not i['resolved']]
            major=any(i['severity']=='major' for i in active)
            fatal=any(i['severity']=='fatal' for i in active)
            held=bool(v['hold_reason']) or any(i['severity']=='hold' for i in active)
            pts=v['process_points']
            p=20*sum(i['score'] for i in pts)/len(pts)+sum(x for k in ('state','checklist','decision') for x in v[k].values())
            reward=None if held else 0 if fatal or v['process_task_failure'] else p/100*(.49 if major else 1)
            require(abs(p-s['P'])<1e-10 and ((reward is None and r['R'] is None) or
                    reward is not None and r['R'] is not None and abs(reward-r['R'])<1e-10),'reward_formula_changed')
    selected=acceptance_groups(groups,c['acceptance_questions'])
    report=dict(status='passed',questions=len(groups),records=sum(len(g['rollouts']) for g in groups),
        held_questions=sum(g['held'] for g in groups),usable_questions=sum(not g['held'] for g in groups),
        planned_batches=len(batches(groups,c)),dataset_sha256=h,
        acceptance_questions=[g['question_id'] for g in selected],
        includes_longest_update_input=max(g['max_sequence_tokens'] for g in selected)==max(g['max_sequence_tokens'] for g in groups if not g['held']),
        model_loads=0,api_calls=0,optimizer_steps=0,reserved_evaluation_overlap=0,
        server_GPU_acceptance='pending',formal_training=False)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();import json
    print(json.dumps(run(a.package),ensure_ascii=False,indent=2))
