"""Whole-grid summaries, unscored review templates and private read-only export.

No model, retrieval or Judge is called. Missing/failed results remain explicit.
"""
import argparse
from contextlib import closing
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sqlite3
import tarfile
from eval_common import (ARMS, digest, lock, manifest, read, record_folder, require,
                         safe, save, sha, verify_done, verify_package)

PROTOCOL = 'process_prefinal_reward_v4_offline'
CAPS = dict(p1=40, p2=30, p3=20, p4=10)


def questions(package):
    rows = read(Path(package)/'data/questions.json')
    require(len(rows)==100 and {q['index'] for q in rows}==set(range(1,101)), 'all_100_questions_required')
    require(len({q['question_id'] for q in rows})==100, 'duplicate_question_id')
    return sorted(rows,key=lambda q:q['index'])


def delivery(folder, result):
    """Only an actually issued first Final request is a delivered input."""
    folder = Path(folder)
    events = read(folder/'EVENTS.json') if (folder/'EVENTS.json').exists() else []
    first = next((e['data']['request'] for e in events if e['kind']=='model_request'
                  and e['data']['request'].get('role')=='final'), None)
    stored = read(folder/'FINAL_INPUT.json') if (folder/'FINAL_INPUT.json').exists() else None
    if stored is not None:
        require(first is not None and stored==first,'Final_input_not_original_first_request')
    request = first
    if request is None:
        require(result.get('final_input_sha256') in (None,''),'Final_hash_without_request')
        return dict(status='no_final_request',request=None,source='EVENTS.json',
                    note='Raw PRE_FINAL_PACKAGE is diagnostic only; no delivered input invented.')
    require(result.get('final_input_sha256')==digest(request['messages']),'delivered_Final_input_hash_changed')
    return dict(status='actual_first_Final_request',request=request,
                final_input_sha256=result['final_input_sha256'],source='EVENTS.json:model_request:first_Final')


def _benchmark_score(grade_root, arm, index, folder, result_hash, issues):
    base = Path(grade_root)/'scores'/arm/f'Q{index:02d}'
    if not (base/'STATUS.json').is_file():
        return None
    status = read(base/'STATUS.json')
    if status.get('status')!='scored' or status.get('valid_score') is not True:
        issues.append('Benchmark:'+str(status.get('status','unclassified')))
        return None
    require((base/'SCORE.json').is_file(),'scored_status_missing_SCORE')
    require(status.get('score_sha256')==sha(base/'SCORE.json'),'HealthBench_SCORE_hash_changed')
    score = read(base/'SCORE.json')
    identity=read(Path(grade_root)/'IDENTITY.json')
    require(identity.get('work_identity')==digest(read(Path(folder).parents[2]/'IDENTITY.json')) and
            score.get('identity')==digest(identity),'HealthBench_grading_identity_changed')
    result=read(Path(folder)/'RESULT.json')
    require(score.get('index')==index and score.get('question_id')==result.get('question_id'),
            'HealthBench_question_changed')
    require(score.get('final_sha256')==result_hash and score.get('process_arm')==arm,
            'HealthBench_score_source_changed')
    require(score.get('final_variant')=='base','HealthBench_Final_variant_changed')
    require(type(score.get('score')) in (int,float) and math.isfinite(score['score']), 'invalid_HealthBench_score')
    return score['score']


def _process_review(reviews, arm, q, result_hash, issues,source_dossier_sha256=None):
    p = Path(reviews)/'records'/arm/f"Q{q['index']:02d}"/'review.json'
    if not p.exists():
        return None
    r = read(p)
    if r.get('status')!='reviewed':
        return None
    bound=r.get('source_record_sha256',r.get('source_RESULT_sha256'))
    require(r.get('protocol')==PROTOCOL and r.get('question_id')==q['question_id']
            and r.get('arm')==arm and bound==result_hash,
            'Process_review_source_or_protocol_changed')
    require(source_dossier_sha256 is None or r.get('source_dossier_sha256')==source_dossier_sha256,
            'historical_review_dossier_changed')
    require(isinstance(r.get('notes'),str) and r['notes'].strip(),'review_requires_specific_text_record')
    require(isinstance(r.get('issues'),list) and all(isinstance(x,str) and x.strip() for x in r['issues']),
            'review_issues_must_be_text')
    flags = ('major','fatal','model_task_failure','engineering_hold')
    require(all(type(r.get(k)) is bool for k in flags),'review_flags_must_be_explicit')
    require(not any(r[k] for k in flags) or r['issues'],'error_flag_requires_specific_problem')
    values = {k:r.get(k) for k in CAPS}
    require(not r['engineering_hold'] or all(v is None for v in values.values()),
            'engineering_hold_scores_must_remain_null')
    scored = all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=CAPS[k]
                 for k,v in values.items())
    if scored:
        cp = read(Path(reviews)/'questions'/f"Q{q['index']:02d}.json")
        require(cp.get('question_id')==q['question_id'] and cp.get('status')=='approved'
                and isinstance(cp.get('content_points'),list) and cp['content_points'],
                'shared_content_points_not_approved')
        total = sum(values.values())
    else:
        require(all(v is None for v in values.values()) and
                any(r[k] for k in ('fatal','model_task_failure','engineering_hold')),
                'partial_or_invalid_Process_score')
        total = None
    reward = (None if r['engineering_hold'] else 0. if r['fatal'] or r['model_task_failure'] else
              (0.49 if r['major'] else 1.)*total/100)
    if r.get('reward') is not None:
        require(reward is not None and abs(r['reward']-reward)<=1e-12,'review_reward_formula_changed')
    issues.extend(r['issues'])
    return dict(process_score=total,process_reward=reward,notes=r['notes'],
                review_issues=r['issues'],**{k:r[k] for k in flags})


def grid(package, work, grades=None, reviews=None):
    package,work = Path(package),Path(work)
    identity = read(work/'IDENTITY.json')
    require(identity.get('final_variant')=='base' and identity.get('arms')==list(ARMS), 'four_arm_frozen_Base_Final_required')
    require(identity.get('dataset_sha256')==sha(package/'data/questions.json'),'summary_dataset_identity_changed')
    identity_hash = digest(identity)
    require(not (package/'data/HISTORICAL_HEALTHBENCH.json').exists(),'historical_results_forbidden')
    grades = Path(grades) if grades else work/'grades'
    reviews = Path(reviews) if reviews else work/'review'
    cells = []
    for q in questions(package):
        for arm in ARMS:
            folder = record_folder(work,arm,q['index'])
            item = dict(index=q['index'],question_id=q['question_id'],benchmark=q['benchmark'],source_id=q['source_id'],cluster=q['cluster'],arm=arm,status='missing',
                result_source='new',configuration_qualification='current_frozen_runtime',
                source_integrity=False,process_status=None,process_termination=None,final_status=None,
                answer_present=False,benchmark_score=None,process_score=None,process_reward=None,
                issues=[],source_RESULT_sha256=None,delivery_status=None)
            if not folder.exists():
                cells.append(item);continue
            if not (folder/'RESULT.json').is_file():
                item.update(status='interrupted',issues=['unsealed_or_interrupted_directory'])
                cells.append(item);continue
            try:
                require(verify_done(folder,identity_hash),'result_not_sealed')
                result = read(folder/'RESULT.json')
                require(result.get('status') in ('complete','process_failure','final_failure','engineering_failure'),
                        'unknown_result_status')
                require(result.get('question_id',q['question_id'])==q['question_id'],'result_question_changed')
                d = delivery(folder,result)
                result_hash = sha(folder/'RESULT.json')
                item.update(status=result['status'],source_integrity=True,
                    process_status=result.get('process_status'),process_termination=result.get('process_termination'),
                    final_status=result.get('final_status'),answer_present=bool(result.get('answer')),
                    source_RESULT_sha256=result_hash,source_record_sha256=result_hash,delivery_status=d['status'])
                item['issues'].extend(str(x) for x in result.get('issues',[]))
            except (OSError,ValueError,KeyError,TypeError) as error:
                item.update(status='source_integrity_hold',issues=['source_integrity:'+type(error).__name__])
                cells.append(item);continue
            try:
                if item['status']=='complete' and item['final_status']=='complete' and item['answer_present']:
                    item['benchmark_score'] = _benchmark_score(grades,arm,q['index'],folder,result_hash,item['issues'])
            except (OSError,ValueError,KeyError,TypeError):
                item['issues'].append('Benchmark_score_integrity_hold')
            try:
                review = _process_review(reviews,arm,q,result_hash,item['issues'])
                if review:item.update(review)
            except (OSError,ValueError,KeyError,TypeError):
                item['issues'].append('Process_review_integrity_hold')
            cells.append(item)
    require(len(cells)==400,'whole_400_grid_required')
    return cells


def bootstrap_delta(deltas):
    if not deltas:return None
    import random
    rng=random.Random(20261026);n=len(deltas)
    means=sorted(sum(rng.choice(deltas) for _ in range(n))/n for _ in range(2000))
    return dict(low=means[49],high=means[1949],level=0.95,resamples=2000,
                method='paired question-cluster percentile bootstrap; one run per arm; not run-to-run variance')

def summarize(cells):
    out=dict(planned_records=400,planned_questions=100,missing_or_failed_not_scored_zero=True,
             no_cross_benchmark_average=True,independent_unseen_test=True,
             novelty_scope='known local pools; pretraining contamination and semantic paraphrases unknown',benchmarks={})
    for bench in ('healthbench','medbrowsecomp'):
        subset=[x for x in cells if x['benchmark']==bench]
        arms={};paired={}
        for arm in ARMS:
            rows=[x for x in subset if x['arm']==arm];valid=[x['benchmark_score'] for x in rows if x['benchmark_score'] is not None]
            status={s:sum(x['status']==s for x in rows) for s in sorted({x['status'] for x in rows})}
            data=dict(planned=50,statuses=status,complete=sum(x['status']=='complete' for x in rows),
                scored=len(valid),unscored=50-len(valid),completion_rate=sum(x['status']=='complete' for x in rows)/50,
                raw_mean=sum(valid)/len(valid) if valid else None,
                process_scored=sum(x['process_score'] is not None for x in rows),
                process_mean=None)
            ps=[x['process_score'] for x in rows if x['process_score'] is not None]
            data['process_mean']=sum(ps)/len(ps) if ps else None
            if bench=='healthbench':data['clipped_mean']=max(0,data['raw_mean']) if valid else None
            else:
                data['valid_answer_accuracy']=data['raw_mean']
                data['correct_deliveries_per_planned_question']=sum(valid)/50
                data['yield_denominator_note']='engineering delivery yield; failed/unscored records retain null native scores'
            arms[arm]=data
        for a,b in combinations(ARMS,2):
            metrics={}
            for key in ('benchmark_score','process_score','process_reward'):
                aa={x['question_id']:x[key] for x in subset if x['arm']==a and x[key] is not None}
                bb={x['question_id']:x[key] for x in subset if x['arm']==b and x[key] is not None}
                ids=sorted(aa.keys()&bb.keys());ds=[bb[i]-aa[i] for i in ids]
                metrics[key]=dict(common_questions=len(ids),question_ids=ids,a_scored=len(aa),b_scored=len(bb),
                    a_mean=sum(aa[i] for i in ids)/len(ids) if ids else None,
                    b_mean=sum(bb[i] for i in ids)/len(ids) if ids else None,
                    delta_b_minus_a=sum(ds)/len(ds) if ds else None,paired_bootstrap95=bootstrap_delta(ds),
                    b_wins=sum(d>0 for d in ds),a_wins=sum(d<0 for d in ds),ties=sum(d==0 for d in ds))
            paired[a+'__'+b]=metrics
        out['benchmarks'][bench]=dict(arms=arms,paired=paired)
    return out


def write_summary(package,work,output,grades=None,reviews=None,*,_cells=None):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    cells = grid(package,work,grades,reviews) if _cells is None else _cells
    save(output/'ALL_400_RECORDS.json',cells)
    summary = summarize(cells);save(output/'SUMMARY.json',summary)
    columns = ('index','question_id','benchmark','source_id','cluster','arm','result_source','configuration_qualification','status','source_integrity','process_status','process_termination',
               'final_status','answer_present','benchmark_score','process_score','process_reward','issues')
    with (output/'ALL_400_RECORDS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for c in cells:
            row = {k:c.get(k) for k in columns};row['issues']=' | '.join(c['issues']);writer.writerow(row)
    return summary


def build_review(package,work,output,*,_cells=None,historical_review_root=None):
    output = Path(output)
    require(not output.exists(),'fresh_review_template_output_required')
    output.mkdir(parents=True)
    rows = {q['index']:q for q in questions(package)}
    approved={}
    for q in rows.values():
        save(output/'questions'/f"Q{q['index']:02d}.json",dict(question_id=q['question_id'],
            index=q['index'],question=q['question'],status='approved' if q['index'] in approved else 'pending',
            content_points=approved[q['index']]['points'] if q['index'] in approved else [],
            benchmark=q['benchmark'],reference=q.get('reference_answer'),native_rubrics=q['rubrics'],
            source_id=q['source_id'],content_point_review_before_comparing_arms=True,
            approval_note='Frozen question-based points approved before new GRPO outputs.' if q['index'] in approved else
                'Approve same-question content points before scoring all four arms; no trajectory-derived requirements.'))
    cells = grid(package,work) if _cells is None else _cells
    for c in cells:
        folder = record_folder(work,c['arm'],c['index'])
        q=rows[c['index']]
        request=delivery(folder,read(folder/'RESULT.json')) if c['source_integrity'] else dict(status='unavailable',request=None)
        target=output/'records'/c['arm']/f"Q{c['index']:02d}"
        save(target/'delivery.json',dict(question=q['question'],question_id=q['question_id'],**request,
            source_record=f"work/results/{c['arm']}/Q{c['index']:02d}",raw_prefinal_is_diagnostic_only=True))
        save(target/'review.json',dict(protocol=PROTOCOL,status='pending',question_id=q['question_id'],
            index=c['index'],arm=c['arm'],source_RESULT_sha256=c['source_RESULT_sha256'],
            source_record_sha256=c.get('source_record_sha256'),result_source=c['result_source'],
            source_dossier_sha256=c.get('historical_source',{}).get('source_dossier_sha256'),
            observed_result_status=c['status'],p1=None,p2=None,p3=None,p4=None,reward=None,
            major=None,fatal=None,model_task_failure=None,engineering_hold=None,issues=[],notes='',
            shared_content_points=f"../../../questions/Q{c['index']:02d}.json"))
    save(output/'TEMPLATE.json',dict(schema='unscored_process_review_v4',records=400,questions=100,
         evaluation_only=True,four_arms_are_model_comparisons_not_training_rollout_group=True,
         formula='P=sum(p1,p2,p3,p4); hold:null; fatal/model_failure:0; major:0.49*P/100; else:P/100',
         automatic_scoring=False,template_identity=digest(read(Path(work)/'IDENTITY.json'))))


SECRET = re.compile(rb'-{5}BEGIN[A-Z0-9 ]{0,60}PRIVATE KEY-{5}|\bGOCSPX-[A-Za-z0-9_-]{12,}|'
    rb'\bsk-[A-Za-z0-9_-]{20,}|\bya29\.[A-Za-z0-9._-]{20,}|'
    rb'(?i:Bearer\s+[A-Za-z0-9._-]{20,})|'
    rb'(?i:(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)["\s]*[:=]["\s]*[a-f0-9]{32,64}\b)')


def scan_file(p):
    tail=b''
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(65536),b''):
            require(not SECRET.search(tail+block),'potential_secret_pattern_export_hold')
            tail=(tail+block)[-1024:]


def sqlite_check(p):
    p=Path(p)
    for suffix in ('-wal','-shm'):
        extra=Path(str(p)+suffix)
        require(not extra.exists() or extra.stat().st_size==0,'nonempty_SQLite_WAL_or_SHM_export_hold')
    with closing(sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        require(db.execute('PRAGMA integrity_check').fetchall()==[('ok',)],'SQLite_integrity_failure')


def _collect_tree(source,prefix):
    source=Path(source);files={};sources={}
    for p in sorted(source.rglob('*')):
        require(not p.is_symlink(),'export_symlink_forbidden')
        if not p.is_file() or p.name in ('EVAL.lock',):continue
        if p.name.endswith(('-wal','-shm')):
            require(p.stat().st_size==0,'nonempty_SQLite_WAL_or_SHM_export_hold')
        require(p.suffix.lower() not in ('.safetensors','.pt','.bin') and p.name not in ('secrets.env','rclone.conf','gdrive-transfer.conf'),
                'weights_or_credentials_not_allowed_in_results_export')
        scan_file(p)
        if p.suffix in ('.sqlite','.db'):sqlite_check(p)
        name=p.relative_to(source).as_posix()
        safe(source,name)
        files[name]=sha(p);sources[prefix+'/'+name]=p
    return files,sources


def export_bundle(package,work,output,grades=None,reviews=None):
    package,work,output=Path(package).resolve(),Path(work).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(work) and not output.is_relative_to(package),
            'fresh_external_export_output_required')
    for extra in (grades,reviews):
        require(extra is None or not output.is_relative_to(Path(extra).resolve()),'export_output_overlaps_external_input')
    package_hash=verify_package(package)
    output.mkdir(parents=True)
    with lock(work/'EVAL.lock'):
        snapshot=output/'snapshot';snapshot.mkdir()
        copied={};sources={}
        for key,source in (('package',package),('work',work)):
            copied[key],added=_collect_tree(source,key);sources.update(added)
        for key,value in (('external_grades',grades),('external_reviews',reviews)):
            if value and not Path(value).resolve().is_relative_to(work):
                copied[key],added=_collect_tree(Path(value).resolve(),key);sources.update(added)
        cells=grid(package,work,grades,reviews)
        summary=write_summary(package,work,snapshot/'summary',grades,reviews,_cells=cells)
        build_review(package,work,snapshot/'review-template',_cells=cells)
        save(snapshot/'EXPORT.json',dict(schema='process_grpo_eval_private_export_v1',
            package_manifest_sha256=package_hash,work_identity=digest(read(work/'IDENTITY.json')),
            copied_files=copied,records=400,summary=summary,weights_included=False,
            secret_pattern_check='passed_not_full_privacy_review',sqlite_read_only_integrity_checked=True,
            originals_preserved=True,archive_streams_sources_without_full_work_copy=True))
        expected={prefix+'/'+name:value for prefix,files in copied.items() for name,value in files.items()}
        expected.update(manifest(snapshot))
        save(snapshot/'FILES.sha256.json',expected)
        for p in snapshot.rglob('*'):
            if p.is_file():scan_file(p)
        archive=output/'process_grpo_eval_results.tar.gz'
        temporary=output/(archive.name+'.partial')
        with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
            for name,source in sorted(sources.items()):
                require(sha(source)==expected[name],'source_changed_before_export')
                tar.add(source,arcname='process_grpo_eval_results/'+name,recursive=False)
                require(sha(source)==expected[name],'source_changed_during_export')
            for source in sorted(snapshot.rglob('*')):
                if source.is_file():
                    tar.add(source,arcname='process_grpo_eval_results/'+source.relative_to(snapshot).as_posix(),recursive=False)
        expected['FILES.sha256.json']=sha(snapshot/'FILES.sha256.json')
        with tarfile.open(temporary,'r:gz') as tar:
            members=[m for m in tar if m.isfile()]
            require(len(members)==len(expected),'archive_file_count_changed')
            require({m.name.removeprefix('process_grpo_eval_results/') for m in members}==set(expected),
                    'archive_member_set_changed')
            import hashlib
            for m in members:
                relative=m.name.removeprefix('process_grpo_eval_results/')
                require(relative in expected and not m.issym() and not m.islnk(),'unexpected_archive_member')
                h=hashlib.sha256()
                with tar.extractfile(m) as f:
                    for b in iter(lambda:f.read(1048576),b''):h.update(b)
                require(h.hexdigest()==expected[relative],'archive_file_hash_changed')
        for key,files in copied.items():
            root=package if key=='package' else work if key=='work' else Path(grades) if key=='external_grades' else Path(reviews)
            for name,value in files.items():require(sha(safe(root,name))==value,'source_changed_after_export')
        temporary.replace(archive)
        checksum=sha(archive)
        (output/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('summary','review-template','export'))
    p.add_argument('--package',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grades',type=Path);p.add_argument('--reviews',type=Path)
    p.add_argument('--historical-review-root',type=Path)
    a=p.parse_args()
    if a.command=='export':
        archive=export_bundle(a.package,a.work,a.output,a.grades,a.reviews)
        print('ARCHIVE='+str(archive));print('SHA256_FILE='+str(archive)+'.sha256')
        print('EXPORT_HASHES_AND_SQLITE=passed; secret-pattern scan is not full privacy review')
    else:
        with lock(a.work/'EVAL.lock'):
            if a.command=='summary':write_summary(a.package,a.work,a.output,a.grades,a.reviews)
            else:build_review(a.package,a.work,a.output,historical_review_root=a.historical_review_root)
        print('OFFLINE_OUTPUT_COMPLETE; planned_records=400; MODEL=0; API=0; no automatic Process scores')


if __name__=='__main__':main()
