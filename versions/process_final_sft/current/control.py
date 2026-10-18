"""Compact read-only status, or explicit human acceptance of the initial pilot."""
import argparse
from contextlib import closing
from collections import Counter
from common import read,save,load_work,result_path,verify_done

def status(work):
    work,c,p=load_work(work,verify=False);counts=Counter();failures=[]
    for q in read(p/'QUESTIONS.json'):
        f=result_path(work,q)
        if not f.exists():continue
        counts['started']+=1
        if (f/'DONE.json').is_file():
            counts['finished']+=1;r=read(f/'RESULT.json')
            if r['process_status']=='process_complete':counts['normal']+=1
            elif r['completion_mode']=='degraded':counts['degraded']+=1
            else:counts['failed']+=1;failures.append(q['question_id'])
            if r['teacher_input_ready']:
                counts['ready']+=1;counts[q['split']+'_ready']+=1
                if r.get('quality_hold'):counts['no_evidence_hold']+=1
        else:counts['incomplete']+=1
    activity=read(work/'ACTIVE.json') if (work/'ACTIVE.json').exists() else {'stage':'PREPARED'}
    inner=''
    if activity.get('question_id'):
        import sqlite3,json
        db=result_path(work,activity)/'run/trajectory.sqlite'
        if db.is_file():
            try:
                with closing(sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=1)) as conn:
                    row=conn.execute('SELECT data FROM checkpoint LIMIT 1').fetchone()
                    if row:inner=json.loads(row[0]).get('next_node','unknown')
            except sqlite3.Error:inner='unknown'
    print('Activity:',activity.get('question_id','-'),activity.get('stage','-'),'| inner:',inner or 'not observed')
    print(f"Started {counts['started']}/600 | Finished {counts['finished']} | Normal {counts['normal']} | Degraded {counts['degraded']} | Failed {counts['failed']}")
    print(f"Frozen Final inputs {counts['ready']} | Train {counts['train_ready']}/500 | Dev {counts['dev_ready']}/100")
    print(f"No-evidence quality hold {counts['no_evidence_hold']} | Interrupted/pending {counts['incomplete']}")
    print('No Final/Judge/training; engineering completion is not gold-data approval.')
    if failures:print('Process failures:',', '.join(failures)[:350])
    return counts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('mode',choices=['status','accept-pilot']);a=p.parse_args()
    if a.mode=='status':status(a.work)
    else:
        work,c,package=load_work(a.work)
        for q in read(package/'QUESTIONS.json'):
            if q['question_id'] in read(package/'SPLIT.json')['pilot']:
                f=result_path(work,q);assert verify_done(f)
                r=read(f/'RESULT.json');assert r['teacher_input_ready'] and not r.get('quality_hold')
        save(work/'PILOT_ACCEPTED.json',dict(package_sha256=read(work/'IDENTITY.json')['package_sha256'],
            explicit_user_acceptance=True,clinical_quality_certified=False))
        print('PILOT_ACCEPTED; no full run submitted')
