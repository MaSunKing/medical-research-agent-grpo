"""Acceptance receipts and concise read-only status, no model or network."""
import argparse
from pathlib import Path
from eval_common import ARMS,read,save,digest,sha,require,verify_done,record_folder,verify_package,lock

def validate_pilot(package,work):
    package,work=Path(package),Path(work)
    identity=read(work/'IDENTITY.json');rid=digest(identity)
    require(verify_package(package)==identity['package_sha256'],'pilot_package_changed')
    results=[]
    with lock(work/'EVAL.lock'):
        for arm in ARMS:
            for i in (1,51):
                folder=record_folder(work,arm,i)
                require(verify_done(folder,rid),'pilot_record_missing_or_unsealed')
                r=read(folder/'RESULT.json')
                require(r['status'] in ('complete','process_failure','final_failure'),'pilot_engineering_failure')
                results.append(dict(arm=arm,index=i,status=r['status'],result_sha256=sha(folder/'RESULT.json')))
        for i in (1,51):
            require(any(r['index']==i and r['status']=='complete' for r in results),'pilot_no_complete_record_for_benchmark')
        receipt=dict(run_identity=rid,package_sha256=identity['package_sha256'],records=results,
            accepted_for_generation=True,medical_quality_approved=False,judge_API_accepted=False,training=False)
        target=work/'PILOT_ACCEPTANCE.json'
        if target.exists():require(read(target)==receipt,'pilot_acceptance_changed')
        else:save(target,receipt,exclusive=True)
    print('GENERATION_PILOT=passed; 8 sealed records; semantic/judge acceptance separate',flush=True)
    return receipt

def require_pilot(work,identity):
    work=Path(work);r=read(work/'PILOT_ACCEPTANCE.json')
    require(r['run_identity']==digest(identity) and r['package_sha256']==identity['package_sha256']
            and r['accepted_for_generation'] is True and len(r['records'])==8,'pilot_acceptance_missing_or_changed')
    require({(x['arm'],x['index']) for x in r['records']}=={(a,i) for a in ARMS for i in (1,51)},'pilot_grid_changed')
    for x in r['records']:
        folder=record_folder(work,x['arm'],x['index'])
        require(verify_done(folder,digest(identity)) and sha(folder/'RESULT.json')==x['result_sha256'],'pilot_source_changed')

def status(work):
    work=Path(work)
    for filename in ('ACTIVE.json','STATUS.json'):
        if (work/filename).exists():
            x=read(work/filename)
            if filename=='ACTIVE.json':print('Activity (last saved):',x.get('arm','-'),x.get('index','-'),x.get('stage','unknown'))
            else:print('Finished',str(x.get('finished',0))+'/'+str(x.get('expected',400)),'Counts',x.get('counts',{}))
    grades=work/'grades/PASS_STATUS.json'
    if grades.exists():print('Judge (last saved):',read(grades).get('counts',{}))

def validate_judge(package,work):
    from grading import result_for
    from review_export import _benchmark_score
    package,work=Path(package),Path(work);identity=read(work/'IDENTITY.json');rid=digest(identity)
    require(verify_package(package)==identity['package_sha256'],'judge_package_changed')
    rows={q['index']:q for q in read(package/'data/questions.json')};scored=[]
    with lock(work/'EVAL.lock'):
        for arm in ARMS:
            for i in (1,51):
                result,folder=result_for(work,rows[i],arm,rid)
                require(result is not None,'judge_pilot_generation_missing')
                if result['status']!='complete':continue
                issues=[];value=_benchmark_score(work/'grades',arm,i,folder,sha(folder/'RESULT.json'),issues)
                require(value is not None,'judge_pilot_score_missing_or_deferred')
                scored.append(dict(arm=arm,index=i,score=value))
        require({x['index'] for x in scored}=={1,51},'judge_pilot_must_cover_both_protocols')
        save(work/'JUDGE_PILOT_ACCEPTANCE.json',dict(run_identity=rid,scored=scored,
            API_schema_cache_engineering_passed=True,medical_judgment_approved=False))
    print('JUDGE_PILOT=passed; protocols=2; semantic calibration remains local',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('pilot-check','judge-check','status'))
    p.add_argument('--package',type=Path);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args()
    if a.command=='pilot-check':validate_pilot(a.package,a.work)
    elif a.command=='judge-check':validate_judge(a.package,a.work)
    else:status(a.work)
