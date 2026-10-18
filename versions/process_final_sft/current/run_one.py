"""One Process-only trajectory and deterministic real Final-input freeze; no Final/Judge."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from common import read, save, sha

sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime/src'))

async def run(a):
    from trainable_agent.cli import run as collect
    from trainable_agent.storage import digest
    from freeze_final_input import freeze
    question=read(a.question); folder=a.output; folder.mkdir(parents=True,exist_ok=True)
    config=read(a.config)
    args=SimpleNamespace(config=a.config,output=folder/'run',run_id=a.run_id,
                         question=question['question'],process_only=True)
    code=await collect(args)
    t=read(folder/'run/TRAJECTORY.json'); pre=read(folder/'run/PRE_FINAL_PACKAGE.json')
    assert t['question']==pre['question']==question['question']
    assert pre==t['prefinal'] and pre['opened_evidence']==t['evidence']
    assert pre['evidence_snapshot_sha256']==digest(t['evidence'])
    assert t['final'] is None and t['final_status']=='not_requested'
    result=dict(question_id=question['question_id'],split=question['split'],exit_code=code,
        process_status=t['run_status'],termination=t['termination'],completion_mode=t.get('completion_mode','normal'),
        evidence_count=len(t['evidence']),final_generated=False,teacher_input_ready=False,semantic_review='PENDING')
    if code==0 and t['run_status']=='process_complete' and result['completion_mode']=='normal':
        try:
            receipt=await freeze(folder,a.config)
            result.update(teacher_input_ready=True,quality_hold=receipt['quality_hold'])
        except Exception as error:
            result['final_input_error_type']=type(error).__name__
            save(folder/'FINAL_INPUT_FAILURE.json',dict(error_type=type(error).__name__,raw_records_preserved=True))
    save(folder/'RESULT.json',result)
    names=['RESULT.json','agent.config.json','run/TRAJECTORY.json','run/PRE_FINAL_PACKAGE.json','run/EVENTS.json']
    if result['teacher_input_ready']:
        names+=['teacher_input/INPUT_IDENTITY.json','teacher_input/FINAL_REQUEST.json','teacher_input/MESSAGES.json','teacher_input/TEACHER_INPUT.txt','teacher_input/PREPARATION_EVENTS.json']
    save(folder/'DONE.json',dict(files={n:sha(folder/n) for n in names},question_sha256=digest(question)))
    print(json.dumps(result),flush=True)
    # A recorded failed trajectory does not abort the whole question queue.
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--question',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-id',required=True);raise SystemExit(asyncio.run(run(p.parse_args())))
