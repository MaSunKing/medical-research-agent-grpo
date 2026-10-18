"""Serial Process-only queue; immutable attempts, budgeted allocation, no automatic failed retries."""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from common import read,save,load_work,locked,result_path,verify_done,verify_adapter,verify_base

def make_config(c,q):
    url=f"http://127.0.0.1:{c['port']}/v1"
    return dict(tokenizer=c['base_model'],context_tokens=12288,soft_context_tokens=12288,hard_context_tokens=15000,
        repair_reserve_tokens=384,context_safety_margin=128,evidence_cards=True,compression_max_calls=12,
        compression_context_tokens=10240,decision_budget=dict(max_rejected=6),
        tool_budget=dict(normal_calls=6,failure_exemptions=5),final=None,
        process=dict(base_url=url,model='process-policy',temperature=1.,seed=q['seed']),
        compression=dict(base_url=url,model='frozen-base-selector',temperature=.1,seed=q['seed']+10000),
        retrieval=dict(snapshot='retrieval_snapshot_v2',worker=dict(kind='native',python=c['retrieval_python'],
            secrets=c['secrets_file'],timeouts=dict(startup=300,operation=360,bridge=390),
            models=dict(tokenizer=c['base_model'],**c['retrieval_models']))))

def collect(a):
    work,c,p=load_work(a.work);verify_adapter(p,c['process_adapter']);verify_base(p,c['base_model'])
    if a.phase=='all':
        pilot=read(work/'PILOT_ACCEPTED.json')
        assert pilot['package_sha256']==read(work/'IDENTITY.json')['package_sha256']
    assert not any(k in c for k in ['api_key','SERPER_API_KEY','DASHSCOPE_API_KEY'])
    questions=read(p/'QUESTIONS.json')
    if a.phase=='pilot':
        ids=set(read(p/'SPLIT.json')['pilot']);questions=[q for q in questions if q['question_id'] in ids]
    start=time.monotonic(); stop=[False]
    for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    with locked(work):
        # Never attach to an unknown model service already using this port.
        import socket
        with socket.socket() as sock:
            sock.settimeout(1)
            assert sock.connect_ex(('127.0.0.1',c['port']))!=0,'port_in_use'
        model_log=(work/f'model_{time.time_ns()}.log').open('x',encoding='utf8')
        model=subprocess.Popen([c['python'],'-B',str(p/'serve_policy.py'),'--base-model',c['base_model'],
            '--process-adapter',c['process_adapter'],'--port',str(c['port']),'--context-tokens','15000'],
            stdout=model_log,stderr=model_log)
        try:
            import json
            deadline=time.monotonic()+900
            while True:
                assert model.poll() is None,'model_service_exited'
                assert time.monotonic()<deadline,'model_startup_timeout'
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{c['port']}/health",timeout=2) as response:health=json.load(response)
                except OSError:time.sleep(2);continue
                assert health['context_tokens']==15000 and health['adapters']=='provided_frozen'
                assert health['precision']=='NF4_double_quant_BF16_compute'
                assert health['role_adapters']['process-policy']=='process' and health['role_adapters']['frozen-base-selector'] is None
                save(work/'MODEL_HEALTH.json',health);break
            for q in questions:
                if stop[0] or time.monotonic()-start>a.seconds-3900:break
                folder=result_path(work,q)
                if verify_done(folder):continue
                if folder.exists():
                    # Interrupted/pending calls require receipt/lease reconciliation, never blind replay.
                    print('NEEDS_RECONCILIATION',q['question_id'],flush=True);continue
                folder.mkdir(parents=True,exist_ok=False)
                save(folder/'QUESTION.json',q);save(folder/'agent.config.json',make_config(c,q))
                run_id='prefinal600-'+q['question_id']
                save(folder/'START.json',dict(run_id=run_id,started=time.time()))
                save(work/'ACTIVE.json',dict(stage='PROCESS_TO_PREFINAL',question_id=q['question_id'],split=q['split'],started=time.time()))
                env={k:v for k,v in os.environ.items() if not k.startswith('MEDGAP_')}
                env.update(PYTHONPATH=str(p/'runtime/src'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1',LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',
                    MEDGAP_EVIDENCE_READER_BASE_URL='',MEDGAP_EVIDENCE_READER_MODEL='',MEDGAP_EVIDENCE_READER_LOCAL_MODEL='')
                cmd=[c['python'],'-B',str(p/'run_one.py'),'--question',str(folder/'QUESTION.json'),
                    '--config',str(folder/'agent.config.json'),'--output',str(folder),'--run-id',run_id]
                with (folder/'launcher.log').open('x',encoding='utf8') as out,(folder/'launcher.err').open('x',encoding='utf8') as err:
                    child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
                    try:code=child.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        code=124
                if not (folder/'DONE.json').is_file():
                    save(folder/'INTERRUPTED.json',dict(exit_code=code,manual_reconciliation_required=True))
                save(work/'ACTIVE.json',dict(stage='IDLE',last_question=q['question_id']))
                print('QUESTION_FINISHED',q['question_id'],'exit='+str(code),flush=True)
            save(work/'ACTIVE.json',dict(stage='IDLE'))
            print('QUEUE_SLICE_COMPLETE; resume same WORK; no failed resampling',flush=True)
        finally:
            model.terminate()
            try:model.wait(timeout=30)
            except subprocess.TimeoutExpired:model.kill();model.wait()
            model_log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('phase',choices=['pilot','all']);p.add_argument('--seconds',type=int,default=18000)
    collect(p.parse_args())
