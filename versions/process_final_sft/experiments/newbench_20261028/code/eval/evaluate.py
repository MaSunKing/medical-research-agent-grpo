"""Four-arm Process evaluation. Real tools/models require allocated explicit opt-in."""
import argparse
import asyncio
import contextlib
import importlib.metadata
import os
from pathlib import Path
import signal
import sys
import time
from eval_common import (ARMS,read,save,sha,digest,require,safe,verify_files,
                         verify_package,verify_done,seal,lock,record_folder)

def validate_config(c):
    require(set(c)=={'base_model','original_adapter','training_output','retrieval','seed'},'unexpected_config_keys')
    require(type(c['seed']) is int and 0<=c['seed']<2**32,'invalid_seed')
    def walk(v):
        if isinstance(v,dict):
            for k,x in v.items():
                require(not any(z in k.lower() for z in ('api_key','access_token','refresh_token','password','authorization','client_secret')),'inline_credentials_forbidden')
                walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(c)
    require(c['retrieval']['worker']['kind']=='native','native_worker_required')
    require(c['retrieval'].get('snapshot','retrieval_snapshot_v2')=='retrieval_snapshot_v2','bundled_snapshot_required')

def frozen_imports(package):
    runtime=Path(package)/'runtime'
    for p in (runtime/'src',runtime/'training/process_grpo'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))

def bind_models(package,c):
    original=read(Path(package)/'data/EXPECTED_ORIGINAL972.json')['adapter_files']
    verify_files(Path(c['original_adapter']),original)
    models={'base':dict(path=c['original_adapter'],files=original,adapter_disabled=True),
            'sft':dict(path=c['original_adapter'],files=original,adapter_disabled=False)}
    out=Path(c['training_output']);done=read(out/'TRAINING_COMPLETE.json')
    require(done['completed_batches']==25 and done['checkpoint']=='checkpoint-0025'
            and done['mid_checkpoint']=='checkpoint-0013','complete_first_training_pass_required')
    expected=read(Path(package)/'data/EXPECTED_TRAINING.json')
    for arm,step in (('grpo13',13),('grpo25',25)):
        cp=out/f'checkpoint-{step:04d}';state=read(cp/'STATE.json')
        require(state['step']==step and state['identity']==done['identity'],'training_checkpoint_identity_changed')
        require(state['identity']['dataset_sha256']==expected['dataset_sha256'] and
                state['identity']['initial_adapter']['adapter_model.safetensors']==original['adapter_model.safetensors'],
                'wrong_training_lineage')
        require(state['identity']['config']==expected['config'] and state['identity']['code']==expected['code'],
                'wrong_training_configuration_or_source')
        require(state['identity']['environment']['deterministic_algorithms'] is True,'deterministic_training_required')
        verify_files(cp,read(cp/'COMPLETE.json'))
        models[arm]=dict(path=str(cp),files={n:sha(cp/n) for n in original},adapter_disabled=False,
                         training_step=step,complete_sha256=sha(cp/'COMPLETE.json'))
    expected_models=read(Path(package)/'data/EXPECTED_EVAL_MODELS.json')
    for arm in ARMS:
        require(models[arm]['files']==expected_models[arm]['files'] and
                models[arm]['adapter_disabled']==expected_models[arm]['adapter_disabled'],
                'adapter_differs_from_prior_verified_checkpoint:'+arm)
    return models,done['identity']['environment']

def prepare(package,config,work,include_fresh_baselines=True):
    package,work=Path(package),Path(work);c=read(config);validate_config(c)
    require(not work.exists(),'new_eval_work_required')
    for p in (package,Path(c['base_model']),Path(c['original_adapter']),Path(c['training_output'])):
        require(not work.resolve().is_relative_to(p.resolve()),'work_overlaps_immutable_input')
    pkg=verify_package(package);models,training_env=bind_models(package,c)
    frozen_imports(package)
    from trainable_agent.runtime_paths import verified_snapshot,worker_command
    snapshot,manifest=verified_snapshot(c['retrieval'],package/'runtime')
    worker_command(c['retrieval'],snapshot,work/'preflight-path-check-only')
    identity=dict(protocol='process_benchmark_eval_v1',dataset_sha256=sha(package/'data/questions.json'),
        package_sha256=pkg,config_sha256=digest(c),arms=list(ARMS),final_variant='base',models=models,
        generated_arms=list(ARMS), benchmark_protocols=read(package/'data/DATASET.json')['benchmarks'],
        historical_baselines_default=False,historical_configuration_mismatch=False,
        training_environment=training_env,process_head='fp32_linear_unmerged',final_head='original_BF16',
        policy_replay=False,training_eligible=False,rollouts_per_question=1,independent_unseen_test=True, novelty_scope="known_local_training_and_evaluation_pools_only",
        schedule='five_question_blocks_rotating_arms_v1',live_retrieval_time_variation_possible=True,
        retrieval_snapshot_sha256=sha(snapshot/'MANIFEST.json'),
        environment={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','bitsandbytes','fastmcp','langgraph')},
        budgets=dict(context=12288,hard=15000,repair=384,safety=128,normal_tools=6,exemptions=5,
                     invalid_decisions=6,compression_calls=12),
        thread_policy=dict(allocated_cpus=8,passage_torch_threads=4,frozen_history_rank_threads=2))
    work.mkdir(parents=True)
    save(work/'CONFIG.json',c,exclusive=True);save(work/'IDENTITY.json',identity,exclusive=True)
    save(work/'STATUS.json',dict(stage='PREPARED',expected=400,finished=0))
    print('PREPARED; questions=100; new_records=400; historical=0; MODEL=0; API=0',flush=True)
    return identity

class EvalProcessPolicy:
    def __init__(self,model,tokenizer,arm,seed,output,identity,active):
        self.model,self.tokenizer,self.arm=model,tokenizer,arm
        self.seed,self.output,self.ordinal,self.active=seed,Path(output),0,active
        self.identity=digest(dict(mode='evaluation_without_replay_v1',arm=arm,run=identity,seed=seed))

    async def generate(self,request):
        import native
        from decoupled import process_head
        self.active(request['stage']);self.ordinal+=1
        prompt=self.tokenizer.apply_chat_template(request['messages'],tokenize=True,return_dict=False,
                    add_generation_prompt=True,enable_thinking=False)
        require(prompt==request['prompt_token_ids'],'prompt_tokens_changed')
        require(len(prompt)+request['max_new_tokens']<=15000,'context_exceeded')
        folder=self.output/f'{self.ordinal:04d}_{request["stage"]}'
        save(folder/'REQUEST.json',request,exclusive=True)
        seed=int(digest([self.seed,self.ordinal,request['input_sha256']])[:8],16)
        scope=self.model.disable_adapter() if self.arm=='base' else contextlib.nullcontext()
        began=time.monotonic();audit={}
        self.model.eval().requires_grad_(False)
        try:
            with scope,process_head(self.model):
                ids,logp=native.sample(self.model,self.tokenizer,prompt,request['max_new_tokens'],seed,1.,audit=audit)
            require(audit.get('effective_parameters_verified') is True,'effective_sampling_not_verified')
            result=dict(text=self.tokenizer.decode(ids,skip_special_tokens=True),model='eval-'+self.arm,
                token_ids=ids,behavior_logprobs=logp,sampling=dict(seed=seed,temperature=1.,**audit),
                policy_identity=self.identity,process_head='fp32_linear_unmerged',training_eligible=False,
                grpo_capture_ready=False,probability_replay='not_requested_for_evaluation',
                finish_reason='eos' if ids[-1] in native.stop_tokens(self.model,self.tokenizer) else 'length',
                usage=dict(prompt_tokens=len(prompt),completion_tokens=len(ids)),seconds=time.monotonic()-began)
            save(folder/'RESPONSE.json',result,exclusive=True)
            return result
        except Exception as e:
            save(folder/'FAILURE.json',dict(error_type=type(e).__name__,raw_records_preserved=True),exclusive=True)
            raise

def classify(t):
    term=t.get('process_termination') or t.get('termination')
    engineering=t.get('failure')
    if engineering and engineering.get('stage')=='final':
        return 'final_failure',('degraded' if t.get('completion_mode')=='degraded' else 'complete')
    if engineering or term=='engineering_failure':return 'engineering_failure','engineering_failure'
    if term in ('model_contract_failure','context_budget_exhausted'):return 'process_failure','failed'
    process='degraded' if t.get('completion_mode')=='degraded' else 'complete'
    return ('complete' if t.get('final_status')=='complete' else 'final_failure'),process

def result_from(t,events,row,arm,identity,seconds):
    status,ps=classify(t)
    first=[e['data']['request'] for e in events if e['kind']=='model_request' and
           e['data']['request']['role']=='final'][:1]
    answer=t.get('final') if t.get('final_status')=='complete' else None
    require(answer is None or isinstance(answer,str),'invalid_final_answer')
    result=dict(schema='process_grpo_eval_result_v1',index=row['index'],question_id=row['question_id'],arm=arm,benchmark=row['benchmark'],source_id=row['source_id'],
        run_identity=identity,status=status,process_status=ps,process_termination=t.get('process_termination') or t.get('termination'),
        final_status=t.get('final_status'),answer=answer,answer_sha256=digest(answer) if answer is not None else None,
        final_input_sha256=first[0]['input_sha256'] if first else None,issues=[],seconds=seconds,
        evidence_count=len(t.get('evidence',{})),final_variant='base',local_review='PENDING')
    if status!='complete':result['issues'].append(result['process_termination'] if status=='process_failure' else status)
    return result,first[0] if first else None

async def run_one(package,work,c,identity,row,arm,model,tokenizer):
    from fastmcp import Client
    from trainable_agent.context import ChatCodec,ContextBuilder
    from trainable_agent.engine import Engine,initial_state
    from trainable_agent.runtime_paths import prepare_worker,retrieval_timeouts
    from trainable_agent.storage import Store
    from trainable_agent.tools import WorkerBridge,create_server,MCPTools
    import native
    folder=record_folder(work,arm,row['index']);folder.mkdir(parents=True,exist_ok=False)
    rid=digest(identity);seed=int(digest(['multibench-paired-process',c['seed'],row['question_id']])[:8],16)
    def active(stage):save(work/'ACTIVE.json',dict(arm=arm,index=row['index'],question_id=row['question_id'],stage=stage,time=time.time()))
    save(folder/'START.json',dict(run_identity=rid,arm=arm,index=row['index'],seed=seed,time=time.time()),exclusive=True)
    process=EvalProcessPolicy(model,tokenizer,arm,seed,folder/'captures/process',rid,active)
    final=native.NativePolicy(model,tokenizer,process=False,seed=int(digest(['paired-final',c['seed'],row['question_id']])[:8],16),
        output=folder/'captures/final',policy_identity='frozen-base',active=active)
    compressor=native.NativePolicy(model,tokenizer,process=False,seed=int(digest(['paired-reader',c['seed'],row['question_id']])[:8],16),
        output=folder/'captures/compression',policy_identity='frozen-base-reader')
    context=ContextBuilder(ChatCodec(tokenizer),12288,evidence_cards=True,safety_margin=128,
        repair_reserve_tokens=384,soft_context_tokens=12288,hard_context_tokens=15000,compression_max_calls=12)
    run_id=f'eval-{arm}-Q{row["index"]:02d}';store=Store(folder/'trajectory.sqlite');bridge=None;began=time.monotonic()
    try:
        command,manifest=prepare_worker(c['retrieval'],Path(package)/'runtime',folder/'retrieval')
        deadlines=retrieval_timeouts(c['retrieval'])
        bridge=WorkerBridge(command,folder/'retrieval.stderr.log',timeout=deadlines['bridge'],
            startup_timeout=deadlines['startup'],operation_timeout=deadlines['operation'])
        state=initial_state(run_id,row['question'],process.identity,'frozen-base-final',digest(manifest),
            max_charged=6,max_exemptions=5,max_invalid_decisions=6)
        async with Client(create_server(bridge)) as client:
            tools=MCPTools(client)
            t=await Engine(store,process,final,tools,context,tools.rank,compressor=compressor,
                           compression_context_tokens=10240).run(state)
        events=store.events(run_id)
        save(folder/'TRAJECTORY.json',t,exclusive=True)
        save(folder/'PRE_FINAL_PACKAGE.json',t.get('prefinal'),exclusive=True)
        save(folder/'EVENTS.json',events,exclusive=True)
        result,request=result_from(t,events,row,arm,rid,time.monotonic()-began)
        if request:save(folder/'FINAL_INPUT.json',request,exclusive=True)
        else:save(folder/'DELIVERY_STATUS.json',dict(status='no_final_request',not_reconstructed=True),exclusive=True)
        save(folder/'RESULT.json',result,exclusive=True)
    finally:
        try:
            if bridge is not None:await bridge.close()
        finally:store.db.close()
    seal(folder,rid)
    print(f'QUESTION_FINISHED Q{row["index"]:02d} {arm} status={result["status"]}',flush=True)
    return result

def update_status(package,work,identity):
    counts={k:0 for k in ('complete','process_failure','final_failure','engineering_failure','interrupted','missing')}
    for q in read(Path(package)/'data/questions.json'):
        for arm in identity['generated_arms']:
            folder=record_folder(work,arm,q['index'])
            if not verify_done(folder,digest(identity)):
                counts['interrupted' if folder.exists() else 'missing']+=1
            else:counts[read(folder/'RESULT.json')['status']]+=1
    finished=sum(counts[k] for k in ('complete','process_failure','final_failure','engineering_failure'))
    expected=len(identity['generated_arms'])*100
    save(Path(work)/'STATUS.json',dict(stage='EVALUATION',expected=expected,finished=finished,counts=counts))
    if finished==expected:
        save(Path(work)/'EVALUATION_COMPLETE.json',dict(run_identity=digest(identity),finished=finished,
            failures_preserved=True,medical_quality_approved=False))

async def run(a):
    require(a.real_gpu and os.environ.get('SLURM_JOB_ID'),'explicit_allocated_GPU_required')
    require(600<=a.seconds<=20100,'invalid_slice_seconds')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=8,'eight_CPU_allocation_required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        MEDGAP_RETRIEVAL_TORCH_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    package,work=a.package,a.work;identity=read(work/'IDENTITY.json');c=read(work/'CONFIG.json')
    from ops import require_pilot
    if a.pilot_only:
        require(a.indices==[1,51] and set(a.arms)==set(ARMS) and a.limit is None,'pilot_grid_must_cover_both_benchmarks_all_arms')
    else:
        require_pilot(work,identity)
    require(set(a.arms)<=set(identity['generated_arms']),'arm_not_authorized_by_eval_plan')
    require(verify_package(package)==identity['package_sha256'] and digest(c)==identity['config_sha256'],'eval_identity_changed')
    frozen_imports(package)
    require(identity['environment']=={n:importlib.metadata.version(n) for n in identity['environment']},'runtime_versions_changed')
    import torch,native
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16_CUDA_required')
    require(torch.cuda.get_device_name(0)==identity['training_environment']['gpu'],'GPU_type_changed')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    stopped=[False];began=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    questions=read(package/'data/questions.json');limit=a.limit or 100
    require(1<=limit<=100,'invalid_question_limit')
    with lock(work/'EVAL.lock'):
        verify_files(c['base_model'],read(package/'data/MODEL_IDENTITY.json')['files'])
        for selected,arm in work_units(selected_questions(questions,limit,a.indices),a.arms):
            binding=identity['models'][arm];verify_files(binding['path'],binding['files'])
            pending=[]
            for q in selected:
                folder=record_folder(work,arm,q['index'])
                if verify_done(folder,digest(identity)):continue
                require(not folder.exists(),'interrupted_record_requires_reconciliation:'+str(folder))
                pending.append(q)
            if not pending:continue
            if stopped[0] or time.monotonic()-began>a.seconds-600:break
            print('STAGE=LOAD_ARM; arm='+arm,flush=True)
            model,tokenizer=native.load(dict(base_model=c['base_model'],process_adapter=binding['path']))
            require(all(not p.requires_grad for p in model.parameters()),'eval_parameters_must_be_frozen')
            require(all(p.dtype==torch.float32 for n,p in model.named_parameters() if 'lora_A.' in n or 'lora_B.' in n),'FP32_LoRA_required')
            try:
                for q in pending:
                    if stopped[0] or time.monotonic()-began>a.seconds-600:
                        print('PAUSED_RESUME_SAME_WORK; completed failures also retained',flush=True);break
                    result=await run_one(package,work,c,identity,q,arm,model,tokenizer)
                    update_status(package,work,identity)
                    if result['status']=='engineering_failure':
                        raise ValueError('engineering_failure_preserved_inspect_before_continuing')
            finally:
                del model;torch.cuda.empty_cache()
            if stopped[0] or time.monotonic()-began>a.seconds-600:break
        update_status(package,work,identity);save(work/'ACTIVE.json',dict(stage='IDLE',time=time.time()))
        print('EVALUATION_SLICE_COMPLETE; no training or judge',flush=True)

def selected_questions(rows,limit,indices=None):
    if indices is None:return rows[:limit]
    require(len(set(indices))==len(indices) and set(indices)<=set(q['index'] for q in rows),'invalid_selected_indices')
    return [q for q in rows if q['index'] in indices]

def work_units(rows,arms):
    """Short balanced blocks limit cross-arm live-web time separation."""
    for block,start in enumerate(range(0,len(rows),5)):
        offset=block%len(arms)
        order=list(arms[offset:])+list(arms[:offset])
        for arm in order:yield rows[start:start+5],arm

def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    v=s.add_parser('prepare');v.add_argument('--package',type=Path,required=True);v.add_argument('--config',type=Path,required=True);v.add_argument('--work',type=Path,required=True)
    v.add_argument('--include-fresh-baselines',action='store_true')
    v=s.add_parser('run');v.add_argument('--package',type=Path,required=True);v.add_argument('--work',type=Path,required=True)
    v.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));v.add_argument('--limit',type=int);v.add_argument('--indices',type=int,nargs='+')
    v.add_argument('--seconds',type=int,default=19800);v.add_argument('--real-gpu',action='store_true')
    v.add_argument('--pilot-only',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.package,a.config,a.work,a.include_fresh_baselines)
    else:asyncio.run(run(a))

if __name__=='__main__':main()
