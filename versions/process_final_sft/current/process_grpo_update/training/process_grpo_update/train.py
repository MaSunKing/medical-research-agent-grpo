"""Explicit allocated-GPU acceptance or first-batch Process-only GRPO update."""
import argparse
import math
import os
from pathlib import Path
import signal
import time
from contract import (acceptance_groups,batches,digest,load_data,read,require,save,sha,
                      validate_config,verify_files)

HERE=Path(__file__).resolve().parent


def check_alignment(model,data,groups,reference,c):
    import torch
    import runtime as rt
    count=0;maximum=0.;stages=set();nonempty_state=False;rejected=0;repaired=0;longest=0
    for g in groups:
        for r in g['rollouts']:
            for call in read(rt.call_path(data,r)):
                v=rt.logprobs(model,call,gradient=True,chunk=c['logit_chunk_tokens'])
                old=rt.reference_values(reference,r,call)
                err=float((v.detach()-torch.tensor(old,device=v.device)).abs().max())
                maximum=max(maximum,err);count+=1;stages.add(call['request']['stage'])
                longest=max(longest,len(call['request']['prompt_token_ids'])+len(call['response']['token_ids']))
                rejected+=not call['accepted'];repaired+=call['json_repaired']
                if call['request']['stage']=='state' and call['accepted']:
                    # Not used as a label or an admission filter. Only diagnostic coverage.
                    nonempty_state |= 'requirement_id' in call['response']['text']
                require(err<=c['forward_tolerance'],'initial_grad_forward_alignment_hold')
                del v
    require(stages=={'checklist','decision','state'} and nonempty_state,'acceptance_stage_coverage_incomplete')
    return dict(calls=count,maximum_absolute_error=maximum,stages=sorted(stages),
                nonempty_state_observed=nonempty_state,rejected_attempts=rejected,
                json_repaired_attempts=repaired,longest_sequence_tokens=longest)


def accept(model,tok,params,optimizer,a,groups,c,identity,stop):
    import torch
    import runtime as rt
    chosen=acceptance_groups(groups,c['acceptance_questions'])
    ref=a.output/'reference'
    save(a.output/'SELECTION.json',dict(question_ids=[g['question_id'] for g in chosen],
        includes_global_longest_nonheld=True,synthetic_rewards=False))
    require(rt.prepare_reference(model,tok,a.data,chosen,ref,identity,c,stop),'acceptance_reference_interrupted')
    before=rt.state_digest(rt.adapter_state(model))
    print('STAGE=INITIAL_GRAD_FORWARD_ALIGNMENT',flush=True)
    alignment=check_alignment(model,a.data,chosen,ref,c)
    save(a.output/'ALIGNMENT.json',alignment)
    plan=[chosen[:4],chosen[4:]]
    results=[]; first=None;final=None
    for i,batch in enumerate(plan,1):
        print('STAGE=DIAGNOSTIC_UPDATE_'+str(i),flush=True)
        metrics=rt.update_batch(model,params,optimizer,a.data,ref,batch,c,stop=stop)
        require(metrics is not None and metrics['optimizer_step'],'diagnostic_update_interrupted_or_empty')
        cp=rt.save_checkpoint(model,optimizer,a.output,i,identity,metrics)
        if i==1:first=cp
        results.append(metrics)
    final=rt.adapter_state(model)
    final_optimizer=rt.optimizer_snapshot(optimizer)
    require(before!=rt.state_digest(final),'optimizer_did_not_change_adapter')
    rt.restore(model,optimizer,first,identity)
    print('STAGE=RESUME_EQUIVALENCE_UPDATE',flush=True)
    replay=rt.update_batch(model,params,optimizer,a.data,ref,plan[1],c,stop=stop)
    require(replay is not None,'resume_check_interrupted')
    restored=rt.adapter_state(model)
    comparison=rt.resume_comparison(final,restored,results[-1],replay,
                                     final_optimizer,rt.optimizer_snapshot(optimizer))
    # Persist the observation BEFORE a failing gate. Never erase its exact values.
    save(a.output/'RESUME_COMPARISON.json',comparison)
    print('RESUME_COMPARISON='+str({k:comparison[k] for k in
        ('adapter_max_abs','loss_abs_difference','parameter_passed','loss_passed','optimizer_exact_match')}),flush=True)
    error=comparison['adapter_max_abs']
    observed=rt.save_checkpoint(model,optimizer,a.output/'resume_observed',2,identity,replay)
    require(comparison['parameter_passed'] and comparison['loss_passed'],'resume_update_mismatch')
    require(comparison['optimizer_exact_match'],
            'resume_optimizer_state_mismatch')
    # Reload the resumed state in a fresh adapter object; no in-memory-only save claim.
    last=observed
    expected_digest=rt.state_digest(restored)
    rt.restore(model,optimizer,last,identity)
    require(rt.state_digest(rt.adapter_state(model))==expected_digest,'saved_adapter_reload_failed')
    save(a.output/'DETAILS.json',dict(alignment=alignment,updates=results,resume=replay,
        resume_adapter_max_abs=error,resume_optimizer_exact_match=True,initial_adapter_unchanged_on_disk=True,formal_training=False,
        optimizer_steps=3,diagnostic_only=True))
    files={p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob('*') if p.is_file() and
           p.name not in ('ACCEPTANCE.json','.lock')}
    save(a.output/'ACCEPTANCE.json',dict(status='passed',identity=identity,files=files,
        output_role='diagnostic_only_never_formal_initialization',formal_training=False,
        optimizer_steps=3,questions=8,rollouts=32,synthetic_rewards=False))
    print('GPU_ACCEPTANCE=passed; diagnostic updates only; formal training not started',flush=True)


def train(model,tok,params,optimizer,a,groups,c,identity,stop):
    import runtime as rt
    require(a.acceptance is not None,'separate_GPU_acceptance_required')
    accepted=read(a.acceptance)
    require(accepted['status']=='passed' and accepted['identity']==identity and
            accepted['output_role']=='diagnostic_only_never_formal_initialization',
            'acceptance_identity_or_environment_mismatch')
    verify_files(a.acceptance.parent,accepted['files'])
    ref=a.output/'reference';plan=batches(groups,c)
    save(a.output/'PLAN.json',dict(batches=[[g['question_id'] for g in b] for b in plan],passes=1,
        held_questions=[g['question_id'] for g in groups if g['held']],
        mid_after_batch=math.ceil(len(plan)/2),first_batch_only=True))
    # Model is still original972 here, including when resuming a partial run.
    print('STAGE=PREPARE_INITIAL_TRAINING_PATH_REFERENCE',flush=True)
    if not rt.prepare_reference(model,tok,a.data,[g for b in plan for g in b],ref,identity,c,stop):
        print('PAUSED_REFERENCE; resume same output; optimizer_steps=0',flush=True);return
    checkpoints=sorted(a.output.glob('checkpoint-[0-9][0-9][0-9][0-9]'))
    completed=rt.restore(model,optimizer,checkpoints[-1],identity) if checkpoints else 0
    if (a.output/'TRAINING_COMPLETE.json').exists():
        require(completed==len(plan),'invalid_completion');print('ALREADY_COMPLETE',flush=True);return
    for i,batch in enumerate(plan):
        if i<completed:continue
        if stop():print('PAUSED_RESUME_SAME_OUTPUT',flush=True);return
        print('STAGE=UPDATE; batch='+str(i+1)+'/'+str(len(plan)),flush=True)
        metrics=rt.update_batch(model,params,optimizer,a.data,ref,batch,c,stop=stop)
        if metrics is None:
            print('PAUSED_PARTIAL_GRADIENTS_DISCARDED; resume last sealed checkpoint',flush=True);return
        cp=rt.save_checkpoint(model,optimizer,a.output,i+1,identity,metrics)
        save(a.output/'STATUS.json',dict(completed_batches=i+1,total_batches=len(plan),
            checkpoint=cp.name,metrics=metrics,formal_training=True))
        print('UPDATE_SAVED='+str(i+1)+'; loss='+str(round(metrics['loss'],6)),flush=True)
    save(a.output/'TRAINING_COMPLETE.json',dict(identity=identity,completed_batches=len(plan),
        checkpoint=f'checkpoint-{len(plan):04d}',mid_checkpoint=f'checkpoint-{math.ceil(len(plan)/2):04d}',
        evaluation_pending=True,medical_quality_approved=False,new_collection_required_before_next_round=True))
    print('FIRST_BATCH_UPDATE_COMPLETE; evaluation pending; no automatic next round',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'CONFIG.json')
    p.add_argument('--mode',choices=('accept','train'),required=True)
    p.add_argument('--acceptance',type=Path);p.add_argument('--real-gpu',action='store_true')
    p.add_argument('--seconds',type=int,default=19800)
    a=p.parse_args();require(a.real_gpu and os.environ.get('SLURM_JOB_ID'),'explicit_allocated_GPU_required')
    require(a.mode!='train' or a.acceptance is not None and a.acceptance.is_file(),
            'separate_GPU_acceptance_file_required_before_loading')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        LANGCHAIN_TRACING_V2='false',LANGSMITH_TRACING='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    require(600<=a.seconds<=20100,'invalid_slice_seconds')
    for root in (a.data,a.base,a.adapter,HERE.parents[1]):
        require(not a.output.resolve().is_relative_to(root.resolve()),'output_overlaps_immutable_inputs')
    if a.acceptance:
        require(not a.output.resolve().is_relative_to(a.acceptance.parent.resolve()),'formal_output_overlaps_diagnostic')
    groups,m,manifest=load_data(a.data);c=read(a.config);validate_config(c)
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(a.mode!='accept' or not (a.output/'RUN_IDENTITY.json').exists(),'fresh_acceptance_output_required')
    interrupted=[False];began=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:interrupted.__setitem__(0,True))
    stop=lambda:interrupted[0] or time.monotonic()-began>a.seconds
    import torch,runtime as rt
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model,tok,params,env=rt.load(a.data,a.base,a.adapter)
    identity=dict(schema='process_grpo_update_run_v1',dataset_sha256=manifest,config=c,
        code={x.name:sha(x) for x in sorted(HERE.glob('*.py'))},environment=env,
        initial_adapter=read(a.data/'EXPECTED_ORIGINAL972.json')['adapter_files'],
        base_identity_sha256=sha(a.data/'MODEL_IDENTITY.json'))
    run=dict(identity=identity,mode=a.mode)
    if (a.output/'RUN_IDENTITY.json').exists():require(read(a.output/'RUN_IDENTITY.json')==run,'run_identity_changed')
    else:save(a.output/'RUN_IDENTITY.json',run)
    optimizer=torch.optim.AdamW(list(params.values()),lr=c['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    print('ENVIRONMENT='+str(env),flush=True)
    fn=accept if a.mode=='accept' else train
    fn(model,tok,params,optimizer,a,groups,c,identity,stop)
    # Reverify the immutable initialization files after any diagnostic/formal work.
    verify_files(a.adapter,identity['initial_adapter'])


if __name__=='__main__':main()
