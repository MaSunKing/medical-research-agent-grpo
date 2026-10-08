"""Offline single-A40 sparse-mask QLoRA candidate; no automatic training approval."""
import argparse
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from core import checkpoint_complete, masked_loss, sha, token_mass, validate_record

def read_data(root, mode):
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if Path(name).name != name or sha(root/name)!=digest:
            raise ValueError('data manifest mismatch')
    report=json.loads((root/'BUILD_REPORT.json').read_text())
    if mode=='mixed' and not report['mixed_training_eligible']:
        raise ValueError('No verified continuous samples; mixed training is prohibited')
    examples=[]; masses={}
    for kind in ('stepwise','trajectory'):
        if mode=='baseline' and kind=='trajectory':continue
        rows=[json.loads(x) for x in (root/(kind+'.jsonl')).read_text().split('\n') if x.strip()]
        for row in rows:validate_record(row)
        masses[kind]=sum(token_mass(r) for r in rows)
        examples.extend((kind,r) for r in rows)
    if not examples or masses['stepwise']<=0:
        raise ValueError('empty supervision')
    # Mean over N rows has expected contribution 0.8/0.2 by weighted target tokens.
    shares={'stepwise':1.0} if mode=='baseline' else {'stepwise':0.8,'trajectory':0.2}
    scales={kind:len(examples)*shares[kind]/mass for kind,mass in masses.items()}
    return examples,scales,report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['baseline'],required=True)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--diagnostic-stop',type=int)
    p.add_argument('--diagnostic-authorized',action='store_true')
    p.add_argument('--approval',type=Path)
    p.add_argument('--initial-adapter',type=Path)
    p.add_argument('--steps',type=int,default=843)
    p.add_argument('--accum',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--seconds',type=int,default=42300)
    p.add_argument('--prepare-research-approval',action='store_true')
    a=p.parse_args()
    if a.initial_adapter or a.steps!=843 or a.accum!=8 or a.lr!=1e-4 or a.save_every!=200 or a.seconds!=42300:
        raise ValueError('Frozen raw-base Process700 configuration mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Required CUBLAS_WORKSPACE_CONFIG missing')

    if min(a.steps,a.accum,a.save_every,a.seconds)<=0 or a.lr<=0:
        raise ValueError('invalid hyperparameters')
    examples,scales,report=read_data(a.data,a.mode)
    from process_checks import verify_data
    verify_data(a.data)
    for name,digest in report['tokenizer_files'].items():
        if sha(a.model/name)!=digest:raise ValueError('server tokenizer identity mismatch')
    base_identity={name:sha(a.model/name) for name in ('config.json','model.safetensors.index.json')}
    index=json.loads((a.model/'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        f=a.model/name
        if Path(name).name!=name or not f.is_file():raise ValueError('missing or invalid base model shard')
    print(json.dumps({'preflight':'passed','mode':a.mode,'rows':len(examples),
        'max_tokens':max(len(r['input_ids']) for _,r in examples),'steps':a.steps,
        'stage_share':'weighted_target_tokens','scales':scales,'model_loaded':False}),flush=True)
    if a.preflight:return
    identity={'dataset':sha(a.data/'DATA_MANIFEST.json'),'code':sha(Path(__file__)),
        'core':sha(Path(__file__).with_name('core.py')),'checks':sha(Path(__file__).with_name('process_checks.py')),'base':base_identity,
        'tokenizer':report['tokenizer_files'],'model_path':str(a.model.resolve()),
        'mode':a.mode,'steps':a.steps,'accum':a.accum,'lr':a.lr,'save_every':a.save_every,
        'precision':'NF4_double_quant_BF16_base_FP32_LoRA_no_prepare',
        'deterministic_math_sdpa':False,'attention_backend':'flash_deterministic','deterministic_algorithms':True,'scales':scales}
    if a.initial_adapter:
        identity['initial_adapter']={n:sha(a.initial_adapter/n) for n in ('adapter_config.json','adapter_model.safetensors')}
    if a.prepare_research_approval:
        raise ValueError('No automatic approval: finish new Process700 semantic review and new-data GPU/resume tests first')
    diagnostic=a.diagnostic_stop is not None
    if diagnostic and (not a.diagnostic_authorized or not 1<=a.diagnostic_stop<=20):
        raise ValueError('Explicit diagnostic authorization and stop between 1 and 20 required')
    if not a.gpu_smoke and not diagnostic:
        if not a.approval:raise ValueError('Explicit candidate-specific approval required')
        approved=json.loads(a.approval.read_text())
        semantic=json.loads((a.data.parent/'audit/SELECTION_ACCEPTANCE.json').read_text())
        if not semantic.get('semantic_full_review_complete') and approved.get('incomplete_semantic_review_acknowledged') is not True:
            raise ValueError('Incomplete medical semantic review must be explicitly acknowledged')
        if approved.get('identity')!=identity or approved.get('research_training_authorized') is not True:
            raise ValueError('Approval identity mismatch')
        for key in ('gpu_smoke','resume_parity'):
            evidence=Path(approved[key]['path'])
            if sha(evidence)!=approved[key]['sha256']:raise ValueError('New-data evidence hash mismatch')
            receipt=json.loads(evidence.read_text())
            if receipt.get('data_manifest_sha256')!=identity['dataset'] or receipt.get('status')!='passed':
                raise ValueError('New-data GPU or resume test not passed')
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    output=a.output.resolve()
    if output in (a.data.resolve(),a.model.resolve(),Path.home().resolve()) or output==Path('/'):
        raise ValueError('unsafe output')
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    candidates=sorted(output.glob('checkpoint-[0-9]'+'[0-9]'*7))
    for c in candidates:checkpoint_complete(c,identity)
    latest=candidates[-1] if candidates else None
    if a.gpu_smoke and latest:raise ValueError('smoke output must be fresh')
    if (output/'TRAINING_COMPLETE').exists():
        if latest is None:raise ValueError('completion marker without checkpoint')
        saved=torch.load(latest/'state.pt',map_location='cpu',weights_only=False)
        if saved['step']!=a.steps:raise ValueError('completion state mismatch')
        print('ALREADY_COMPLETE',flush=True);return
    base=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    for parameter in base.parameters():parameter.requires_grad_(False)
    base.config.use_cache=False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    base.enable_input_require_grads()
    adapter=latest or a.initial_adapter
    model=(PeftModel.from_pretrained(base,str(adapter),is_trainable=True) if adapter else
        get_peft_model(base,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')))
    trainables=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    if not trainables or any('lora_' not in n or v.dtype!=torch.float32 for n,v in trainables):
        raise ValueError('unexpected trainable precision/parameters')
    if any(v.is_floating_point() and v.dtype!=torch.bfloat16 for n,v in model.named_parameters() if 'lora_' not in n):
        raise ValueError('base floating precision drift')
    model.train()
    if a.gpu_smoke:
        _,row=max(examples,key=lambda x:len(x[1]['input_ids']))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            numerator,mass=masked_loss(model,torch.tensor([row['input_ids']],device='cuda'),row['target_spans'])
            loss=numerator/mass
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(v.grad).all() for _,v in trainables if v.grad is not None):
            raise FloatingPointError('nonfinite smoke')
        info={'status':'passed','longest_forward_backward':'passed','new_mask_resume_parity':'NOT_TESTED',
            'data_manifest_sha256':identity['dataset'],'tokens':len(row['input_ids']),
            'gpu':torch.cuda.get_device_name(),'loss':float(loss.detach()),
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
            'identity':identity,'optimizer_updates':0}
        (output/'GPU_ACCEPTANCE.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info),flush=True);return
    optimizer=torch.optim.AdamW([v for _,v in trainables],lr=a.lr)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:
        min(1.0,(s+1)/max(1,int(a.steps*.05)))*.5*(1+math.cos(math.pi*min(s,a.steps)/a.steps)))
    step=epoch=position=0
    if latest:
        state=torch.load(latest/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step,epoch,position=state['step'],state['epoch'],state['position']
        random.setstate(state['random']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        print('RESUMED',step,epoch,position,flush=True)
    stopping=[False]
    for name in ('SIGTERM','SIGUSR1','SIGINT'):
        signal.signal(getattr(signal,name),lambda *_:stopping.__setitem__(0,True))
    started=time.monotonic()
    def save():
        dest=output/f'checkpoint-{step:08d}'
        if dest.exists():checkpoint_complete(dest,identity);return
        temp=output/f'.saving-{step:08d}-{os.getpid()}';temp.mkdir()
        model.save_pretrained(temp,safe_serialization=True)
        torch.save({'step':step,'epoch':epoch,'position':position,'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'random':random.getstate(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()},temp/'state.pt')
        (temp/'COMPLETE.json').write_text(json.dumps({'identity':identity,'files':{n:sha(temp/n) for n in
            ('adapter_config.json','adapter_model.safetensors','state.pt')}},sort_keys=True)+'\n')
        checkpoint_complete(temp,identity);os.replace(temp,dest)
        print('CHECKPOINT='+str(dest),flush=True)
    while step<a.steps:
        order=list(range(len(examples)));random.Random(42+epoch).shuffle(order)
        batch=order[position:position+a.accum]
        if not batch:epoch+=1;position=0;continue
        optimizer.zero_grad(set_to_none=True);value=0.0
        for index in batch:
            kind,row=examples[index]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                numerator,_=masked_loss(model,torch.tensor([row['input_ids']],device='cuda'),row['target_spans'])
                loss=numerator*scales[kind]/len(batch)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();value+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
        optimizer.step();scheduler.step();step+=1;position+=len(batch)
        print(json.dumps({'step':step,'total_steps':a.steps,'epoch':epoch,'position':position,
            'loss':value,'grad_norm':float(norm),'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}),flush=True)
        if diagnostic:save()
        if diagnostic and step>=a.diagnostic_stop:
            print('DIAGNOSTIC_STOP='+str(step),flush=True);return
        if step%a.save_every==0:save()
        if step<a.steps and (stopping[0] or time.monotonic()-started>=a.seconds):
            save();print('PAUSED_RESUBMIT_SAME_COMMAND',flush=True);return
    save();(output/'TRAINING_COMPLETE').write_text(str(step)+'\n');print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
