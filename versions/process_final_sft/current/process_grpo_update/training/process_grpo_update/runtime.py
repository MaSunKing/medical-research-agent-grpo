"""Allocated-GPU training path. No generation, retrieval, Final, or network."""
import contextlib
import importlib.metadata
import os
from pathlib import Path
import time
from contract import digest,read,require,save,sha,verify_files


def environment(torch):
    return dict(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
        cuda=torch.version.cuda,packages={n:importlib.metadata.version(n)
            for n in ('torch','transformers','peft','bitsandbytes','safetensors')},
        tf32=False,mode='NF4_double_BF16_body_FP32_LoRA_FP32_unmerged_head_eval_checkpoint_deterministic_v2',
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)


def deterministic_setup(torch):
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8',
            'deterministic_CUBLAS_workspace_required_before_CUDA_initialization')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def load(data,base,adapter):
    import torch
    deterministic_setup(torch)
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    require(os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and
            torch.cuda.is_bf16_supported(),'allocated_BF16_CUDA_required')
    versions={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','bitsandbytes')}
    require(versions==dict(torch='2.6.0+cu124',transformers='4.52.4',peft='0.15.2',bitsandbytes='0.46.0'),
            'unvalidated_server_dependency_versions')
    verify_files(base,read(Path(data)/'MODEL_IDENTITY.json')['files'])
    verify_files(adapter,read(Path(data)/'EXPECTED_ORIGINAL972.json')['adapter_files'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    raw=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=PeftModel.from_pretrained(raw,adapter,is_trainable=True,local_files_only=True)
    model.requires_grad_(False)
    trainable={n:p for n,p in model.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    require(len(trainable)==504,'original972_adapter_shape_changed')
    for p in trainable.values():
        p.data=p.data.float();p.requires_grad_(True)
    core=model.get_base_model()
    require(type(core.lm_head) is torch.nn.Linear and not core.lm_head.weight.requires_grad,
            'frozen_unmerged_linear_head_required')
    # Same FP32 linear computation as collection. Cast frozen head once, avoiding
    # one retained FP32 vocabulary weight allocation per output chunk.
    core.lm_head.to(dtype=torch.float32)
    core.config.use_cache=False
    model.eval()
    install_checkpointing(model)
    require(all(not m.training for m in model.modules()),'dropout_must_be_disabled')
    return model,tok,trainable,environment(torch)


def install_checkpointing(model):
    """Checkpoint decoder layers without enabling dropout or HF train-mode branches."""
    import torch
    from torch.utils.checkpoint import checkpoint
    for layer in model.get_base_model().model.layers:
        original=layer.forward
        def wrapped(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            return _original(*args,**kwargs)
        layer.forward=wrapped


def logprobs(model,call,gradient=False,chunk=8):
    import torch
    from torch.utils.checkpoint import checkpoint
    q,r=call['request'],call['response']; prompt=q['prompt_token_ids']; completion=r['token_ids']
    core=model.get_base_model();device=next(model.parameters()).device
    ids=torch.tensor([prompt+completion],dtype=torch.long,device=device)
    scope=torch.enable_grad() if gradient else torch.no_grad()
    with scope,torch.autocast(device_type='cuda',enabled=False):
        hidden=core.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
        values=[]
        def selected(h,t):
            logits=core.lm_head(h.float()).float()
            return logits.log_softmax(-1).gather(-1,t[...,None]).squeeze(-1)
        for begin in range(len(prompt),ids.shape[1],chunk):
            end=min(begin+chunk,ids.shape[1]);h=hidden[:,begin-1:end-1];t=ids[:,begin:end]
            v=checkpoint(selected,h,t,use_reentrant=False) if gradient else selected(h,t)
            values.append(v[0])
        return torch.cat(values)


def call_path(data,rollout):
    from contract import safe
    return safe(data,rollout['calls_file'])


def reference_key(r,c):
    return r['calls_file'].removesuffix('.json')+'/'+c['call_id']+'.json'


def prepare_reference(model,tok,data,groups,output,identity,c,stop=None):
    """Freeze this GPU path's initial-policy probabilities BEFORE any update.

    Original behavior and original full-forward values remain unchanged. New old
    values are explicitly a separate training-path reference, never relabelled
    as behavior. Crash-safe, one original call per file.
    """
    import math
    from contract import safe
    output=Path(output)
    if (output/'IDENTITY.json').exists():
        require(read(output/'IDENTITY.json')==identity,'reference_identity_changed')
    else:save(output/'IDENTITY.json',identity)
    if (output/'COMPLETE.json').exists():
        complete=read(output/'COMPLETE.json')
        require(complete['identity']==identity,'reference_completion_identity_changed')
        verify_files(output,complete['files'])
    total=0;capped=0;sw=sw2=0.;files={};counts=0
    for g in groups:
        for r in g['rollouts']:
            for call in read(call_path(data,r)):
                if stop and stop():return False
                req=call['request']
                ids=tok.apply_chat_template(req['messages'],tokenize=True,return_dict=False,
                        add_generation_prompt=True,enable_thinking=False)
                require(ids==req['prompt_token_ids'],'actual_template_token_mismatch')
                name=reference_key(r,call);dest=safe(output,name)
                if dest.exists():x=read(dest)
                else:
                    old=logprobs(model,call,chunk=c['logit_chunk_tokens']).tolist()
                    x=dict(call_digest=digest(call),logprobs=old,reference_identity=digest(identity))
                    save(dest,x)
                require(x['call_digest']==digest(call) and x['reference_identity']==digest(identity),
                        'reference_call_changed')
                old=x['logprobs'];beh=call['response']['behavior_logprobs']
                require(len(old)==len(beh),'reference_length_changed')
                for a,b in zip(old,beh):
                    require(math.isfinite(a) and a<=1e-6 and abs(a-b)<50,'invalid_initial_reference')
                    w=math.exp(a-b);wc=min(w,c['is_cap']);total+=1;capped+=w>c['is_cap'];sw+=wc;sw2+=wc*wc
                files[name]=sha(dest);counts+=1
        print('REFERENCE_GROUP='+g['question_id']+'; calls='+str(counts),flush=True)
    stats=dict(tokens=total,calls=counts,capped_fraction=capped/max(1,total),
               weight_ess_fraction=sw*sw/max(1e-30,total*sw2))
    save(output/'WEIGHT_DIAGNOSTIC.json',stats)
    require(stats['capped_fraction']<=c['max_capped_fraction'] and
            stats['weight_ess_fraction']>=c['minimum_weight_ess_fraction'],'importance_weight_quality_hold')
    save(output/'COMPLETE.json',dict(identity=identity,files=files,stats=stats))
    return True


def reference_values(reference,r,call):
    from contract import safe
    x=read(safe(reference,reference_key(r,call)))
    require(x['call_digest']==digest(call),'reference_call_changed')
    return x['logprobs']


def update_batch(model,trainable,optimizer,data,reference,batch,c,forward_check=False,stop=None):
    import torch
    from objective import backward_call
    optimizer.zero_grad(set_to_none=True);loss=0.;calls=0;tokens=0;clipped=0;stages=set()
    # A missing measurement is not a measured zero. Alignment calls include
    # zero-advantage rollouts when the optional probe actually runs.
    maxdiff=None;alignment_calls=0
    base_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    for g in batch:
        for r in g['rollouts']:
            for call in read(call_path(data,r)):
                if stop and stop():
                    optimizer.zero_grad(set_to_none=True);return None
                old=reference_values(reference,r,call)
                if forward_check:
                    current=logprobs(model,call,gradient=True,chunk=c['logit_chunk_tokens'])
                    diff=float((current.detach()-torch.tensor(old,device=current.device)).abs().max())
                    maxdiff=diff if maxdiff is None else max(maxdiff,diff)
                    alignment_calls+=1
                    require(diff<=c['forward_tolerance'],'initial_grad_forward_alignment_hold')
                    del current
                if r['A']==0:continue
                # Equal question/rollout weights. Every sampled token, including
                # rejected and pre-repair outputs, contributes to this denominator.
                denom=len(batch)*4*r['policy_tokens']
                item=backward_call(model,call,old,r['A'],denom,c,
                    lambda m,x,gradient:logprobs(m,x,gradient=gradient,chunk=c['logit_chunk_tokens']))
                loss+=item['loss'];tokens+=item['tokens'];calls+=1;stages.add(call['request']['stage']);clipped+=item['clipped']
    require(all(base_versions[n]==p._version for n,p in model.named_parameters() if n in base_versions),
            'frozen_base_changed')
    alignment=dict(initial_alignment_checked=alignment_calls>0,
        initial_alignment_calls=alignment_calls,initial_alignment_max_abs=maxdiff)
    if not calls:
        return dict(optimizer_step=False,loss=0.,calls=0,tokens=0,zero_advantage_group=True,**alignment)
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable.values()),
            'missing_or_nonfinite_adapter_gradient')
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'base_gradient_detected')
    norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),c['max_grad_norm'],error_if_nonfinite=True)
    require(float(norm)>0,'zero_gradient_hold')
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in trainable.values()),'nonfinite_updated_adapter')
    optimizer.zero_grad(set_to_none=True)
    return dict(optimizer_step=True,loss=loss,calls=calls,tokens=tokens,clipped_fraction=clipped/max(1,tokens),
        stages=sorted(stages),gradient_norm=float(norm),**alignment,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}


def state_digest(state):
    import torch,hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def optimizer_snapshot(optimizer):
    return {str(i):{k:(v.detach().cpu().clone() if hasattr(v,'detach') else v)
                    for k,v in s.items()} for i,s in optimizer.state_dict()['state'].items()}


def optimizer_equal(a,b):
    import torch
    return a.keys()==b.keys() and all(a[i].keys()==b[i].keys() and all(
        torch.equal(a[i][k],b[i][k]) if torch.is_tensor(a[i][k]) else a[i][k]==b[i][k]
        for k in a[i]) for i in a)


def resume_comparison(expected,observed,expected_metrics,observed_metrics,expected_optimizer,observed_optimizer):
    """Diagnostic values only; retain the original gates without rounding."""
    import torch
    require(expected.keys()==observed.keys(),'resume_adapter_keys_changed')
    errors={k:float((expected[k]-observed[k]).abs().max()) for k in expected}
    maximum=max(errors.values(),default=0.)
    loss_delta=abs(expected_metrics['loss']-observed_metrics['loss'])
    return dict(adapter_max_abs=maximum,loss_abs_difference=loss_delta,
        parameter_tolerance=1e-6,loss_tolerance=1e-6,
        parameter_passed=maximum<=1e-6,loss_passed=loss_delta<=1e-6,
        optimizer_exact_match=optimizer_equal(expected_optimizer,observed_optimizer),
        different_adapter_tensors=sum(x>0 for x in errors.values()),
        adapter_tensors_above_tolerance=sum(x>1e-6 for x in errors.values()),
        largest_adapter_differences=sorted(errors.items(),key=lambda x:x[1],reverse=True)[:10],
        expected_metrics=expected_metrics,observed_metrics=observed_metrics,
        formal_training=False,thresholds_changed=False)


def save_checkpoint(model,optimizer,output,step,identity,metrics):
    import torch,uuid
    output=Path(output);dest=output/f'checkpoint-{step:04d}'
    require(not dest.exists(),'checkpoint_already_exists')
    tmp=output/('.partial-'+uuid.uuid4().hex);tmp.mkdir(parents=True)
    model.save_pretrained(tmp,safe_serialization=True)
    torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all()),tmp/'optimizer.pt')
    save(tmp/'STATE.json',dict(step=step,identity=identity,metrics=metrics,
        adapter_tensor_digest=state_digest(adapter_state(model)),created_at=time.time()))
    save(tmp/'COMPLETE.json',{p.name:sha(p) for p in sorted(tmp.iterdir()) if p.is_file()})
    tmp.rename(dest)
    return dest


def restore(model,optimizer,checkpoint,identity):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    verify_files(checkpoint,read(Path(checkpoint)/'COMPLETE.json'))
    s=read(Path(checkpoint)/'STATE.json');require(s['identity']==identity,'checkpoint_identity_changed')
    set_peft_model_state_dict(model,load_file(str(Path(checkpoint)/'adapter_model.safetensors')))
    require(state_digest(adapter_state(model))==s['adapter_tensor_digest'],'adapter_reload_mismatch')
    x=torch.load(Path(checkpoint)/'optimizer.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(x['optimizer']);torch.set_rng_state(x['torch_rng']);torch.cuda.set_rng_state_all(x['cuda_rng'])
    return s['step']
