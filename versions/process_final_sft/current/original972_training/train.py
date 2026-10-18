"""Process-only QLoRA. Explicit GPU tests before formal training; no tools/APIs."""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
from contract import batch_order, lr_scale, preflight, read, save, sha
from training_core import masked_loss, checkpoint_complete


def digest_tensors(values):
    import hashlib
    h = hashlib.sha256()
    for name, value in values:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--diagnostic-stop', type=int, choices=(1, 2))
    ap.add_argument('--acceptance', type=Path)
    ap.add_argument('--acknowledge-research-data-not-full-medical-review', action='store_true')
    ap.add_argument('--seconds', type=int, default=18000)
    a = ap.parse_args()
    began = time.monotonic()
    stopping = [False]
    for name in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
        signal.signal(getattr(signal, name), lambda *_: stopping.__setitem__(0, True))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    rows, c, identity = preflight(a.package, a.base)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    diagnostic = a.diagnostic_stop is not None
    assert not (a.smoke and diagnostic)
    if not a.smoke and not diagnostic:
        assert a.acknowledge_research_data_not_full_medical_review
        assert a.acceptance, 'New-data GPU/resume acceptance required'
        accepted = read(a.acceptance)
        assert accepted['status'] == 'passed' and accepted['identity'] == identity
        assert accepted['medical_semantic_certification'] is False
        for receipt in accepted['receipts']:
            assert sha(Path(receipt['path'])) == receipt['sha256']
    import torch
    import fcntl
    import importlib.metadata
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda,
        packages={n: importlib.metadata.version(n) for n in ('transformers', 'peft', 'bitsandbytes', 'safetensors')},
        gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
    if not a.smoke and not diagnostic:
        assert accepted['environment'] == environment, 'GPU/library environment differs from acceptance'
    output = a.output.resolve()
    assert not output.is_relative_to(a.base.resolve()) and not output.is_relative_to(a.package.resolve())
    assert output not in (Path('/'), Path.home())
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing_env = output / 'ENVIRONMENT.json'
    if existing_env.exists():
        assert read(existing_env) == environment, 'Runtime changed; rerun acceptance in new output'
    else:
        save(existing_env, environment)
    checkpoints = sorted(output.glob('checkpoint-' + '[0-9]' * 8))
    for cp in checkpoints:
        checkpoint_complete(cp, identity)
    latest = checkpoints[-1] if checkpoints else None
    assert not a.smoke or latest is None
    if (output / 'TRAINING_COMPLETE').exists():
        assert latest and int(latest.name.split('-')[-1]) == c['steps']
        assert read(output / 'TRAINING_COMPLETE') == dict(step=c['steps'], samples=c['rows'], identity=identity)
        final_state = torch.load(latest / 'state.pt', map_location='cpu', weights_only=False)
        assert final_state['step'] == c['steps'] and final_state['epoch'] == c['epochs'] and final_state['position'] == 0
        assert final_state['diagnostic'] is False
        print('ALREADY_COMPLETE', flush=True)
        return
    random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    base = AutoModelForCausalLM.from_pretrained(a.base, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    for p in base.parameters():
        p.requires_grad_(False)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    model = (PeftModel.from_pretrained(base, latest, is_trainable=True, local_files_only=True) if latest else
        get_peft_model(base, LoraConfig(r=c['lora_r'], lora_alpha=c['lora_alpha'], lora_dropout=c['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM')))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and p.dtype == torch.float32 for n, p in trainable)
    assert all(not p.is_floating_point() or p.dtype == torch.bfloat16 for n, p in model.named_parameters() if 'lora_' not in n)
    model.train()
    if a.smoke:
        row = max(rows, key=lambda r: len(r['input_ids']))
        spans = [[row['target_start'], len(row['input_ids']), 1.0]]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            numerator, mass = masked_loss(model, torch.tensor([row['input_ids']], device='cuda'), spans)
            loss = numerator / mass
        loss.backward()
        grads = [p.grad for _, p in trainable if p.grad is not None]
        assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
        save(output / 'GPU_SMOKE.json', dict(status='passed', identity=identity, environment=environment,
            longest_tokens=len(row['input_ids']), loss=float(loss.detach()), optimizer_updates=0,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        print('LONGEST_FORWARD_BACKWARD=passed', flush=True)
        return
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=c['lr'], betas=(.9, .999), eps=1e-8, weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: lr_scale(u, c['steps'], c['warmup_steps']))
    step = epoch = position = 0
    initial = digest_tensors(trainable) if latest is None else None
    if latest:
        state = torch.load(latest / 'state.pt', map_location='cpu', weights_only=False)
        assert state['diagnostic'] == diagnostic
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step, epoch, position = state['step'], state['epoch'], state['position']
        assert 0 <= epoch <= c['epochs'] and 0 <= position < c['rows']
        assert step == (c['steps'] if epoch == c['epochs'] else position // c['accum'])
        initial = state['initial_lora_sha256']
        random.setstate(state['random'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        print(f'RESUMED step={step} epoch={epoch} position={position}', flush=True)

    def checkpoint():
        dest = output / f'checkpoint-{step:08d}'
        if dest.exists():
            checkpoint_complete(dest, identity)
            return
        tmp = output / f'.saving-{step:08d}-{time.time_ns()}'
        tmp.mkdir()
        model.save_pretrained(tmp, safe_serialization=True)
        torch.save(dict(step=step, epoch=epoch, position=position, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), random=random.getstate(), torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(), diagnostic=diagnostic, initial_lora_sha256=initial), tmp / 'state.pt')
        save(tmp / 'COMPLETE.json', dict(identity=identity, files={n: sha(tmp / n) for n in
            ('adapter_config.json', 'adapter_model.safetensors', 'state.pt')}))
        checkpoint_complete(tmp, identity)
        os.replace(tmp, dest)
        print('CHECKPOINT=' + str(dest), flush=True)

    longest = max(range(len(rows)), key=lambda i: len(rows[i]['input_ids'])) if diagnostic else None
    while epoch < c['epochs']:
        order = batch_order(len(rows), c['seed'], epoch, longest)
        batch = order[position:position+c['accum']]
        assert batch
        mass = sum(len(rows[i]['input_ids']) - rows[i]['target_start'] for i in batch)
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        per_stage = {}
        for index in batch:
            row = rows[index]
            spans = [[row['target_start'], len(row['input_ids']), 1.0]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                numerator, token_count = masked_loss(model, torch.tensor([row['input_ids']], device='cuda'), spans)
                loss = numerator / mass
            assert torch.isfinite(loss)
            loss.backward()
            value += float(loss.detach())
            stage = per_stage.setdefault(row['stage'], dict(numerator=0., tokens=0))
            stage['numerator'] += float(numerator.detach())
            stage['tokens'] += token_count
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], c['max_grad_norm'])
        assert torch.isfinite(norm)
        update_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        step += 1
        position += len(batch)
        if position == len(rows):
            epoch += 1
            position = 0
        info = dict(step=step, total_steps=c['steps'], epoch=epoch, position=position,
            samples_seen=c['rows'] if epoch else position, loss=value, supervised_tokens=mass,
            stage_token_mean={s: v['numerator']/v['tokens'] for s, v in per_stage.items()},
            grad_norm=float(norm), update_lr=update_lr, peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        print(json.dumps(info), flush=True)
        with (output / 'loss.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(info) + '\n')
        if step % c['save_every'] == 0 or diagnostic or epoch == c['epochs']:
            checkpoint()
        if diagnostic and step >= a.diagnostic_stop:
            assert digest_tensors(trainable) != initial, 'LoRA did not change'
            print('DIAGNOSTIC_UPDATE=passed; not formal training', flush=True)
            return
        if step < c['steps'] and (stopping[0] or time.monotonic() - began >= a.seconds):
            checkpoint()
            print('PAUSED_RESUBMIT_SAME_COMMAND', flush=True)
            return
    assert step == c['steps']
    checkpoint()
    save(output / 'TRAINING_COMPLETE', dict(step=step, samples=c['rows'], identity=identity))
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
