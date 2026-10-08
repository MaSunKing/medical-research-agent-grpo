"""Fresh Final LoRA, token-average completion-only CE; resumable epoch checkpoints."""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from evaluate import read, save, sha
from training_core import masked_loss, checkpoint_complete


def data_and_identity(package, base):
    config = read(package / 'TRAIN_CONFIG.json')
    expected = dict(rows=226, epochs=3, accum=8, steps=87, lr=5e-5, warmup_steps=5,
        lora_r=32, lora_alpha=64, lora_dropout=0.05, weight_decay=0.01, max_grad_norm=1.0,
        context_tokens=10240, output_reserve=2400, seed=42, targets='Final_only',
        loss='valid_target_token_mean_per_effective_batch', scheduler='cosine_after_warmup',
        precision='NF4_double_quant_BF16_base_FP32_LoRA_no_prepare', attention='flash_deterministic')
    assert config == expected, 'Frozen Final training config changed'
    for name, digest in read(package / 'FILES.sha256.json').items():
        assert sha(package / name) == digest, name
    rows = [json.loads(line) for line in (package / 'data/train.final_sft.encoded.jsonl').read_text(encoding='utf-8').splitlines()]
    text = [json.loads(line) for line in (package / 'data/train.final_sft.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len(text) == 226
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    plan = read(package / 'PLAN.json')
    for name, digest in plan['tokenizer_hashes'].items():
        assert sha(base / name) == digest, name
    from final_citation_alias_v1 import validate_answer, digest
    train_manifest = read(package / 'data/INPUT_MANIFEST.json')['rows']
    lookup = {r['question_id']: r for r in train_manifest}
    dev_ids = {c['question_id'] for c in plan['cases']}
    packed = []
    for encoded, raw in zip(rows, text):
        qid = encoded['question_id']
        assert qid == raw['question_id'] and qid not in dev_ids
        original = read(package / 'data' / lookup[qid]['input_file'])
        assert raw['input_contract_sha256'] == original['input_contract_sha256']
        assert digest({k: v for k, v in original.items() if k not in {'input_contract_sha256', 'allowed_citations'}}) == original['input_contract_sha256']
        for key in ('messages', 'assistant_prefix', 'template_options'):
            assert raw[key] == original[key], (qid, key)
        validate_answer(raw['completion'], read(package / 'data' / lookup[qid]['citation_map_file']))
        prompt = tok.apply_chat_template(raw['messages'], tokenize=False, add_generation_prompt=True,
                                        **raw['template_options']) + raw['assistant_prefix']
        prefix = tok.encode(prompt, add_special_tokens=False)
        target = tok.encode(raw['completion'], add_special_tokens=False) + [tok.eos_token_id]
        assert encoded['input_ids'] == prefix + target
        # Independent whole-string tokenization catches prefix/target BPE boundary drift.
        assert tok.encode(prompt + raw['completion'] + tok.eos_token, add_special_tokens=False) == prefix + target
        assert encoded['labels'] == [-100] * len(prefix) + target
        assert encoded['attention_mask'] == [1] * len(encoded['input_ids'])
        assert len(prefix) + 2400 <= 10240 and len(target) <= 2400
        assert target[-1] == tok.eos_token_id and prefix
        packed.append({'input_ids': encoded['input_ids'], 'target_spans': [[len(prefix), len(prefix) + len(target), 1.0]]})
    assert len(lookup) == len(packed) == 226
    model_index = read(base / 'model.safetensors.index.json')
    for name in set(model_index['weight_map'].values()):
        assert Path(name).name == name and (base / name).is_file(), name
    identity = dict(role='Final_only_fresh_LoRA', config=config,
        data={n: sha(package / 'data' / n) for n in ('train.final_sft.jsonl', 'train.final_sft.encoded.jsonl', 'INPUT_MANIFEST.json')},
        code={n: sha(package / n) for n in ('train_final.py', 'training_core.py')},
        tokenizer=plan['tokenizer_hashes'], base_metadata={n: sha(base / n) for n in ('config.json', 'model.safetensors.index.json')},
        base_path=str(base.resolve()))
    return packed, config, identity


def lr_scale(update, total=87, warmup=5):
    if update < warmup:
        return (update + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def tensor_digest(values):
    h = hashlib.sha256()
    for name, v in values:
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--diagnostic-stop', type=int, choices=(1, 2))
    ap.add_argument('--acceptance', type=Path)
    ap.add_argument('--seconds', type=int, default=18000)
    a = ap.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    rows, config, identity = data_and_identity(a.package, a.base)
    print(json.dumps({'preflight': 'passed', 'rows': 226, 'epochs': 3, 'steps': 87,
        'max_tokens': max(len(r['input_ids']) for r in rows), 'targets': 'Final_only', 'weights_loaded': False}), flush=True)
    if a.preflight:
        return
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    if not a.smoke and a.diagnostic_stop is None:
        assert a.acceptance, 'Run new-data GPU acceptance first'
        accepted = read(a.acceptance)
        assert accepted['status'] == 'passed' and accepted['identity'] == identity
        for receipt in accepted['receipts']:
            assert sha(Path(receipt['path'])) == receipt['sha256']
        assert accepted['medical_semantic_certification'] is False
    import torch
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
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = a.output.resolve()
    assert output not in (a.base.resolve(), a.package.resolve(), Path('/'), Path.home())
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoints = sorted(output.glob('checkpoint-' + '[0-9]' * 8))
    for cp in checkpoints:
        checkpoint_complete(cp, identity)
    latest = checkpoints[-1] if checkpoints else None
    if a.smoke:
        assert latest is None
    if (output / 'TRAINING_COMPLETE').exists():
        assert latest
        state = torch.load(latest / 'state.pt', map_location='cpu', weights_only=False)
        assert state['step'] == 87 and state['epoch'] == 3 and state['position'] == 0
        print('ALREADY_COMPLETE', flush=True)
        return
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
        get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type='CAUSAL_LM')))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and p.dtype == torch.float32 for n, p in trainable)
    assert all(not p.is_floating_point() or p.dtype == torch.bfloat16 for n, p in model.named_parameters() if 'lora_' not in n)
    model.train()
    if a.smoke:
        row = max(rows, key=lambda r: len(r['input_ids']))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            numerator, mass = masked_loss(model, torch.tensor([row['input_ids']], device='cuda'), row['target_spans'])
            loss = numerator / mass
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for _, p in trainable if p.grad is not None)
        save(output / 'GPU_SMOKE.json', dict(status='passed', identity=identity,
            longest_tokens=len(row['input_ids']), loss=float(loss.detach()),
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30, optimizer_updates=0,
            gpu=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            slurm_node=os.environ.get('SLURMD_NODENAME')))
        print('LONGEST_FORWARD_BACKWARD=passed', flush=True)
        return
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    step = epoch = position = 0
    diagnostic = a.diagnostic_stop is not None
    initial = tensor_digest(trainable) if latest is None else None
    if latest:
        state = torch.load(latest / 'state.pt', map_location='cpu', weights_only=False)
        assert state['diagnostic'] == diagnostic
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step, epoch, position = state['step'], state['epoch'], state['position']
        initial = state['initial_lora_sha256']
        random.setstate(state['random'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        print(f'RESUMED step={step} epoch={epoch} position={position}', flush=True)
    stopping = [False]
    for name in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
        signal.signal(getattr(signal, name), lambda *_: stopping.__setitem__(0, True))
    began = time.monotonic()
    session_id = f'{os.getpid()}-{time.time_ns()}'
    resumed_from_step = step

    def checkpoint():
        dest = output / f'checkpoint-{step:08d}'
        if dest.exists():
            checkpoint_complete(dest, identity)
            return
        temp = output / f'.saving-{step:08d}-{os.getpid()}'
        temp.mkdir()
        model.save_pretrained(temp, safe_serialization=True)
        torch.save(dict(step=step, epoch=epoch, position=position, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), random=random.getstate(), torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(), diagnostic=diagnostic, initial_lora_sha256=initial), temp / 'state.pt')
        save(temp / 'COMPLETE.json', dict(identity=identity, files={n: sha(temp / n) for n in
            ('adapter_config.json', 'adapter_model.safetensors', 'state.pt')}))
        checkpoint_complete(temp, identity)
        os.replace(temp, dest)
        print('CHECKPOINT=' + str(dest), flush=True)

    while epoch < 3:
        order = list(range(226))
        random.Random(42 + epoch).shuffle(order)
        if diagnostic:
            longest = max(range(226), key=lambda i: len(rows[i]['input_ids']))
            order.remove(longest)
            order.insert(0, longest)
        batch = order[position:position + 8]
        assert batch
        mass = sum(rows[i]['target_spans'][0][1] - rows[i]['target_spans'][0][0] for i in batch)
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for index in batch:
            row = rows[index]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                numerator, _ = masked_loss(model, torch.tensor([row['input_ids']], device='cuda'), row['target_spans'])
                loss = numerator / mass
            assert torch.isfinite(loss)
            loss.backward()
            value += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        assert torch.isfinite(norm)
        update_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        step += 1
        position += len(batch)
        completed_epoch = position == 226
        if completed_epoch:
            epoch += 1
            position = 0
        info = dict(step=step, total_steps=87, completed_epochs=epoch, position=position,
            samples_this_update=len(batch), supervised_tokens=mass, loss=value, grad_norm=float(norm),
            update_lr=update_lr, peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            session_id=session_id, resumed_from_step=resumed_from_step)
        print(json.dumps(info), flush=True)
        with (output / 'loss.jsonl').open('a') as f:
            f.write(json.dumps(info) + '\n')
        if completed_epoch or diagnostic:
            checkpoint()
        if diagnostic and step >= a.diagnostic_stop:
            assert tensor_digest(trainable) != initial, 'LoRA weights did not change'
            print('DIAGNOSTIC_UPDATE=passed; not formal training', flush=True)
            return
        if step < 87 and (stopping[0] or time.monotonic() - began >= a.seconds):
            checkpoint()
            print('PAUSED_RESUBMIT_SAME_COMMAND', flush=True)
            return
    assert step == 87
    checkpoint()
    (output / 'TRAINING_COMPLETE').write_text('87\n')
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
