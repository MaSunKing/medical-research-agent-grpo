"""Frozen-input Final evaluation. No retrieval, teacher, judge or training."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

STOP_REQUESTED = False
ACTIVE_WORKER = None


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    if ACTIVE_WORKER is not None and ACTIVE_WORKER.poll() is None:
        ACTIVE_WORKER.send_signal(signal.SIGUSR1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def preflight(package, base=None):
    from final_citation_alias_v1 import digest
    for name, expected in read(package / 'FILES.sha256.json').items():
        assert sha(package / name) == expected, name
    plan = read(package / 'PLAN.json')
    assert len(plan['cases']) == 38 and len(plan['checkpoints']) == 3
    assert plan['planned_answers'] == len(arms(plan)) * len(plan['cases'])
    assert len({c['question_id'] for c in plan['cases']}) == 38
    train_ids = set(read(package / 'TRAIN_QUESTION_IDS.json'))
    for case in plan['cases']:
        obj = read(package / case['input_file'])
        assert obj['question_id'] == case['question_id'] and obj['question_id'] not in train_ids
        assert digest({k: v for k, v in obj.items() if k not in {'input_contract_sha256', 'allowed_citations'}}) == obj['input_contract_sha256']
        ids = read(package / case['token_file'])
        assert len(ids) == obj['input_tokens'] and len(ids) + 2400 <= 10240
        ledger = read(package / case['citation_map_file'])
        assert ledger['mapping_sha256'] == digest(ledger['alias_to_source_id'])
        assert set(obj['allowed_citations']) == set(ledger['alias_to_source_id'])
    if base is not None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        for name, expected in plan['tokenizer_hashes'].items():
            assert sha(base / name) == expected, name
        for case in plan['cases']:
            obj = read(package / case['input_file'])
            prompt = tok.apply_chat_template(obj['messages'], tokenize=False, add_generation_prompt=True,
                                            **obj['template_options']) + obj['assistant_prefix']
            assert tok.encode(prompt, add_special_tokens=False) == read(package / case['token_file']), case['question_id']
    return plan


def adapter_identity(path):
    assert path.is_dir() and not path.is_symlink(), path
    identity = {name: sha(path / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}
    complete = read(path / 'COMPLETE.json')
    assert complete['identity']['role'] == 'Final_only_fresh_LoRA', 'Not a new Final-only adapter'
    assert complete['identity']['config']['steps'] == 87
    for name, expected in complete['files'].items():
        assert Path(name).name == name and sha(path / name) == expected, name
    cfg = read(path / 'adapter_config.json')
    assert cfg['r'] == 32 and cfg['lora_alpha'] == 64
    return identity


def arms(plan):
    return (['base'] if plan['base_final_included'] else []) + [c['name'] for c in plan['checkpoints']]


def arm_identities(plan, training, base):
    result = {c['name']: adapter_identity(training / c['directory']) for c in plan['checkpoints']}
    if plan['base_final_included']:
        result['base'] = {'role': 'Base_Final_no_adapter', 'base_metadata': {
            n: sha(base / n) for n in ('config.json', 'model.safetensors.index.json')}}
    return result


def finished(work, arm, case, identity):
    path = work / 'results' / arm / f"q{case['index']:03d}.json"
    if not path.exists():
        return False
    result = read(path)
    assert result['adapter_identity'] == identity and result['seed'] == case['seed']
    assert result['input_contract_sha256'] == case['input_contract_sha256']
    assert result['generation'] == read(work / 'EVALUATION_PLAN.json')['generation']
    capture = work / result['capture_file']
    assert sha(capture) == result['capture_sha256']
    return True


def worker(args, plan):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, StoppingCriteria
    from peft import PeftModel
    from final_citation_alias_v1 import validate_answer
    from final_repetition_guard import repetition_event
    cp = ({'name': 'base'} if args.worker == 'base' else
          next(c for c in plan['checkpoints'] if c['name'] == args.worker))
    adapter = args.training / cp['directory'] if args.worker != 'base' else None
    identity = arm_identities(plan, args.training, args.base)[args.worker]
    # Adapter files are immutable during this worker; check once more at exit.
    pending = [c for c in plan['cases'] if not finished(args.work, cp['name'], c, identity)]
    if not pending:
        return
    assert torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    for name, p in model.named_parameters():
        if 'lora_' in name:
            assert p.dtype == torch.float32, (name, str(p.dtype))
    start = time.monotonic()
    for case in pending:
        if STOP_REQUESTED or time.monotonic() - start > args.seconds - 300:
            print('PAUSED_EVALUATION; rerun same command', flush=True)
            return
        obj = read(args.package / case['input_file'])
        ids = read(args.package / case['token_file'])
        torch.manual_seed(case['seed'])
        torch.cuda.manual_seed_all(case['seed'])
        class End(StoppingCriteria):
            reason = None
            def __call__(self, sequence, scores, **kwargs):
                if STOP_REQUESTED:
                    self.reason = 'interrupted'
                    return True
                text = tok.decode(sequence[0, len(ids):], skip_special_tokens=True)
                if repetition_event(text):
                    self.reason = 'abnormal_repetition'
                    return True
                if '</answer>' in text:
                    self.reason = 'answer_complete'
                    return True
                return False
        end = End()
        gen = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0, max_new_tokens=2400, num_beams=1,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id, use_cache=True)
        tokens = torch.tensor([ids], device='cuda')
        began = time.monotonic()
        print(f"FINAL_STARTED {cp['name']} Q{case['index']:03d} seed={case['seed']}", flush=True)
        with torch.inference_mode():
            outputs = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                     generation_config=gen, stopping_criteria=[end])
        output_ids = outputs[0, len(ids):].tolist()
        raw = tok.decode(output_ids, skip_special_tokens=True)
        if STOP_REQUESTED:
            save(args.work / 'interrupted' / cp['name'] / f"q{case['index']:03d}-{time.time_ns()}.json",
                dict(question_id=case['question_id'], input_ids=ids, output_ids=output_ids,
                     raw_completion=raw, seed=case['seed'], adapter_identity=identity,
                     input_contract_sha256=obj['input_contract_sha256'], accepted=False,
                     stop_reason='interrupted; rerun this case with original seed'))
            print('INTERRUPTED_CAPTURE_PRESERVED; not counted as completed', flush=True)
            return
        try:
            protocol = validate_answer(raw, read(args.package / case['citation_map_file']))
        except ValueError as e:
            protocol = {'format': 'failed', 'error': str(e), 'semantic_support': 'NOT_CHECKED'}
        capture_name = f"captures/{cp['name']}/q{case['index']:03d}.json"
        capture = dict(question_id=case['question_id'], question=obj['question'], input_ids=ids,
            output_ids=output_ids, raw_completion=raw, messages=obj['messages'],
            assistant_prefix=obj['assistant_prefix'], template_options=obj['template_options'],
            generation=plan['generation'], seed=case['seed'], adapter_identity=identity,
            input_contract_sha256=obj['input_contract_sha256'])
        save(args.work / capture_name, capture)
        result = dict(index=case['index'], question_id=case['question_id'], checkpoint=cp['name'],
            seed=case['seed'], adapter_identity=identity, generation=plan['generation'],
            input_contract_sha256=obj['input_contract_sha256'], capture_file=capture_name,
            capture_sha256=sha(args.work / capture_name), protocol=protocol,
            answer=raw, output_tokens=len(output_ids), seconds=round(time.monotonic() - began, 2),
            stop_reason=end.reason or ('eos' if output_ids and output_ids[-1] == tok.eos_token_id else 'token_budget'),
            semantic_review='PENDING', gpu=torch.cuda.get_device_name(0))
        save(args.work / 'results' / cp['name'] / f"q{case['index']:03d}.json", result)
        completed = len(list((args.work / 'results').glob('*/q*.json')))
        save(args.work / 'STATUS.json', {'state': 'COLLECTING', 'completed': completed, 'total': plan['planned_answers'],
            'checkpoint': cp['name'], 'latest_index': case['index'], 'semantic_review': 'PENDING'})
        print(f"FINAL_FINISHED {cp['name']} Q{case['index']:03d} format={protocol['format']}", flush=True)
    assert arm_identities(plan, args.training, args.base)[args.worker] == identity


def main():
    global ACTIVE_WORKER
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--training', type=Path)
    ap.add_argument('--work', type=Path)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--tokenizer-check', action='store_true')
    ap.add_argument('--worker')
    ap.add_argument('--seconds', type=int, default=18600)
    args = ap.parse_args()
    for name in ('SIGUSR1', 'SIGTERM', 'SIGINT'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    plan = preflight(args.package, args.base if args.tokenizer_check or not args.preflight_only else None)
    if args.preflight_only:
        print(f"DEV38_PREFLIGHT=passed; questions=38; planned_answers={plan['planned_answers']}; weights/API/training=0")
        return
    assert args.training and args.work
    if args.worker:
        return worker(args, plan)
    import fcntl
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = args.work / 'EVALUATION_PLAN.json'
        if existing.exists():
            assert read(existing) == plan
        else:
            save(existing, plan)
        # Require all three immutable epoch checkpoints BEFORE any generation.
        identities = arm_identities(plan, args.training, args.base)
        base_identity = {n: sha(args.base / n) for n in ('config.json', 'model.safetensors.index.json')}
        identity_file = args.work / 'IDENTITY.json'
        identity = {'adapters': identities, 'base_metadata': base_identity,
                    'base_path': str(args.base.resolve()), 'training_path': str(args.training.resolve())}
        if identity_file.exists():
            assert read(identity_file) == identity
        else:
            save(identity_file, identity)
        started = time.monotonic()
        for arm in arms(plan):
            remaining = args.seconds - int(time.monotonic() - started)
            done = sum(finished(args.work, name, case, identities[name])
                       for name in arms(plan) for case in plan['cases'])
            save(args.work / 'STATUS.json', {'state': 'COLLECTING', 'completed': done, 'total': plan['planned_answers']})
            if STOP_REQUESTED or remaining < 600:
                save(args.work / 'STATUS.json', {'state': 'PAUSED', 'completed': done, 'total': plan['planned_answers']})
                return
            ACTIVE_WORKER = subprocess.Popen([sys.executable, '-B', '-u', __file__, '--package', str(args.package),
                '--base', str(args.base), '--training', str(args.training), '--work', str(args.work),
                '--worker', arm, '--seconds', str(remaining)])
            code = ACTIVE_WORKER.wait()
            ACTIVE_WORKER = None
            if code:
                raise RuntimeError(f'Evaluation worker {arm} failed exit={code}; preserve records and inspect logs')
        done = sum(finished(args.work, name, case, identities[name])
                   for name in arms(plan) for case in plan['cases'])
        save(args.work / 'STATUS.json', {'state': 'COMPLETED' if done == plan['planned_answers'] else 'PAUSED',
            'completed': done, 'total': plan['planned_answers'], 'semantic_review': 'PENDING', 'judge_calls': 0,
            'retrieval_calls': 0, 'training': False})
        if done == plan['planned_answers']:
            subprocess.run([sys.executable, '-B', str(args.package / 'export_results.py'),
                            '--package', str(args.package), '--work', str(args.work)], check=True)


if __name__ == '__main__':
    main()
