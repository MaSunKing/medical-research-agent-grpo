"""Final-only experiment contracts. No model, network, or server imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile


def require(value, code):
    if not value:
        raise ValueError(code)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8', newline='\n') as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def child(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in str(name), 'unsafe_relative_path')
    path = root / relative
    require(path.resolve().is_relative_to(root), 'path_outside_root')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent), 'symlink_not_allowed')
    require(path.is_file(), 'missing_source_file')
    return path


def referenced(root, entry):
    require(set(entry) == {'path', 'sha256'}, 'file_reference_requires_path_and_sha256')
    path = child(root, entry['path'])
    require(sha(path) == entry['sha256'], 'source_file_hash_mismatch')
    return path


def verify_files(root, files):
    for name, value in files.items():
        path = child(root, name)
        expected = value if isinstance(value, str) else value['sha256']
        require(sha(path) == expected, 'file_identity_changed:' + name)
        if isinstance(value, dict) and 'size' in value:
            require(path.stat().st_size == value['size'], 'file_size_changed:' + name)


def manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'FILES.sha256.json'}


def load_dataset(root):
    root = Path(root)
    files = read(root / 'FILES.sha256.json')
    verify_files(root, files)
    require(set(manifest(root)) == set(files), 'unexpected_dataset_files')
    meta = read(root / 'DATASET.json')
    require(meta['schema'] == 'final_sft_dataset_v1', 'wrong_dataset_schema')
    rows = [json.loads(line, object_pairs_hook=unique_object)
            for line in (root / 'records.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    require(len({r['record_id'] for r in rows}) == len(rows), 'duplicate_record_id')
    require(all(isinstance(r['record_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', r['record_id'])
                for r in rows), 'unsafe_record_id')
    require(digest(rows) == meta['records_sha256'], 'records_changed')
    return rows, meta, digest(files)


def flat_ids(value):
    require(isinstance(value, (list, tuple)) and value and
            all(type(i) is int and i >= 0 for i in value), 'flat_integer_token_ids_required')
    return list(value)


def encode_prompt(tokenizer, messages):
    return flat_ids(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False))


def visible_evidence(request):
    messages = request['messages']
    require(messages and all(m.get('role') in {'system', 'user'} and isinstance(m.get('content'), str)
                            for m in messages), 'frozen_final_messages_required')
    require(messages[-1]['role'] == 'user', 'final_user_context_required')
    context = json.loads(messages[-1]['content'], object_pairs_hook=unique_object)
    evidence = context['opened_evidence']
    require(isinstance(evidence, dict) and all(re.fullmatch(r'E[0-9]+', x) for x in evidence), 'invalid_visible_evidence')
    return sorted(evidence), context['question']


def validate_final(text, evidence_ids):
    require(isinstance(text, str) and re.fullmatch(r'\s*<answer>\s*\S[\s\S]*?</answer>\s*', text)
            and text.count('<answer>') == text.count('</answer>') == 1, 'expected_one_nonempty_answer_block')
    for group in re.findall(r'\[([^\]]+)\]', text):
        for eid in re.findall(r'\bE[0-9]+\b', group):
            require(eid in evidence_ids, 'unknown_citation_id:' + eid)
    return text


def check_request(request, tokenizer):
    ids = encode_prompt(tokenizer, request['messages'])
    require(ids == flat_ids(request['prompt_token_ids']), 'frozen_prompt_token_mismatch')
    require(request['input_sha256'] == digest(request['messages']), 'frozen_messages_hash_mismatch')
    require(request.get('exact_tokenizer') is True, 'exact_tokenizer_required')
    require(request.get('role', 'final') == request.get('stage', 'final') == 'final', 'Final_request_only')
    reserve = request['max_new_tokens']
    require(type(reserve) is int and 0 < reserve <= 2400, 'invalid_output_budget')
    require(len(ids) + reserve + 384 + 128 <= 15000, 'frozen_request_over_hard_budget')
    visible_evidence(request)
    return ids


def encode_target(tokenizer, request, answer):
    prefix = check_request(request, tokenizer)
    eids, _ = visible_evidence(request)
    validate_final(answer, eids)
    require(not any(marker in answer for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')), 'chat_control_tokens_not_allowed_in_target')
    require(tokenizer.eos_token_id is not None, 'EOS_required')
    target = flat_ids(list(tokenizer.encode(answer, add_special_tokens=False)) + [tokenizer.eos_token_id])
    require(len(target) <= request['max_new_tokens'], 'target_exceeds_output_reserve_no_truncation')
    return dict(input_ids=prefix + target, target_start=len(prefix), labels=[-100] * len(prefix) + target)


def update_plan(n, accum=8):
    """Two epochs; split batches at 1, 1.5, 2 epoch milestones, never lose samples."""
    require(type(n) is int and n >= 2 and type(accum) is int and accum > 0, 'invalid_training_size')
    plan = []
    for epoch in range(2):
        position = 0
        for boundary in ([n] if epoch == 0 else [(n + 1) // 2, n]):
            while position < boundary:
                stop = min(position + accum, boundary)
                label = None
                if stop == boundary:
                    label = 'epoch1' if epoch == 0 else 'epoch2' if boundary == n else 'epoch1_5'
                plan.append(dict(epoch=epoch, start=position, stop=stop, checkpoint=label))
                position = stop
    return plan


def batch_order(n, seed, epoch):
    result = list(range(n))
    random.Random(seed + epoch).shuffle(result)
    return result


def lr_factor(step, steps, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, steps - warmup))))


def repair_request(request, error, tokenizer):
    import copy
    detail = str(error)[:1500]
    initial = len(request['prompt_token_ids'])
    allowance = min(384, 15000 - initial - request['max_new_tokens'] - 128)
    while True:
        result = copy.deepcopy(request)
        result['messages'].append(dict(role='user', content='Previous output rejected; no action executed. Correct this error: ' + detail))
        ids = encode_prompt(tokenizer, result['messages'])
        if len(ids) - initial <= allowance:
            result.update(prompt_token_ids=ids, input_sha256=digest(result['messages']))
            return result
        require(len(detail) > 16, 'repair_context_overflow')
        detail = detail[:max(16, len(detail) // 2)]


def acquire_lock(path):
    """Keep returned handle alive; POSIX server and Windows local judge supported."""
    handle = Path(path).open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle

