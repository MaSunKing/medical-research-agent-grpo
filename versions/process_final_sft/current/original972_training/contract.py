"""Offline immutable-data and exact-model gates. Never loads model weights."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_child(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = root / relative
    assert not path.is_symlink() and path.resolve().is_relative_to(root)
    return path


def verify_files(root, manifest):
    for name, meta in manifest.items():
        p = safe_child(root, name)
        assert p.is_file() and p.stat().st_size == meta['size'], name
        assert sha(p) == meta['sha256'], name


def lr_scale(update, total, warmup):
    if update < warmup:
        return (update + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def batch_order(count, seed, epoch, longest=None):
    import random
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    if longest is not None:
        order.remove(longest)
        order.insert(0, longest)
    return order


def preflight(package, base):
    package, base = Path(package).resolve(), Path(base).resolve()
    verify_files(package, read(package / 'FILES.sha256.json'))
    identity_file = read(package / 'MODEL_IDENTITY.json')
    verify_files(base, identity_file['files'])
    config = read(package / 'TRAIN_CONFIG.json')
    assert config['initial_adapter'] is None
    assert config['rows'] == 10575 and config['questions'] == 972
    assert config['epochs'] == 1 and config['steps'] == math.ceil(config['rows'] / config['accum'])
    assert config['stage_weights'] == dict(checklist=1., decision=1., state=1.)
    assert config['targets'] == 'accepted_Process_only'
    assert read(package / 'data/BUILD_REPORT.json')['final_targets'] == 0
    questions = read(package / 'data/QUESTIONS.json')
    qids = {q['question_id'] for q in questions}
    assert len(qids) == len(questions) == 972 and all(q['split'] == 'train' for q in questions)
    for q in questions:
        assert sha(safe_child(package, q['prefinal_file'])) == q['prefinal_sha256']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows = []
    seen = set()
    with (package / 'data/train.text.jsonl').open(encoding='utf8') as raw_file, (package / 'data/train.encoded.jsonl').open(encoding='utf8') as enc_file:
        for raw_line, enc_line in itertools.zip_longest(raw_file, enc_file):
            assert raw_line is not None and enc_line is not None, 'row count mismatch'
            raw, enc = json.loads(raw_line), json.loads(enc_line)
            assert raw['row_id'] == enc['row_id'] and raw['question_id'] == enc['question_id']
            assert raw['row_id'] not in seen and raw['question_id'] in qids
            seen.add(raw['row_id'])
            assert raw['stage'] == enc['stage'] and raw['stage'] in config['stage_weights']
            assert raw['template_options'] == dict(add_generation_prompt=True, enable_thinking=False)
            prefix_text = tok.apply_chat_template(raw['messages'], tokenize=False, **raw['template_options'])
            prefix = tok.encode(prefix_text, add_special_tokens=False)
            target = tok.encode(raw['target'], add_special_tokens=False) + [tok.eos_token_id]
            assert enc['input_ids'] == prefix + target and enc['target_start'] == len(prefix)
            assert tok.encode(prefix_text + raw['target'] + tok.eos_token, add_special_tokens=False) == prefix + target
            assert 1 <= len(prefix) < len(enc['input_ids']) <= config['context_tokens']
            assert len(target) <= raw['max_new_tokens']
            assert len(prefix) + raw['max_new_tokens'] <= config['context_tokens']
            rows.append(enc)
    assert len(rows) == config['rows']
    identity = dict(role='Process_only_fresh_LoRA', config=config,
        package_manifest_sha256=sha(package / 'FILES.sha256.json'),
        model_identity_sha256=sha(package / 'MODEL_IDENTITY.json'),
        base_path=str(base), data_manifest_sha256=sha(package / 'DATA_MANIFEST.json'))
    print(json.dumps(dict(preflight='passed', questions=len(qids), rows=len(rows), steps=config['steps'],
        max_tokens=max(map(lambda r: len(r['input_ids']), rows)), model_weights_loaded=False,
        API=0, training=False)), flush=True)
    return rows, config, identity


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--base', type=Path, required=True)
    a = ap.parse_args()
    preflight(a.package, a.base)
