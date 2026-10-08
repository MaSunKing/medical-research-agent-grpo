"""Exact-prefix encoding and sparse, weighted next-token supervision."""
import hashlib
import json
from pathlib import Path

CONTEXT = 10240
RESERVES = {'checklist_init': 1200, 'decision': 240, 'decision_stop': 240,
            'state_update': 1200, 'final': 2400}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def encode(tok, row):
    prompt = tok.apply_chat_template(row['messages'], tokenize=False,
        add_generation_prompt=True, **row['template_options']) + row['assistant_prefix']
    target = row['completion']
    if not target.endswith(tok.eos_token):
        target += tok.eos_token
    combined = prompt + target
    encoded = tok(combined, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    active = []
    for i, ((a, b), ident) in enumerate(zip(encoded['offset_mapping'], encoded['input_ids'])):
        if a < len(prompt) < b:
            raise ValueError('token crosses prompt/target boundary')
        if (a >= len(prompt) and b > a) or (i == len(encoded['input_ids'])-1 and ident == tok.eos_token_id):
            active.append(i)
    if not active or active[0] == 0 or active != list(range(active[0], len(encoded['input_ids']))):
        raise ValueError('invalid completion mask')
    if len(encoded['input_ids']) > CONTEXT:
        raise ValueError('context overflow; no truncation allowed')
    spans.append([active[0], len(encoded['input_ids']), 1.0])
    return {'input_ids': encoded['input_ids'], 'target_spans': spans}, prompt, combined

def extend(previous, previous_text, next_encoded, next_prompt):
    """Only admit continuous samples whose next visible prefix is EXACTLY preserved.

    Added observations remain context-only. No chat role rewriting, synthesized
    bridge, hidden previous-stage transcript or future evidence is permitted.
    """
    first = next_encoded['target_spans'][0][0]
    if not next_prompt.startswith(previous_text):
        return None
    ids = next_encoded['input_ids']
    if ids[:len(previous['input_ids'])] != previous['input_ids'] or len(previous['input_ids']) > first:
        return None
    return {'input_ids': ids,
            'target_spans': previous['target_spans'] + next_encoded['target_spans']}

def validate_record(row):
    if set(row) != {'input_ids', 'target_spans'}:
        raise ValueError('training record contains non-training fields')
    ids = row['input_ids']
    if not ids or len(ids) > CONTEXT or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('invalid input IDs')
    last = 0
    for start, end, weight in row['target_spans']:
        if not (type(start) is int and type(end) is int and 1 <= start < end <= len(ids)
                and start >= last and weight > 0):
            raise ValueError('invalid or overlapping target spans')
        last = end
    if not row['target_spans']:
        raise ValueError('no targets')

def token_mass(row):
    return sum((end-start)*weight for start, end, weight in row['target_spans'])

def masked_loss(model, tokens, spans):
    import torch
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    hidden = base.model(input_ids=tokens, use_cache=False).last_hidden_state
    total = hidden.sum() * 0
    mass = 0.0
    for start, end, weight in spans:
        mass += (end-start)*weight
        for begin in range(start, end, 64):
            stop = min(begin+64, end)
            def ce(x, y):
                logits = base.lm_head(x).float()
                return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1), reduction='sum')
            total = total + weight * checkpoint(ce, hidden[:, begin-1:stop-1],
                tokens[:, begin:stop], use_reentrant=False)
    return total, mass

def checkpoint_complete(path, identity):
    p = Path(path)
    if p.is_symlink() or not p.is_dir():
        raise ValueError('unsafe checkpoint')
    manifest = p/'COMPLETE.json'
    if manifest.is_symlink():
        raise ValueError('unsafe manifest')
    data = json.loads(manifest.read_text())
    required = {'adapter_config.json', 'adapter_model.safetensors', 'state.pt'}
    if data['identity'] != identity or set(data['files']) != required:
        raise ValueError('checkpoint identity/required files mismatch')
    for name, digest in data['files'].items():
        f = p/name
        if f.is_symlink() or not f.is_file() or f.stat().st_size == 0 or sha(f) != digest:
            raise ValueError('checkpoint integrity mismatch: '+name)
    return True
