"""Private grouped-rollout contracts; no model, network or training on import."""
import hashlib
import json
import math
import re
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf8', newline='\n') as f:
            f.write(text)
    else:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(text, encoding='utf8', newline='\n')
        tmp.replace(path)


def require(ok, code):
    if not ok:
        raise ValueError(code)


def normalize(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def questions(path):
    p = Path(path)
    rows = ([json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
            if p.suffix == '.jsonl' else read(p))
    require(isinstance(rows, list) and rows, 'question_list_required')
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('question'), str)
                and row['question'].strip(), 'invalid_question')
        qid = row.get('question_id', '')
        require(isinstance(qid, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', qid),
                'unsafe_question_id')
    return rows


def validate_training_questions(rows, exclusions):
    require(1 <= len(rows) <= 100, 'between_one_and_100_questions_required')
    require(len({r['question_id'] for r in rows}) == len(rows), 'duplicate_question_id')
    require(len({normalize(r['question']) for r in rows}) == len(rows), 'duplicate_question_text')
    banned_ids = {r['question_id'] for r in exclusions}
    banned_texts = {normalize(r['question']) for r in exclusions}
    for r in rows:
        require(r.get('split') == 'train', 'only_training_questions_allowed')
        require(r['question_id'] not in banned_ids and normalize(r['question']) not in banned_texts,
                'question_overlaps_reserved_evaluation')
        provenance = r.get('provenance', {})
        require(isinstance(provenance, dict) and provenance.get('source_url'), 'source_provenance_required')
        # HealthBench is reserved for evaluation, even if a new example was not used before.
        require('healthbench' not in json.dumps(provenance, ensure_ascii=False).casefold()
                and not r['question_id'].casefold().startswith(('healthbench', 'hb_')),
                'healthbench_training_forbidden')


def seed_for(question_id, rollout):
    return int(digest(['process-grpo-v1', question_id, rollout])[:8], 16)


def verify_done(folder):
    folder = Path(folder)
    if not (folder / 'DONE.json').is_file():
        return False
    for name, expected in read(folder / 'DONE.json')['files'].items():
        p = folder / name
        require(not p.is_symlink() and p.resolve().is_relative_to(folder.resolve())
                and p.is_file() and sha(p) == expected, 'completed_record_changed')
    return True


def capture_admission(events):
    """All sampled Process attempts, including rejected/repaired JSON, must be present."""
    process = [e['data'] for e in events if e['kind'] == 'generation'
               and e['data']['request']['role'] == 'process']
    requests = [e for e in events if e['kind'] == 'model_request'
                and e['data']['request']['role'] == 'process']
    require(process and len(process) == len(requests), 'missing_process_generation_receipt')
    for g in process:
        r = g['response']
        ids, probs = r.get('token_ids'), r.get('behavior_logprobs')
        require(r.get('grpo_capture_ready') is True and ids and isinstance(probs, list)
                and len(ids) == len(probs), 'missing_exact_behavior_capture')
        require(all(type(i) is int and i >= 0 for i in ids)
                and all(type(p) in (int, float) and math.isfinite(p) and p <= 1e-6 for p in probs),
                'invalid_behavior_capture')
        require(r.get('policy_token_mask') == [1]*len(ids), 'invalid_process_policy_token_mask')
        require(r.get('replay_check', {}).get('passed') is True, 'rollout_replay_parity_not_passed')
        require(r.get('sampling', {}).get('temperature') == 1.0, 'process_temperature_must_be_one')
        require(r.get('sampling', {}).get('effective_parameters_verified') is True,
                'effective_sampling_parameters_not_verified')
    return dict(process_attempts=len(process),
                rejected_attempts=sum(not g['accepted'] for g in process),
                repaired_attempts=sum(bool(g.get('json_repair', {}).get('applied')) for g in process),
                process_tokens=sum(len(g['response']['token_ids']) for g in process),
                collection_capture_complete=True, trainer_precision_acceptance=False)
