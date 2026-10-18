"""Integrity and immutable run identities for the private Pre-Final collection."""
import contextlib
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    os.replace(tmp, p)

def verify_package(p=PACKAGE):
    p = Path(p).resolve(); manifest = read(p/'FILES.sha256.json')
    for name, expected in manifest.items():
        target = p/name
        if not target.resolve().is_relative_to(p) or target.is_symlink() or sha(target) != expected:
            raise ValueError('package_hash_mismatch:'+name)
    questions = read(p/'QUESTIONS.json'); split = read(p/'SPLIT.json')
    assert len(questions) == len({q['question_id'] for q in questions}) == 600
    assert {q['split'] for q in questions} == {'train', 'dev'}
    assert sum(q['split']=='train' for q in questions)==500
    assert sum(q['split']=='dev' for q in questions)==100
    assert set(split['train']) == {q['question_id'] for q in questions if q['split']=='train'}
    assert set(split['dev']) == {q['question_id'] for q in questions if q['split']=='dev'}
    return manifest

def load_work(work, verify=True):
    work = Path(work).resolve(); c = read(work/'CONFIG.json')
    p = Path(c['package']).resolve()
    if verify: verify_package(p)
    identity = read(work/'IDENTITY.json')
    assert identity['package_sha256'] == sha(p/'FILES.sha256.json')
    assert identity['config_sha256'] == sha(work/'CONFIG.json')
    return work, c, p

@contextlib.contextmanager
def locked(work, name='COLLECT.lock'):
    # Nonblocking allocation-wide lock; never delete another process's lock file.
    import fcntl
    with (Path(work)/name).open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def result_path(work, q):
    return Path(work)/'results'/q['question_id']/'attempt-001'

def verify_done(folder):
    receipt = Path(folder)/'DONE.json'
    if not receipt.is_file(): return False
    for n,h in read(receipt)['files'].items():
        target=Path(folder)/n
        if target.is_symlink() or not target.resolve().is_relative_to(Path(folder).resolve()) or sha(target)!=h:
            raise ValueError('completed_result_changed')
    return True

def verify_adapter(package, checkpoint):
    expected=read(Path(package)/'EXPECTED_ORIGINAL972.json'); cp=Path(checkpoint)
    assert cp.name=='checkpoint-00001322', 'original972_checkpoint1322_required'
    assert sha(cp/'COMPLETE.json')==expected['adapter_complete_sha256']
    assert sha(cp.parent/'TRAINING_COMPLETE')==expected['training_complete_sha256']
    t=read(cp.parent/'TRAINING_COMPLETE')
    assert t['step']==1322 and t['samples']==10575
    for n,h in expected['adapter_files'].items(): assert sha(cp/n)==h, n

def verify_base(package, base):
    # Full shards are checked on the allocated node, not just config/tokenizer.
    for n,v in read(Path(package)/'MODEL_IDENTITY.json')['files'].items():
        path=Path(base)/n
        assert path.is_file() and not path.is_symlink() and path.stat().st_size==v['size'], n
        assert sha(path)==v['sha256'], 'base_hash_mismatch:'+n
