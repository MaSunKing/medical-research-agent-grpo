"""Portable offline evaluation identities. No model/network work on import."""
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

ARMS = ('base','sft','grpo13','grpo25')

def require(ok, code):
    if not ok:
        raise ValueError(code)

def unique(pairs):
    result = {}
    for k,v in pairs:
        require(k not in result, 'duplicate_JSON_key')
        result[k] = v
    return result

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique)

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def save(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)
        return
    fd,tmp=tempfile.mkstemp(prefix='.writing-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8',newline='\n') as f:
            f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def safe(root,name):
    root=Path(root).resolve();r=PurePosixPath(str(name))
    require(not r.is_absolute() and '..' not in r.parts and ':' not in str(name)
            and '\\' not in str(name), 'unsafe_relative_path')
    p=root.joinpath(*r.parts)
    require(p != root and p.resolve().is_relative_to(root),'path_outside_root')
    require(not any(x.is_symlink() for x in [p,*p.parents] if x.is_relative_to(root)), 'symlink_not_allowed')
    return p

def verify_files(root,files):
    for name,value in files.items():
        p=safe(root,name);h=value if isinstance(value,str) else value['sha256']
        require(p.is_file() and sha(p)==h,'file_identity_changed:'+name)

def manifest(root,excludes=('FILES.sha256.json',)):
    root=Path(root)
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).as_posix() not in excludes}

def verify_package(package):
    package=Path(package)
    files=read(package/'FILES.sha256.json');verify_files(package,files)
    require(manifest(package)==files,'unexpected_or_changed_package_file')
    return sha(package/'FILES.sha256.json')

def verify_done(folder,expected_identity):
    folder=Path(folder)
    if not (folder/'DONE.json').is_file():return False
    done=read(folder/'DONE.json')
    require(done['identity']==expected_identity,'result_identity_changed')
    require('RESULT.json' in done['files'],'RESULT_not_sealed')
    verify_files(folder,done['files'])
    actual=manifest(folder,excludes=('DONE.json',))
    require(actual==done['files'],'unsealed_result_file')
    require(read(folder/'RESULT.json')['run_identity']==expected_identity,'RESULT_run_identity_changed')
    return True

def seal(folder,identity):
    folder=Path(folder)
    require(not any(p.stat().st_size for p in folder.rglob('*')
        if p.is_file() and p.name.endswith(('-wal','-shm'))), 'SQLite_not_closed_before_seal')
    save(folder/'DONE.json',dict(identity=identity,files=manifest(folder,excludes=('DONE.json',))),exclusive=True)
    verify_done(folder,identity)

@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        f.close()

def record_folder(work,arm,index):
    require(arm in ARMS and type(index) is int and 1<=index<=100,'invalid_record_coordinates')
    return Path(work)/'results'/arm/f'Q{index:02d}'
