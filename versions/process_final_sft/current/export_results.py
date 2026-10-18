"""Private transfer snapshot with consistent SQLite backup and credential-value guard."""
import argparse
from contextlib import closing
import json
import shlex
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from common import read,save,sha,load_work,locked,verify_done,result_path

def secret_values(path):
    values=[]
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line=line.strip().removeprefix('export ')
        if not line or line.startswith('#') or '=' not in line:continue
        key,value=line.split('=',1)
        if not any(x in key.upper() for x in ['KEY','TOKEN','SECRET','PASSWORD']):continue
        parsed=shlex.split(value,comments=True)
        if len(parsed)==1 and len(parsed[0])>=8:values.append(parsed[0].encode())
    return values

def export(work):
    work,c,p=load_work(work)
    with locked(work):
        export_root=work.parent/'exports';export_root.mkdir(exist_ok=True)
        target=Path(tempfile.mkdtemp(prefix='prefinal600-',dir=export_root))
        payload=target/'prefinal600_results';payload.mkdir()
        copied=[]
        for source in (work/'results').rglob('*') if (work/'results').exists() else []:
            if not source.is_file():continue
            assert not source.is_symlink() and source.resolve().is_relative_to(work/'results')
            if source.name.endswith(('-wal','-shm')):continue
            rel=source.relative_to(work);dest=payload/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if source.suffix=='.sqlite':
                with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as original:
                    with closing(sqlite3.connect(dest)) as backup:original.backup(backup)
                with closing(sqlite3.connect(dest)) as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            else:shutil.copyfile(source,dest)
            copied.append(dest)
        for name in ['IDENTITY.json','CONFIG.json','MODEL_HEALTH.json']:
            if (work/name).is_file():shutil.copyfile(work/name,payload/name);copied.append(payload/name)
        for name in ['QUESTIONS.json','SPLIT.json','PLAN.json','CODE_PROVENANCE.json','DEDUP_AUDIT.json','MODEL_IDENTITY.json','EXPECTED_ORIGINAL972.json','README.zh.md','GOLD_REVIEW_RULES.zh.md']:
            shutil.copyfile(p/name,payload/name);copied.append(payload/name)
        values=secret_values(c['secrets_file'])
        for file in copied:
            raw=file.read_bytes()
            if any(v in raw for v in values):raise ValueError('export_contains_credential_value:'+file.name)
            assert file.suffix not in {'.safetensors','.pt','.bin','.env'}
        manifest={f.relative_to(payload).as_posix():sha(f) for f in copied}
        save(payload/'FILES.sha256.json',manifest)
        archive=target/'prefinal600_results.tar.gz'
        with tarfile.open(archive,'x:gz') as stream:
            for f in sorted(payload.rglob('*')):
                if f.is_file():stream.add(f,arcname=f.relative_to(target).as_posix(),recursive=False)
        with tarfile.open(archive,'r:gz') as stream:
            for name,h in manifest.items():
                import hashlib
                assert hashlib.sha256(stream.extractfile('prefinal600_results/'+name).read()).hexdigest()==h
        Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
        print('EXPORT_HASHES_AND_SQLITE=passed; secrets not printed')
        print('ARCHIVE='+str(archive));print('SHA256_FILE='+str(archive)+'.sha256')
        return archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);export(p.parse_args().work)
