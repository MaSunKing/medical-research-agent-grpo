"""Refresh only the already allowlisted version-three source export hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'versions/process_final_sft/current'


def main():
    path=ROOT/'SOURCE_MANIFEST.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for name in ('README.md','check_public_source.py'):
        manifest['files'].setdefault(name,dict(source_sha256=None,changes='Public documentation/source-only verifier'))
    for name,row in manifest['files'].items():
        p=ROOT/name
        assert not p.is_symlink() and p.resolve().is_relative_to(ROOT)
        row['published_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest['files']['prepare.py']['changes']='Explicit --process-adapter replaces private hardcoded server path; original972 identity gate unchanged'
    manifest['files']['original972_training/run_phase.sh']['changes']='Require explicit PROCESS972_BASE and PROCESS972_PYTHON instead of personal server path defaults; trainer unchanged'
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.name!='SOURCE_MANIFEST.json'}
    assert actual==set(manifest['files']), 'unreviewed_or_missing_export_file'
    path.write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('CURRENT_SOURCE_MANIFEST=refrozen; files='+str(len(actual)))


if __name__=='__main__':main()
