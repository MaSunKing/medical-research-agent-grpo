"""Allocated-CPU preparation, paths only; no LLM, search, Final, or training."""
import argparse
import subprocess
import sys
from pathlib import Path
from common import PACKAGE, read, save, sha, verify_package, verify_adapter, verify_base

def prepare(reference, work, process_adapter):
    verify_package(); assert not work.exists(), 'fresh_work_required'
    old=read(reference/'CONFIG.json')
    fields=['base_model','python','retrieval_python','secrets_file','retrieval_models','partition','qos']
    c={k:old[k] for k in fields}
    c.update(package=str(PACKAGE),port=30206,process_adapter=str(process_adapter))
    verify_adapter(PACKAGE,c['process_adapter']);verify_base(PACKAGE,c['base_model'])
    # Import dependency checks without importing secrets into this process.
    for py,names in [(c['python'],['torch','transformers','peft','bitsandbytes','langgraph','fastmcp','httpx']),
                     (c['retrieval_python'],['torch','transformers','trafilatura','rank_bm25','numpy','requests'])]:
        for name in names:
            subprocess.run([py,'-B','-c',f'import importlib;importlib.import_module({name!r})'],
                check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    work.mkdir(parents=True,exist_ok=False);(work/'logs').mkdir()
    save(work/'CONFIG.json',c);(work/'CONFIG.json').chmod(0o600)
    save(work/'IDENTITY.json',dict(package_sha256=sha(PACKAGE/'FILES.sha256.json'),
        config_sha256=sha(work/'CONFIG.json'),split_sha256=sha(PACKAGE/'SPLIT.json'),
        original972_verified=True,full_base_verified=True,training=False))
    from collect import make_config
    q=read(PACKAGE/'QUESTIONS.json')[0]
    check=work/'preflight.agent.json';save(check,make_config(c,q))
    subprocess.run([c['python'],'-B',str(PACKAGE/'runtime/scripts/check_runtime.py'),
        '--config',str(check),'--output',str(work/'runtime_preflight')],check=True)
    print('PREPARE=passed; MODEL=0; SEARCH=0; FINAL=0; TRAINING=0')
    print('WORK='+str(work))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-work',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--process-adapter',type=Path,required=True,
                   help='Verified original972 checkpoint-00001322; private deployment path')
    a=p.parse_args();prepare(a.reference_work.resolve(),a.work.resolve(),a.process_adapter.resolve())
