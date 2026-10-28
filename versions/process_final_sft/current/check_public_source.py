"""Public source identity and optional synthetic runtime tests. No private data needed."""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODULES=['test_agent','test_model_json','test_tokenizer_return_contract','test_search_cache',
 'test_search_repeat','test_decision_recovery','test_failure_feedback','test_engineering_failures',
 'test_decision_observations','test_browse_no_progress','test_browse_language','test_browse_novelty',
 'test_browse_quality_regression','test_cleaning','test_cleaning_adaptation','test_context',
 'test_context_pipeline','test_context_capacity_v2','test_context_receipt_projection',
 'test_context_empty_projection','test_search_context','test_soft_hard_context',
 'test_source_coordinates','test_deterministic_context','test_table_context_provenance',
 'test_nonbody_block_cleaning','test_bounded_browser_noise',
 'test_short_body_protection','test_worker_source_registry']


def verify():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,row in manifest['files'].items():
        p=ROOT/name
        assert not p.is_symlink() and p.resolve().is_relative_to(ROOT),name
        raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['published_sha256'],name
        assert b'\r\n' not in raw,name
        text=raw.decode('utf-8')
        private_path_pattern='/home/' + r'msai/|C:[/\\]Users[/\\]|172\.21\.26\.100'
        assert not re.search(private_path_pattern,text),name
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{20,}',text),name
        if p.suffix=='.py':ast.parse(text,filename=name)
    print('PUBLIC_CURRENT_SOURCE=passed; files='+str(len(manifest['files'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tests',action='store_true');a=p.parse_args()
    verify()
    subprocess.run([sys.executable,'-B',str(ROOT/'runtime/scripts/check_source_snapshot.py')],check=True)
    if a.tests:
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'runtime/src'),str(ROOT/'runtime/tests')]),
                 PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable,'-B','-m','unittest',*MODULES,'-q'],cwd=ROOT,env=env,check=True)
    print('NO_PRIVATE_QUESTIONS; MODEL=0; RETRIEVAL_API=0; TRAINING=0',flush=True)


if __name__=='__main__':main()
