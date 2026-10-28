"""Source-only GRPO check; optional synthetic CPU tests, no private data or API."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    for name, row in manifest['files'].items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('unsafe_source_path')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['published_sha256']:
            raise ValueError('public_source_hash_mismatch:' + name)
        if b'\r\n' in raw:
            raise ValueError('source_must_use_LF:' + name)
        if path.suffix == '.py':
            ast.parse(raw.decode('utf8'), filename=name)
    print('PUBLIC_GRPO_SOURCE=passed; files=' + str(len(manifest['files'])), flush=True)
    if args.tests:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable, '-B', str(ROOT / 'training/process_grpo_update/test_update.py')],
                       cwd=ROOT, env=env, check=True)
    print('PRIVATE_DATA=0; QWEN_LOADS=0; RETRIEVAL_API=0; FORMAL_TRAINING=0', flush=True)


if __name__ == '__main__':
    main()
