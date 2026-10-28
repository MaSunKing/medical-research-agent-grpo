"""Verify frozen source export; optional synthetic-only evaluator tests."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    assert manifest['source_only'] and len(manifest['files']) == 152
    for name, row in manifest['files'].items():
        p = ROOT / name
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['published_sha256'], name
        assert b'\r' not in raw, name
        if p.suffix == '.py':
            ast.parse(raw, filename=name)
    print('FROZEN_PUBLIC_SOURCE=passed; files=152; syntax and hashes; MODEL=0; API=0', flush=True)
    if args.tests:
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT / 'eval')
        env['NEWBENCH_TEST_PACKAGE'] = str(ROOT)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONUTF8'] = '1'
        scratch = ROOT / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='newbench-public-', dir=scratch) as tmp:
            env['TMP'] = env['TEMP'] = env['TMPDIR'] = tmp
            subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'test_binding', 'test_evaluate', 'test_grading', 'test_protocol', '-q'],
                           cwd=tmp, env=env, check=True)
        print('EVALUATOR_FIXTURES=passed; synthetic only; no weights, paid APIs or training')


if __name__ == '__main__':
    main()
