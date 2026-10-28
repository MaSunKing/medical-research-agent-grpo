"""Offline source integrity and dependency presence only. No API calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):raise ValueError('unsafe source path')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('source mismatch: '+name)
        if name.endswith('.py'):ast.parse(raw.decode('utf-8-sig'),filename=name)
    print('SOURCE_INTEGRITY=passed files='+str(len(manifest['files'])))
    for module in ('torch','transformers','fastmcp','starlette','aiohttp','requests','dotenv',
                   'pydantic','diskcache','numpy','bs4','lxml','rank_bm25'):
        print(module, 'PRESENT' if importlib.util.find_spec(module) else 'MISSING')
    print('SCOPE=source_and_dependency_presence_only; no model/API/GPU validation')

if __name__=='__main__':main()
