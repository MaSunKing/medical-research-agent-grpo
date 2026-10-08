"""Source, links and frozen-copy checks. No network, models or private data."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for p in ROOT.rglob('*.py'):
        if '.git' not in p.parts:ast.parse(p.read_bytes(),filename=str(p))
    frozen=ROOT/'versions/legacy_retrieval'
    for name,value in json.loads((frozen/'SOURCE_HASHES.json').read_text()).items():
        assert sha(frozen/name)==value, 'legacy drift: '+name
    current=json.loads((ROOT/'SOURCE_EXPORT_CURRENT.json').read_text())
    for row in current['files']:
        assert sha(ROOT/row['path'])==row['published_sha256'],row['path']
    docs=[ROOT/'README.md',*(ROOT/'versions').rglob('*.md'),
          ROOT/'retrieval/README.md',ROOT/'retrieval/runtime/README.md',
          *(ROOT/'docs').glob('*.md')]
    for p in docs:
        text=p.read_text(encoding='utf8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            path=target.split('#',1)[0]
            assert (p.parent/path).exists(),f'broken link: {p.relative_to(ROOT)} -> {target}'
    # New source export must contain source, not private experiment products.
    for row in current['files']:
        p=ROOT/row['path']
        assert p.suffix in ('.py','.json'),p
        text=p.read_text(encoding='utf8')
        assert not re.search(r'/home/msai/|C:/Users/|172\.21\.26\.100',text),p
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{20,}',text),p
    print('PUBLIC_ARCHITECTURE=passed; source hashes, legacy freeze, links, syntax; MODEL=0; API=0')

if __name__=='__main__':main()
