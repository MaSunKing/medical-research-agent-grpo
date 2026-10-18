"""Read-only status. No torch, model, tools or network."""
import argparse
import json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('work', type=Path)
a = ap.parse_args()
path = a.work/'training/loss.jsonl'
latest = None
if path.exists():
    for line in path.read_text(encoding='utf8').splitlines():
        try:
            row = json.loads(line)
            if latest is None or row['step'] >= latest['step']: latest = row
        except (ValueError, KeyError): pass
print('GPU acceptance:', (a.work/'tests/ACCEPTANCE.json').exists())
if latest:
    print(f"Optimizer steps: {latest['step']} / {latest['total_steps']} ({100*latest['step']/latest['total_steps']:.1f}%)")
    print(f"Stage rows: {latest['samples_seen']} / 10575; latest target-token-mean loss: {latest['loss']:.6f}")
else: print('No formal optimizer update recorded.')
print('Training complete:', (a.work/'training/TRAINING_COMPLETE').exists())
