"""Compare actual new-data uninterrupted vs fresh-process resumed checkpoints."""
import argparse
from pathlib import Path
from contract import preflight, read, save, sha
from training_core import checkpoint_complete


def equal(a, b, path='root'):
    import torch
    assert type(a) is type(b), path
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f'{path}[{i}]')
    else:
        assert a == b, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--tests', type=Path, required=True)
    a = ap.parse_args()
    _, _, identity = preflight(a.package, a.base)
    import torch
    from safetensors.torch import load_file
    smoke = read(a.tests / 'smoke/GPU_SMOKE.json')
    assert smoke['status'] == 'passed' and smoke['identity'] == identity and smoke['optimizer_updates'] == 0
    receipts = [dict(path=str((a.tests/'smoke/GPU_SMOKE.json').resolve()), sha256=sha(a.tests/'smoke/GPU_SMOKE.json'))]
    environment = smoke['environment']
    for mode in ('reference', 'resumed'):
        assert read(a.tests / mode / 'ENVIRONMENT.json') == environment
    for step in (1, 2):
        left = a.tests / 'reference' / f'checkpoint-{step:08d}'
        right = a.tests / 'resumed' / f'checkpoint-{step:08d}'
        for cp in (left, right):
            checkpoint_complete(cp, identity)
            receipts.append(dict(path=str((cp/'COMPLETE.json').resolve()), sha256=sha(cp/'COMPLETE.json')))
        equal(load_file(left/'adapter_model.safetensors'), load_file(right/'adapter_model.safetensors'), f'weights_step{step}')
        states = [torch.load(p/'state.pt', map_location='cpu', weights_only=False) for p in (left, right)]
        equal(*states, path=f'state_step{step}')
        assert states[0]['step'] == step and states[0]['position'] == step*8 and states[0]['diagnostic']
    report = dict(status='passed', identity=identity, environment=environment, receipts=receipts,
        comparisons=[dict(step=i, LoRA_bitwise_equal=True, optimizer_scheduler_RNG_equal=True) for i in (1, 2)],
        longest_forward_backward=True, full_training_complete=False, medical_semantic_certification=False)
    save(a.tests/'ACCEPTANCE.json', report)
    print('GPU_ACCEPTANCE=passed; not medical quality certification')


if __name__ == '__main__':
    main()
