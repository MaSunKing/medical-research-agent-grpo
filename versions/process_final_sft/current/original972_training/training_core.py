"""Sparse causal CE: no loss on history/tool/input tokens; logits only on targets."""
from pathlib import Path
from contract import read, sha


def masked_loss(model, tokens, spans):
    import torch
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    hidden = base.model(input_ids=tokens, use_cache=False).last_hidden_state
    total = hidden.sum() * 0
    mass = 0.
    for start, end, weight in spans:
        assert 1 <= start < end <= tokens.shape[1] and weight > 0
        mass += (end-start)*weight
        for begin in range(start, end, 64):
            stop = min(begin+64, end)
            def ce(x, y):
                logits = base.lm_head(x).float()
                return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1), reduction='sum')
            total = total + weight * checkpoint(ce, hidden[:, begin-1:stop-1],
                tokens[:, begin:stop], use_reentrant=False)
    return total, mass


def checkpoint_complete(path, identity):
    path = Path(path)
    assert path.is_dir() and not path.is_symlink()
    assert not (path / 'COMPLETE.json').is_symlink()
    manifest = read(path / 'COMPLETE.json')
    assert manifest['identity'] == identity
    assert set(manifest['files']) == {'adapter_config.json', 'adapter_model.safetensors', 'state.pt'}
    for name, digest in manifest['files'].items():
        p = path / name
        assert p.is_file() and not p.is_symlink() and p.stat().st_size and sha(p) == digest, name
    return True
