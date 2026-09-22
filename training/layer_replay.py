"""Qwen3-specific replay path used by the RL trainer.

Validated for the pinned Qwen3/Transformers SDPA configuration.  This module
is not a general-purpose replay implementation: it intentionally relies on
the inspected Qwen3 model structure and a Transformers internal mask helper.
"""
import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache


class LayerCache:
    """Fresh per-layer cache, created inside every checkpoint invocation."""
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.k = k if self.k is None else torch.cat((self.k, k), dim=-2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=-2)
        return self.k, self.v


def replay_hidden(model, input_ids, output_ids, checkpoint_layers=True):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    body = base.model
    if body.config._attn_implementation != 'sdpa':
        raise ValueError('Only the inspected SDPA path is supported')
    if body.gradient_checkpointing:
        raise ValueError('Disable built-in gradient checkpointing first')
    if any(m.training and getattr(m, 'p', 0) > 0
           for m in model.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError('Replay requires dropout disabled')
    if any(layer.self_attn.training and layer.self_attn.attention_dropout
           for layer in body.layers):
        raise ValueError('Attention dropout must be zero')
    if not input_ids or not output_ids:
        raise ValueError('Empty input/output')
    device = body.embed_tokens.weight.device
    feeds = [input_ids] + [[x] for x in output_ids[:-1]]
    chunks, metadata = [], []
    offset = 0
    for feed in feeds:
        ids = torch.tensor([feed], device=device)
        emb = body.embed_tokens(ids)
        positions = torch.arange(offset, offset + len(feed), device=device)
        position_ids = positions.unsqueeze(0)
        # Only cache length is consulted by the inspected dynamic SDPA mask path.
        length_cache = DynamicCache()
        if offset:
            dummy = emb.new_zeros((1, 1, offset, 1))
            length_cache.update(dummy, dummy, 0)
        mask = body._update_causal_mask(
            torch.ones((1, offset + len(feed)), dtype=torch.long, device=device),
            emb, positions, length_cache, False)
        rope = body.rotary_emb(emb, position_ids)
        metadata.append((offset, len(feed), mask, position_ids, positions, rope))
        chunks.append(emb)
        offset += len(feed)
    hidden = torch.cat(chunks, dim=1)
    for layer in body.layers:
        def run_layer(h, layer=layer):
            state = ()
            results = []
            for start, count, mask, posids, positions, rope in metadata:
                def step(x, *past, mask=mask, posids=posids,
                         positions=positions, rope=rope):
                    cache = LayerCache()
                    if past:
                        cache.k, cache.v = past
                    result = layer.forward(
                        x, attention_mask=mask, position_ids=posids,
                        past_key_value=cache, use_cache=True,
                        output_attentions=False, cache_position=positions,
                        position_embeddings=rope)
                    return result[0], cache.k, cache.v
                x = h[:, start:start+count, :].contiguous()
                if checkpoint_layers and torch.is_grad_enabled():
                    values = checkpoint(step, x, *state, use_reentrant=False)
                else:
                    values = step(x, *state)
                results.append(values[0])
                state = values[1:]
            return torch.cat(results, dim=1)
        hidden = checkpoint(run_layer, hidden, use_reentrant=False) \
            if checkpoint_layers and torch.is_grad_enabled() else run_layer(hidden)
    # Preserve original prefill/decode normalization shapes too.
    selected = []
    for start, count, *_ in metadata:
        selected.append(body.norm(hidden[:, start:start+count, :])[:, -1, :])
    return selected


def replay_logps(model, rec, checkpoint_layers=True):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hidden = replay_hidden(model, rec['input_ids'], rec['output_ids'], checkpoint_layers)
    temperature = float(rec['sampling']['temperature'])
    if not temperature > 0:
        raise ValueError('Invalid temperature')
    support = rec.get('support')
    if rec['sampling'].get('grammar') is not None and support is None:
        raise ValueError('Missing captured grammar support')
    logps = []
    for i, h in enumerate(hidden):
        def head(x, i=i):
            logits = base.lm_head(x).float()[0] / temperature
            target = rec['output_ids'][i]
            if support is not None:
                allowed = support['pool'][support['refs'][i]]
                if target not in allowed:
                    raise ValueError('Target outside captured support')
                indices = torch.tensor(allowed, device=logits.device)
                return logits[target] - torch.logsumexp(logits[indices], dim=0)
            return logits[target] - torch.logsumexp(logits, dim=0)
        logps.append(checkpoint(head, h, use_reentrant=False)
                     if torch.is_grad_enabled() else head(h))
    return torch.stack(logps)
