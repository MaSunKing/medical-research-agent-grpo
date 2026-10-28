"""Token-level truncated IS times clipped GRPO surrogate; no implicit reward grading."""
import torch
from contract import require


def terms(current, old, behavior, advantage, epsilon=.2, cap=2.):
    old=torch.as_tensor(old,dtype=torch.float32,device=current.device).detach()
    behavior=torch.as_tensor(behavior,dtype=torch.float32,device=current.device).detach()
    require(current.shape == old.shape == behavior.shape and current.numel()>0,'logp_shape')
    require(all(torch.isfinite(x).all().item() for x in (current,old,behavior)),'nonfinite_logprob')
    w=(old-behavior).exp().detach()
    ratio=(current-old).exp()
    require(torch.isfinite(w).all().item() and torch.isfinite(ratio).all().item(),'nonfinite_ratio')
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    loss_tokens=-w.clamp(max=cap)*surrogate
    return loss_tokens,dict(raw_weight=w,ratio=ratio.detach(),
                           clipped=((ratio<1-epsilon)|(ratio>1+epsilon)).detach())


def backward_call(model,call,old,advantage,denominator,config,logp_fn):
    current=logp_fn(model,call,gradient=True)
    losses,stats=terms(current,old,call['response']['behavior_logprobs'],advantage,
                       config['epsilon'],config['is_cap'])
    loss=losses.sum()/denominator
    require(torch.isfinite(loss).item(),'nonfinite_loss')
    loss.backward()
    return dict(loss=float(loss.detach()),tokens=current.numel(),
        clipped=int(stats['clipped'].sum()),weight_min=float(stats['raw_weight'].min()),
        weight_max=float(stats['raw_weight'].max()))
