"""Loss/data boundaries; semantic rewards must be supplied by an audited evaluator."""
import math
from .storage import digest


def sft_samples(store, run_id, codec, role="process"):
    if not codec.exact:
        raise ValueError("SFT_requires_exact_deployed_tokenizer")
    seen = set()
    for event in store.events(run_id, "generation"):
        row = event["data"]
        if row["receipt_key"] in seen:
            continue
        seen.add(row["receipt_key"])
        if not row["accepted"] or row["request"]["role"] != role:
            continue
        # A program-normalized answer is not a verbatim valid model target.
        # Preserve the receipt; any curated target requires a separate reviewed export.
        if row.get("json_repair", {}).get("applied"):
            continue
        req = row["request"]
        if not req["exact_tokenizer"] or codec.encode(req["messages"]) != req["prompt_token_ids"]:
            raise ValueError("runtime_training_prompt_mismatch")
        target = codec.target(row["response"]["text"])
        prompt = req["prompt_token_ids"]
        if len(target) > req["max_new_tokens"]:
            raise ValueError("target_exceeds_reserved_budget")
        yield dict(stage=row["stage"], messages=req["messages"], target=row["response"]["text"],
                   input_ids=prompt+target, labels=[-100]*len(prompt)+target,
                   input_sha256=req["input_sha256"], policy_identity=row["policy_identity"])


def stage_ce(logits, labels):
    """Causal, target-token-mean CE. Caller aggregates stage losses with explicit weights."""
    import torch.nn.functional as F
    shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    if not bool((shift_labels != -100).any()):
        raise ValueError("no_supervised_tokens")
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)


def group_advantages(rewards, epsilon=1e-8):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("invalid_reward_group")
    mean = sum(rewards)/len(rewards)
    sd = math.sqrt(sum((x-mean)**2 for x in rewards)/len(rewards))
    return [(x-mean)/(sd+epsilon) for x in rewards]


def require_grpo_capture(records):
    # Include rejected model attempts too: do not select only successful action tokens.
    for row in records:
        if row["request"]["role"] != "process":
            continue
        response = row["response"]
        ids, probs = response.get("token_ids"), response.get("behavior_logprobs")
        if not response.get("grpo_capture_ready") or not ids or probs is None or len(ids) != len(probs):
            raise ValueError("missing_behavior_token_logprobs")
        if not row["request"]["exact_tokenizer"]:
            raise ValueError("inexact_prompt_tokens")


def clipped_surrogate(new_logprobs, old_logprobs, advantage, mask, clip=0.2):
    """Reference scalar GRPO surrogate only (KL/optimizer/replay supplied by trainer)."""
    if not (len(new_logprobs) == len(old_logprobs) == len(mask)) or not any(mask):
        raise ValueError("invalid_policy_mask")
    terms = []
    for new, old, selected in zip(new_logprobs, old_logprobs, mask):
        if selected:
            ratio = math.exp(new-old)
            terms.append(min(ratio*advantage, min(1+clip, max(1-clip, ratio))*advantage))
    return -sum(terms)/len(terms)


def combine_reward(answer_score, process_score, avoidable_cost, process_weight=0.2, cost_weight=0.01):
    # Never turn self-reported State transitions into evaluator-certified progress.
    values = (answer_score, process_score, avoidable_cost, process_weight, cost_weight)
    if not all(math.isfinite(v) for v in values) or avoidable_cost < 0:
        raise ValueError("invalid_reward_components")
    return {"total": answer_score+process_weight*process_score-cost_weight*avoidable_cost,
            "answer": answer_score, "process": process_score, "avoidable_cost": avoidable_cost,
            "weights": {"process": process_weight, "cost": cost_weight}, "credit_assignment": "trajectory"}
