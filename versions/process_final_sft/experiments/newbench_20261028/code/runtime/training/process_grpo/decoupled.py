"""Collection-only decoupled probability receipts. No optimizer or rewards."""
import contextlib
import math
import time
from types import MethodType
from common import digest, require, save
import native

MODE = 'decoupled_capture_v1'
CACHE_TOL = 1e-4


def difference(a, b):
    require(isinstance(a, list) and isinstance(b, list) and a and len(a) == len(b),
            'probability_length_mismatch')
    require(all(type(x) in (float, int) and math.isfinite(x) and x <= 1e-6
                for x in a + b), 'invalid_logprob')
    values = [abs(x-y) for x, y in zip(a, b)]
    return dict(maximum_absolute_error=max(values), mean_absolute_error=sum(values)/len(values))


@contextlib.contextmanager
def process_head(model):
    import torch
    import torch.nn.functional as F
    head = model.get_base_model().lm_head
    require(type(head) is torch.nn.Linear and not head.weight.requires_grad, 'frozen_linear_head_required')
    original = head.forward
    def fp32(module, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), module.weight.float(),
                            None if module.bias is None else module.bias.float())
    head.forward = MethodType(fp32, head)
    try:
        yield
    finally:
        head.forward = original


def validate_response(r, request):
    require(r.get('capture_mode') == MODE and r.get('training_eligible') is False,
            'wrong_decoupled_capture_mode')
    require(r.get('request_digest') == digest(request), 'capture_request_changed')
    require(request['input_sha256'] == digest(request['messages']), 'input_hash_changed')
    ids, behavior = r.get('token_ids'), r.get('behavior_logprobs')
    require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids), 'invalid_token_ids')
    require(len(ids) <= request['max_new_tokens'] and isinstance(behavior, list)
            and len(ids) == len(behavior), 'invalid_capture_length')
    cached, full = r.get('cached_replay_logprobs'), r.get('old_train_logprobs')
    cache_diff, full_diff = difference(behavior, cached), difference(behavior, full)
    require(cache_diff['maximum_absolute_error'] <= CACHE_TOL, 'cached_behavior_replay_failed')
    require(r.get('cached_replay_check') == dict(passed=True, tolerance=CACHE_TOL, **cache_diff),
            'cached_replay_receipt_changed')
    check = r.get('replay_check', {})
    require(type(check.get('tolerance')) in (int, float) and 0 < check['tolerance'] <= .05,
            'invalid_full_diagnostic_tolerance')
    require(check.get('maximum_absolute_error') == full_diff['maximum_absolute_error']
            and check.get('passed') == (full_diff['maximum_absolute_error'] <= check['tolerance'])
            and check.get('mode') == 'full_forward_diagnostic_not_collection_gate',
            'full_replay_receipt_changed')
    require(r.get('replay_logprobs') == full and r.get('policy_token_mask') == [1]*len(ids),
            'invalid_replay_or_mask')
    sampling = r.get('sampling', {})
    effective = sampling.get('effective', {})
    expected = dict(do_sample=True, temperature=1., top_p=1., top_k=0, num_beams=1,
                    repetition_penalty=1., max_new_tokens=request['max_new_tokens'],
                    use_cache=True, num_return_sequences=1)
    require(sampling.get('temperature') == 1. and sampling.get('effective_parameters_verified') is True
            and effective == expected, 'unverified_effective_sampling')
    require(r.get('process_head') == 'fp32_linear_unmerged' and r.get('correction_required') is True
            and r.get('grpo_capture_ready') is True, 'capture_not_ready')
    return dict(cache=cache_diff, full=full_diff)


def capture_admission(events):
    generations = [e['data'] for e in events if e['kind'] == 'generation'
                   and e['data']['request']['role'] == 'process']
    requests = [e['data'] for e in events if e['kind'] == 'model_request'
                and e['data']['request']['role'] == 'process']
    require(generations and len(generations) == len(requests), 'missing_process_generation_receipt')
    require(sorted(digest(g['request']) for g in generations)
            == sorted(digest(r['request']) for r in requests), 'generation_request_binding_failed')
    for g in generations:
        validate_response(g['response'], g['request'])
    return dict(capture_mode=MODE, process_attempts=len(generations),
        rejected_attempts=sum(not g['accepted'] for g in generations),
        repaired_attempts=sum(bool(g.get('json_repair', {}).get('applied')) for g in generations),
        process_tokens=sum(len(g['response']['token_ids']) for g in generations),
        collection_capture_complete=True, correction_required=True,
        trainer_precision_acceptance=False, training_eligible=False)


class DecoupledPolicy(native.NativePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = digest(dict(mode=MODE, process=self.process, seed=self.seed,
                                    policy=self.policy_identity, process_head='fp32_linear_unmerged'))

    async def generate(self, request):
        # Frozen Base Final/Reader retain the original implementation and head.
        if not self.process:
            return await super().generate(request)
        began = time.monotonic()
        if self.active:
            self.active(request['stage'])
        require(request['role'] == 'process' and request['stage'] in ('checklist','decision','state'),
                'unexpected_process_stage')
        prompt = self.tokenizer.apply_chat_template(request['messages'], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
        require(prompt == request['prompt_token_ids'], 'actual_prompt_token_mismatch')
        require(len(prompt)+request['max_new_tokens'] <= 15000, 'native_context_exceeded')
        folder = self.output / f'{self.ordinal:04d}_{request["stage"]}'
        self.ordinal += 1
        save(folder/'REQUEST.json', request, exclusive=True)
        seed = int(digest([self.seed, self.ordinal, request['input_sha256']])[:8], 16)
        self.model.eval().requires_grad_(False)
        try:
            audit = {}
            with process_head(self.model):
                ids, behavior = native.sample(self.model, self.tokenizer, prompt,
                    request['max_new_tokens'], seed, 1., audit=audit)
                # Persist true behavior before either replay; failures never erase it.
                save(folder/'SAMPLING.json', dict(token_ids=ids, behavior_logprobs=behavior,
                    sampling=audit, seed=seed, request_digest=digest(request), capture_mode=MODE,
                    process_head='fp32_linear_unmerged', training_eligible=False), exclusive=True)
                full = native.replay_logprobs(self.model, prompt, ids)
                cached = native.replay_cached_logprobs(self.model, prompt, ids)
            cd, fd = difference(behavior, cached), difference(behavior, full)
            cache_ok = cd['maximum_absolute_error'] <= CACHE_TOL
            result = dict(text=self.tokenizer.decode(ids, skip_special_tokens=True),
                model='original972-process', capture_mode=MODE, request_digest=digest(request),
                sampling=dict(temperature=1., top_p=1., top_k=0, seed=seed, thinking=False,
                    repetition_penalty=1., num_beams=1,
                    stop_token_ids=native.stop_tokens(self.model,self.tokenizer), **audit),
                token_ids=ids, behavior_logprobs=behavior, policy_token_mask=[1]*len(ids),
                grpo_capture_ready=cache_ok and audit.get('effective_parameters_verified') is True,
                replay_logprobs=full, old_train_logprobs=full, cached_replay_logprobs=cached,
                replay_check=dict(passed=fd['maximum_absolute_error'] <= self.tolerance,
                    tolerance=self.tolerance, maximum_absolute_error=fd['maximum_absolute_error'],
                    mode='full_forward_diagnostic_not_collection_gate'),
                cached_replay_check=dict(passed=cache_ok, tolerance=CACHE_TOL, **cd),
                correction_required=True, training_eligible=False, process_head='fp32_linear_unmerged',
                policy_identity=self.policy_identity,
                finish_reason='eos' if ids[-1] in native.stop_tokens(self.model,self.tokenizer) else 'length',
                usage=dict(prompt_tokens=len(prompt), completion_tokens=len(ids)), seconds=time.monotonic()-began)
            save(folder/'RESPONSE.json', result, exclusive=True)
            validate_response(result, request)
            return result
        except Exception as error:
            save(folder/'FAILURE.json', dict(error_type=type(error).__name__, raw_records_preserved=True), exclusive=True)
            raise
