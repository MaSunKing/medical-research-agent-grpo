"""Opt-in native Qwen sampler. Captures actual sampled IDs; never retokenizes answers.

Only sampling temperature=1 is admitted as Process capture. A single-vocabulary
buffer captures selected-token logprobs without retaining scores for all steps.
This is a collection implementation, not a GRPO trainer or GPU acceptance receipt.
"""
import contextlib
import inspect
import time
from pathlib import Path
from common import digest, require, save


class SelectedScores:
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
        self.previous = None
        self.values = []
        self.last_length = prompt_length - 1

    def __call__(self, input_ids, scores):
        require(input_ids.shape[0] == 1 and input_ids.shape[1] == self.last_length + 1,
                'unexpected_generation_batch_or_sequence')
        if self.previous is not None:
            self.values.append(float(self.previous[0, int(input_ids[0, -1])].item()))
        self.previous = scores.float().log_softmax(-1)
        self.last_length = input_ids.shape[1]
        return scores

    def finish(self, full_ids):
        require(self.previous is not None and full_ids.shape[1] == self.last_length + 1,
                'generation_did_not_append_exactly_one_token')
        self.values.append(float(self.previous[0, int(full_ids[0, -1])].item()))
        require(len(self.values) == full_ids.shape[1] - self.prompt_length, 'capture_length_mismatch')
        return self.values


def stop_tokens(model, tokenizer):
    eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    eos = tokenizer.eos_token_id if eos is None else eos
    ids = [eos] if type(eos) is int else list(eos)
    require(ids and all(type(i) is int and i >= 0 for i in ids), 'invalid_stop_token_configuration')
    return ids


def generation_setup(model, tokenizer, maximum, temperature=1.0):
    """Resolve the installed HF merge rules before recording sampling parameters.

    Transformers 4.x may treat values equal to global defaults as unset and
    replace them with model defaults. Newer versions removed that argument.
    Never use the requested values alone as evidence of effective settings.
    """
    from transformers import GenerationConfig
    core = model.get_base_model() if hasattr(model, 'get_base_model') else model
    config = GenerationConfig(do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
        num_beams=1, num_return_sequences=1, repetition_penalty=1.0, max_new_tokens=maximum, use_cache=True,
        eos_token_id=stop_tokens(model, tokenizer), pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False, output_scores=False)
    flags = {}
    if 'use_model_defaults' in inspect.signature(core.generate).parameters:
        flags['use_model_defaults'] = False
    prepare = core._prepare_generation_config
    preparation_flags = {}
    if 'use_model_defaults' in inspect.signature(prepare).parameters:
        preparation_flags['use_model_defaults'] = False
        require(flags.get('use_model_defaults') is False,
                'generation_default_override_cannot_be_disabled')
    effective, _ = prepare(config, **preparation_flags)
    expected = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
        num_beams=1, repetition_penalty=1.0, max_new_tokens=maximum, use_cache=True,
        num_return_sequences=1)
    actual = {k: getattr(effective, k, None) for k in expected}
    require(actual == expected, 'effective_generation_parameters_changed')
    # No later processor may silently alter the distribution observed by our tracker.
    for name in ('forced_bos_token_id', 'forced_eos_token_id', 'suppress_tokens',
                 'begin_suppress_tokens', 'bad_words_ids', 'sequence_bias',
                 'watermarking_config', 'min_p', 'top_h', 'guidance_scale'):
        require(getattr(effective, name, None) is None,
                'unexpected_generation_distribution_modifier:' + name)
    for name, default in (('typical_p', 1.), ('epsilon_cutoff', 0.), ('eta_cutoff', 0.),
                          ('no_repeat_ngram_size', 0), ('min_new_tokens', 0)):
        require(getattr(effective, name, None) in (None, default),
                'unexpected_generation_distribution_modifier:' + name)
    audit = dict(effective_parameters_verified=True, effective=actual,
                 use_model_defaults=flags.get('use_model_defaults', 'argument_not_supported'),
                 verification='installed_transformers_generation_config_resolution')
    return config, flags, audit


def sample(model, tokenizer, prompt, maximum, seed, temperature=1.0, *, audit=None):
    import torch
    from transformers import LogitsProcessorList, set_seed
    device = next(model.parameters()).device
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    tracker = SelectedScores(len(prompt))
    config, flags, verified = generation_setup(model, tokenizer, maximum, temperature)
    if audit is not None:
        audit.update(verified)
    set_seed(seed)
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=config, logits_processor=LogitsProcessorList([tracker]), **flags)
        values = tracker.finish(output)
    return output[0, len(prompt):].tolist(), values


def replay_logprobs(model, prompt, completion):
    """Teacher-forced, eval/no-autocast replay; training must use/validate this path."""
    import torch
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    core = model.get_base_model() if hasattr(model, 'get_base_model') else model
    result = []
    with torch.inference_mode():
        hidden = core.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                            use_cache=False).last_hidden_state
        for begin in range(len(prompt), len(prompt) + len(completion), 32):
            end = min(begin + 32, len(prompt) + len(completion))
            logits = core.lm_head(hidden[:, begin - 1:end - 1]).float()
            values = logits.log_softmax(-1).gather(-1, tokens[:, begin:end, None]).squeeze(-1)
            result.extend(values[0].tolist())
    return result


def replay_cached_logprobs(model, prompt, completion):
    """Diagnostic only: force existing tokens through the incremental cache path.

    This does not replace the full teacher-forced collection/training parity gate.
    It generates no new tokens and does not call retrieval, Final or an optimizer.
    """
    import torch
    require(prompt and completion, 'nonempty_replay_tokens_required')
    core = model.get_base_model() if hasattr(model, 'get_base_model') else model
    device = next(model.parameters()).device
    all_tokens = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    past = None
    values = []
    model.eval()
    with torch.inference_mode():
        for index, target in enumerate(completion):
            end = len(prompt) + index
            tokens = all_tokens[:, :end] if past is None else all_tokens[:, end-1:end]
            result = core.model(input_ids=tokens,
                attention_mask=torch.ones_like(all_tokens[:, :end]),
                past_key_values=past, use_cache=True)
            logits = core.lm_head(result.last_hidden_state[:, -1:, :]).float()
            values.append(float(logits.log_softmax(-1)[0, 0, target].item()))
            past = result.past_key_values
            require(past is not None, 'cached_replay_did_not_return_cache')
    return values


class NativePolicy:
    def __init__(self, model, tokenizer, *, process, seed, output, policy_identity,
                 tolerance=0.03, active=None):
        self.model, self.tokenizer, self.process = model, tokenizer, process
        self.seed, self.output, self.policy_identity = seed, Path(output), policy_identity
        self.tolerance, self.active, self.ordinal = tolerance, active, 0
        self.identity = digest(dict(native_sampler='process_grpo_sampling_lock_v2', process=process,
                                    seed=seed, policy=policy_identity))

    async def generate(self, request):
        started = time.monotonic()
        if self.active:
            self.active(request['stage'])
        prompt = self.tokenizer.apply_chat_template(request['messages'], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
        require(prompt == request['prompt_token_ids'], 'actual_prompt_token_mismatch')
        require(len(prompt) + request['max_new_tokens'] <= 15000, 'native_context_exceeded')
        folder = self.output / f'{self.ordinal:04d}_{request["stage"]}'
        self.ordinal += 1
        save(folder / 'REQUEST.json', request, exclusive=True)
        seed_input = [self.seed, self.ordinal, request['input_sha256']] if self.process else [self.seed, self.ordinal, request['stage']]
        seed = int(digest(seed_input)[:8], 16)
        # Final uses an independent fixed seed for the question; policy sampling uses rollout seed.
        temperature = 0.1 if request['stage'] not in {'checklist', 'decision', 'state', 'final'} else 1.0
        scope = contextlib.nullcontext() if self.process else self.model.disable_adapter()
        self.model.eval().requires_grad_(False)
        try:
            sampling_audit = {}
            with scope:
                ids, old = sample(self.model, self.tokenizer, prompt, request['max_new_tokens'],
                                  seed, temperature, audit=sampling_audit)
                replay = replay_logprobs(self.model, prompt, ids) if self.process else []
            maximum = max((abs(a - b) for a, b in zip(old, replay)), default=0.0)
            passed = len(old) == len(replay) and maximum <= self.tolerance if self.process else False
            result = dict(text=self.tokenizer.decode(ids, skip_special_tokens=True),
                model='original972-process' if self.process else 'frozen-base-final-or-reader',
                sampling=dict(temperature=temperature, top_p=1.0, top_k=0, seed=seed,
                              thinking=False, repetition_penalty=1.0, num_beams=1,
                              stop_token_ids=stop_tokens(self.model,self.tokenizer), **sampling_audit),
                token_ids=ids, behavior_logprobs=old if self.process else None,
                policy_token_mask=[1 if self.process else 0]*len(ids),
                grpo_capture_ready=passed and temperature == 1.0
                    and sampling_audit.get('effective_parameters_verified') is True,
                replay_logprobs=replay if self.process else None,
                replay_check=dict(passed=passed, maximum_absolute_error=maximum,
                                  tolerance=self.tolerance, mode='eval_no_autocast_teacher_forced'),
                policy_identity=self.policy_identity,
                finish_reason='eos' if ids[-1] in stop_tokens(self.model,self.tokenizer) else 'length',
                usage=dict(prompt_tokens=len(prompt), completion_tokens=len(ids)),
                seconds=time.monotonic() - started)
            save(folder / 'RESPONSE.json', result, exclusive=True)
            if self.process:
                require(result['grpo_capture_ready'], 'process_rollout_replay_parity_failed')
            return result
        except Exception as error:
            save(folder / 'FAILURE.json', dict(error_type=type(error).__name__,
                 raw_records_preserved=True), exclusive=True)
            raise


def load(config):
    import os
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16_CUDA_required')
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(config['base_model'], local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(base, config['process_adapter'], is_trainable=False,
                                      local_files_only=True)
    model.eval().requires_grad_(False)
    return model, tokenizer
