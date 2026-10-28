"""Immutable, offline Process update inputs. Importing this module does no work."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                   encoding='utf8', newline='\n')
    tmp.replace(path)


def safe(root, name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and ':' not in name and
            '\\' not in name, 'unsafe_relative_path')
    out = Path(root) / name
    require(not out.is_symlink() and out.resolve().is_relative_to(Path(root).resolve()), 'unsafe_file')
    return out


def verify_files(root, files):
    for name, value in files.items():
        h = value if isinstance(value, str) else value['sha256']
        require(sha(safe(root, name)) == h, 'file_identity_changed:'+name)


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def validate_call(c):
    q, r = c['request'], c['response']
    require(q['role'] == 'process' and q['stage'] in ('checklist','decision','state'), 'non_process_target')
    require(q['exact_tokenizer'] is True and q['input_sha256'] == digest(q['messages']), 'prompt_identity_changed')
    require(r['request_digest'] == digest(q), 'request_response_binding_changed')
    prompt, ids = q['prompt_token_ids'], r['token_ids']
    require(prompt and ids and all(type(t) is int and t >= 0 for t in prompt+ids), 'invalid_token_ids')
    require(len(ids) <= q['max_new_tokens'] and len(prompt)+len(ids) <= 15000, 'context_or_output_exceeded')
    require(r['policy_token_mask'] == [1]*len(ids), 'changed_policy_mask')
    require(r['capture_mode'] == 'decoupled_capture_v1' and r['training_eligible'] is False and
            r['correction_required'] is True and r['grpo_capture_ready'] is True and
            r['process_head'] == 'fp32_linear_unmerged', 'capture_contract_changed')
    b, old, cache = [r[k] for k in ('behavior_logprobs','old_train_logprobs','cached_replay_logprobs')]
    require(all(len(v) == len(ids) and all(finite(x) and x <= 1e-6 for x in v)
                for v in (b, old, cache)), 'invalid_logprob')
    diff = [abs(x-y) for x,y in zip(b,cache)]
    require(max(diff) <= 1e-4 and r['cached_replay_check'] == dict(passed=True,tolerance=1e-4,
            maximum_absolute_error=max(diff),mean_absolute_error=sum(diff)/len(diff)), 'behavior_cache_mismatch')
    check=r['replay_check']; fullmax=max(abs(x-y) for x,y in zip(b,old))
    require(r['replay_logprobs'] == old and check['maximum_absolute_error'] == fullmax and
            check['mode'] == 'full_forward_diagnostic_not_collection_gate' and
            check['passed'] == (fullmax <= check['tolerance']), 'changed_full_diagnostic')
    s=r['sampling']
    require(s['effective_parameters_verified'] is True and s['temperature'] == 1 and
            s['effective'] == dict(do_sample=True,temperature=1.,top_p=1.,top_k=0,num_beams=1,
                repetition_penalty=1.,max_new_tokens=q['max_new_tokens'],use_cache=True,num_return_sequences=1),
            'effective_sampling_changed')
    require(c['accepted'] in (True,False) and isinstance(c['json_repaired'],bool), 'attempt_status_missing')


def load_data(root):
    root=Path(root); m=read(root/'MANIFEST.json'); verify_files(root,m['files'])
    rows=read(root/'GROUPS.json'); seen=set(); records=0
    for g in rows:
        require(g['question_id'] not in seen and len(g['rollouts']) == 4, 'incomplete_or_duplicate_group')
        seen.add(g['question_id']); rs=[r['R'] for r in g['rollouts']]
        require([r['rollout'] for r in g['rollouts']] == list(range(4)), 'rollout_order_changed')
        require(all(x is None or finite(x) and 0<=x<=1 for x in rs), 'invalid_reward')
        held=any(x is None for x in rs); mean=None if held else sum(rs)/4
        require(g['held'] == held, 'hold_flag_changed')
        for r in g['rollouts']:
            expected=None if held else r['R']-mean
            require((expected is None and r['A'] is None) or expected is not None and
                    r['A'] is not None and abs(r['A']-expected)<1e-10, 'advantage_changed')
            calls=read(safe(root,r['calls_file']))
            require(calls and len({c['call_id'] for c in calls}) == len(calls), 'duplicate_or_missing_calls')
            for c in calls:
                validate_call(c)
                require(c['response']['policy_identity'] == r['policy_identity'], 'policy_binding_changed')
            require(sum(len(c['response']['token_ids']) for c in calls) == r['policy_tokens'], 'token_denominator_changed')
            records+=1
    require(len(rows)==m['questions'] and records==m['records'], 'dataset_count_changed')
    return rows,m,sha(root/'MANIFEST.json')


def validate_config(c):
    require(c['schema']=='process_grpo_update_v1' and c['passes']==1 and c['beta']==0,
            'first_batch_one_pass_no_KL_contract')
    require(c['groups_per_update']==4 and c['acceptance_questions']==8 and c['seed']==42,
            'group_plan_changed')
    require(0<c['lr']<=1e-5 and 0<c['epsilon']<1 and 1<c['is_cap']<=3 and
            c['max_grad_norm']>0 and c['forward_tolerance']<=1e-4, 'invalid_update_config')
    require(c['rollout_reduction']=='mean_policy_tokens_per_rollout_then_equal_rollout_mean' and
            c['advantage']=='reward_minus_question_mean_no_std' and c['gradient_checkpointing'] is True,
            'objective_or_precision_changed')


def batches(groups,c):
    import random
    selected=[g for g in groups if not g['held']]
    random.Random(c['seed']).shuffle(selected)
    return [selected[i:i+c['groups_per_update']] for i in range(0,len(selected),c['groups_per_update'])]


def acceptance_groups(groups,n=8):
    valid=[g for g in groups if not g['held']]
    require(len(valid)>=n,'insufficient_acceptance_groups')
    # Longest actual update input, model failure, then both collection strata.
    chosen=[]
    for pool in ([max(valid,key=lambda g:g['max_sequence_tokens'])],
                 [g for g in valid if any(r['process_task_failure'] for r in g['rollouts'])],
                 [g for g in valid if g['stratum']=='existing600_train_short'],
                 [g for g in valid if g['stratum']=='external_long_liveqa_train'],valid):
        for g in pool:
            if g not in chosen:
                chosen.append(g)
                break
    for g in valid:
        if len(chosen)==n: break
        if g not in chosen:chosen.append(g)
    return chosen
