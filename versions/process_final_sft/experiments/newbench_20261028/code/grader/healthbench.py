"""Pinned native HealthBench rubric grading; one retry, defer rather than abort batch."""
import argparse
import ast
import http.client
import json
import os
from pathlib import Path
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from common import read, save, sha, digest, require, load_dataset, acquire_lock


class Deferred(Exception):
    pass


class AuthenticationFailure(Exception):
    pass


def official_parts(path, expected_sha):
    require(sha(path) == expected_sha, 'pinned_HealthBench_protocol_changed')
    tree = ast.parse(Path(path).read_text(encoding='utf8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GRADER_TEMPLATE' for t in node.targets):
            nodes.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ('RubricItem', 'calculate_score'):
            nodes.append(node)
    require(len(nodes) == 3, 'unexpected_upstream_protocol_schema')
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-official-HealthBench>', 'exec'), ns)
    return ns


def parse_grade(text):
    from common import unique_object
    require(isinstance(text, str), 'judge_content_must_be_text')
    text = re.sub(r'^```json\s*|\s*```$', '', text.strip())
    value = json.loads(text, object_pairs_hook=unique_object)
    require(isinstance(value, dict) and type(value.get('criteria_met')) is bool and isinstance(value.get('explanation'), str), 'invalid_judge_schema')
    return dict(criteria_met=value['criteria_met'], explanation=value['explanation'])


def credential(path):
    key = os.environ.get('DASHSCOPE_API_KEY')
    if not key and path:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            line = line.strip().removeprefix('export ')
            if line.startswith('DASHSCOPE_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"').strip("'")
    require(bool(key), 'DASHSCOPE_API_KEY_missing')
    return key


def post(payload, endpoint, key):
    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode('utf8'),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = json.loads(response.read())
    choice = raw['choices'][0]
    require(choice['finish_reason'] == 'stop', 'incomplete_judge_response')
    # Do not persist hidden reasoning, HTTP headers, or raw exceptions.
    return dict(content=choice['message']['content'], returned_model=raw.get('model'),
                response_id=raw.get('id'), usage=raw.get('usage'), finish_reason=choice['finish_reason'])


def grade_one(prompt, answer, rubric, config, key, folder, official, *, deadline=float('inf'),
              stopping=None, post_fn=post, allow_uncertain_retry=False):
    folder = Path(folder)
    stopping = stopping or [False]
    endpoint = config['base_url'].rstrip('/') + '/chat/completions'
    url = urllib.parse.urlsplit(endpoint)
    require(url.scheme == 'https' and not url.username and not url.password and not url.query and not url.fragment,
            'clean_HTTPS_judge_endpoint_required')
    conversation = '\n\n'.join(f"{m['role']}: {m['content']}" for m in prompt + [dict(role='assistant', content=answer)])
    item = official['RubricItem'].from_dict(rubric)
    text = official['GRADER_TEMPLATE'].replace('<<conversation>>', conversation).replace('<<rubric_item>>', str(item))
    payload = dict(model=config['model'], messages=[dict(role='system', content='You are a helpful assistant.'),
                   dict(role='user', content=text)], temperature=config['temperature'],
                   max_tokens=config['max_tokens'], enable_thinking=False, stream=False)
    identity = digest(dict(payload=payload, endpoint=endpoint, protocol=config['protocol_sha256']))
    if (folder / 'GRADE.json').exists():
        cached = read(folder / 'GRADE.json')
        require(cached['identity'] == identity, 'judge_cache_identity_changed')
        return cached['grade']
    for started in folder.glob('requests/*/STARTED.json'):
        if not (started.parent / 'FINISHED.json').exists() and not allow_uncertain_retry:
            raise Deferred('unknown_prior_judge_request_requires_explicit_reconciliation')
    save(folder / 'INPUT.json', dict(identity=identity, payload=payload, endpoint=endpoint))
    for attempt in range(2):
        if stopping[0] or time.monotonic() + 125 > deadline:
            raise Deferred('insufficient_request_time_remaining')
        capture = folder / 'requests' / (str(time.time_ns()) + '-' + str(attempt))
        save(capture / 'STARTED.json', dict(identity=identity, attempt=attempt, time=time.time()))
        try:
            response = post_fn(payload, endpoint, key)
            save(capture / 'RESPONSE.json', response)
            grade = parse_grade(response['content'])
        except urllib.error.HTTPError as exc:
            save(capture / 'FINISHED.json', dict(status='error', error_type='HTTPError', http_status=exc.code))
            if exc.code in (401, 403):
                raise AuthenticationFailure('judge_authentication_failed') from None
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise AuthenticationFailure('judge_nonretryable_HTTP_status') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException,
                json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
            save(capture / 'FINISHED.json', dict(status='error', error_type=type(exc).__name__))
        else:
            save(capture / 'FINISHED.json', dict(status='complete'))
            save(folder / 'GRADE.json', dict(identity=identity, grade=grade, receipt=str(capture.relative_to(folder))))
            return grade
        if attempt == 0 and not stopping[0] and time.monotonic() + 127 <= deadline:
            time.sleep(2)
    raise Deferred('judge_failed_after_one_retry; next_question_allowed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--grader-config', type=Path, required=True)
    p.add_argument('--secrets-file', type=Path)
    p.add_argument('--real-judge-api', action='store_true')
    p.add_argument('--acknowledge-uncertain-retry', action='store_true')
    p.add_argument('--seconds', type=int, default=19000)
    a = p.parse_args()
    require(a.real_judge_api, 'explicit_potentially_paid_judge_opt_in_required')
    require(not any(a.output.resolve().is_relative_to(x.resolve()) for x in (a.data, a.generation)), 'grading_output_overlaps_immutable_input')
    rows, meta, data_hash = load_dataset(a.data)
    require(meta['kind'] == 'healthbench50_development_ablation', 'native_HealthBench_rubrics_required_not_dev100')
    config = read(a.grader_config)
    allowed = {'base_url', 'model', 'temperature', 'max_tokens', 'enable_thinking', 'official_original_grader',
               'official_protocol', 'official_model'}
    require(set(config) <= allowed, 'unexpected_or_sensitive_grader_config_field')
    config['protocol_sha256'] = a.protocol_sha256
    official = official_parts(a.protocol, a.protocol_sha256)
    key = credential(a.secrets_file)
    generation_identity = read(a.generation / 'IDENTITY.json')
    require(generation_identity['dataset'] == data_hash, 'generation_dataset_mismatch')
    identity = dict(dataset=data_hash, generation=digest(generation_identity), config=config,
                    score_protocol_sha256=a.protocol_sha256, source_sha256=sha(__file__))
    a.output.mkdir(parents=True, exist_ok=True)
    with acquire_lock(a.output / '.lock'):
        run_grading(a, rows, config, official, key, generation_identity, identity)


def run_grading(a, rows, config, official, key, generation_identity, identity):
    if (a.output / 'IDENTITY.json').exists():
        require(read(a.output / 'IDENTITY.json') == identity, 'grading_identity_changed')
    else:
        save(a.output / 'IDENTITY.json', identity)
    parts = a.generation.resolve().parts
    relative = '/'.join(parts[parts.index('generation'):]) if 'generation' in parts else None
    save(a.output / 'GENERATION_SOURCE.json', dict(path=str(a.generation.resolve()), work_relative_path=relative))
    stopping = [False]
    for name in ('SIGTERM', 'SIGINT', 'SIGUSR1'):
        if hasattr(signal, name):  # Local Windows has no SIGUSR1.
            signal.signal(getattr(signal, name), lambda *_: stopping.__setitem__(0, True))
    deadline = time.monotonic() + a.seconds
    for row in rows:
        if stopping[0] or time.monotonic() + 125 > deadline:
            print('PAUSED_GRADING_RESUME_SAME_WORK', flush=True)
            return
        folder = a.output / 'scores' / row['record_id']
        final_path = a.generation / 'results' / row['record_id'] / 'RESULT.json'
        if not final_path.exists():
            save(folder / 'STATUS.json', dict(status='generation_not_finished'))
            continue
        final = read(final_path)
        require(final['run_identity'] == digest(generation_identity), 'answer_generation_identity_changed')
        if final['status'] != 'complete':
            save(folder / 'STATUS.json', dict(status=final['status'], issues=final['issues']))
            continue
        require(final['input_sha256'] == row['input_sha256'] and digest(final['answer']) == final['answer_sha256'], 'answer_input_or_content_mismatch')
        if (folder / 'SCORE.json').exists():
            old = read(folder / 'SCORE.json')
            require(old['identity'] == digest(identity) and old['final_sha256'] == sha(final_path), 'score_cache_changed')
            continue
        grades = []
        try:
            for i, rubric in enumerate(row['rubrics']):
                grades.append(grade_one(row['native_prompt'], final['answer'], rubric, config, key,
                    folder / f'rubric-{i+1:03d}', official, deadline=deadline, stopping=stopping,
                    allow_uncertain_retry=a.acknowledge_uncertain_retry))
                print('GRADE ' + row['record_id'] + f' {i+1}/{len(row["rubrics"])}', flush=True)
        except Deferred as exc:
            save(folder / 'STATUS.json', dict(status='deferred', reason=str(exc), completed_rubrics=len(grades)))
            print('DEFERRED ' + row['record_id'], flush=True)
            continue
        score = official['calculate_score']([official['RubricItem'].from_dict(x) for x in row['rubrics']], grades)
        require(isinstance(score, (int, float)), 'official_score_missing')
        save(folder / 'SCORE.json', dict(record_id=row['record_id'], question_id=row['question_id'],
            process_arm=row['process_arm'], variant=generation_identity['variant'], score=score,
            rubric_grades=grades, rubrics=row['rubrics'], final_sha256=sha(final_path), identity=digest(identity),
            official_leaderboard=False, replacement_judge=config['model'], native_HealthBench_rubrics=True))
        save(folder / 'STATUS.json', dict(status='scored'))
    print('GRADING_PASS_COMPLETE; deferred/missing/failures are not scored zero', flush=True)


if __name__ == '__main__':
    main()
