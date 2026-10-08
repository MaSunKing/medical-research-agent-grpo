"""Public execution facts, not evidence coverage or reward judgments."""
import copy
import functools
import html
import json
import re

OUTPUT = re.compile(r'<tool_output\b[^>]*>(.*?)</tool_output>', re.S)
CALL = re.compile(r'<call_tool\s+name=["\'](?P<tool>[^"\']+)["\'](?P<attrs>[^>]*)>(?P<body>.*?)</call_tool>', re.S)
ATTR = re.compile(r'(\w+)\s*=\s*(["\'])(.*?)\2', re.S)
SEARCH = {'pubmed_search', 'medical_web_search'}


def result_facts(output):
    if not isinstance(output, dict):
        raise ValueError('Search output must be an object')
    if 'data' not in output:
        return {'returned_ids': None, 'result_count': None,
                'result_observability': 'not_recorded'}
    rows = output['data']
    if not isinstance(rows, list):
        raise ValueError('Search data must be a list, not an implicit empty result')
    ids = [row['source_id'] for row in rows
           if isinstance(row, dict) and isinstance(row.get('source_id'), str)
           and row['source_id']]
    return {'returned_ids': ids, 'result_count': len(rows),
            'result_observability': 'observed'}


class SearchHistoryLedger:
    """One instance per runtime arm; observe, never repeat, actual calls."""
    def __init__(self):
        self.events = []

    def record(self, output, arguments):
        facts = result_facts(output)
        self.events.append({'tool': arguments['tool'], 'query': arguments['query'],
                            **facts})
        return output

    def enrich(self, history):
        out = copy.deepcopy(history)
        index = 0
        for event in out:
            if event.get('tool') not in SEARCH:
                continue
            # Compatibility/rejection feedback is never another execution.
            if event.get('executed') is False:
                continue
            if index >= len(self.events):
                raise ValueError('Search history has no matching actual execution')
            actual = self.events[index]
            index += 1
            if (event.get('tool'), event.get('query')) != (actual['tool'], actual['query']):
                raise ValueError('Search history tool/query differs from actual execution')
            for key in ('returned_ids', 'result_count', 'result_observability'):
                if key in event and event[key] != actual[key]:
                    raise ValueError('Search history conflicts with actual result facts')
                event[key] = copy.deepcopy(actual[key])
        if index != len(self.events):
            raise ValueError('Actual Search execution missing from public history')
        return out

    def install(self, collection):
        previous = collection.policy_prompt
        @functools.wraps(previous)
        def prompt(**kwargs):
            if kwargs.get('role') == 'decision':
                kwargs['interface_history'] = self.enrich(kwargs.get('interface_history', []))
            return previous(**kwargs)
        collection.policy_prompt = prompt


def summarize_history(prefix, view):
    observations = list(OUTPUT.finditer(prefix))
    calls = [m for m in CALL.finditer(prefix)
             if not any(o.start() <= m.start() < o.end() for o in observations)]
    history = []
    for index, call in enumerate(calls):
        boundary = calls[index + 1].start() if index + 1 < len(calls) else len(prefix)
        observation = next((o for o in observations
                            if call.end() <= o.start() < boundary), None)
        if observation is None:
            continue
        try:
            body = json.loads(observation[1])
        except ValueError:
            continue
        if not isinstance(body, dict):
            raise ValueError('Executed observation must be an object')
        if 'current_view' in body or 'runtime_working_memory' in body:
            continue
        attrs = {k: html.unescape(v) for k, _, v in ATTR.findall(call['attrs'])}
        argument = html.unescape(call['body']).strip()
        tool = call['tool']
        query = argument if tool in SEARCH else attrs.get('query')
        if body.get('tool') is not None and body['tool'] != tool:
            raise ValueError('Observation tool differs from captured action')
        if body.get('query') is not None and body['query'] != query:
            raise ValueError('Observation query differs from captured action')
        feedback = body.get('runtime_feedback')
        executed = not (body.get('executed') is False or
                        isinstance(feedback, dict) and feedback.get('executed') is False)
        record = {'tool': tool, 'argument': argument, 'query': query, 'executed': executed}
        if executed and 'returned_ids' in body and 'result_count' in body:
            ids, count = body['returned_ids'], body['result_count']
            if ids is None and count is None:
                facts = {'returned_ids': None, 'result_count': None,
                         'result_observability': 'not_recorded'}
            else:
                if not (isinstance(ids, list) and all(isinstance(x, str) and x for x in ids)
                        and type(count) is int and count >= len(ids)):
                    raise ValueError('Malformed explicit execution result facts')
                facts = {'returned_ids': list(ids), 'result_count': count,
                         'result_observability': 'observed'}
        elif executed:
            facts = result_facts(body)
        else:
            facts = {'returned_ids': None, 'result_count': None,
                     'result_observability': 'not_executed'}
        record.update(facts)
        if 'new_source_ids' in body:
            record['new_candidate_ids'] = copy.deepcopy(body['new_source_ids'])
        record['failed'] = bool(body.get('failed') or body.get('error') or body.get('environment_failure'))
        if body.get('error'):
            record['error'] = body['error']
        history.append(record)
    return history
