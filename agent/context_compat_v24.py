"""Inference-only context presentation; never executes or fabricates tool actions."""
import json
import re


def render_input(tokenizer, messages, assistant_prefix=None, template_options=None):
    if assistant_prefix is not None:
        if not isinstance(assistant_prefix, str) or not messages or messages[-1]['role'] != 'user':
            raise ValueError('assistant prefix requires final user message')
        if any(t in assistant_prefix for t in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
            raise ValueError('chat control token in assistant prefix')
    # Append verbatim AFTER the template. Qwen's assistant-message rendering can
    # strip previous think sections; a full trajectory prefix must not be lost.
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, **(template_options or {}))
    return rendered + (assistant_prefix or '')


def public_action(text):
    text = str(text)
    if re.search(r'<think>', text, re.I) and not re.search(r'</think>', text, re.I):
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.I | re.S).strip()


def decision_complete(text):
    # Share execution-parser handoff rules, including orphan closing-think.
    # Lazy imports avoid the public_action import cycle.
    from stop_state_v29 import stop_action
    from stop_prefix_v40 import recover
    if stop_action(text) is not None or recover(text) is not None:return True
    from action_boundary_v46 import normalize
    if normalize(text) is not None:return True
    public = public_action(text)
    return re.fullmatch(r'<call_tool\s+[^>]+>.*?</call_tool>', public, re.S) is not None


def classify(text, candidates):
    public = public_action(text)
    if public == 'FINAL_READY' or public.startswith('<answer>'):
        return {'action': 'stop', 'valid': True, 'handoff': 'separate_final',
                'answer_adopted': False}
    match = re.fullmatch(r'<call_tool\s+name="([^"]+)"([^>]*)>(.*?)</call_tool>', public, re.S)
    if not match:
        return {'action': 'unparsed', 'valid': False}
    tool, attrs, body = match.groups()
    body = body.strip()
    if tool in ('pubmed_search', 'medical_web_search'):
        return {'action': 'search', 'tool': tool, 'body': body, 'valid': bool(body), 'attrs': attrs}
    candidate = next((r for r in candidates if r['source_id'] == body), None)
    expected = 'browse_document' if body.startswith(('PMID:', 'S2:')) else 'browse_webpage'
    return {'action': 'browse' if tool in ('browse_document', 'browse_webpage') else 'unknown',
            'tool': tool, 'body': body, 'valid': bool(candidate) and tool == expected, 'attrs': attrs}


def envelope(value):
    return '<tool_output>\n' + json.dumps(value, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e') + '\n</tool_output>\n'


def matched_contexts(saved_prompt, previous_completions):
    """Accepted actions only; no new/future tool results or teacher labels.

    Both arms contain byte-identical HELP, question, checklist, candidates and
    compact executed observations. History is NOT expanded to full abstracts.
    Fail closed rather than silently treating retry/failed actions as executed.
    """
    help_text, tail = saved_prompt.split('\n\nQuestion: ', 1)
    question_line, tail = tail.split('\n\n', 1)
    checklist, rest = tail.split('\n\nExecuted tool history ', 1)
    hist_raw = rest.split('<tool_output id="history">\n', 1)[1]
    history = json.JSONDecoder().raw_decode(hist_raw)[0]
    current_raw = rest.split('<tool_output id="current_view">\n', 1)[1]
    current = json.JSONDecoder().raw_decode(current_raw)[0]
    if len(history) != len(previous_completions):
        raise ValueError('history/action length mismatch')
    prefix = ''
    for completion, event in zip(previous_completions, history):
        action = classify(completion, [])
        if action.get('tool') != event['tool']:
            raise ValueError('accepted tool differs from executed observation')
        expected_body = event.get('query') if action['action']=='search' else event.get('source_id')
        if action.get('body') != expected_body or action['action'] not in ('search','browse'):
            raise ValueError('accepted action differs from executed observation')
        prefix += completion.strip() + '\n' + envelope(event)
    prefix += envelope({'runtime_working_memory': checklist, 'current_view': current})
    suffix=current_raw.split('</tool_output>',1)[1].strip()
    if suffix:
        # Existing validation feedback is correction data, not an executed result.
        prefix+='\n'+suffix+'\n'
    question = 'Question: ' + question_line
    # Same literal content in both arms; only chat-role boundaries differ.
    return {
        'matched_user': {'messages': [{'role': 'user', 'content': help_text+'\n\n'+question+'\n\n'+prefix}], 'assistant_prefix': None},
        'assistant_continuation': {'messages': [{'role': 'system', 'content': help_text}, {'role': 'user', 'content': question}], 'assistant_prefix': prefix},
    }, current


def install_live(collection):
    """Per-run accepted-action ledger. State generation remains untouched."""
    accepted=[]
    sample=collection.sample_runtime_action
    parse=collection.parse_agent_action
    def sample_action(*args,**kwargs):
        result=sample(*args,**kwargs)
        if result['action_type']!='final_ready':accepted.append(result.get('execution_completion',result['completion']))
        return result
    def parse_action(raw,candidates,v71):
        if public_action(raw).startswith('<answer>'):
            return dict(action_type='final_ready',completion=raw,
                action_completion='FINAL_READY',legacy_answer_handoff=True,answer_adopted=False)
        return parse(raw,candidates,v71)
    collection.sample_runtime_action=sample_action
    collection.parse_agent_action=parse_action
    return accepted
