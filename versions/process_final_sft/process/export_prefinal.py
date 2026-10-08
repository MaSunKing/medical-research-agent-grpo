"""Export the real shared Final input, without generating a Final or loading weights."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(live, trajectory, records, tokenizer, output, *, allow_resident_cards=False,
           source_cache=None, reserve=2400, citation_format='original'):
    """Same renderer/history/packing as Runtime; target and audit notes never enter messages."""
    live, output = Path(live).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('new_export_directory_required')
    output.mkdir(parents=True)
    os.environ.update(V42_LIVE='1', MEDGAP_CONTEXT_LIMIT='10240')
    sys.path.insert(0, str(live))
    alignment = load('prefinal_shared_alignment', live/'alignment_v4_live.py')
    handoff = load('prefinal_evidence_handoff', live/'evidence_handoff_v3.py')
    from evidence_freshness_v57 import install as install_freshness
    install_freshness(handoff)
    from context_compat_v24 import render_input
    from process_history_v1 import rendered
    raw, history = copy.deepcopy(trajectory), copy.deepcopy(records)
    if not isinstance(raw.get('question'), str) or not raw['question'].strip():
        raise ValueError('missing_original_question')
    state = raw.get('policy_visible_state') or raw.get('final_policy_state')
    if not isinstance(state, dict):
        raise ValueError('missing_actual_state')
    if any(r.get('stage') == 'final' for r in history):
        raise ValueError('future_final_in_process_history')
    cache = output/'budget_cache'
    if source_cache and Path(source_cache).is_dir():
        cache.mkdir()
        for p in Path(source_cache).glob('*.card.json'):
            if p.is_symlink():
                raise ValueError('symlink_in_card_cache')
            shutil.copyfile(p, cache/p.name)
    auxiliary = []

    def cards_only(messages, **kwargs):
        if not allow_resident_cards:
            # RuntimeError intentionally cannot be treated as a failed model card.
            raise RuntimeError('UNCACHED_EVIDENCE_CARD_REQUIRES_MODEL; offline export stopped')
        import resident_sft
        kwargs['temperature'] = 1.0
        result = resident_sft.request_generation(messages, **kwargs)
        auxiliary.append({'purpose': 'existing_evidence_span_extraction',
                          'final_generated': False, 'sampling_capture': result.get('sampling_capture')})
        return result

    messages = [{'role': 'user', 'content': handoff.final_prompt(raw)}]
    prepared = alignment.prepare_shared(messages, tokenizer=tokenizer, max_tokens=reserve,
        cache=cache, generate=cards_only, assistant_prefix=None, json_mode=False,
        final_boundary=True, context_limit=10240, history_records=history)
    if prepared is None or prepared['obj']['stage'] != 'final':
        raise ValueError('final_stage_not_recognized')
    obj = prepared['obj']
    build = alignment.load_contract()
    options = build.template_options('final')
    text = render_input(tokenizer, obj['messages'], obj['assistant_prefix'], options)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if text != rendered(tokenizer, dict(stage='final', messages=obj['messages'],
            assistant_prefix=obj['assistant_prefix'], template_options=options)):
        raise ValueError('runtime_sft_template_mismatch')
    if len(input_ids)+reserve > 10240:
        raise ValueError('final_context_overflow')
    visible = json.loads(obj['messages'][-1]['content'])
    if visible.get('termination') != raw.get('termination'):
        raise ValueError('termination_drift')
    if visible.get('checklist_update') != handoff.status_for_view(
            raw.get('opened_evidence'), raw.get('checklist_update')):
        raise ValueError('runtime_freshness_drift')
    expected_state = {k: state[k] for k in ('requirements', 'optional_requirements') if k in state}
    if visible['question'] != raw['question'] or visible['policy_evidence_state'] != expected_state:
        raise ValueError('question_or_latest_state_drift')
    original = {c['source_id']: c for d in raw.get('opened_evidence', [])
                for c in (d.get('evidence') or d).get('chunks', [])}
    citable, omitted = [], []
    for c in visible['opened_evidence']:
        if c['source_id'] not in original:
            raise ValueError('evidence_without_original_receipt')
        if c.get('final_budget_receipt') or c.get('decision_budget_receipt'):
            omitted.append(c['source_id'])
        else:
            citable.append(c['source_id'])
            if not c.get('model_extraction') and c['text'] != original[c['source_id']]['text']:
                raise ValueError('unexplained_evidence_text_change')
    past_digest = hashlib.sha256(json.dumps(history, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    final_input = {'schema_version': 'actual_final_input_v1', 'question_id': raw['question_id'],
        'question': raw['question'], 'messages': obj['messages'],
        'assistant_prefix': obj['assistant_prefix'], 'template_options': options,
        'input_ids': input_ids, 'input_tokens': len(input_ids),
        'reserved_output_tokens': reserve, 'context_tokens': 10240,
        'rendered_input_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'citable_evidence_ids': sorted(set(citable)), 'noncitable_receipt_ids': sorted(set(omitted)),
        'history_audit': obj.get('process_history_audit'), 'budget_audit': prepared['audit'],
        'source_history_sha256': past_digest, 'final_generated': False,
        'semantic_review': 'PENDING'}
    if citation_format == 'short':
        from final_citation_alias_v1 import convert_input
        final_input, text = convert_input(final_input, tokenizer)
        save(output/'CITATION_MAP.json', final_input['citation_mapping'])
    elif citation_format != 'original':
        raise ValueError('unknown_citation_format')
    save(output/'FINAL_INPUT.json', final_input)
    (output/'FINAL_INPUT.txt').write_text(text, encoding='utf-8')
    # This separate ledger is not inserted into model-visible messages.
    save(output/'PRE_FINAL_PACKAGE.json', {'question_id': raw['question_id'],
        'question': raw['question'], 'latest_state': state, 'termination': raw.get('termination'),
        'opened_evidence': raw.get('opened_evidence', []), 'actual_trajectory': raw,
        'actual_stage_history': history, 'final_input_sha256': sha(output/'FINAL_INPUT.json'),
        'final_generated': False, 'semantic_review': 'PENDING'})
    save(output/'EXPORT_AUDIT.json', {'status': 'passed', 'model_weights_loaded_by_exporter': False,
        'final_generated': False, 'retrieval_api_calls': 0, 'training': False,
        'resident_card_calls': len(auxiliary), 'auxiliary_captures': auxiliary,
        'source_modules': {n: sha(live/n) for n in ('alignment_v4_live.py', 'process_history_v1.py',
            'shared_budget_v52.py', 'evidence_handoff_v3.py', 'evidence_freshness_v57.py',
            'evidence_exact_v31.py')},
        'final_input_sha256': sha(output/'FINAL_INPUT.json'),
        'input_plus_reserve': final_input['input_tokens']+reserve,
        'citation_format': citation_format,
        'history_audit': obj.get('process_history_audit'), 'semantic_review': 'PENDING'})
    if raw != trajectory or history != records:
        raise ValueError('source_modified_in_export')
    return final_input


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--allow-resident-cards', action='store_true')
    p.add_argument('--citation-format', choices=('original', 'short'), default='original')
    a = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    data = export(a.live, read(a.run/'retrieval/preview_relaxed.jsonl'),
                  read(a.run/'retrieval/alignment_v4_requests.json'), tok,
                  a.output or a.run/'prefinal', allow_resident_cards=a.allow_resident_cards,
                  source_cache=a.run/'retrieval/shared_evidence_cards', citation_format=a.citation_format)
    export_dir = a.output or a.run/'prefinal'
    report = read(export_dir/'EXPORT_AUDIT.json')
    report['tokenizer_files'] = {n: sha(a.tokenizer/n) for n in
        ('tokenizer_config.json', 'tokenizer.json', 'merges.txt', 'vocab.json', 'config.json')
        if (a.tokenizer/n).is_file()}
    save(export_dir/'EXPORT_AUDIT.json', report)
    print('ACTUAL_FINAL_INPUT_EXPORT=passed; tokens='+str(data['input_tokens'])+
          '; Final=0; weights_loaded_by_exporter=0', flush=True)


if __name__ == '__main__':
    main()
