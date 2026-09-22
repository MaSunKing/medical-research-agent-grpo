"""Format/provenance selection only; not a semantic Judge or training approval."""
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def repetition_audit(raw):
    """Detect repeated visible claims without editing or rejecting the output."""
    public = re.sub(r'(?is)<think>.*?</think>', '', str(raw or ''))
    match = re.search(r'(?is)<answer>(.*?)</answer>', public)
    answer = match.group(1) if match else public
    seen = {}
    repeated = []
    for index, paragraph in enumerate(re.split(r'\n\s*\n', answer)):
        visible = re.sub(r'(?is)\s*<cite\b[^>]*>.*?</cite>', '', paragraph)
        visible = re.sub(r'\s+', ' ', visible).strip()
        if len(visible) < 80:
            continue
        value = hashlib.sha256(visible.encode('utf-8')).hexdigest()
        if value in seen:
            repeated.append({'first_index':seen[value], 'repeat_index':index,
                             'visible_text_sha256':value})
        else:
            seen[value] = index
    return {'detector':'citation_insensitive_paragraph_v1',
            'repeated_paragraphs':repeated,
            'low_quality_behavior_observed':bool(repeated)}


def select_final(records, artifact, load_capture, parse_final, maximum=2400):
    def need(ok, message):
        if not ok:
            raise ValueError(message)

    finals = [r for r in records if r.get('stage') == 'final']
    attempts = artifact.get('validation_attempts')
    need(isinstance(attempts, list) and 1 <= len(finals) <= 3
         and len(attempts) == len(finals), 'Final attempt count mismatch')
    ids = artifact.get('opened_ids')
    need(isinstance(ids, list) and all(isinstance(x, str) for x in ids)
         and len(set(ids)) == len(ids), 'opened IDs malformed')
    rows = []
    seen = set()
    for index, (record, attempt) in enumerate(zip(finals, attempts)):
        ref = record['sampling_capture']
        capture = load_capture(ref)
        need(attempt.get('attempt') == index, 'Final attempt order mismatch')
        raw = record['completion']
        count = len(capture['output_ids'])
        need(raw == capture['raw_completion'] == attempt.get('raw_completion'),
             'Final text binding mismatch')
        need(capture['capture_id'] == ref['capture_id']
             and capture['capture_id'] not in seen, 'Final capture reused/mismatched')
        seen.add(capture['capture_id'])
        need(count == ref['output_tokens'] == attempt.get('completion_tokens'),
             'Final token count mismatch')
        need(capture['token_digest'] == digest([capture['input_ids'], capture['output_ids']]),
             'Final token digest mismatch')
        parsed = parse_final(raw, set(ids), count >= maximum)
        valid = parsed['strict_protocol_passed'] is True
        need(attempt.get('strict_protocol_passed') is valid
             and attempt.get('unknown_citation_ids') == parsed['unknown_citation_ids'],
             'Final validation receipt mismatch')
        abnormal = (capture.get('abnormal_termination') is True
                    or capture.get('generation_stop_reason') == 'abnormal_repetition')
        need(not valid or not abnormal, 'Abnormal Final cannot be selected as valid')
        need(index == len(finals)-1 or not valid, 'Unexplained regeneration after valid Final')
        repeat = repetition_audit(raw)
        rows.append(dict(attempt=index, sampling_capture=ref,
            input_ids_sha256=digest(capture['input_ids']),
            output_ids_sha256=digest(capture['output_ids']),
            behavior_logps_sha256=digest(capture['behavior_logps']),
            format_valid=valid, abnormal_termination=abnormal,
            repetition_audit=repeat,
            selection_valid=(valid and not abnormal
                             and not repeat['low_quality_behavior_observed'])))
    last = attempts[-1]
    need(artifact.get('raw_completion') == last['raw_completion']
         and artifact.get('completion_tokens') == last['completion_tokens'],
         'Final artifact does not bind last attempt')
    need(artifact.get('strict_protocol_passed') is rows[-1]['format_valid']
         and artifact.get('hit_token_limit') is (last['completion_tokens'] >= maximum),
         'Final artifact status mismatch')
    selected = rows[-1]['selection_valid']
    return dict(schema='final_attempt_selection_v2', attempts=rows,
        correction_count=len(rows)-1, maximum_corrections=2,
        selected_final_capture=(rows[-1]['sampling_capture'] if selected else None),
        state=('FORMAT_AUDIT_PASSED' if selected else 'NEEDS_ATTENTION'),
        semantic_review='PENDING',
        low_quality_behavior_observed=rows[-1]['repetition_audit']['low_quality_behavior_observed'],
        reward_export_authorized=False, training_ready=False)
