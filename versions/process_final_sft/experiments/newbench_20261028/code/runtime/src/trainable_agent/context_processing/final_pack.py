"""Reversible State/Final factoring of repeated engineering feedback, never evidence."""
import copy
from collections import Counter
from ..storage import canonical

NOTICE = (
    "Final-only lossless packing: a shared_feedback_ref resolves to the exact value in "
    "history_shared_feedback; an unresolved state_ref resolves to that requirement's full "
    "entry in state, including gap and conflict. Read referenced values as present, not omitted. "
    "All actions, per-evidence scope assignments and warnings remain; full originals are archived."
)
STATE_NOTICE = (
    "Lossless history packing: shared_feedback_ref resolves to the exact value in "
    "history_shared_feedback. Read each referenced scope/limitation as present for that action. "
    "All actions, E-ID assignments, gaps, conflicts and original evidence remain; originals are archived."
)


def pack_final(payload):
    return _pack(payload, 'compact_history', NOTICE, 'final_pack_v1')


def pack_state(payload):
    return _pack(payload, 'history', STATE_NOTICE, 'state_pack_v1')


def _pack(payload, history_key, notice, version):
    """Factor exact repeats only; unknown history shapes and unequal State entries stay."""
    original = copy.deepcopy(payload)
    if any(k in payload for k in ('history_shared_feedback', 'final_packing_notice')):
        return original, {'version': version, 'applied': False}
    def reserved(value):
        if isinstance(value, dict):
            return bool({'shared_feedback_ref', 'state_ref'} & set(value)) or any(reserved(v) for v in value.values())
        return isinstance(value, list) and any(reserved(v) for v in value)
    if reserved(payload):
        return original, {'version': version, 'applied': False}
    result = copy.deepcopy(payload)
    history = result.get(history_key, [])
    if not isinstance(history, list):
        return original, {'version': version, 'applied': False}
    slots = []
    for entry in history:
        if not isinstance(entry, dict) or not isinstance(entry.get('feedback'), dict):
            continue
        feedback = entry['feedback']
        if isinstance(feedback.get('retrieved_scope_message'), str):
            slots.append((feedback, 'retrieved_scope_message'))
        if isinstance(feedback.get('content_scope_message'), str):
            slots.append((feedback, 'content_scope_message'))
        scopes = feedback.get('retrieved_content_scopes', [])
        if isinstance(scopes, list):
            for row in scopes:
                if isinstance(row, dict) and isinstance(row.get('scope'), dict):
                    slots.append((row, 'scope'))
    counts = Counter(canonical(parent[key]) for parent, key in slots)
    pool, names = {}, {}
    for parent, key in slots:
        encoded = canonical(parent[key])
        # Short values are not worth the reference overhead.
        if counts[encoded] < 2 or len(encoded) < 120:
            continue
        if encoded not in names:
            name = 'F' + str(len(names) + 1)
            names[encoded] = name
            pool[name] = copy.deepcopy(parent[key])
        parent[key] = {'shared_feedback_ref': names[encoded]}
    references = []
    if isinstance(result.get('state'), dict) and isinstance(result.get('unresolved'), dict):
        for rid, value in result['unresolved'].items():
            if rid in result['state'] and value == result['state'][rid] and isinstance(value, dict):
                # Avoid colliding with an upstream format we do not own.
                if 'state_ref' not in value:
                    result['unresolved'][rid] = {'state_ref': rid}
                    references.append(rid)
    if pool:
        result['history_shared_feedback'] = pool
    if pool or references:
        result['final_packing_notice'] = notice
    if len(canonical(result)) >= len(canonical(original)):
        return original, {'version': version, 'applied': False}
    assert _unpack(result, history_key, notice) == original, 'history_pack_not_reversible'
    return result, dict(version=version, applied=True,
                        shared_feedback_count=len(pool), unresolved_state_refs=references)


def unpack_final(payload):
    return _unpack(payload, 'compact_history', NOTICE)


def unpack_state(payload):
    return _unpack(payload, 'history', STATE_NOTICE)


def _unpack(payload, history_key, notice):
    """Witness decoder for verification; clinical text is never regenerated."""
    result = copy.deepcopy(payload)
    if result.get('final_packing_notice') != notice:
        return result
    pool = result.pop('history_shared_feedback', {})
    for entry in result.get(history_key, []):
        if not isinstance(entry, dict) or not isinstance(entry.get('feedback'), dict):
            continue
        feedback = entry['feedback']
        value = feedback.get('retrieved_scope_message')
        if isinstance(value, dict) and set(value) == {'shared_feedback_ref'}:
            feedback['retrieved_scope_message'] = copy.deepcopy(pool[value['shared_feedback_ref']])
        value = feedback.get('content_scope_message')
        if isinstance(value, dict) and set(value) == {'shared_feedback_ref'}:
            feedback['content_scope_message'] = copy.deepcopy(pool[value['shared_feedback_ref']])
        scopes = feedback.get('retrieved_content_scopes', [])
        for row in scopes if isinstance(scopes, list) else []:
            if isinstance(row, dict):
                value = row.get('scope')
                if isinstance(value, dict) and set(value) == {'shared_feedback_ref'}:
                    row['scope'] = copy.deepcopy(pool[value['shared_feedback_ref']])
    unresolved = result.get('unresolved', {})
    for rid, value in unresolved.items() if isinstance(unresolved, dict) else []:
        if isinstance(value, dict) and value == {'state_ref': rid}:
            result['unresolved'][rid] = copy.deepcopy(result['state'][rid])
    result.pop('final_packing_notice')
    return result
