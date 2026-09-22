# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Version actual returned evidence, not the reusable pre-budget chunk slot."""
import hashlib
import json


def bind_returned_window(chunk):
    parent = chunk.get('canonical_chunk_id') or chunk['chunk_id']
    text = chunk['text']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Cannot identify empty evidence')
    identity = dict(version='returned_window_identity_v1', canonical_chunk_id=parent,
        start_char=chunk['start_char'], end_char=chunk['end_char'],
        text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    if identity['end_char'] - identity['start_char'] != len(text):
        raise ValueError('Returned evidence offsets do not match exact text')
    value = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    result = dict(chunk, canonical_chunk_id=parent,
        chunk_id=parent+'-v'+value[:16], window_identity=identity,
        window_identity_sha256=value)
    if 'source_id' in result:
        result['source_id'] = result['chunk_id']
    return result


def patch_handoff(source):
    anchor = '        out.append(aligned)'
    if source.count(anchor) != 1:
        raise ValueError('Unexpected evidence handoff; refuse identity patch')
    replacement = ('        from .returned_window_identity import bind_returned_window\n'
                   '        aligned = bind_returned_window(aligned)\n'
                   "        audit[-1]['returned_chunk_id'] = aligned['chunk_id']\n"
                   "        audit[-1]['window_identity_sha256'] = aligned['window_identity_sha256']\n"
                   '        out.append(aligned)')
    updated = source.replace(anchor, replacement)
    compile(updated, 'evidence_handoff_v3.py', 'exec')
    return updated
