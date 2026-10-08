"""Lossless handoff of previously opened chunks; never silently budget evidence."""
import copy

def pack(opened, max_docs=None, per_doc_chars=None, **kwargs):
    docs = {}
    seen = {}
    audit = []
    for event in opened or []:
        evidence = event.get('evidence') or {}
        if evidence.get('failed'):
            continue
        for chunk in evidence.get('chunks') or []:
            sid, text = chunk.get('source_id'), chunk.get('text')
            if not sid or not isinstance(text, str) or not text.strip():
                continue
            if sid in seen:
                if seen[sid] != text:
                    raise ValueError('conflicting opened text for chunk ID: ' + sid)
                audit.append(dict(source_id=sid, selected=False, reason='exact_id_text_duplicate'))
                continue
            seen[sid] = text
            docs.setdefault(sid.split('#')[0], []).append(copy.deepcopy(chunk))
            audit.append(dict(source_id=sid, selected=True, original_chars=len(text),
                              selected_chars=len(text), reason='verbatim_opened_text'))
    return [dict(source_id=sid, chunks=chunks) for sid, chunks in docs.items()], audit
