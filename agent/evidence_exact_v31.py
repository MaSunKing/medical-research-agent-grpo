"""Lossless handoff of previously opened chunks; never silently budget evidence."""
import copy
import hashlib
from evidence_contract_v14 import validate_structure_chunk

def pack(opened, max_docs=None, per_doc_chars=None, **kwargs):
    docs = {}
    seen = {}
    selected_ids = set()
    audit = []
    for event in opened or []:
        evidence = event.get('evidence') or {}
        if evidence.get('failed'):
            continue
        for chunk in evidence.get('chunks') or []:
            validate_structure_chunk(chunk)
            sid, text = chunk.get('source_id'), chunk.get('text')
            if not sid or not isinstance(text, str) or not text.strip():
                continue
            if sid in seen and seen[sid] != text:
                raise ValueError('conflicting opened text for chunk ID: ' + sid)
            seen.setdefault(sid, text)
            if sid in selected_ids:
                audit.append(dict(source_id=sid, selected=False, reason='exact_id_text_duplicate'))
                continue
            structure_kind = chunk['structure_kind']
            table_like = structure_kind == 'table_like'
            integrity = chunk.get('table_integrity_verified')
            if table_like and integrity is not True:
                audit.append(dict(
                    source_id=sid, selected=False,
                    original_chars=len(text), selected_chars=0,
                    original_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    structure_kind='table_like',
                    table_integrity_verified=integrity,
                    reason=('incomplete_table_boundary' if chunk.get('boundary_incomplete')
                            else 'unverified_table_integrity'),
                    provenance_retained=True))
                continue
            docs.setdefault(sid.split('#')[0], []).append(copy.deepcopy(chunk))
            selected_ids.add(sid)
            audit.append(dict(source_id=sid, selected=True, original_chars=len(text),
                              selected_chars=len(text), reason='verbatim_opened_text',
                              structure_kind=structure_kind,
                              table_integrity_verified=integrity,
                              provenance_retained=True))
    return [dict(source_id=sid, chunks=chunks) for sid, chunks in docs.items()], audit
