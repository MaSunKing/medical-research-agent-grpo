"""Live-only freshness: exact chunk IDs/text, independent of display metadata."""
import copy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def snapshot(opened):
    from evidence_exact_v31 import pack
    docs, _ = pack(opened)
    pairs = sorted((c['source_id'], c['text']) for d in docs for c in d['chunks'])
    return digest({'schema': 'chunk_id_exact_text_v57', 'evidence': pairs})


def install(module):
    module.snapshot = snapshot

    def status_for_view(opened, status):
        result = copy.deepcopy(status or {'status': 'unverified', 'checklist_stale': True})
        stored = result.get('evidence_snapshot_sha256')
        if stored == snapshot(opened):
            return result
        # Read-only compatibility with the two known historical projections.
        # Never accept an arbitrary mismatch or rewrite the saved audit record.
        from evidence_exact_v31 import pack as exact
        from runtime_handoff_v53 import pack as enriched
        if stored and stored in {digest(exact(opened)[0]), digest(enriched(opened)[0])}:
            result['freshness_compatibility'] = 'verified_legacy_projection_v57'
            return result
        result.update(status='stale_evidence_mismatch', checklist_stale=True)
        return result

    module.status_for_view = status_for_view
