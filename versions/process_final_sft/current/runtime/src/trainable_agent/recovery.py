"""Explicit recovery operations; no provider/tool calls and no lease stealing."""
import copy
from .storage import canonical

def inspect(store, run_id):
    state = store.load(run_id)
    return {"run_id": run_id, "run_status": (state or {}).get("run_status"),
            "next_node": (state or {}).get("next_node"),
            "receipts": [{"key": k, "status": s} for k, s in store.db.execute("SELECT key,status FROM receipts")],
            "active_or_stale_lease": bool(store.db.execute("SELECT 1 FROM leases WHERE run=?", (run_id,)).fetchone())}

def prepare_resume(store, run_id, retry_failed=()):
    owner = store.acquire(run_id)
    try:
        failures = store.events(run_id, "engineering_failure")
        state = store.load(run_id)
        if not failures or not state or state.get("run_status") != "failed":
            raise ValueError("no_engineering_failure_to_resume")
        rows = dict(store.db.execute("SELECT key,status FROM receipts"))
        if "pending" in rows.values():
            raise ValueError("uncertain_calls_require_receipt_reconciliation_before_resume")
        allowed = set(retry_failed)
        failed = {k for k, status in rows.items() if status == "failed"}
        if allowed != failed:
            raise ValueError("explicit_retry_keys_must_match_definitely_failed_calls")
        restored = copy.deepcopy(failures[-1]["data"]["resume_state"])
        with store.db:
            for key in allowed:
                store.db.execute("INSERT INTO events(run,kind,data) VALUES(?,?,?)",
                    (run_id, "retry_authorized", canonical({"receipt_key": key, "previous_status": "failed"})))
                store.db.execute("UPDATE receipts SET status='retry_ready' WHERE key=?", (key,))
            store.db.execute("INSERT INTO events(run,kind,data) VALUES(?,?,?)",
                (run_id, "resume_prepared", canonical({"restored_node": restored["next_node"]})))
            store.db.execute("INSERT OR REPLACE INTO checkpoint VALUES(?,?)", (run_id, canonical(restored)))
        return restored
    finally:
        store.release(run_id, owner)
