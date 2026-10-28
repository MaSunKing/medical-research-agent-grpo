"""Transactional local records. Pending external calls are never silently retried."""
import json
import sqlite3
import hashlib
import uuid
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class CallFailure(RuntimeError):
    execution_outcome = "unknown"
    def __init__(self, code):
        import re
        self.error_code = code if re.fullmatch(r"(?:codex_cli_[a-z_]+|model_http_status:[0-9]{3})", code) else "provider_failure"
        super().__init__(self.error_code)


class DefiniteCallFailure(CallFailure):
    execution_outcome = "failed"


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS checkpoint(run TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run TEXT, kind TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS leases(run TEXT PRIMARY KEY, owner TEXT NOT NULL);
        """)

    def event(self, run, kind, value):
        with self.db:
            self.db.execute("INSERT INTO events(run,kind,data) VALUES(?,?,?)", (run, kind, canonical(value)))

    def events(self, run, kind=None):
        sql = "SELECT kind,data FROM events WHERE run=?"
        args = [run]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return [{"kind": k, "data": json.loads(v)} for k, v in self.db.execute(sql + " ORDER BY seq", args)]

    def transition(self, state, node):
        with self.db:
            self.db.execute("INSERT INTO events(run,kind,data) VALUES(?,?,?)",
                (state["run_id"], "transition", canonical({"node": node, "state": state})))
            self.db.execute("INSERT OR REPLACE INTO checkpoint VALUES(?,?)",
                            (state["run_id"], canonical(state)))

    def checkpoint(self, state):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO checkpoint VALUES(?,?)", (state["run_id"], canonical(state)))

    def load(self, run):
        row = self.db.execute("SELECT data FROM checkpoint WHERE run=?", (run,)).fetchone()
        return json.loads(row[0]) if row else None

    def acquire(self, run):
        owner = uuid.uuid4().hex
        try:
            with self.db:
                self.db.execute("INSERT INTO leases VALUES(?,?)", (run, owner))
        except sqlite3.IntegrityError:
            raise RuntimeError("run_active_or_crashed_requires_manual_lease_reconciliation") from None
        return owner

    def release(self, run, owner):
        with self.db:
            self.db.execute("DELETE FROM leases WHERE run=? AND owner=?", (run, owner))

    async def once(self, key, function):
        row = self.db.execute("SELECT status,data FROM receipts WHERE key=?", (key,)).fetchone()
        if row:
            if row[0] == "complete":
                return json.loads(row[1])
            if row[0] != "retry_ready":
                raise RuntimeError("ambiguous_pending_call_requires_manual_reconciliation")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO receipts VALUES(?,'pending',NULL)", (key,))
        try:
            value = await function()
        except Exception as exc:
            status = "failed" if getattr(exc, "execution_outcome", None) == "failed" else "pending"
            # Exception messages may contain credentials: persist type/outcome only.
            failure = {"exception_type": type(exc).__name__, "outcome": status,
                       "error_code": getattr(exc, "error_code", "unclassified_call_failure")}
            with self.db:
                self.db.execute("UPDATE receipts SET status=?,data=? WHERE key=?", (status, canonical(failure), key))
                self.db.execute("INSERT INTO events(run,kind,data) VALUES(?,?,?)",
                    ("_receipt", "call_failure", canonical({"receipt_key": key, **failure})))
            raise
        with self.db:
            self.db.execute("UPDATE receipts SET status='complete',data=? WHERE key=?", (canonical(value), key))
        return value
