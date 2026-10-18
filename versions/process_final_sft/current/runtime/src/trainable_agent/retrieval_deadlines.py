"""Bounded retrieval work and durable, credential-free operation receipts.

The deadline is local to the worker clock. A transport timeout is not proof
that a remote PDF task failed; its saved identity must be reconciled first.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time

_deadline = ContextVar("retrieval_operation_deadline", default=None)
_progress = ContextVar("retrieval_operation_progress", default=None)


class RetrievalDeadlineExceeded(BaseException):
    """A worker-owned cancellation signal, not an arbitrary tool bug.

    Legacy content routes catch Exception/OSError to try alternate sources.
    They must not swallow this operation deadline. Only the worker catches it.
    """


class RetrievalIntegrityError(BaseException):
    """Worker-only local receipt/identity fault; content fallbacks cannot hide it."""
    def __init__(self, code):
        import re
        self.error_code = code if re.fullmatch(r"[a-z_]{1,100}", code) else "retrieval_integrity_fault"
        super().__init__(self.error_code)


def bounded_timeout(seconds):
    deadline = _deadline.get()
    if deadline is None:
        return seconds
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RetrievalDeadlineExceeded()
    return max(.001, min(float(seconds), remaining))


def operation_progress(stage, **details):
    callback = _progress.get()
    if callback:
        allowed = {k: v for k, v in details.items() if k in {
            "task_id", "task_state", "pdf_sha256", "resume_identity_sha256",
            "task_ledger", "model_version", "adapter_version"}}
        callback(stage, allowed)


@contextmanager
def operation_deadline(seconds, progress=None):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid_operation_deadline")
    deadline_token = _deadline.set(time.monotonic() + seconds)
    progress_token = _progress.set(progress)
    alarm = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    prior_handler = prior_timer = None
    if alarm:
        prior_handler = signal.getsignal(signal.SIGALRM)
        prior_timer = signal.getitimer(signal.ITIMER_REAL)
        def expired(*_):
            raise RetrievalDeadlineExceeded()
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        if alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior_handler)
            if prior_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *prior_timer)
        _deadline.reset(deadline_token)
        _progress.reset(progress_token)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def receipt_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class OperationJournal:
    def __init__(self, root, operation_id, operation, arguments):
        # Caller-generated ID is a hash, not a path supplied by source material.
        import re
        if not re.fullmatch(r"[a-f0-9]{64}", operation_id):
            raise ValueError("invalid_worker_operation_id")
        self.root = Path(root).resolve()
        self.folder = self.root / "worker_operations" / operation_id
        self.identity = dict(operation=operation, arguments_sha256=receipt_digest(arguments))
        self.state = dict(operation_id=operation_id, identity=self.identity, state="started",
                          stage="dispatch", outcome="unknown")

    def progress(self, stage, details):
        self.state.update(stage=stage, progress={**self.state.get("progress", {}), **details})
        atomic_json(self.folder / "state.json", self.state)
        with (self.folder / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(stage=stage, details=details, monotonic=time.monotonic())) + "\n")

    def start(self):
        self.state["attempt_started_monotonic"] = time.monotonic()
        atomic_json(self.folder / "state.json", self.state)

    def finish(self, value):
        atomic_json(self.folder / "result.json", value)
        self.state.update(state="complete", outcome="returned", result_sha256=receipt_digest(value))
        atomic_json(self.folder / "state.json", self.state)


def reconcile_operation(root, operation_id, operation, arguments):
    """Return verified completed value or a safely resumable *existing* PDF task."""
    root = Path(root).resolve()
    folder = root / "worker_operations" / operation_id
    path = folder / "state.json"
    if not path.is_file():
        return None, "unknown"
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("operation_id") != operation_id or row.get("identity") != dict(
            operation=operation, arguments_sha256=receipt_digest(arguments)):
        raise ValueError("worker_journal_identity_mismatch")
    if row.get("state") == "complete":
        value = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if receipt_digest(value) != row.get("result_sha256"):
            raise ValueError("worker_journal_result_hash_mismatch")
        return value, "completed_receipt"
    progress = row.get("progress", {})
    ledger_name = progress.get("task_ledger")
    if not ledger_name:
        return None, "unknown"
    ledger = (root / ledger_name).resolve()
    if not ledger.is_relative_to(root / "mineru_cache") or not ledger.is_file():
        raise ValueError("unsafe_worker_task_ledger")
    task = json.loads(ledger.read_text(encoding="utf-8"))
    identity = task.get("request_identity", {})
    if (task.get("task_id") != progress.get("task_id")
            or receipt_digest(identity) != progress.get("resume_identity_sha256")
            or identity.get("pdf_sha256") != progress.get("pdf_sha256")):
        raise ValueError("worker_task_identity_mismatch")
    # Pending/running does not imply the cloud task is still queued right now.
    if task.get("state") == "pending" and task.get("task_id"):
        return None, "resume_same_task"
    return None, "unknown"


def timeout_value(operation_id, phase, resume_mode):
    return dict(failed=True, failure_type="environment_failure",
                error_code="retrieval_operation_timeout", retryable=resume_mode == "resume_same_task",
                retry_after_seconds=5.0 if resume_mode == "resume_same_task" else 0.0,
                execution=dict(outcome="unknown", phase=phase, receipt_id=operation_id,
                               resume_mode=resume_mode), data=[])
