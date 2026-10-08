# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Content-addressed immutable tool receipts for paired Base/SFT evaluation.

This module is evaluation-only.  Normal rollout collection leaves it disabled.
The stored payload is the public model-visible value after redaction/projection;
credentials and process environment are never part of the receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


SCHEMA = "paired_retrieval_receipt_v1"


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class PairedReceiptStore:
    def __init__(self, root, *, release_id, mode="off", wait_seconds=30):
        self.mode = str(mode or "off").lower()
        if self.mode not in {"off", "freeze", "replay"}:
            raise ValueError("paired receipt mode must be off, freeze, or replay")
        self.root = Path(root).resolve() if root else None
        self.release_id = str(release_id or "").strip()
        self.wait_seconds = int(wait_seconds)
        if self.mode != "off":
            if not self.release_id:
                raise ValueError("paired retrieval release identity is required")
            if self.root is None:
                raise ValueError("paired receipt directory is required")
            self.root.mkdir(parents=True, exist_ok=True)

    def request(self, operation, arguments):
        if operation not in {"search", "browse"} or not isinstance(arguments, dict):
            raise ValueError("invalid paired retrieval request")
        return {
            "schema": "paired_retrieval_request_v1",
            "release_id": self.release_id,
            "operation": operation,
            "arguments": arguments,
        }

    def _path(self, key):
        return self.root / key[:2] / (key + ".json")

    def _read(self, path, request, key):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not (
            receipt.get("schema") == SCHEMA
            and receipt.get("request_key") == key
            and receipt.get("request") == request
            and receipt.get("request_sha256") == digest(request)
            and receipt.get("response_sha256") == digest(receipt.get("response"))
        ):
            raise ValueError("paired retrieval receipt identity mismatch")
        return receipt["response"], {
            "request_key": key,
            "response_sha256": receipt["response_sha256"],
            "disposition": "replayed",
        }

    def resolve(self, operation, arguments, execute):
        if self.mode == "off":
            return execute(), None
        request = self.request(operation, arguments)
        key = digest(request)
        path = self._path(key)
        if path.is_file():
            return self._read(path, request, key)
        if self.mode == "replay":
            raise ValueError("paired retrieval receipt missing")

        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        deadline = time.monotonic() + self.wait_seconds
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if path.is_file():
                    return self._read(path, request, key)
                if time.monotonic() >= deadline:
                    raise TimeoutError("paired retrieval receipt lock timeout")
                time.sleep(0.05)
        try:
            os.close(descriptor)
            if path.is_file():
                return self._read(path, request, key)
            response = execute()
            receipt = {
                "schema": SCHEMA,
                "request_key": key,
                "request": request,
                "request_sha256": digest(request),
                "response": response,
                "response_sha256": digest(response),
            }
            temporary = path.with_suffix(".tmp")
            encoded = canonical_bytes(receipt) + b"\n"
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return response, {
                "request_key": key,
                "response_sha256": receipt["response_sha256"],
                "disposition": "recorded",
            }
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
