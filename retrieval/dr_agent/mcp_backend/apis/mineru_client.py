# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Optional MinerU PDF adapter for the V28 retrieval backend."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import requests


MINERU_ADAPTER_VERSION = "medgap_mineru_pdf_v28"


def _find_markdown(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("markdown", "md_content", "content", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for candidate in value.values():
            found = _find_markdown(candidate)
            if found:
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _find_markdown(candidate)
            if found:
                return found
    return ""


class MinerUClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = str(base_url or os.getenv("MEDGAP_MINERU_BASE_URL") or "").rstrip("/")
        self.timeout = float(os.getenv("MEDGAP_MINERU_TIMEOUT", "300"))
        self.max_bytes = int(os.getenv("MEDGAP_MINERU_MAX_PDF_BYTES", str(50 * 1024 * 1024)))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def parse_url(self, url: str) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("MinerU is not configured")
        pdf = requests.get(
            url,
            headers={"User-Agent": "DR-Tulu-Medical-Agent/1.0"},
            timeout=min(self.timeout, 60),
        )
        pdf.raise_for_status()
        if len(pdf.content) > self.max_bytes:
            raise ValueError("PDF exceeds MEDGAP_MINERU_MAX_PDF_BYTES")
        if not (pdf.content.startswith(b"%PDF") or "pdf" in pdf.headers.get("content-type", "").lower()):
            raise ValueError("MinerU input is not a PDF")
        response = requests.post(
            self.base_url + "/file_parse",
            files={"files": ("document.pdf", pdf.content, "application/pdf")},
            data={"return_md": "true"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        markdown = _find_markdown(payload)
        task_id = str(payload.get("task_id") or payload.get("id") or "") if isinstance(payload, dict) else ""
        deadline = time.monotonic() + self.timeout
        while not markdown and task_id and time.monotonic() < deadline:
            time.sleep(1)
            status = requests.get(
                self.base_url + f"/tasks/{task_id}/result", timeout=min(30, self.timeout)
            )
            status.raise_for_status()
            payload = status.json()
            markdown = _find_markdown(payload)
            state = str(payload.get("status") or payload.get("state") or "").lower()
            if state in {"failed", "error"}:
                raise RuntimeError(f"MinerU task failed: {state}")
        if not markdown:
            raise RuntimeError("MinerU returned no markdown")
        return markdown, {
            "adapter_version": MINERU_ADAPTER_VERSION,
            "source_url": re.sub(r"[?#].*$", "", url),
            "bytes": len(pdf.content),
        }


_CLIENT: MinerUClient | None = None


def get_mineru_client() -> MinerUClient:
    from .pdf_policy_v54 import require_pdf_enabled
    require_pdf_enabled()
    if os.getenv("MINERU_API_TOKEN") and not os.getenv("MEDGAP_MINERU_BASE_URL"):
        from .mineru_cloud import MinerUCloudClient
        return MinerUCloudClient()
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MinerUClient()
    return _CLIENT
