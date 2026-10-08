# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Low-cost anti-bot preflight for model-selected medical webpages."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse


CHALLENGE_CUES = (
    "please wait 5 seconds",
    "verify you are human",
    "checking your browser",
    "enable javascript and cookies to continue",
    "attention required! | cloudflare",
    "access denied",
    "robot check",
    "captcha",
)

_PREFLIGHT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_PREFLIGHT_CACHE_LOCK = threading.Lock()
_FAILURE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_FAILURE_CACHE_LOCK = threading.Lock()


def _request_headers(*, last_byte: int) -> dict[str, str]:
    """Use one browser-shaped request contract for probe and direct fetch."""

    return {
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Range": f"bytes=0-{max(8191, int(last_byte))}",
        "Cache-Control": "no-cache",
    }


def _curl_cffi_get(url: str, *, timeout: float, headers: dict[str, str]):
    from curl_cffi import requests as curl_requests

    return curl_requests.get(
        url,
        timeout=timeout,
        headers=headers,
        impersonate="chrome",
        allow_redirects=True,
    )


def fetch_web_content_consistent(
    url: str,
    *,
    timeout_seconds: float | None = None,
    max_bytes: int | None = None,
    request_get: Callable[..., Any] | None = None,
):
    """Fetch bounded HTML with the same client fingerprint as preflight.

    The previous path probed with curl-cffi and then fetched with ``requests``.
    Some sites (notably CDC in V34) returned 206 to the former and 403 to the
    latter.  Keeping both stages on one request contract makes the probe
    predictive without downloading an unbounded page.
    """

    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.getenv("MEDICAL_DIRECT_HTTP_TIMEOUT", "12")
    )
    timeout = max(1.0, min(timeout, 60.0))
    byte_budget = int(
        max_bytes
        if max_bytes is not None
        else os.getenv("MEDGAP_WEB_DIRECT_MAX_BYTES", str(2 * 1024 * 1024))
    )
    byte_budget = max(8192, min(byte_budget, 8 * 1024 * 1024))
    getter = request_get or _curl_cffi_get
    return getter(
        url,
        timeout=timeout,
        headers=_request_headers(last_byte=byte_budget - 1),
    )


def recent_web_failure(url: str) -> dict[str, Any] | None:
    """Return a recent terminal fetch failure for this exact URL."""

    ttl = max(
        0.0,
        min(float(os.getenv("MEDGAP_WEB_FAILURE_CACHE_TTL", "900")), 7200.0),
    )
    if not ttl:
        return None
    now = time.monotonic()
    with _FAILURE_CACHE_LOCK:
        cached = _FAILURE_CACHE.get(str(url))
        if cached is None:
            return None
        if now - cached[0] > ttl:
            _FAILURE_CACHE.pop(str(url), None)
            return None
        return dict(cached[1])


def record_web_failure(
    url: str,
    *,
    reason: str,
    http_status: int | None = None,
) -> None:
    """Remember only terminal accessibility failures, never relevance misses."""

    terminal = http_status in {401, 403, 429, 451} or reason in {
        "challenge_page",
        "preflight_blocked",
        "direct_http_blocked",
    }
    if not terminal:
        return
    with _FAILURE_CACHE_LOCK:
        _FAILURE_CACHE[str(url)] = (
            time.monotonic(),
            {"reason": str(reason), "http_status": http_status},
        )


def clear_web_failure(url: str) -> None:
    with _FAILURE_CACHE_LOCK:
        _FAILURE_CACHE.pop(str(url), None)


def web_accessibility_penalty(url: str) -> float:
    """Soft search-rank penalty from recent URL/domain accessibility failures."""

    ttl = max(
        0.0,
        min(float(os.getenv("MEDGAP_WEB_FAILURE_CACHE_TTL", "900")), 7200.0),
    )
    if not ttl:
        return 0.0
    exact = recent_web_failure(url)
    host = (urlparse(str(url)).hostname or "").casefold().removeprefix("www.")
    now = time.monotonic()
    domain_failures = 0
    with _FAILURE_CACHE_LOCK:
        for failed_url, (recorded_at, _row) in list(_FAILURE_CACHE.items()):
            if ttl and now - recorded_at > ttl:
                _FAILURE_CACHE.pop(failed_url, None)
                continue
            failed_host = (
                urlparse(failed_url).hostname or ""
            ).casefold().removeprefix("www.")
            if host and failed_host == host:
                domain_failures += 1
    return (120.0 if exact else 0.0) + min(60.0, 20.0 * domain_failures)


def probe_web_readability(
    url: str,
    *,
    timeout_seconds: float | None = None,
    request_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Probe a small byte range without escalating to a browser.

    A Cloudflare header alone is not treated as blocking. Only an explicit
    terminal HTTP response or challenge-page content causes a skip.
    """

    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.getenv("MEDGAP_WEB_PREFLIGHT_TIMEOUT", "2.5")
    )
    timeout = max(0.5, min(timeout, 10.0))
    getter = request_get or _curl_cffi_get
    cache_ttl = max(
        0.0,
        min(float(os.getenv("MEDGAP_WEB_PREFLIGHT_CACHE_TTL", "600")), 3600.0),
    )
    if request_get is None and cache_ttl:
        with _PREFLIGHT_CACHE_LOCK:
            cached = _PREFLIGHT_CACHE.get(url)
        if cached is not None and time.monotonic() - cached[0] <= cache_ttl:
            return {**cached[1], "cache_hit": True, "elapsed_ms": 0.0}
    started = time.perf_counter()
    headers = _request_headers(last_byte=8191)
    try:
        response = getter(url, timeout=timeout, headers=headers)
    except ImportError:
        return {
            "stage": "anti_bot_preflight",
            "status": "unavailable",
            "blocked": False,
            "reason": "curl_cffi_not_installed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:
        return {
            "stage": "anti_bot_preflight",
            "status": "inconclusive",
            "blocked": False,
            "reason": f"probe_{type(exc).__name__}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    status_code = int(getattr(response, "status_code", 0) or 0)
    body = bytes(getattr(response, "content", b"") or b"")[:8192]
    lowered = body.decode("utf-8", errors="ignore").casefold()
    challenge = next((cue for cue in CHALLENGE_CUES if cue in lowered), None)
    terminal_http = status_code in {401, 403, 451}
    rate_limited = status_code == 429
    blocked = bool(terminal_http or rate_limited or challenge)
    if terminal_http:
        reason = f"http_{status_code}"
    elif rate_limited:
        reason = "http_429_rate_limited"
    elif challenge:
        reason = "challenge_page"
    elif 200 <= status_code < 400:
        reason = "probe_passed"
    else:
        reason = f"http_{status_code or 'unknown'}_inconclusive"
    result = {
        "stage": "anti_bot_preflight",
        "status": (
            "rate_limited" if rate_limited
            else "blocked" if blocked
            else "passed" if reason == "probe_passed"
            else "inconclusive"
        ),
        "blocked": blocked,
        "reason": reason,
        "http_status": status_code or None,
        "cloudflare_header_present": bool(
            getattr(response, "headers", {}).get("cf-ray")
        ),
        "challenge_cue": challenge,
        "sampled_bytes": len(body),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "cache_hit": False,
    }
    if request_get is None and cache_ttl:
        with _PREFLIGHT_CACHE_LOCK:
            _PREFLIGHT_CACHE[url] = (time.monotonic(), dict(result))
    return result
