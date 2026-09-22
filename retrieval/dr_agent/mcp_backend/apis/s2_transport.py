# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Key-scoped, cross-process S2 pacing. Never log credentials or response bodies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from email.utils import parsedate_to_datetime
from contextlib import contextmanager
import requests


class S2RequestError(RuntimeError):
    def __init__(self, status=None, kind='http'):
        self.status_code = status
        super().__init__(f'S2 request failed: kind={kind} status={status}')


@contextmanager
def locked(path):
    with path.open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def retry_delay(value, now):
    try:
        return max(0., float(value))
    except (TypeError, ValueError):
        try:
            return max(0., parsedate_to_datetime(value).timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return 0.


class Transport:
    def __init__(self, directory=None, send=None, clock=None, sleep=None, send_post=None):
        self.directory = Path(directory or os.getenv('MEDGAP_S2_RATE_DIR') or
                              Path(tempfile.gettempdir()) / 'medgap-s2-rate-v1')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.send = send or requests.get
        self.send_post = send_post or requests.post
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep

    def post(self, url, **kwargs):
        return self.get(url, _post=True, **kwargs)

    def get(self, url, _post=False, **kwargs):
        if not url.startswith('https://api.semanticscholar.org/'):
            raise ValueError('S2 transport requires official HTTPS endpoint')
        key = (kwargs.get('headers') or {}).get('x-api-key', '')
        kwargs['allow_redirects'] = False  # Never forward x-api-key to a redirect host.
        ident = hashlib.sha256(key.encode()).hexdigest()
        state = self.directory / (ident + '.json')
        # Serialize requests/retries for the same key across local workers.
        with locked(self.directory / (ident + '.lock')):
            next_at = json.loads(state.read_text())['next_at'] if state.exists() else 0.
            for attempt in range(3):
                wait = max(0., next_at - self.clock())
                if wait > 60:
                    raise S2RequestError(429, 'cooldown_pending')
                self.sleep(wait)
                next_at = self.clock() + 1.1
                state.write_text(json.dumps({'next_at': next_at}))
                response = None
                try:
                    response = (self.send_post if _post else self.send)(url, **kwargs)
                    status = response.status_code
                except (requests.Timeout, requests.ConnectionError):
                    status = None
                except requests.RequestException:
                    raise S2RequestError(None, 'transport') from None
                if status is not None and 200 <= status < 300:
                    return response
                retryable = status in (None, 429, 500, 502, 503, 504)
                if retryable:
                    delay = max(2 ** (attempt + 1), retry_delay(
                        response.headers.get('Retry-After') if response is not None else None,
                        self.clock()))
                    next_at = max(next_at, self.clock() + delay)
                    state.write_text(json.dumps({'next_at': next_at}))
                if response is not None:
                    response.close()
                if not retryable or attempt == 2:
                    raise S2RequestError(status, 'retry_exhausted' if retryable else 'http') from None


_transport = None
def get(url, **kwargs):
    global _transport
    if _transport is None:
        _transport = Transport()
    return _transport.get(url, **kwargs)


def post(url, **kwargs):
    global _transport
    if _transport is None:
        _transport = Transport()
    return _transport.post(url, **kwargs)
