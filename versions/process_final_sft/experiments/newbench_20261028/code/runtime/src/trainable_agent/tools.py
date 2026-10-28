"""MCP is an execution boundary, not a second autonomous Agent."""
import asyncio
import json
import copy
import hashlib
import uuid
from pathlib import Path
from .contracts import Observation
from .storage import digest


class WorkerBridge:
    def __init__(self, command, log_path, timeout=300, *, startup_timeout=None, operation_timeout=None):
        self.command, self.log_path, self.timeout = command, log_path, timeout
        self.startup_timeout = timeout if startup_timeout is None else startup_timeout
        self.operation_timeout = max(.001, timeout - min(30, timeout / 5)) if operation_timeout is None else operation_timeout
        if not 0 < self.operation_timeout < timeout or self.startup_timeout <= 0:
            raise ValueError("retrieval_deadlines_need_outer_margin")
        self.process = None
        self.lock = asyncio.Lock()
        self.session_lock = asyncio.Lock()
        self.log = None

    async def request(self, operation, arguments):
        async with self.session_lock:
            self.active_operation_id = digest([uuid.uuid4().hex, operation, arguments])
            self.active_phase = "worker_startup" if self.process is None else "worker_transport"
            try:
                return await self._request(operation, arguments)
            except Exception as exc:
                self.record_protocol("transport_failure", {"exception_type": type(exc).__name__,
                                     "outcome": "unknown"})
                await self.close()  # Discard the stream containing any late reply.
                if isinstance(exc, asyncio.TimeoutError) and operation in {"search", "browse"}:
                    from .retrieval_deadlines import reconcile_operation, timeout_value
                    # Reconcile after discarding the old stream, never consume a late
                    # reply as the next request. Corrupt/foreign receipts still fail.
                    value, mode = reconcile_operation(Path(self.log_path).parent / "retrieval",
                                                      self.active_operation_id, operation, arguments)
                    self.record_protocol("timeout_reconciliation", dict(operation_id=self.active_operation_id,
                                                                       resume_mode=mode))
                    if value is not None:
                        if isinstance(value, dict):
                            attach_failure_details(value, Path(self.log_path).parent / "retrieval/tool_observations")
                        return value
                    return timeout_value(self.active_operation_id, self.active_phase, mode)
                raise

    async def _request(self, operation, arguments):
        async with self.lock:
            if self.process is None:
                self.log = open(self.log_path, "ab")
                self.process = await asyncio.create_subprocess_exec(*self.command, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=self.log, limit=16 * 1024 * 1024)
                line = await asyncio.wait_for(self.process.stdout.readline(), self.startup_timeout)
                ready = json.loads(line)
                self.record_protocol("ready", ready)
                if ready.get("status") != "ready":
                    raise RuntimeError("retrieval_worker_not_ready")
            request_id = uuid.uuid4().hex
            self.active_phase = "worker_transport"
            request = {"request_id": request_id, "operation_id": self.active_operation_id,
                       "deadline_seconds": self.operation_timeout, "operation": operation, "arguments": arguments}
            packet = json.dumps(request).encode() + b"\n"
            self.record_protocol("request", request)
            self.process.stdin.write(packet)
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout)
            result = json.loads(line)
            self.record_protocol("response", result)
            if result.get("request_id") != request_id:
                raise RuntimeError("retrieval_reply_request_id_mismatch")
            if result.get("status") != "ok":
                # A worker/program error is NOT an ordinary unreadable webpage.
                raise RuntimeError("retrieval_worker_contract_or_program_failure")
            value = result["value"]
            if isinstance(value, dict):
                attach_failure_details(value, Path(self.log_path).parent / "retrieval/tool_observations")
            return value

    def record_protocol(self, direction, value):
        from datetime import datetime, timezone
        path = Path(self.log_path).with_name("retrieval.protocol.jsonl")
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "direction": direction, "value": value}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.log:
            self.log.close()
        self.process = None
        self.log = None


def attach_failure_details(raw, root):
    """Read only a content-addressed failure receipt inside this worker's output."""
    ref = raw.get("_retrieval_observability", {})
    name, expected = ref.get("manifest_path"), ref.get("manifest_sha256")
    if not (raw.get("failed") and name and expected):
        return
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("failure_receipt_path_escape")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("failure_receipt_hash_mismatch")
    receipt = json.loads(data)
    if receipt.get("schema_version") != "medgap_retrieval_failure_v1":
        raise ValueError("unknown_failure_receipt_schema")
    # Do not propagate fetch errors, URLs or credentials into policy feedback.
    failures = receipt.get("document_metadata", {}).get("fetch_errors", [])
    route_error = next((code for code in ("archive_http_403", "archive_http_401", "archive_redirect_limit")
                        if any(str(e).endswith(code) for e in failures)), None)
    raw["verified_failure_details"] = dict(error_type=receipt.get("error_type"), route_error=route_error,
        access_blocked=any(a.get("blocked") is True and a.get("http_status") in (401, 403)
                           for a in receipt.get("fetch_attempts", [])))


def normalize(raw, channel, requested_source=None):
    if raw.get("failed") or raw.get("error"):
        # Only explicit backend classifications; unknown failure text does NOT earn a free call.
        if raw.get("error_code") in {"invalid_api_key", "quota_exhausted", "model_missing", "configuration_error"}:
            raise RuntimeError("retrieval_configuration_failure_requires_attention")
        details = raw.get("verified_failure_details", {})
        route_error = details.get("route_error") or raw.get("error_code")
        content_failure = raw.get("error") in {"no_evidence_bearing_chunks", "no_readable_biomedical_document", "webpage_content_parse_failed"} or details.get("error_type") in {"content_quality_gate", "no_evidence_bearing_chunks"}
        terminal = route_error in {"archive_http_403", "archive_http_401", "archive_redirect_limit"}
        environment = not (content_failure or terminal) and (raw.get("environment_failure") is True or raw.get("failure_type") == "environment_failure")
        barrier = requested_source is not None and (route_error in {"archive_http_403", "archive_http_401"} or raw.get("verified_failure_details", {}).get("access_blocked") is True
            or raw.get("failure_type") == "access_blocked" or raw.get("error_code") in {
            "captcha_required", "anti_bot_blocked", "authentication_required", "paywall_required", "access_denied"})
        import math
        delay = raw.get("retry_after_seconds", raw.get("retry_after", 0))
        try:
            delay = float(delay)
        except (ValueError, TypeError):
            delay = 0.0
        delay = delay if math.isfinite(delay) and 0 <= delay <= 300 else 0.0
        if raw.get("retry_after_seconds", raw.get("retry_after", 0)) and not delay:
            environment = False  # Cannot promise an unsupported wait; choose another source.
        return Observation(status="access_blocked" if barrier else "environment_failure" if environment else "source_unreadable",
                           error_code="confirmed_source_access_barrier" if barrier else "no_evidence_bearing_chunks" if content_failure else "archive_redirect_limit" if terminal else "retrieval_operation_timeout" if environment and raw.get("error_code") == "retrieval_operation_timeout" else "retrieval_transport_failure" if environment else "source_content_unavailable",
                           retryable=environment and not barrier and raw.get("retryable", True) is True,
                           retry_after_seconds=delay if environment else 0.0,
                           execution={k: v for k, v in raw.get("execution", {}).items()
                                      if k in {"outcome", "phase", "receipt_id", "resume_mode"}},
                           backend_return=copy.deepcopy(raw)).model_dump()
    if requested_source:
        chunks = []
        for item in raw.get("data", []):
            if not str(item.get("text") or "").strip():
                continue
            import re
            parent = str(item.get("parent_source_id") or item.get("source_id") or "")
            match = re.match(r"^(PMID:[0-9]+|S2:[a-zA-Z0-9]+|WEB:[a-zA-Z0-9]+)(?:$|[:#])", parent)
            if match and match.group(1) != requested_source:
                raise ValueError("backend_browse_source_mismatch")
            # Retain original backend passage identity; parent source ID is separate.
            chunk_id = str(item.get("chunk_id") or item.get("source_id") or "chunk:" + digest(item))
            chunks.append(dict(source_id=requested_source, chunk_id=chunk_id,
                               text=item["text"], title=str(item.get("title") or ""),
                               metadata={k: copy.deepcopy(item[k]) for k in
                                   ("url", "authors", "author", "date", "year", "publication_date", "doi", "journal", "novelty_projection", "content_level", "evidence_scope", "abstract_only", "content_scope", "table_window", "window_identity", "window_identity_sha256")
                                   if k in item}))
        return Observation(status="ok" if chunks else "empty", chunks=chunks,
                           backend_return=copy.deepcopy(raw)).model_dump()
    rows = []
    seen = set()
    for item in raw.get("data", []):
        sid = str(item.get("source_id") or item.get("pmid") or "")
        if sid.isdigit():
            sid = "PMID:" + sid
        if not sid or sid in seen:
            continue
        seen.add(sid)
        preview = next((str(item[x]) for x in ("abstract", "snippet", "text") if item.get(x)), "")
        rows.append(dict(source_id=sid, channel=channel, title=str(item.get("title") or ""),
                         preview=preview, url=str(item.get("url") or "")))
    return Observation(status="ok" if rows else "empty", candidates=rows).model_dump()


def create_server(bridge):
    from fastmcp import FastMCP
    server = FastMCP("medical-retrieval-v1")

    async def search(channel, question, query):
        tool = "pubmed_search" if channel == "paper" else "medical_web_search"
        raw = await bridge.request("search", {"tool": tool, "question": question, "query": query})
        return normalize(raw, channel)

    @server.tool
    async def paper_search(question: str, query: str) -> dict:
        """Search papers; titles/abstracts are previews, NOT opened evidence."""
        return await search("paper", question, query)

    @server.tool
    async def web_search(question: str, query: str) -> dict:
        """Search webpages; snippets are previews, NOT opened evidence."""
        return await search("web", question, query)

    @server.tool
    async def browse(question: str, source_id: str, channel: str, reading_focus: str, opened_windows: list[dict] | None = None) -> dict:
        """Read ONE registered source and return at most three original passages."""
        if channel not in ("paper", "web"):
            raise ValueError("invalid_channel")
        raw = await bridge.request("browse", dict(search_tool="pubmed_search" if channel == "paper" else "medical_web_search",
                                                  source_id=source_id, query=reading_focus, question=question, opened_windows=opened_windows or []))
        return normalize(raw, channel, source_id)

    @server.tool
    async def rank_history(question: str, texts: list[str]) -> dict:
        """Runtime-only MiniLM ranking. NOT an action exposed in policy prompts."""
        return {"scores": await bridge.request("rank_history", {"question": question, "texts": texts})}

    return server


class MCPTools:
    def __init__(self, client):
        self.client = client

    async def call(self, name, args):
        result = await self.client.call_tool(name, args)
        if result.is_error:
            raise RuntimeError("MCP_tool_execution_error")
        value = result.data
        if not isinstance(value, dict):
            raise ValueError("MCP_structured_result_required")
        return value

    async def execute(self, args):
        a, question = args["action"], args["question"]
        if a["action"] == "search":
            name = "paper_search" if a["channel"] == "paper" else "web_search"
            return await self.call(name, dict(question=question, query=a["query"]))
        return await self.call("browse", dict(question=question, source_id=a["source_id"],
                               channel=args["channel"], reading_focus=a["reading_focus"], opened_windows=args.get("opened_windows", [])))

    async def rank(self, question, rows):
        value = await self.call("rank_history", dict(question=question,
                                    texts=[c["title"] + "\n" + c["preview"] for c in rows]))
        return value["scores"]
