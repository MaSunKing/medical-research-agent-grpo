import asyncio
import inspect

def registered_callable(value):
    if callable(value):
        return value
    for name in ("fn", "function", "run"):
        candidate = getattr(value, name, None)
        if callable(candidate):
            return candidate
    raise TypeError(f"registered MCP tool is not callable: {value!r}")

def run_callable(function, **kwargs):
    result = registered_callable(function)(**kwargs)
    return asyncio.run(result) if inspect.isawaitable(result) else result

def _call_search_unchecked(main, anchored, *, tool: str, query: str, question: str) -> dict:
    encoded = anchored(question, query)
    if tool == "pubmed_search":
        return dict(run_callable(main.pubmed_search, query=encoded, limit=10, offset=0))
    if tool == "medical_web_search":
        return dict(
            run_callable(
                main.medical_web_search,
                query=encoded,
                source_types="guideline,regulatory,public_health,evidence_review",
                limit=8,
                gl="us",
                hl="en",
            )
        )
    raise ValueError(f"unsupported Search tool: {tool}")

def call_search(main, anchored, *, tool: str, query: str, question: str) -> dict:
    from requests.exceptions import HTTPError, Timeout, ConnectionError
    try:
        return _call_search_unchecked(
            main, anchored, tool=tool, query=query, question=question
        )
    except (HTTPError, Timeout, ConnectionError) as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        # Do not expose request URLs, credentials, or exception text.
        error = f"search_transport_failure:{type(exc).__name__}:HTTP={status}"
        print("V71_SEARCH_TRANSPORT_FAILURE=" + error, flush=True)
        return {
            "failed": True,
            "error": error,
            "data": [],
            "environment_failure": True,
        }

def call_browse(main, anchored, *, search_tool: str, source_id: str, query: str, question: str) -> dict:
    encoded = anchored(question, query)
    if search_tool == "pubmed_search":
        return dict(
            run_callable(
                main.browse_document,
                source_id=source_id,
                query=encoded,
                top_k=3,
                max_chars=4200,
                max_output_tokens=1000,
            )
        )
    return dict(
        run_callable(
            main.browse_medical_webpage,
            url=source_id,
            query=encoded,
            top_k=3,
            max_chars=4200,
            max_output_tokens=1000,
        )
    )
