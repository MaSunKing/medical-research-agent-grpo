# Adapted public research release; see THIRD_PARTY_NOTICES.md.
from dr_agent.mcp_backend.apis.pdf_policy_v54 import pdf_enabled, is_pdf_url
import argparse
import asyncio
import hashlib
import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Annotated, List, Optional

import aiohttp
import requests
from transformers import AutoTokenizer
import dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from dr_agent.anchored_query import decode_anchored_query

from .apis.data_model import Crawl4aiApiResult
from .apis.massive_serve_apis import parse_massive_serve_results, search_massive_serve
from .apis.medical_document_parser import (
    parse_medical_html,
    parse_medical_markdown,
    select_relevant_chunks,
)
from .apis.retrieval_observability import (
    ENV_STORE_DIR as RETRIEVAL_OBSERVABILITY_ENV,
    persist_retrieval_failure,
    persist_retrieval_observation,
)
from .apis.medical_web_apis import (
    is_allowed_medical_url,
    is_readable_medical_web_content,
    is_supported_medical_web_url,
    medical_web_source_id,
    resolve_medical_web_source,
    search_medical_web,
    web_fetch_failure_retryable,
)
from .apis.pubmed_apis import (
    load_medical_document,
    resolve_web_article_to_pmid,
    search_pubmed,
)
from .apis.web_preflight import (
    clear_web_failure,
    fetch_web_content_consistent,
    probe_web_readability,
    recent_web_failure,
    record_web_failure,
)
from .apis.reranker_apis import RerankerResult
from .apis.semantic_scholar_apis import (
    SemanticScholarSearchQueryParams,
    SemanticScholarSnippetSearchQueryParams,
    search_semantic_scholar_keywords,
    search_semantic_scholar_snippets,
)
from .apis.serper_apis import (
    ScholarResponse,
    SearchResponse,
    WebpageContentResponse,
    fetch_webpage_content,
    search_serper,
    search_serper_scholar,
)
from .apis.jina_apis import JinaWebpageResponse, fetch_webpage_content_jina
from .cache import set_cache_enabled
from .local.crawl4ai_fetcher import Crawl4AiResult
from .local.search import SearcherType
from .local.search.base import LocalSearchResponse
from dr_agent.runtime_policy import is_likely_blocked_publisher_url

# Global instance for local search
local_searcher = None
snippet_tokenizer = None
snippet_max_tokens = 0
medical_tokenizer = None
medical_tokenizer_path = None
_pubmed_provenance: dict[str, dict] = {}
_pubmed_provenance_lock = threading.RLock()

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


def get_medical_tokenizer():
    global medical_tokenizer, medical_tokenizer_path
    configured_path = os.getenv("MEDICAL_TOKENIZER_PATH")
    if configured_path and (
        medical_tokenizer is None or medical_tokenizer_path != configured_path
    ):
        medical_tokenizer = AutoTokenizer.from_pretrained(
            configured_path, local_files_only=True, trust_remote_code=True
        )
        medical_tokenizer_path = configured_path
    return medical_tokenizer


def _persist_medical_retrieval(
    result: dict,
    *,
    document: dict,
    query: str,
    tool_name: str,
    fetch_method: str | None = None,
    original_question: str | None = None,
) -> None:
    diagnostics = result.pop("_retrieval_observability", None)
    if not isinstance(diagnostics, dict):
        return
    pointer = persist_retrieval_observation(
        document=document,
        focused_query=query,
        original_question=original_question,
        tool_name=tool_name,
        retrieval=diagnostics,
        document_metadata=result.get("document_metadata") or document.get("metadata") or {},
        fetch_method=fetch_method,
    )
    if pointer is not None:
        # MCPTool removes this backend-only field before constructing the
        # policy-visible observation, while carrying the pointer as diagnostics.
        result["_retrieval_observability"] = pointer

mcp = FastMCP(
    "RL-RAG MCP",
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,browse,rerank").split(","),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8000/health
    """
    return PlainTextResponse("OK")


@mcp.tool(tags={"search", "necessary"})
def semantic_scholar_search(
    query: Annotated[str, "Search query string"],
    year: Annotated[
        Optional[str], "Year range filter (e.g., '2015-2020', '2015-', '-2015')"
    ] = None,
    min_citation_count: Annotated[Optional[int], "Minimum number of citations"] = None,
    sort: Annotated[
        Optional[str], "Sort order (e.g., 'citationCount:asc', 'publicationDate:desc')"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Maximum number of results to return (max: 100)"] = 25,
) -> dict:
    """
    Search for academic papers using Semantic Scholar API.

    Returns:
        Dictionary containing search results
    """
    query_params = SemanticScholarSearchQueryParams(
        query=query,
        year=year,
        minCitationCount=min_citation_count,
        sort=sort,
        venue=venue,
    )

    results = search_semantic_scholar_keywords(
        query_params=query_params,
        limit=min(limit, 100),  # Ensure limit doesn't exceed API maximum
    )

    return results


@mcp.tool(tags={"search"})
def semantic_scholar_snippet_search(
    query: Annotated[str, "Search query string to find within paper content"],
    year: Annotated[
        Optional[str],
        "Publication year filter - single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
    ] = None,
    paper_ids: Annotated[
        Optional[str], "Comma-separated list of specific paper IDs to search within"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Number of snippets to retrieve"] = 10,
) -> dict:
    """
    Focused snippet retrieval from scientific papers using Semantic Scholar API.

    Purpose: Search for specific text snippets within academic papers to find relevant passages, quotes,
    or mentions from scientific literature. Returns focused snippets from existing papers rather than
    full paper metadata.

    Returns:
        Dictionary containing snippets from existing papers with text passages and their source papers.
        Each snippet includes the relevant text passage and metadata about the source paper.

    Example:
        Search for LLM evaluation snippets published between 2021-2025 in CS/Medicine:
        query="large language model retrieval evaluation", year="2021-2025", limit=8
    """
    # Convert comma-separated string to list if provided
    paper_ids_list = None
    if paper_ids:
        paper_ids_list = [pid.strip() for pid in paper_ids.split(",")]

    query_params = SemanticScholarSnippetSearchQueryParams(
        query=query,
        year=year,
        paperIds=paper_ids_list,
        venue=venue,
    )

    results = search_semantic_scholar_snippets(
        query_params=query_params,
        limit=limit,
    )

    return results


@mcp.tool(tags={"search"})
def pubmed_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search for medical and scientific papers using PubMed API.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
        offset: Starting position for pagination (default: 0)

    Returns:
        Dictionary containing search results with the following fields:
        - total: Total number of results
        - offset: Current offset
        - next: Next offset for pagination
        - data: List of paper details including:
            - paperId: PubMed ID
            - title: Paper title
            - authors: List of authors
            - abstract: Paper abstract
            - year: Publication year
            - venue: Journal name
            - url: Link to PubMed page
            - citationCount: Number of citations (if available from Semantic Scholar)
    """
    query_context = decode_anchored_query(query)
    focused_query = query_context.focused_query
    if "_dual_search" in globals():
        return _dual_search(query,limit=limit,offset=offset)
    results = search_pubmed(
        keywords=focused_query,
        limit=limit,
        offset=offset,
        original_question=query_context.original_question or None,
    )

    # Search results are candidates, not evidence, but their structured PubMed
    # provenance is authoritative metadata for a later Browse of the same PMID.
    with _pubmed_provenance_lock:
        for item in results.get("data") or []:
            source_id = str(item.get("source_id") or "")
            if not source_id.startswith("PMID:"):
                continue
            _pubmed_provenance[source_id] = {
                key: item[key]
                for key in (
                    "source_id", "title", "url", "publication_types",
                    "publicationDate", "year", "journal", "doi",
                )
                if item.get(key) not in (None, "", [], {})
            }
            _pubmed_provenance[source_id]["discovery_query"] = focused_query
            if query_context.original_question:
                _pubmed_provenance[source_id]["original_question"] = (
                    query_context.original_question
                )

    # Search rows are candidates rather than evidence.  Keep only compact
    # selection metadata so Search cannot consume the budget needed to Browse
    # and produce a Final answer.  The full abstract remains available inside
    # the backend for ranking and through browse_document.
    compact_results = dict(results)
    compact_results["data"] = [
        {
            key: item[key]
            for key in (
                "source_id", "pmid", "title", "url", "year", "journal",
                "doi", "pmcid", "publication_types", "publicationDate",
                "study_design", "abstract_available", "full_text_hint", "abstract",
            )
            if item.get(key) not in (None, "", [], {})
        }
        for item in results.get("data") or []
        if isinstance(item, dict)
    ]
    compact_results["candidate_observation"] = "native_abstract_for_bounded_preview_not_opened_evidence"
    return compact_results


@mcp.tool(tags={"search", "browse"})
def browse_document(
    source_id: Annotated[str, "PubMed source ID, for example PMID:38914124"],
    query: Annotated[str, "Focused question used to select relevant passages"],
    top_k: Annotated[int, "Maximum number of relevant chunks"] = 3,
    max_chars: Annotated[int, "Hard character budget before token limiting"] = 4200,
    max_output_tokens: Annotated[int, "Maximum content tokens returned"] = 700,
) -> dict:
    """Read a PubMed/PMC document and return only query-relevant passages."""
    query_context = decode_anchored_query(query)
    focused_query = query_context.focused_query
    if not focused_query:
        raise ValueError("browse_document requires a valid focused query")
    try:
        document = load_medical_document(source_id)
    except Exception as exc:
        error_code = str(
            getattr(exc, "error_code", "paper_document_load_failed")
            or "paper_document_load_failed"
        )
        fetch_attempts = list(getattr(exc, "fetch_attempts", []) or [])
        result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": error_code,
            "document_metadata": {
                "source_id": str(source_id).strip().split("#", 1)[0],
            },
        }
        pointer = persist_retrieval_failure(
            source_id=result["document_metadata"]["source_id"],
            focused_query=focused_query,
            tool_name="browse_document",
            error_code=error_code,
            error_type=type(exc).__name__,
            fetch_attempts=fetch_attempts,
        )
        if pointer is not None:
            result["_retrieval_observability"] = pointer
        logger.warning(
            "Paper document load failed for %s (%s)", source_id, type(exc).__name__
        )
        return result
    canonical_source_id = document["metadata"]["source_id"]
    with _pubmed_provenance_lock:
        provenance = dict(_pubmed_provenance.get(canonical_source_id) or {})
    discovery_query = str(provenance.pop("discovery_query", "") or focused_query)
    original_question = str(
        query_context.original_question
        or provenance.pop("original_question", "")
        or discovery_query
    )
    if provenance:
        publication_date = provenance.pop("publicationDate", None)
        provenance.pop("source_id", None)
        document["metadata"].update(provenance)
        if publication_date:
            document["metadata"]["publication_date"] = publication_date
    if not any(
        str(section.get("text") or "").strip()
        for section in document.get("sections") or []
        if isinstance(section, dict)
    ):
        result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": "no_readable_biomedical_document",
            "document_metadata": dict(document.get("metadata") or {}),
        }
        pointer = persist_retrieval_failure(
            source_id=canonical_source_id,
            focused_query=focused_query,
            tool_name="browse_document",
            error_code="no_readable_biomedical_document",
            fetch_attempts=list(document["metadata"].get("fetch_attempts") or []),
            document_metadata=document.get("metadata") or {},
        )
        if pointer is not None:
            result["_retrieval_observability"] = pointer
        return result
    result = select_relevant_chunks(
        document,
        focused_query,
        source_id=document["metadata"]["source_id"],
        # V36 returns three bounded passages: MedCPT anchor, Reader complement,
        # and one section-diverse high-scoring passage.
        top_k=max(1, min(top_k, 3)),
        max_chars=max(200, min(max_chars, 12000)),
        max_output_tokens=max(1, min(max_output_tokens, 4096)),
        tokenizer=get_medical_tokenizer(),
        include_observability=bool(os.getenv(RETRIEVAL_OBSERVABILITY_ENV)),
        original_question=original_question,
    )
    selected_chunks = result.pop("chunks")
    if not selected_chunks:
        failure_result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": "no_evidence_bearing_chunks",
            "retryable": False,
            "document_metadata": dict(document.get("metadata") or {}),
        }
        pointer = persist_retrieval_failure(
            source_id=canonical_source_id,
            focused_query=focused_query,
            tool_name="browse_document",
            error_code="no_evidence_bearing_chunks",
            error_type="content_quality_gate",
            fetch_attempts=list(document["metadata"].get("fetch_attempts") or []),
            document_metadata=failure_result["document_metadata"],
        )
        if pointer is not None:
            failure_result["_retrieval_observability"] = pointer
        return failure_result
    url = document["metadata"].get("url") or ""
    result["data"] = [
        {
            "source_id": chunk["chunk_id"],
            "canonical_chunk_id": chunk.get("canonical_chunk_id"),
            "window_identity": chunk.get("window_identity"),
            "window_identity_sha256": chunk.get("window_identity_sha256"),
            "title": document.get("title", ""),
            "heading": chunk["heading"],
            "text": chunk["text"],
            "snippet": chunk["text"],
            "url": url,
            "score": chunk["score"],
            "pmid": document["metadata"].get("pmid"),
            "pmcid": document["metadata"].get("pmcid"),
            "content_level": document["metadata"].get("content_level"),
            "evidence_scope": document["metadata"].get("evidence_scope"),
            "abstract_only": bool(document["metadata"].get("abstract_only")),
            "quality_class": chunk.get("quality_class"),
            "quality_eligible": bool(chunk.get("quality_eligible", True)),
            "quality_evidence_signal": bool(chunk.get("quality_evidence_signal")),
            "structure_kind": chunk.get("structure_kind"),
            "structure_audit": chunk.get("structure_audit"),
            "boundary_incomplete": bool(chunk.get("boundary_incomplete")),
            "table_integrity_verified": chunk.get("table_integrity_verified"),
            "start_char": chunk.get("start_char"),
            "end_char": chunk.get("end_char"),
            "content_span_start_char": chunk.get("content_span_start_char"),
            "content_span_end_char": chunk.get("content_span_end_char"),
        }
        for chunk in selected_chunks
    ]
    from dr_agent.medical_source_metadata import infer_medical_source_metadata

    structured_metadata = infer_medical_source_metadata(
        title=document.get("title", ""),
        url=url,
        publication_types=document["metadata"].get("publication_types") or [],
        evidence_text="\n".join(item["text"] for item in result["data"]),
    )
    result["document_metadata"] = {**document["metadata"], **structured_metadata}
    if result["document_metadata"].get("abstract_only"):
        result["evidence_limitation"] = (
            "Only the PubMed abstract was readable; absence of an outcome or "
            "safety detail must not be treated as evidence that it was not reported."
        )
    _persist_medical_retrieval(
        result,
        document=document,
        query=focused_query,
        tool_name="browse_document",
        fetch_method=str(
            document["metadata"].get("fetch_method")
            or document["metadata"].get("source_format")
            or "pubmed_loader"
        ),
        original_question=original_question,
    )
    return result


@mcp.tool(tags={"search"})
def medical_web_search(
    query: Annotated[str, "Medical web evidence query"],
    source_types: Annotated[
        str,
        "Comma-separated evidence types: guideline, regulatory, public_health, evidence_review",
    ] = "guideline",
    limit: Annotated[int, "Maximum number of authoritative results"] = 5,
    gl: Annotated[str, "Search geolocation"] = "us",
    hl: Annotated[str, "Search language"] = "en",
) -> dict:
    """Search broad public medical Web sources and rank evidence candidates."""
    query_context = decode_anchored_query(query)
    focused_query = query_context.focused_query
    try:
        return search_medical_web(
            query=focused_query,
            source_types=source_types,
            limit=limit,
            gl=gl,
            hl=hl,
            original_question=query_context.original_question or None,
        )
    except Exception as exc:
        error_code = "all_medical_web_search_routes_failed"
        pointer = persist_retrieval_failure(
            source_id="SEARCH:" + hashlib.sha256(
                str(focused_query).encode("utf-8", errors="replace")
            ).hexdigest()[:16],
            focused_query=focused_query,
            tool_name="medical_web_search",
            error_code=error_code,
            error_type=type(exc).__name__,
            document_metadata={"source_types": str(source_types)},
        )
        result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": error_code,
            "retryable": True,
        }
        if pointer is not None:
            result["_retrieval_observability"] = pointer
        logger.warning("Medical web search failed (%s)", type(exc).__name__)
        return result


async def _try_pubmed_web_alternate(
    *,
    url: str,
    title: str,
    stage: str = "pubmed_open_alternate",
) -> tuple[str, dict, dict, str | None]:
    """Resolve a publisher URL to readable PubMed/PMC content when possible."""

    started = time.perf_counter()
    try:
        alternate_pmid = await asyncio.to_thread(
            resolve_web_article_to_pmid,
            url,
            title,
        )
        metadata: dict = {}
        if alternate_pmid:
            alternate = await asyncio.to_thread(
                load_medical_document,
                f"PMID:{alternate_pmid}",
            )
            markdown = "\n\n".join(
                f"## {section.get('heading') or 'Section'}\n{section.get('text') or ''}"
                for section in alternate.get("sections") or []
                if str(section.get("text") or "").strip()
            )
            if is_readable_medical_web_content(markdown):
                metadata = {
                    "alternate_source_id": f"PMID:{alternate_pmid}",
                    "alternate_fetch_method": alternate.get("metadata", {}).get(
                        "fetch_method"
                    ),
                }
                status = "success"
            else:
                markdown = ""
                status = "no_content"
        else:
            markdown = ""
            status = "not_available"
        attempt = {
            "stage": stage,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return markdown, metadata, attempt, None
    except Exception as exc:
        attempt = {
            "stage": stage,
            "status": "error",
            "error_type": type(exc).__name__,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return "", {}, attempt, f"{stage}: {type(exc).__name__}: {exc}"


@mcp.tool(tags={"browse"})
async def browse_medical_webpage(
    url: Annotated[str, "WEB source ID returned by medical_web_search"],
    query: Annotated[str, "Focused evidence question"],
    top_k: Annotated[int, "Maximum relevant chunks"] = 3,
    max_chars: Annotated[int, "Hard character budget"] = 4200,
    max_output_tokens: Annotated[int, "Maximum content tokens returned"] = 1000,
) -> dict:
    """Read a medical webpage and return only focused, bounded evidence chunks."""
    from dr_agent.mcp_backend.local.crawl4ai_fetcher import fetch_markdown

    query_context = decode_anchored_query(query)
    focused_query = query_context.focused_query
    if not focused_query:
        raise ValueError("browse_webpage requires a valid focused query")
    requested_source_id = str(url or "").strip()
    search_metadata = {}
    if requested_source_id.startswith("WEB:"):
        from dr_agent.mcp_backend.apis.medical_web_apis import (
            get_medical_web_source_metadata,
        )

        search_metadata = get_medical_web_source_metadata(requested_source_id)
    discovery_query = str(
        search_metadata.pop("discovery_query", "") or focused_query
    )
    original_question = str(
        query_context.original_question
        or search_metadata.pop("original_question", "")
        or discovery_query
    )
    search_metadata.pop("discovery_queries", None)
    url = resolve_medical_web_source(requested_source_id)
    if not is_allowed_medical_url(url):
        raise ValueError("browse_webpage only accepts safe sources returned by medical_web_search")
    if not is_supported_medical_web_url(url):
        raise ValueError("unsupported_pdf: configure V28 MinerU to browse PDF pages")

    fetch_errors = []
    fetch_attempts = []
    source_id = medical_web_source_id(url)
    parsed_url = requests.utils.urlparse(url)
    looks_like_pdf = is_pdf_url(url)
    mineru_metadata = {}
    markdown = ""
    fetch_method = ""
    alternate_preflight_attempted = False
    preflight_terminal = False
    direct_fetch_terminal = False
    response_is_pdf = False
    cached_failure = recent_web_failure(url) if not looks_like_pdf else None
    if cached_failure is not None:
        fetch_attempts.append(
            {
                "stage": "accessibility_failure_cache",
                "status": "blocked",
                "reason": cached_failure.get("reason"),
                "http_status": cached_failure.get("http_status"),
                "cache_hit": True,
                "elapsed_ms": 0.0,
            }
        )
        preflight_terminal = True
    if not looks_like_pdf and is_likely_blocked_publisher_url(url):
        # Avoid paying the predictable browser/HTTP failure cost first.  This
        # is a soft publisher risk route: if no exact literature alternate is
        # found, the normal webpage fallbacks still run.
        alternate_preflight_attempted = True
        (
            markdown,
            alternate_metadata,
            alternate_attempt,
            alternate_error,
        ) = await _try_pubmed_web_alternate(
            url=url,
            title=str(search_metadata.get("title") or ""),
            stage="pubmed_open_alternate_preflight",
        )
        fetch_attempts.append(alternate_attempt)
        mineru_metadata.update(alternate_metadata)
        if alternate_error:
            fetch_errors.append(alternate_error)
        if markdown:
            fetch_method = "pubmed_open_alternate"
    if (
        not looks_like_pdf
        and not markdown
        and not preflight_terminal
        and os.getenv("MEDGAP_WEB_PREFLIGHT_ENABLED", "true").strip().lower()
        in {"1", "true", "yes"}
    ):
        # Probe every source the policy actually chose to Browse, rather than
        # probing every Search candidate.  This catches previously unknown
        # anti-bot sites at one small request per selected URL.  An
        # inconclusive probe deliberately falls through to the normal fetch
        # stack so the preflight cannot turn uncertainty into a false block.
        preflight = await asyncio.to_thread(probe_web_readability, url)
        fetch_attempts.append(preflight)
        preflight_terminal = bool(preflight.get("blocked"))
        if preflight_terminal:
            record_web_failure(
                url,
                reason=str(preflight.get("reason") or "preflight_blocked"),
                http_status=preflight.get("http_status"),
            )
    if looks_like_pdf:
        started = time.perf_counter()
        try:
            from dr_agent.mcp_backend.apis.mineru_client import get_mineru_client

            markdown, mineru_metadata = await asyncio.to_thread(
                get_mineru_client().parse_url, url
            )
            fetch_method = "mineru_pdf"
            status = (
                "success" if is_readable_medical_web_content(markdown) else "no_content"
            )
            if status != "success":
                markdown = ""
            fetch_attempts.append(
                {
                    "stage": "mineru_pdf",
                    "status": status,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        except Exception as exc:
            fetch_errors.append(f"mineru_pdf: {type(exc).__name__}: {exc}")
            fetch_attempts.append(
                {
                    "stage": "mineru_pdf",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    if looks_like_pdf and not markdown:
        # A failed PDF parser must not fall through to an HTML/PDF viewer,
        # whose navigation can masquerade as evidence. A provenance-labelled
        # PubMed alternate below is still allowed; HTML/crawl/scrape are not.
        direct_fetch_terminal = True
    if not markdown and not preflight_terminal and not direct_fetch_terminal:
        # Most public-health, government and guideline pages are static HTML.
        # Parse them before paying the browser-startup cost of Crawl4AI.
        started = time.perf_counter()
        response_is_pdf = False
        try:
            response = await asyncio.to_thread(fetch_web_content_consistent, url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            response_is_pdf = "application/pdf" in content_type or response.content.lstrip().startswith(b"%PDF-")
            if response_is_pdf:
                if not pdf_enabled():
                    raise ValueError("web_pdf_disabled")
                from dr_agent.mcp_backend.apis.mineru_client import get_mineru_client
                markdown, mineru_metadata = await asyncio.to_thread(get_mineru_client().parse_url, url)
                if is_readable_medical_web_content(markdown):
                    mineru_metadata.update(source_format="pdf", content_level="open_pdf_full_text",
                                          abstract_only=False, full_text_available=True,
                                          reader_route="web_pdf_mineru", pdf_parse_status="parsed")
                else:
                    mineru_metadata = {}
            elif "html" in content_type or b"<html" in response.content[:1000].lower():
                parsed = parse_medical_html(response.content)
                markdown = "\n\n".join(
                    f"## {section['heading']}\n{section['text']}"
                    for section in parsed["sections"]
                )
            status = (
                "success"
                if is_readable_medical_web_content(markdown)
                else "unsupported_content"
            )
            if status != "success":
                markdown = ""
            fetch_attempts.append(
                {
                    "stage": "web_pdf_mineru" if response_is_pdf else "direct_html",
                    "status": status,
                    "http_status": int(response.status_code),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            if status == "success":
                fetch_method = "web_pdf_mineru" if response_is_pdf else "direct_html"
        except Exception as exc:
            markdown = ""
            if response_is_pdf:
                direct_fetch_terminal = True
            fetch_errors.append(f"direct_html: {type(exc).__name__}: {exc}")
            response = getattr(exc, "response", None)
            attempt = {
                "stage": "web_pdf_mineru" if response_is_pdf else "direct_html",
                "status": "error",
                "error_type": type(exc).__name__,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            if response is not None and getattr(response, "status_code", None) is not None:
                attempt["http_status"] = int(response.status_code)
                if int(response.status_code) in {401, 403, 451}:
                    direct_fetch_terminal = True
                    record_web_failure(
                        url,
                        reason="direct_http_blocked",
                        http_status=int(response.status_code),
                    )
            fetch_attempts.append(attempt)
    if not markdown and not preflight_terminal and not direct_fetch_terminal:
        # Escalate to browser rendering only when cheap direct HTML could not
        # recover readable content (for example, JavaScript-rendered pages).
        started = time.perf_counter()
        try:
            crawled = await fetch_markdown(
                url=url,
                query=focused_query,
                ignore_links=True,
                use_pruning=False,
                bypass_cache=False,
                headless=True,
                timeout_ms=int(os.getenv("MEDICAL_CRAWL_TIMEOUT_MS", "20000")),
                include_html=False,
            )
            markdown = crawled.markdown if crawled.success else ""
            status = (
                "success" if is_readable_medical_web_content(markdown) else "no_content"
            )
            if status != "success":
                markdown = ""
            fetch_attempts.append(
                {
                    "stage": "crawl4ai",
                    "status": status,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            if status == "success":
                fetch_method = "crawl4ai"
            else:
                fetch_errors.append("crawl4ai returned no content")
        except Exception as exc:
            markdown = ""
            fetch_errors.append(f"crawl4ai: {type(exc).__name__}: {exc}")
            fetch_attempts.append(
                {
                    "stage": "crawl4ai",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    if not markdown and not alternate_preflight_attempted and not preflight_terminal:
        # Publisher HTML frequently blocks automated readers even when the
        # same article has a legal PubMed/PMC representation. Resolve only by
        # exact DOI or conservative title match; keep the model-visible WEB ID
        # and citation contract unchanged.
        (
            markdown,
            alternate_metadata,
            alternate_attempt,
            alternate_error,
        ) = await _try_pubmed_web_alternate(
            url=url,
            title=str(search_metadata.get("title") or ""),
        )
        fetch_attempts.append(alternate_attempt)
        mineru_metadata.update(alternate_metadata)
        if alternate_error:
            fetch_errors.append(alternate_error)
        if markdown:
            fetch_method = "pubmed_open_alternate"
    if not markdown and not preflight_terminal and not direct_fetch_terminal:
        # Serper scrape remains the final fallback for non-article pages that
        # block both browser automation and ordinary HTTP clients.
        started = time.perf_counter()
        try:
            scraped = fetch_webpage_content(url=url, include_markdown=True)
            markdown = scraped.get("markdown") or scraped.get("text") or ""
            status = (
                "success" if is_readable_medical_web_content(markdown) else "no_content"
            )
            if status != "success":
                markdown = ""
            fetch_attempts.append(
                {
                    "stage": "serper_scrape",
                    "status": status,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            if status == "success":
                fetch_method = "serper_scrape"
        except Exception as exc:
            fetch_errors.append(f"serper_scrape: {type(exc).__name__}: {exc}")
            fetch_attempts.append(
                {
                    "stage": "serper_scrape",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    if not markdown.strip():
        retryable = web_fetch_failure_retryable(fetch_attempts)
        if preflight_terminal:
            failure_error_type = "preflight_blocked"
        elif looks_like_pdf or response_is_pdf:
            failure_error_type = "pdf_parse_unavailable"
        elif direct_fetch_terminal:
            failure_error_type = "direct_http_blocked"
        else:
            failure_error_type = "all_fetch_routes_exhausted"
        policy_safe_metadata = {
            key: value for key, value in search_metadata.items() if key != "snippet"
        }
        result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": "no_readable_webpage",
            "retryable": retryable,
            "document_metadata": {
                **policy_safe_metadata,
                "source_id": source_id,
                "url": url,
            },
        }
        pointer = persist_retrieval_failure(
            source_id=source_id,
            focused_query=focused_query,
            tool_name="browse_webpage",
            error_code="no_readable_webpage",
            error_type=failure_error_type,
            fetch_attempts=fetch_attempts,
            document_metadata={
                **search_metadata,
                "source_id": source_id,
                "url": url,
                "fetch_errors": fetch_errors,
                "retryable": retryable,
            },
        )
        if pointer is not None:
            result["_retrieval_observability"] = pointer
        return result

    clear_web_failure(url)

    try:
        parsed_document = parse_medical_markdown(markdown)
    except Exception as exc:
        fetch_attempts.append(
            {
                "stage": "content_parse",
                "status": "error",
                "error_type": type(exc).__name__,
            }
        )
        policy_safe_metadata = {
            key: value for key, value in search_metadata.items() if key != "snippet"
        }
        result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": "webpage_content_parse_failed",
            "retryable": False,
            "document_metadata": {
                **policy_safe_metadata,
                "source_id": source_id,
                "url": url,
            },
        }
        pointer = persist_retrieval_failure(
            source_id=source_id,
            focused_query=focused_query,
            tool_name="browse_webpage",
            error_code="webpage_content_parse_failed",
            error_type=type(exc).__name__,
            fetch_attempts=fetch_attempts,
            document_metadata={
                **search_metadata,
                "source_id": source_id,
                "url": url,
            },
        )
        if pointer is not None:
            result["_retrieval_observability"] = pointer
        return result
    heading_match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    title = parsed_document.get("title") or (heading_match.group(1).strip() if heading_match else url)
    document = {
        "title": title,
        "text": parsed_document.get("text") or markdown,
        "sections": parsed_document.get("sections") or [{"heading": "Webpage", "text": markdown}],
        "metadata": {
            "source_id": source_id,
            "url": url,
            "fetch_method": fetch_method,
            "fetch_attempts": fetch_attempts,
            **mineru_metadata,
        },
    }
    result = select_relevant_chunks(
        document,
        focused_query,
        source_id=source_id,
        # V36 uses the same bounded Top-3 contract for paper and web evidence.
        top_k=max(1, min(top_k, 3)),
        max_chars=max(200, min(max_chars, 12000)),
        max_output_tokens=max(1, min(max_output_tokens, 4096)),
        tokenizer=get_medical_tokenizer(),
        include_observability=bool(os.getenv(RETRIEVAL_OBSERVABILITY_ENV)),
        original_question=original_question,
    )
    selected_chunks = result.pop("chunks")
    if not selected_chunks:
        failure_error_type = "no_evidence_bearing_chunks"
        failure_result = {
            "data": [],
            "failed": True,
            "failure_type": "environment_failure",
            "error": failure_error_type,
            "retryable": False,
            "document_metadata": {
                **search_metadata,
                "source_id": source_id,
                "url": url,
                "fetch_method": fetch_method,
                "fetch_attempts": fetch_attempts,
                **mineru_metadata,
            },
        }
        pointer = persist_retrieval_failure(
            source_id=source_id,
            focused_query=focused_query,
            tool_name="browse_webpage",
            error_code=failure_error_type,
            error_type="content_quality_gate",
            fetch_attempts=fetch_attempts,
            document_metadata=failure_result["document_metadata"],
        )
        if pointer is not None:
            failure_result["_retrieval_observability"] = pointer
        return failure_result
    result["data"] = [
        {
            "source_id": chunk["chunk_id"],
            "canonical_chunk_id": chunk.get("canonical_chunk_id"),
            "window_identity": chunk.get("window_identity"),
            "window_identity_sha256": chunk.get("window_identity_sha256"),
            "title": title,
            "heading": chunk["heading"],
            "text": chunk["text"],
            "snippet": chunk["text"],
            "url": url,
            "score": chunk["score"],
            "source": "medical_web",
            "quality_class": chunk.get("quality_class"),
            "quality_eligible": bool(chunk.get("quality_eligible", True)),
            "quality_evidence_signal": bool(chunk.get("quality_evidence_signal")),
            "structure_kind": chunk.get("structure_kind"),
            "structure_audit": chunk.get("structure_audit"),
            "boundary_incomplete": bool(chunk.get("boundary_incomplete")),
            "table_integrity_verified": chunk.get("table_integrity_verified"),
            "start_char": chunk.get("start_char"),
            "end_char": chunk.get("end_char"),
            "content_span_start_char": chunk.get("content_span_start_char"),
            "content_span_end_char": chunk.get("content_span_end_char"),
        }
        for chunk in selected_chunks
    ]
    from dr_agent.medical_source_metadata import infer_medical_source_metadata

    structured_metadata = infer_medical_source_metadata(
        title=title,
        url=url,
        evidence_text="\n".join(item["text"] for item in result["data"]),
    )
    source_types = list(search_metadata.get("source_types") or [])
    authority_from_search = next(
        (
            value
            for value in ("guideline", "regulatory", "public_health")
            if value in source_types
        ),
        None,
    )
    selected_body = "\n".join(item["text"] for item in result["data"])
    authority_body_verified = bool(
        re.search(
            r"\b(?:recommend(?:s|ed|ation)?|guideline|consensus|should|must|"
            r"contraindicat(?:ed|ion)|boxed warning|public health|eligible persons?)\b",
            selected_body,
            flags=re.IGNORECASE,
        )
    )
    if authority_from_search and authority_body_verified:
        structured_metadata["authority_type"] = authority_from_search
    result["document_metadata"] = {
        **search_metadata,
        "source_id": source_id,
        "url": url,
        "fetch_method": fetch_method,
        "fetch_attempts": fetch_attempts,
        **mineru_metadata,
        **structured_metadata,
        "search_authority_type": authority_from_search or "none",
        "authority_body_verified": authority_body_verified,
    }
    _persist_medical_retrieval(
        result,
        document=document,
        query=focused_query,
        tool_name="browse_webpage",
        fetch_method=fetch_method,
        original_question=original_question,
    )
    return result


@mcp.tool(tags={"necessary", "rerank"})
def vllm_hosted_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> RerankerResult:
    """
    Rerank a list of documents based on their relevance to the query using VLLM hosted reranker.

    Args:
        query: Search query string
        documents: List of document texts to rank
        top_n: Number of top documents to return
        model_name: Name of the reranker model (default: "BAAI/bge-reranker-v2-m3")
        api_url: Base URL for the VLLM reranker API (default: "http://localhost:30002")

    Returns:
        RerankerResult containing reranker results with method, model_name, and ranked results
    """
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_hosted_reranker

    results = vllm_hosted_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
    )

    return results


@mcp.tool(tags={"search", "necessary"})
def massive_serve_search(
    query: str,
    n_docs: int = 10,
    domains: str = "dpr_wiki_contriever_ivfpq",
    base_url: Optional[str] = None,
    nprobe: Optional[int] = None,
) -> dict:
    """
    Search for documents using massive-serve API for dense passage retrieval.

    This tool provides access to large-scale document collections using dense passage
    retrieval with various embedding models and indices.

    Args:
        query: Search query string
        n_docs: Number of documents to return (default: 10)
        domains: Domain/index to search in (default: "dpr_wiki_contriever_ivfpq")
        base_url: Base URL for the massive-serve API (optional, uses default if not provided)
        nprobe: Number of probes for search (optional, uses API default)

    Returns:
        Dictionary containing search results with the following fields:
        - message: Status message
        - query: The original search query
        - n_docs: Number of documents requested
        - results: Dictionary with IDs, passages, and scores
        - data: Parsed list of search results with passage text, scores, and doc IDs
    """
    # Call the massive-serve API
    response = search_massive_serve(
        query=query,
        n_docs=n_docs,
        domains=domains,
        base_url=base_url,
        nprobe=nprobe,
    )

    # Parse the results for easier consumption
    parsed_results = parse_massive_serve_results(response)

    # Add parsed data to the response for convenience
    response["data"] = [
        {
            "passage": result.passage,
            "score": result.score,
            "doc_id": result.doc_id,
        }
        for result in parsed_results
    ]

    return response


@mcp.tool(tags={"search", "necessary"})
def serper_google_webpage_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
    gl: Annotated[
        str,
        "Geolocation - country code to boost search results whose country of origin matches the parameter value",
    ] = "us",
    hl: Annotated[str, "Host language of user interface"] = "en",
):
    """
    General web search using Google Search (based on Serper.dev API). Perform general web search to find relevant webpages, articles, and online resources.

    Returns:
        Dictionary containing web search snippets with the following fields:
        - organic: List of organic search results with title, link, and snippet
        - knowledgeGraph: Knowledge graph information (if available)
        - peopleAlsoAsk: List of related questions
        - relatedSearches: List of related searches
    """
    results = search_serper(
        query=query, num_results=num_results, search_type="search", gl=gl, hl=hl
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
def serper_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    include_markdown: Annotated[
        bool, "Whether to include markdown formatting in the response"
    ] = True,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Returns:
        Dictionary containing the webpage content with the following fields:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
        - url: The original URL that was fetched
        - success: Boolean indicating if the fetch was successful
    """
    try:
        result = fetch_webpage_content(
            url=webpage_url,
            include_markdown=include_markdown,
        )

        return {
            **result,
            "success": True,
        }
    except Exception as e:
        return {
            "text": "",
            "markdown": "",
            "metadata": {},
            "url": webpage_url,
            "success": False,
            "error": str(e),
        }


@mcp.tool(tags={"browse"})
def jina_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    timeout: Annotated[int, "Request timeout in seconds"] = 30,
) -> JinaWebpageResponse:
    """
    Fetch the content of a webpage using Jina Reader API with timeout support.

    Args:
        webpage_url: The URL of the webpage to fetch
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing the webpage content with the following fields:
        - url: The original URL that was fetched
        - title: Page title
        - content: The webpage content as clean text/markdown
        - description: Page description (if available)
        - publishedTime: Published time (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    result = fetch_webpage_content_jina(url=webpage_url, timeout=timeout)
    return result


@mcp.tool(tags={"search", "necessary"})
def serper_google_scholar_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> ScholarResponse:
    """
    Search for academic papers using google scholar (based on Serper.dev API).

    Returns:
        Dictionary containing search results with the following fields:
        - organic: List of organic search results
    """
    results = search_serper_scholar(
        query=query,
        num_results=num_results,
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_fetch_webpage_content(
    url: Annotated[str, "URL to fetch and extract content from"],
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> Crawl4AiResult:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4AiResult with extracted webpage content including markdown-formatted text
    """

    from dr_agent.mcp_backend.local.crawl4ai_fetcher import fetch_markdown

    result = await fetch_markdown(
        url=url,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        bypass_cache=bypass_cache,
        headless=True,
        timeout_ms=timeout_ms,
        include_html=include_html,
    )
    return result


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_docker_fetch_webpage_content(
    url: Annotated[str, "Target URL to crawl and extract content from"],
    base_url: Annotated[
        Optional[str],
        "Base URL for the Crawl4AI Docker API (e.g., 'http://localhost:8000')",
    ] = None,
    api_key: Annotated[Optional[str], "API key for authentication"] = None,
    use_ai2_config: Annotated[
        bool,
        "If True, use AI2 bot configuration with blocklist (requires CRAWL4AI_BLOCKLIST_PATH env var)",
    ] = False,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> Crawl4aiApiResult:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI Docker API.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4aiApiResult with url, success, markdown-formatted text, and optional fit_markdown/html/error fields
    """
    from dr_agent.mcp_backend.apis.crawl4ai_docker_api import crawl_url_docker

    result = await crawl_url_docker(
        url=url,
        base_url=base_url,
        api_key=api_key,
        bypass_cache=bypass_cache,
        include_html=include_html,
        use_ai2_config=use_ai2_config,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        timeout_ms=timeout_ms,
    )
    return result


@mcp.tool(tags={"browse"})
def webthinker_fetch_webpage_content(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import extract_text_from_url

    text = extract_text_from_url(
        url=url,
        snippet=snippet,
        keep_links=keep_links,
    )

    return {"url": url, "text": text}


@mcp.tool(tags={"browse"})
async def webthinker_fetch_webpage_content_async(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Asynchronously extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import (
        extract_text_from_url_async,
    )

    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=240)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.36",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        text = await extract_text_from_url_async(
            url=url,
            session=session,
            snippet=snippet,
            keep_links=keep_links,
        )

    return {"url": url, "text": text}


@mcp.tool(tags={"search", "local"})
def local_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> LocalSearchResponse:
    """
    Perform a search on a local knowledge source.
    Useful for retrieving relevant passages from specific datasets or local indices.
    """
    if local_searcher is None:
        return {
            "results": [],
            "error": "Local searcher not initialized."
        }
    
    try:
        response = local_searcher.search(query, k=num_results)
        
        if snippet_max_tokens > 0 and snippet_tokenizer:
            for cand in response.get("results", []):
                snippet_text = cand["snippet"]
                tokens = snippet_tokenizer.encode(snippet_text, add_special_tokens=False)
                if len(tokens) > snippet_max_tokens:
                    truncated_tokens = tokens[:snippet_max_tokens]
                    cand["snippet"] = snippet_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return response
    except Exception as e:
        logger.error(f"Error during local search: {e}")
        return {"results": [], "error": str(e)}


@mcp.tool(tags={"browse", "local"})
def local_browse(
    url: Annotated[str, "URL to fetch and extract content from"],
) -> dict:
    """
    Retrieve full text content for a given URL from the local knowledge source.
    Useful for reading the full content of documents found via local_search.
    """
    if local_searcher is None:
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": "Local index not initialized."
        }
    
    try:
        text = local_searcher.get_text_by_url(url)
        if text is None:
            return {
                "url": url,
                "success": False,
                "markdown": "",
                "error": f"URL not found: {url}"
            }
            
        return {
            "url": url,
            "success": True,
            "markdown": text
        }
    except Exception as e:
        logger.error(f"Error during local browse: {e}")
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": str(e)
        }



from .apis.dual_paper import install as _install_dual
import sys as _dual_sys
_install_dual(_dual_sys.modules[__name__])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio for local, http for web)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (for HTTP transports)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Path for the HTTP endpoint (default: /mcp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level for the server",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )
    parser.add_argument(
        "--local-searcher-type",
        type=str,
        default=None,
        choices=SearcherType.get_choices(),
        help="Type of local searcher to use (default: None, no local search)",
    )
    parser.add_argument(
        "--local-search-max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens for local search snippets (default: 100, set to 0 to disable truncation)",
    )

    # We need a first pass to get the searcher type to add its specific args
    temp_args, _ = parser.parse_known_args()
    if temp_args.local_searcher_type:
        searcher_cls = SearcherType.get_searcher_class(temp_args.local_searcher_type)
        searcher_cls.parse_args(parser)

    args = parser.parse_args()

    # Initialize local searcher if specified
    if args.local_searcher_type:
        try:
            searcher_cls = SearcherType.get_searcher_class(args.local_searcher_type)
            local_searcher = searcher_cls(args)
            logger.info(f"Initialized local searcher: {args.local_searcher_type}")
            
            snippet_max_tokens = args.local_search_max_tokens
            if snippet_max_tokens > 0:
                logger.info(f"Loading tokenizer for local search truncation (max {snippet_max_tokens} tokens)...")
                snippet_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception as e:
            logger.error(f"Failed to initialize local searcher: {e}")

    # Set cache enabled/disabled based on argument
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Run the server with the provided arguments
    if args.transport == "stdio":
        # stdio transport doesn't accept host/port/path arguments
        # For stdio, we can omit the transport argument since it's the default
        mcp.run(transport="stdio")
    else:
        # HTTP-based transports accept host/port/path/log_level arguments
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
        )
