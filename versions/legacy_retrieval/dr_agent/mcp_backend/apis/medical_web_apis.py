# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Domain-routed web discovery for authoritative medical evidence."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import threading
from typing import Dict, Iterable, List
from urllib.parse import parse_qs, urlparse

from .serper_apis import search_serper
from .web_preflight import web_accessibility_penalty
from ...medical_tool_schema import normalize_source_types
from ...runtime_policy import rank_web_candidates, web_candidate_score_details


SOURCE_TYPE_DOMAINS: Dict[str, tuple[str, ...]] = {
    "guideline": (
        "kdigo.org",
        "nice.org.uk",
        "who.int",
        "acc.org",
        "escardio.org",
        "heart.org",
        "diabetesjournals.org",
        "asco.org",
        "esmo.org",
        "nccn.org",
        "cancer.gov",
        "idsociety.org",
        "aan.com",
        "acog.org",
        "rheumatology.org",
        "uspreventiveservicestaskforce.org",
        "ginasthma.org",
        "goldcopd.org",
        "americangeriatrics.org",
        "auanet.org",
        "uroweb.org",
        "aap.org",
        "hematology.org",
        "cap.org",
        "aasm.org",
        "aad.org",
        "thoracic.org",
        "asahq.org",
        "acponline.org",
        "nih.gov",
    ),
    "regulatory": (
        "fda.gov",
        "ema.europa.eu",
        "efsa.europa.eu",
        "nmpa.gov.cn",
        "gov.uk",
        "canada.ca",
    ),
    "public_health": (
        "who.int",
        "cdc.gov",
        "ecdc.europa.eu",
        "nih.gov",
    ),
    "evidence_review": (
        "cochrane.org",
        "ahrq.gov",
        "nice.org.uk",
    ),
}

MAX_DOMAIN_ATTEMPTS = max(1, int(os.getenv("MEDICAL_MAX_DOMAIN_ATTEMPTS", "3")))

# These are literature records/full-text hosts, not guideline-authority pages.
# Keep them out of web discovery so papers consistently flow through
# pubmed_search -> browse_document and satisfy the biomedical evidence slot.
LITERATURE_HOSTS = {
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
}

WEB_BLOCK_PAGE_CUES = (
    "access denied",
    "attention required! | cloudflare",
    "enable javascript and cookies to continue",
    "please verify you are a human",
    "robot check",
    "temporarily blocked",
)

# Search candidates are identified to the model by opaque WEB:<hash> IDs.  Keep
# the corresponding URL private to the MCP process so browse_webpage can accept
# the public source-ID contract without letting the model bypass the allowlist.
_MEDICAL_WEB_SOURCE_URLS: Dict[str, str] = {}
_MEDICAL_WEB_SOURCE_METADATA: Dict[str, dict] = {}
_MEDICAL_WEB_SOURCE_URLS_LOCK = threading.RLock()


def medical_web_source_id(url: str) -> str:
    return "WEB:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def register_medical_web_sources(items: Iterable[dict]) -> None:
    """Register only valid, hash-consistent candidates returned by search."""
    registrations: Dict[str, str] = {}
    metadata_registrations: Dict[str, dict] = {}
    for item in items:
        source_id = str(item.get("source_id") or "").strip()
        url = str(item.get("url") or "").strip()
        if not source_id or not url:
            continue
        if source_id != medical_web_source_id(url):
            raise ValueError("medical web source ID does not match its URL")
        if not is_allowed_medical_url(url) or not is_supported_medical_web_url(url):
            raise ValueError("cannot register an unsafe or unsupported medical webpage")
        registrations[source_id] = url
        metadata_registrations[source_id] = {
            key: item[key]
            for key in (
                "source_id", "title", "url", "snippet", "domain", "source_types", "date",
                "discovery_query", "discovery_queries", "authority_score",
                "candidate_score", "relevance_score", "readability_score",
                "evidence_type_score", "candidate_rank", "accessibility_penalty",
                "original_question",
            )
            if item.get(key) not in (None, "", [], {})
        }
    with _MEDICAL_WEB_SOURCE_URLS_LOCK:
        _MEDICAL_WEB_SOURCE_URLS.update(registrations)
        _MEDICAL_WEB_SOURCE_METADATA.update(metadata_registrations)


def get_medical_web_source_metadata(source_id: str) -> dict:
    """Return trusted discovery provenance for a registered WEB candidate."""
    value = str(source_id or "").strip()
    with _MEDICAL_WEB_SOURCE_URLS_LOCK:
        metadata = _MEDICAL_WEB_SOURCE_METADATA.get(value)
    if metadata is None:
        raise ValueError(
            f"unknown medical web source ID: {value}; call medical_web_search first"
        )
    return dict(metadata)


def resolve_medical_web_source(source_id_or_url: str) -> str:
    """Resolve an opaque WEB ID, while retaining literal-URL compatibility."""
    value = str(source_id_or_url or "").strip()
    if value.startswith(("http://", "https://")):
        return value
    if not value.startswith("WEB:"):
        raise ValueError("browse_webpage requires a WEB source ID")
    with _MEDICAL_WEB_SOURCE_URLS_LOCK:
        url = _MEDICAL_WEB_SOURCE_URLS.get(value)
    if not url:
        raise ValueError(
            f"unknown medical web source ID: {value}; call medical_web_search first"
        )
    if value != medical_web_source_id(url):
        raise ValueError("registered medical web source ID failed integrity validation")
    return url


def domains_for_source_types(source_types: str | Iterable[str]) -> List[str]:
    normalized = normalize_source_types(source_types)
    return sorted(
        {domain for source_type in normalized for domain in SOURCE_TYPE_DOMAINS[source_type]}
    )


def _host_matches(host: str, domains: Iterable[str]) -> bool:
    host = host.lower().split(":", 1)[0].rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def is_allowed_medical_url(url: str) -> bool:
    if os.getenv("MEDGAP_WEB_DISCOVERY_MODE", "routed").strip().lower() == "broad":
        return is_safe_public_web_url(url)
    host = (urlparse(url).hostname or "").lower()
    all_domains = {
        domain for domains in SOURCE_TYPE_DOMAINS.values() for domain in domains
    }
    return bool(host) and _host_matches(host, all_domains)


def is_safe_public_web_url(url: str) -> bool:
    """Reject local/private targets while allowing broad public discovery."""
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith((".local", ".localhost")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private or address.is_loopback or address.is_link_local
        or address.is_multicast or address.is_reserved or address.is_unspecified
    )


def is_supported_medical_web_url(url: str) -> bool:
    """Return whether the V1 online-training browser can consume this URL.

    MedGap-GRPO V1 deliberately trains on HTML pages plus PubMed/PMC XML. PDF
    parsing (including MinerU) is a deployment extension, so obvious PDF
    candidates must not be handed to the HTML browser.
    """

    parsed = urlparse(url)
    pdf_enabled = bool(os.getenv("MEDGAP_MINERU_BASE_URL")) and (
        os.getenv("MEDGAP_PASSAGE_RETRIEVAL_MODE", "bm25").strip().lower() == "v28"
    )
    if parsed.path.lower().rstrip("/").endswith(".pdf"):
        return pdf_enabled
    query = {
        key.lower(): [value.lower() for value in values]
        for key, values in parse_qs(parsed.query).items()
    }
    is_pdf = any(
        value == "pdf" or value.endswith("/pdf")
        for key in ("format", "type", "output", "download")
        for value in query.get(key, [])
    )
    return pdf_enabled if is_pdf else True


def is_readable_medical_web_content(value: str, *, min_chars: int = 120) -> bool:
    """Reject empty/challenge pages before they enter passage retrieval."""
    compact = " ".join(str(value or "").split())
    if len(compact) < max(1, int(min_chars)):
        return False
    prefix = compact[:4000].casefold()
    return not any(cue in prefix for cue in WEB_BLOCK_PAGE_CUES)


def web_fetch_failure_retryable(fetch_attempts: Iterable[dict]) -> bool:
    """Classify whether one more focused attempt could recover this URL."""
    attempts = [dict(item) for item in fetch_attempts]
    for item in attempts:
        status = item.get("http_status")
        if isinstance(status, int) and 400 <= status < 500 and status not in {408, 429}:
            return False
    terminal_content_states = {
        "no_content", "unsupported_content", "not_available", "blocked"
    }
    observed_states = {
        str(item.get("status") or "") for item in attempts if item.get("status")
    }
    if observed_states and observed_states <= terminal_content_states:
        return False
    return True


def web_query_variants(query: str, source_types: Iterable[str]) -> List[str]:
    """Small deterministic query fan-out; no hidden rubric or paid LLM needed."""
    base = " ".join(str(query or "").split())
    variants = [base]
    normalized = list(source_types)
    if "guideline" in normalized:
        variants.append(f"{base} guideline recommendation consensus")
    if "regulatory" in normalized:
        variants.append(f"{base} safety warning label regulatory")
    if "public_health" in normalized:
        variants.append(f"{base} public health guidance evidence")
    if "evidence_review" in normalized:
        variants.append(f"{base} systematic review evidence")
    variants.append(f"{base} clinical evidence outcomes adverse effects")
    result: List[str] = []
    for value in variants:
        if value not in result:
            result.append(value)
    maximum = max(1, int(os.getenv("MEDGAP_WEB_QUERY_VARIANTS", "3")))
    return result[:maximum]


def _prioritize_domains(query: str, domains: Iterable[str]) -> List[str]:
    """Put the most relevant authority first without exposing routing to the model."""
    available = list(domains)
    lowered = query.lower()
    specialty_routes = (
        (("fda", "regulatory", "approval", "approved"), ("fda.gov",)),
        (("nmpa", "china drug", "\u4e2d\u56fd\u836f\u76d1", "\u56fd\u5bb6\u836f\u76d1"),
         ("nmpa.gov.cn",)),
        (("nmpa", "china drug", "中国药监", "国家药监"), ("nmpa.gov.cn",)),
        (("cancer", "oncology", "nsclc", "tumor", "tumour", "carcinoma"),
         ("cancer.gov", "asco.org", "esmo.org", "nccn.org")),
        (("kidney stone", "renal stone", "urolithiasis", "nephrolithiasis"),
         ("auanet.org", "uroweb.org")),
        (("kidney", "renal", "ckd", "dialysis"), ("kdigo.org",)),
        (("diabetes", "glycemic", "glucose"), ("diabetesjournals.org",)),
        (("heart", "cardiac", "cardiovascular", "coronary"),
         ("acc.org", "heart.org", "escardio.org")),
        (("infant", "neonatal", "newborn", "pediatric", "paediatric", "adolescent"),
         ("aap.org", "nice.org.uk", "who.int")),
        (("hematology", "haematology", "anemia", "anaemia", "hemophilia", "haemophilia",
          "von willebrand", "leukemia", "leukaemia", "spherocytosis"),
         ("hematology.org", "nice.org.uk")),
        (("molecular testing", "biomarker", "egfr", "ntrk", "fusion gene"),
         ("cap.org", "cancer.gov", "asco.org")),
        (("sleep", "insomnia", "apnea", "apnoea"), ("aasm.org", "nice.org.uk")),
        (("skin", "dermatology", "melanoma", "psoriasis"),
         ("aad.org", "cancer.gov", "nice.org.uk")),
        (("thoracic", "ventilation", "respiratory insufficiency"),
         ("thoracic.org", "nice.org.uk")),
        (("anesthesia", "anaesthesia", "epidural", "perioperative"),
         ("asahq.org", "nice.org.uk")),
        (("primary care", "internal medicine", "telehealth"),
         ("acponline.org", "nice.org.uk")),
        (("rare disease", "genetic", "syndrome", "osteogenesis", "krabbe", "noonan",
          "ehlers", "tay-sachs"), ("nih.gov", "nice.org.uk")),
        (("infection", "infectious", "antimicrobial", "antibiotic", "hiv", "tuberculosis"),
         ("idsociety.org", "cdc.gov", "who.int", "ecdc.europa.eu")),
        (("neurology", "neurologic", "epilepsy", "seizure", "migraine", "multiple sclerosis"),
         ("aan.com", "nice.org.uk")),
        (("pregnancy", "obstetric", "gynecology", "maternal", "prenatal"),
         ("acog.org", "nice.org.uk")),
        (("rheumatology", "rheumatoid", "lupus", "vasculitis", "gout"),
         ("rheumatology.org", "nice.org.uk")),
        (("asthma", "pulmonary", "respiratory", "copd", "bronchodilator"),
         ("nice.org.uk", "ginasthma.org", "goldcopd.org")),
        (("geriatrics", "geriatric", "older adult", "beers criteria"),
         ("americangeriatrics.org", "nice.org.uk")),
        (("screening", "preventive", "prevention recommendation"),
         ("uspreventiveservicestaskforce.org", "nice.org.uk", "who.int")),
    )
    preferred: List[str] = []
    for keywords, routed_domains in specialty_routes:
        if any(keyword in lowered for keyword in keywords):
            preferred.extend(domain for domain in routed_domains if domain in available)
    if any("\u4e00" <= char <= "\u9fff" for char in query):
        preferred.extend(
            domain
            for domain in ("nmpa.gov.cn", "who.int", "cdc.gov", "nice.org.uk")
            if domain in available and domain not in preferred
        )
    # For queries without a specialty keyword, do not fall back to alphabetical
    # order (which previously sent generic psychiatry/rehabilitation questions
    # to ACC/ACOG first).  Try broad, authoritative sources before narrower
    # society sites.  The allowlist remains the hard security boundary.
    default_authorities = (
        "nice.org.uk", "who.int", "cdc.gov", "fda.gov", "nih.gov",
        "uspreventiveservicestaskforce.org", "ahrq.gov", "cochrane.org",
    )
    preferred.extend(
        domain for domain in default_authorities
        if domain in available and domain not in preferred
    )
    preferred.extend(domain for domain in available if domain not in preferred)
    return preferred


def search_medical_web(
    query: str,
    source_types: str | Iterable[str] = "guideline",
    limit: int = 5,
    gl: str = "us",
    hl: str = "en",
    original_question: str | None = None,
) -> dict:
    if not query.strip():
        raise ValueError("medical web query must not be empty")
    limit = max(1, min(int(limit), 10))
    normalized_types = normalize_source_types(source_types)
    domains = domains_for_source_types(normalized_types)
    discovery_mode = os.getenv("MEDGAP_WEB_DISCOVERY_MODE", "routed").strip().lower()
    if discovery_mode not in {"routed", "broad"}:
        raise ValueError("MEDGAP_WEB_DISCOVERY_MODE must be routed or broad")
    results = []
    routed_queries = []
    route_errors = []
    seen_urls = set()
    # Serper free accounts reject advanced `site:` query operators. Add one
    # backend-selected authority as a plain discovery hint, then enforce the
    # actual security boundary by strictly filtering returned URL hosts.
    if discovery_mode == "broad":
        routed_queries = [query.strip()]
        if original_question and original_question.casefold() != query.casefold():
            routed_queries.append(
                " ".join(f"{original_question} {query}".split())[:1800]
            )
        for variant in web_query_variants(query, normalized_types):
            if all(variant.casefold() != item.casefold() for item in routed_queries):
                routed_queries.append(variant)
        routed_queries = routed_queries[:3]
        attempted_domains: List[str | None] = [None] * len(routed_queries)
    else:
        attempted_domains = _prioritize_domains(query, domains)[:MAX_DOMAIN_ATTEMPTS]
        routed_queries = [f"{query.strip()} {domain}" for domain in attempted_domains]
    per_domain_limit = (
        limit if discovery_mode == "broad"
        else max(1, math.ceil(limit / max(1, len(attempted_domains))))
    )
    attempted_query_count = 0
    for route_index, (domain, routed_query) in enumerate(
        zip(attempted_domains, routed_queries)
    ):
        if discovery_mode == "broad" and route_index >= 2 and len(results) >= limit:
            break
        attempted_query_count += 1
        try:
            raw = search_serper(
                routed_query,
                num_results=min(20, max(limit * 2, limit)),
                gl=gl,
                hl=hl,
            )
        except Exception as exc:
            route_errors.append({"domain": domain, "query": routed_query, "error": str(exc)})
            continue
        domain_results = 0
        for item in raw.get("organic", []):
            url = str(item.get("link") or "").strip()
            host = (urlparse(url).hostname or "").lower()
            if (
                not url
                or url in seen_urls
                or not is_safe_public_web_url(url)
                or not is_supported_medical_web_url(url)
                or host in LITERATURE_HOSTS
                or (discovery_mode == "routed" and not _host_matches(host, domains))
            ):
                continue
            seen_urls.add(url)
            matched_types = [
                source_type
                for source_type in normalized_types
                if _host_matches(host, SOURCE_TYPE_DOMAINS[source_type])
            ]
            results.append(
                {
                    "source_id": medical_web_source_id(url),
                    "title": str(item.get("title") or "").strip(),
                    "url": url,
                    "snippet": str(item.get("snippet") or "").strip(),
                    "domain": host,
                    "source": "medical_web",
                    "source_types": matched_types,
                    "date": item.get("date"),
                    "discovery_query": routed_query,
                    "discovery_queries": list(routed_queries),
                    "original_question": original_question,
                    # Backend-only, short-lived accessibility signal. It does
                    # not block broad discovery or change the public schema.
                    "accessibility_penalty": web_accessibility_penalty(url),
                }
            )
            domain_results += 1
            if domain_results >= per_domain_limit:
                break

    returned_results = rank_web_candidates(
        query, results, original_question=original_question
    )[:limit]
    for rank, item in enumerate(returned_results, start=1):
        scores = web_candidate_score_details(
            query, item, original_question=original_question
        )
        item.update(
            {
                "candidate_rank": rank,
                "candidate_score": round(scores["score"], 6),
                "relevance_score": round(scores["relevance_score"], 6),
                "focused_relevance_score": round(
                    scores["focused_relevance_score"], 6
                ),
                "original_relevance_score": round(
                    scores["original_relevance_score"], 6
                ),
                "authority_score": round(scores["authority_score"], 6),
                "readability_score": round(scores["readability_score"], 6),
                "evidence_type_score": round(scores["evidence_type_score"], 6),
            }
        )
    register_medical_web_sources(returned_results)
    # Discovery fan-out is backend-only.  Preserve the exact model-visible
    # candidate schema learned during SFT.
    public_results = []
    for item in returned_results:
        public_item = dict(item)
        public_item.pop("discovery_query", None)
        public_item.pop("discovery_queries", None)
        public_item.pop("authority_score", None)
        public_item.pop("candidate_score", None)
        public_item.pop("relevance_score", None)
        public_item.pop("focused_relevance_score", None)
        public_item.pop("original_relevance_score", None)
        public_item.pop("original_question", None)
        public_item.pop("readability_score", None)
        public_item.pop("evidence_type_score", None)
        public_item.pop("candidate_rank", None)
        public_item.pop("accessibility_penalty", None)
        public_results.append(public_item)
    response = {
        "query": query,
        "routed_query": routed_queries[0],
        "routed_queries": routed_queries,
        "attempted_query_count": attempted_query_count,
        "anchored_query_bundle": bool(original_question),
        "route_errors": route_errors,
        "max_domain_attempts": MAX_DOMAIN_ATTEMPTS,
        "source_types": normalized_types,
        "allowed_domains": domains,
        "data": public_results,
        "total_returned": min(len(results), limit),
    }
    if not public_results and route_errors and len(route_errors) == len(routed_queries):
        response.update(
            {
                "failed": True,
                "failure_type": "environment_failure",
                "error": "all_medical_web_search_routes_failed",
            }
        )
    elif not public_results:
        response.update(
            {
                "failed": True,
                "failure_type": "search_no_results",
                "error": "no_supported_medical_web_results",
            }
        )
    return response
