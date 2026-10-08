# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Deployment-time policy gates for the medical Research Agent.

These helpers do not prescribe a tool route.  They make the environment safer
and more reproducible by ranking candidates, rejecting wasteful duplicate
actions, following the user's language, and checking whether a proposed answer
covers the evidence classes implied by the question.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable
from typing import Any, Iterable
from urllib.parse import urlparse

from .medical_tool_schema import normalize_medical_tool_arguments


WORD_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
CITE_RE = re.compile(r'<cite\s+id="([^"]+)">.*?</cite>', re.DOTALL)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?")

# Public regulators, public-health bodies, guideline producers, and independent
# evidence-review organizations.  Journal articles hosted by PMC are useful
# evidence, but PMC hosting alone does not make an item a normative authority.
AUTHORITY_SCORES = {
    "fda.gov": 120,
    "ema.europa.eu": 120,
    "efsa.europa.eu": 120,
    "nmpa.gov.cn": 120,
    "who.int": 115,
    "cdc.gov": 110,
    "ecdc.europa.eu": 110,
    "nice.org.uk": 110,
    "uspreventiveservicestaskforce.org": 110,
    "ahrq.gov": 105,
    "cochrane.org": 105,
    "kdigo.org": 105,
    "cancer.gov": 100,
    "nih.gov": 70,
    # Professional guideline producers accepted by medical_web_search. Keep
    # these at the high-authority threshold so the finalizer does not reject a
    # source that the discovery tool deliberately treats as authoritative.
    "acc.org": 100,
    "heart.org": 100,
    "escardio.org": 100,
    "diabetesjournals.org": 100,
    "asco.org": 100,
    "esmo.org": 100,
    "nccn.org": 100,
    "idsociety.org": 100,
    "aan.com": 100,
    "acog.org": 100,
    "rheumatology.org": 100,
    "ginasthma.org": 100,
    "goldcopd.org": 100,
    "americangeriatrics.org": 100,
    "auanet.org": 100,
    "uroweb.org": 100,
    "aap.org": 100,
    "hematology.org": 100,
    "cap.org": 100,
    "aasm.org": 100,
    "aad.org": 100,
    "thoracic.org": 100,
    "asahq.org": 100,
    "acponline.org": 100,
    "gov.uk": 110,
    "canada.ca": 110,
}

SAFETY_TERMS = {
    "safe", "safety", "harm", "harmful", "harmless", "risk", "risks",
    "adverse", "toxicity", "toxic", "side", "effect", "effects",
}
CURRENT_AUTHORITY_TERMS = {
    "current", "latest", "guideline", "guidelines", "recommendation",
    "recommendations", "approved", "approval", "label", "warning",
    "regulatory", "screening", "policy",
}
NONHUMAN_MARKERS = {
    "in vitro", "mouse", "mice", "rat ", "rats", "animal study",
    "cell line", "murine", "premolars were", "enamel samples",
}
HUMAN_EVIDENCE_MARKERS = {
    "systematic review", "meta-analysis", "randomized", "randomised",
    "clinical trial", "cohort", "case-control", "participants", "patients",
    "adults", "children", "epidemiolog", "follow-up", "prospective",
}

_PUBMED_RANDOMIZED_TERMS = {
    "randomized", "randomised", "trial", "placebo", "controlled",
    "comparative", "versus", "efficacy", "effectiveness", "treatment",
}
_PUBMED_SYNTHESIS_TERMS = {
    "meta", "analysis", "meta-analysis", "systematic", "review", "pooled", "synthesis",
}
_PUBMED_DIAGNOSTIC_TERMS = {
    "diagnostic", "diagnosis", "sensitivity", "specificity", "accuracy",
    "screening", "predictive",
}

_PUBMED_PROGNOSIS_TERMS = {
    "prognosis", "prognostic", "incidence", "prevalence", "epidemiology",
    "cohort", "risk factor", "mortality",
}
_PUBMED_GUIDELINE_TERMS = {
    "guideline", "guidelines", "recommendation", "recommendations",
    "consensus", "practice guideline",
}
_PUBMED_PROTOCOL_TERMS = {"protocol", "design", "rationale", "methods"}

CLAIM_STOP_WORDS = {
    "about", "after", "again", "also", "among", "because", "before",
    "being", "between", "could", "does", "during", "evidence", "from",
    "have", "into", "more", "most", "other", "should", "than", "that",
    "their", "there", "these", "this", "those", "through", "under", "using",
    "were", "which", "with", "would",
}
MAX_BROWSE_ATTEMPTS_PER_SOURCE = 2

LIKELY_BLOCKED_PUBLISHER_HOSTS = {
    "nejm.org", "sciencedirect.com", "onlinelibrary.wiley.com",
    "ahajournals.org", "jacc.org", "jamanetwork.com",
    "publications.aap.org", "endocrinepractice.org",
    # Confirmed by the V30J/V32 fixed-Dev runtime census.
    "diabetesjournals.org", "academic.oup.com", "ard.eular.org",
    "ajkd.org", "bmj.com", "akjournals.com",
}
PREFERRED_READABLE_MEDICAL_HOSTS = {
    "fda.gov", "ema.europa.eu", "who.int", "cdc.gov", "ecdc.europa.eu",
    "nice.org.uk", "uspreventiveservicestaskforce.org", "ahrq.gov",
    "cancer.gov", "nih.gov", "gov.uk", "canada.ca",
}
LOW_VALUE_WEB_PATH_CUES = (
    "/news", "/blog", "/press-release", "/podcast", "/video",
    "/commentary", "/opinion",
)
LOW_EVIDENCE_SOCIAL_HOSTS = {
    "facebook.com", "instagram.com", "x.com", "twitter.com", "tiktok.com",
}
EVIDENCE_WEB_CUES = (
    "guideline", "guidance", "recommendation", "consensus", "statement",
    "safety", "warning", "assessment", "evidence", "clinical-practice",
)


@dataclass(frozen=True)
class MedicalAnswerValidation:
    accepted: bool
    reasons: tuple[str, ...]
    citation_ids: tuple[str, ...]
    opened_evidence_ids: tuple[str, ...]
    coverage_gaps: tuple[str, ...]
    language_ok: bool
    claim_support_checked: bool


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text or "")}


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 1.0 if not a and not b else 0.0
    return len(a & b) / len(a | b)


def _host(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower().rstrip(".")


def authority_score(url_or_host: str) -> int:
    host = _host(url_or_host) if "://" in (url_or_host or "") else (url_or_host or "").lower()
    scores = [score for domain, score in AUTHORITY_SCORES.items()
              if host == domain or host.endswith("." + domain)]
    return max(scores, default=60)


def is_high_authority_url(url: str) -> bool:
    return authority_score(url) >= 100


def is_likely_blocked_publisher_url(url_or_host: str) -> bool:
    """Identify publisher pages that should try a literature alternate first."""

    host = _host(url_or_host) if "://" in str(url_or_host or "") else str(url_or_host or "")
    host = host.casefold().removeprefix("www.")
    return any(
        host == domain or host.endswith("." + domain)
        for domain in LIKELY_BLOCKED_PUBLISHER_HOSTS
    )


def web_candidate_score_details(
    query: str,
    item: dict[str, Any],
    *,
    original_question: str | None = None,
) -> dict[str, float]:
    """Score a public Web candidate for evidence utility and likely readability."""
    query_tokens = _tokens(query)
    url = str(item.get("url") or item.get("link") or "")
    domain = str(item.get("domain") or _host(url)).casefold()
    normalized_domain = domain.removeprefix("www.")
    text = " ".join(str(item.get(key) or "") for key in ("title", "snippet", "text"))
    item_tokens = _tokens(text)
    focused_relevance = len(query_tokens & item_tokens) / max(1, len(query_tokens))
    original_tokens = _tokens(original_question or "")
    original_relevance = (
        len(original_tokens & item_tokens) / max(1, len(original_tokens))
        if original_tokens else focused_relevance
    )
    relevance = 0.6 * focused_relevance + 0.4 * original_relevance
    compact_domain = normalized_domain.split(".", 1)[0]
    direct_domain = 25.0 if compact_domain and compact_domain in query.casefold() else 0.0
    lowered_url = url.casefold()
    lowered_text = text.casefold()
    evidence_bonus = 12.0 if any(
        cue in lowered_url or cue in lowered_text for cue in EVIDENCE_WEB_CUES
    ) else 0.0
    source_type_bonus = 8.0 if item.get("source_types") or item.get("source_type") else 0.0
    literature_host = normalized_domain in {
        "pmc.ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"
    }
    official_readability = 30.0 if not literature_host and any(
        normalized_domain == host or normalized_domain.endswith("." + host)
        for host in PREFERRED_READABLE_MEDICAL_HOSTS
    ) else 0.0
    # This is a rank penalty, not a block. A unique relevant publisher result
    # remains available, while readable official alternatives move ahead.
    publisher_penalty = 80.0 if is_likely_blocked_publisher_url(domain) else 0.0
    low_value_penalty = 10.0 if any(cue in lowered_url for cue in LOW_VALUE_WEB_PATH_CUES) else 0.0
    # Keep broad discovery, but make social posts a last-resort candidate rather
    # than allowing an account page to outrank a readable medical source.
    social_penalty = 140.0 if any(
        normalized_domain == host or normalized_domain.endswith("." + host)
        for host in LOW_EVIDENCE_SOCIAL_HOSTS
    ) else 0.0
    accessibility_penalty = max(
        0.0, min(float(item.get("accessibility_penalty") or 0.0), 180.0)
    )
    score = (
        authority_score(domain)
        + 70.0 * relevance
        + direct_domain
        + evidence_bonus
        + source_type_bonus
        + official_readability
        - publisher_penalty
        - low_value_penalty
        - social_penalty
        - accessibility_penalty
    )
    return {
        "score": score,
        "relevance_score": relevance,
        "focused_relevance_score": focused_relevance,
        "original_relevance_score": original_relevance,
        "authority_score": float(authority_score(domain)),
        "readability_score": (
            official_readability - publisher_penalty - social_penalty
            - accessibility_penalty
        ),
        "evidence_type_score": evidence_bonus + source_type_bonus,
    }


def rank_web_candidates(
    query: str,
    items: Iterable[dict[str, Any]],
    *,
    original_question: str | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates while discouraging a top list dominated by one domain."""
    remaining = [dict(item) for item in items]
    ranked: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    domain_repeat_penalty = max(
        0.0,
        float(os.getenv("MEDGAP_V37_WEB_DOMAIN_DIVERSITY_PENALTY", "30")),
    )
    while remaining:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                web_candidate_score_details(
                    query, remaining[index], original_question=original_question
                )["score"]
                - domain_repeat_penalty * domain_counts.get(
                    str(
                        remaining[index].get("domain")
                        or _host(str(remaining[index].get("url") or remaining[index].get("link") or ""))
                    ).casefold(),
                    0,
                ),
                str(remaining[index].get("title") or ""),
            ),
        )
        selected = remaining.pop(best_index)
        ranked.append(selected)
        domain = str(selected.get("domain") or _host(str(selected.get("url") or ""))).casefold()
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    return ranked


def pubmed_candidate_value_score(query: str, item: dict[str, Any]) -> float:
    """Score how useful a PubMed candidate is for the public search query.

    MedCPT measures semantic similarity.  This deterministic companion score
    measures evidence utility using only public query text and PubMed metadata:
    direct term/phrase overlap, human evidence, requested study design, and
    outcome emphasis such as safety.  It never reads the hidden reward rubric.
    """
    query_tokens = _tokens(query)
    query_sequence = [token.lower() for token in WORD_RE.findall(query or "")]
    query_bigrams = set(zip(query_sequence, query_sequence[1:]))
    query_lower = str(query or "").casefold()
    types = " ".join(
        str(value) for value in item.get("publication_types") or []
    ).casefold()
    text = " ".join(
        str(item.get(key) or "")
        for key in ("title", "abstract", "text", "snippet", "discovery_text")
    ).casefold()
    combined = f"{types} {text}"

    overlap = len(query_tokens & _tokens(text)) / max(1, len(query_tokens))
    text_sequence = [token.lower() for token in WORD_RE.findall(text)]
    text_bigrams = set(zip(text_sequence, text_sequence[1:]))
    phrase_score = 20 * min(2, len(query_bigrams & text_bigrams))

    randomized = any(
        marker in combined
        for marker in (
            "randomized controlled trial", "randomised controlled trial",
            "randomized", "randomised", "placebo-controlled",
        )
    )
    meta_analysis = "meta-analysis" in combined or "meta analysis" in combined
    systematic_review = "systematic review" in combined
    guideline = any(
        marker in combined
        for marker in ("practice guideline", "clinical guideline", "consensus statement")
    )
    diagnostic = any(
        marker in combined
        for marker in ("diagnostic accuracy", "sensitivity", "specificity")
    )
    cohort = any(
        marker in combined
        for marker in ("cohort", "case-control", "prospective", "retrospective")
    )
    protocol = "protocol" in combined or "rationale and design" in combined

    wants_randomized = bool(query_tokens & _PUBMED_RANDOMIZED_TERMS)
    wants_synthesis = bool(query_tokens & _PUBMED_SYNTHESIS_TERMS)
    wants_diagnostic = bool(query_tokens & _PUBMED_DIAGNOSTIC_TERMS)
    wants_prognosis = any(term in query_lower for term in _PUBMED_PROGNOSIS_TERMS)
    wants_guideline = any(term in query_lower for term in _PUBMED_GUIDELINE_TERMS)
    wants_protocol = bool(query_tokens & _PUBMED_PROTOCOL_TERMS)
    wants_safety = bool(query_tokens & SAFETY_TERMS) or any(
        marker in query_lower
        for marker in ("adverse event", "adverse effect", "tolerability", "discontinuation")
    )

    design_score = 0.0
    if wants_guideline:
        design_score += 55 if guideline else 0
        design_score += 12 if systematic_review or meta_analysis else 0
    elif wants_diagnostic:
        design_score += 45 if diagnostic else 0
        design_score += 25 if systematic_review or meta_analysis else 0
    elif wants_prognosis:
        design_score += 38 if cohort else 0
        design_score += 22 if systematic_review or meta_analysis else 0
    elif wants_synthesis:
        design_score += 55 if meta_analysis else 0
        design_score += 45 if systematic_review else 0
        design_score += 10 if randomized else 0
    elif wants_randomized:
        design_score += 48 if randomized else 0
        design_score += 24 if systematic_review or meta_analysis else 0
    else:
        design_score += 30 if meta_analysis else 0
        design_score += 24 if systematic_review else 0
        design_score += 26 if randomized else 0
        design_score += 10 if cohort else 0

    if wants_safety and any(
        marker in combined
        for marker in (
            "adverse event", "adverse effect", "safety", "tolerability",
            "discontinuation", "toxicity", "serious adverse",
        )
    ):
        design_score += 22
    if any(marker in combined for marker in HUMAN_EVIDENCE_MARKERS):
        design_score += 12
    if any(marker in combined for marker in NONHUMAN_MARKERS):
        design_score -= 60
    if protocol and not wants_protocol:
        design_score -= 55

    year = str(item.get("year") or item.get("publicationDate") or "")[:4]
    recency = int(year) - 2000 if year.isdigit() else 0
    abstract_text = str(item.get("abstract") or item.get("text") or "").strip()
    # A candidate with a substantive PubMed abstract is much more likely to
    # remain usable when PMC full text is unavailable.  This is only a soft
    # public-metadata signal: a uniquely relevant title is never hard-dropped.
    readability_score = (
        min(12.0, len(abstract_text) / 120.0) if abstract_text else -12.0
    )
    # Prefer candidates with a known open full-text route when relevance and
    # study design are otherwise similar. This remains a soft signal because a
    # missing PMCID in PubMed does not rule out Europe PMC or a useful abstract.
    if str(item.get("full_text_hint") or "").casefold() == "pmc" or item.get("pmcid"):
        readability_score += 8.0
    elif item.get("abstract_available") is True and not abstract_text:
        # Compact/model-facing candidates may omit the abstract body while
        # retaining the authoritative availability flag.
        readability_score += 4.0
    return (
        80 * overlap
        + phrase_score
        + design_score
        + min(max(recency, 0), 30) * 0.25
        + readability_score
    )


def rank_pubmed_candidates(
    query: str,
    items: Iterable[dict[str, Any]],
    *,
    original_question: str | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates by public-query evidence utility without changing schema."""
    return sorted(
        (dict(item) for item in items),
        key=lambda item: (
            0.6 * pubmed_candidate_value_score(query, item)
            + 0.4 * pubmed_candidate_value_score(original_question or query, item),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )


def canonical_source_id(source_id: str) -> str:
    return str(source_id or "").split("#", 1)[0].strip()


def successful_browse_counts(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count successful Browse observations for each canonical source.

    Failed Browse attempts are deliberately excluded so a transient network or
    parser failure may be retried. A source may be opened successfully twice so
    a second focused query can fill a different evidence gap; later attempts
    must move to another paper or webpage.
    """
    counts: dict[str, int] = {}
    for event in events:
        if event.get("type") != "tool_output" or event.get("tool") not in {
            "browse_document",
            "browse_webpage",
        }:
            continue
        payload = event.get("output") or {}
        if (
            not isinstance(payload, dict)
            or payload.get("failed")
            or payload.get("error")
            or payload.get("duplicate_blocked")
        ):
            continue
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            source = canonical_source_id(str(item.get("source_id") or ""))
            if source:
                counts[source] = counts.get(source, 0) + 1
                break
    return counts


def browse_attempt_counts(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count model-requested Browse actions by canonical source.

    Runtime-blocked third attempts still appear as policy actions, but this
    counter is evaluated before dispatch and therefore guarantees at most two
    real backend executions for one PMID/WEB source.
    """
    counts: dict[str, int] = {}
    for event in events:
        if event.get("type") != "tool_call" or event.get("tool") not in {
            "browse_document",
            "browse_webpage",
        }:
            continue
        arguments = event.get("arguments") or {}
        source = canonical_source_id(
            str(arguments.get("source_id") or arguments.get("url") or "")
        )
        if source:
            counts[source] = counts.get(source, 0) + 1
    return counts


def failed_web_browse_sources(events: Iterable[dict[str, Any]]) -> set[str]:
    """Return WEB sources with deterministic unreadable-page failures.

    HTTP 4xx and terminal no-content results are cached immediately. Transient
    timeouts/5xx may be retried once with a different focused query; the shared
    two-attempt cap still prevents repeated waste.
    """
    failed: set[str] = set()
    pending_source: str | None = None
    for event in events:
        if event.get("type") == "tool_call":
            pending_source = None
            if event.get("tool") == "browse_webpage":
                arguments = event.get("arguments") or {}
                pending_source = canonical_source_id(
                    str(arguments.get("source_id") or arguments.get("url") or "")
                )
            continue
        if event.get("type") != "tool_output" or event.get("tool") != "browse_webpage":
            continue
        payload = event.get("output") or {}
        if (
            pending_source
            and isinstance(payload, dict)
            and not payload.get("duplicate_blocked")
            and payload.get("retryable") is not True
            and (
                payload.get("failed")
                or payload.get("error")
                or not (payload.get("data") or [])
            )
        ):
            failed.add(pending_source)
        pending_source = None
    return failed


_NON_RETRYABLE_DOCUMENT_FAILURE_CUES = (
    "no_readable_biomedical_document",
    "pubmed_record_not_found",
    "no readable biomedical document",
    "pubmed returned no record",
    "no readable pubmed abstract",
    "no open full text",
    "invalid pmid",
    "pmid must",
)
_TRANSIENT_DOCUMENT_FAILURE_CUES = (
    "timeout", "timed out", "connection reset", "connection aborted",
    "temporarily unavailable", "http 429", "http 500", "http 502",
    "http 503", "http 504", "retryerror",
)


def failed_document_browse_sources(events: Iterable[dict[str, Any]]) -> set[str]:
    """Return PMIDs with deterministic no-record/no-readable-content failures."""
    failed: set[str] = set()
    pending_source: str | None = None
    for event in events:
        if event.get("type") == "tool_call":
            pending_source = None
            if event.get("tool") == "browse_document":
                arguments = event.get("arguments") or {}
                pending_source = canonical_source_id(
                    str(arguments.get("source_id") or "")
                )
            continue
        if event.get("type") != "tool_output" or event.get("tool") != "browse_document":
            continue
        payload = event.get("output") or {}
        if not pending_source or not isinstance(payload, dict):
            pending_source = None
            continue
        failure_type = str(payload.get("failure_type") or "").casefold()
        value = " ".join(
            str(payload.get(key) or "") for key in ("error", "message")
        ).casefold()
        deterministic = (
            failure_type not in {"policy_invalid", "budget_violation", "final_answer_reserve"}
            and not any(cue in value for cue in _TRANSIENT_DOCUMENT_FAILURE_CUES)
            and any(cue in value for cue in _NON_RETRYABLE_DOCUMENT_FAILURE_CUES)
        )
        if deterministic and not payload.get("duplicate_blocked"):
            failed.add(pending_source)
        pending_source = None
    return failed


def tool_call_arguments(tool: str, content: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a parsed XML tool call into the canonical medical-tool schema."""
    arguments = dict(parameters or {})
    if tool in {"pubmed_search", "medical_web_search"}:
        arguments.setdefault("query", str(content or "").strip())
    elif tool in {"browse_document", "browse_webpage"}:
        arguments.setdefault("source_id", str(content or "").strip())
    else:
        arguments.setdefault("query", str(content or "").strip())
    return normalize_medical_tool_arguments(tool, arguments)


def normalized_tool_output(output: Any) -> dict[str, Any]:
    """Return the structured payload used by every medical runtime surface."""
    if isinstance(output, dict):
        return dict(output)
    raw = getattr(output, "raw_output", None)
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return {"data": list(raw)}
    error = str(getattr(output, "error", "") or "")
    return {
        "data": [],
        "failed": bool(error),
        "error": error or None,
    }


def opened_evidence_ids(events: Iterable[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in _opened_browse_events(events):
        for item in (event.get("output") or {}).get("data") or []:
            source_id = str(item.get("source_id") or "").strip()
            if source_id:
                ids.add(source_id)
    return ids


def _evidence_by_id(events: Iterable[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for event in _opened_browse_events(events):
        for item in (event.get("output") or {}).get("data") or []:
            source_id = str(item.get("source_id") or "").strip()
            if not source_id:
                continue
            result[source_id] = " ".join(
                str(item.get(key) or "")
                for key in ("title", "heading", "text", "snippet")
            )
    return result


def _claim_before_citation(answer: str, citation_start: int) -> str:
    prefix = answer[max(0, citation_start - 600):citation_start]
    parts = re.split(r"(?<=[.!?。！？;；])\s*", prefix)
    return (parts[-1] if parts else prefix).strip()


def deterministic_claim_support_issues(
    answer: str,
    events: Iterable[dict[str, Any]],
) -> list[str]:
    """Catch obvious claim/citation mismatches without pretending to be an NLI judge.

    Exact numeric claims must occur in the cited chunk.  For English prose, at
    least one non-trivial claim term must overlap.  A later LLM/NLI judge can be
    supplied for full semantic entailment; this deterministic gate is designed
    to reject clear mismatches while avoiding unsafe over-claims.
    """
    evidence = _evidence_by_id(events)
    issues: list[str] = []
    for match in CITE_RE.finditer(answer or ""):
        citation_id = match.group(1)
        cited_text = evidence.get(citation_id, "")
        if not cited_text:
            continue
        claim = _claim_before_citation(answer, match.start())
        claim_numbers = set(NUMBER_RE.findall(claim))
        cited_numbers = set(NUMBER_RE.findall(cited_text))
        missing_numbers = sorted(claim_numbers - cited_numbers)
        if missing_numbers:
            issues.append(
                f"citation {citation_id} does not contain claim numbers: {', '.join(missing_numbers)}"
            )
            continue
        claim_terms = {
            token for token in re.findall(r"[A-Za-z][A-Za-z-]{4,}", claim.lower())
            if token not in CLAIM_STOP_WORDS
        }
        cited_terms = set(re.findall(r"[A-Za-z][A-Za-z-]{4,}", cited_text.lower()))
        if claim_terms and not (claim_terms & cited_terms):
            issues.append(f"citation {citation_id} has no direct lexical support for its claim")
    return issues


def duplicate_action_reason(
    tool: str,
    arguments: dict[str, Any],
    prior_events: Iterable[dict[str, Any]],
) -> str | None:
    """Return a user-visible reason when an action adds no likely information."""
    prior_calls = [event for event in prior_events if event.get("type") == "tool_call"]
    query = str(arguments.get("query") or "").strip()
    if tool in {"pubmed_search", "medical_web_search"}:
        for event in prior_calls:
            if event.get("tool") != tool:
                continue
            old = str((event.get("arguments") or {}).get("query") or "").strip()
            if old and (old.casefold() == query.casefold() or _similarity(old, query) >= 0.85):
                return f"duplicate {tool} query; reformulate around an unmet evidence gap"
        return None

    source = canonical_source_id(
        str(arguments.get("source_id") or arguments.get("url") or "")
    )
    if tool == "browse_webpage" and source in failed_web_browse_sources(prior_events):
        return (
            f"source {source} previously failed to return readable webpage content; "
            "choose a different unopened webpage candidate"
        )
    if tool == "browse_document" and source in failed_document_browse_sources(prior_events):
        return (
            f"paper {source} previously had no readable biomedical document; "
            "choose a different unopened PubMed candidate"
        )
    attempt_count = browse_attempt_counts(prior_events).get(source, 0)
    if source and attempt_count >= MAX_BROWSE_ATTEMPTS_PER_SOURCE:
        return (
            f"source {source} was already attempted twice (two Browse attempts); choose a "
            "different paper or webpage for the remaining evidence gap"
        )
    return None


def unopened_candidates_for_tool(
    tool: str,
    prior_events: Iterable[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Expose discovered, unopened browse candidates without selecting one."""
    if tool not in {"browse_document", "browse_webpage"}:
        return []
    events = list(prior_events)
    expected_prefix = "PMID:" if tool == "browse_document" else "WEB:"
    opened = {
        canonical_source_id(
            str(
                (event.get("arguments") or {}).get("source_id")
                or (event.get("arguments") or {}).get("url")
                or ""
            )
        )
        for event in events
        if event.get("type") == "tool_call"
        and event.get("tool") in {"browse_document", "browse_webpage"}
    }
    failed_sources = failed_web_browse_sources(events) if tool == "browse_webpage" else set()
    source_domains: dict[str, str] = {}
    for event in events:
        if event.get("type") != "tool_output":
            continue
        for item in (event.get("output") or {}).get("data") or []:
            if not isinstance(item, dict):
                continue
            source_id = canonical_source_id(str(item.get("source_id") or ""))
            domain = str(item.get("domain") or _host(str(item.get("url") or "")))
            if source_id and domain:
                source_domains[source_id] = domain
    failed_domains = {
        source_domains[source_id]
        for source_id in failed_sources
        if source_domains.get(source_id)
    }
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "tool_output":
            continue
        for item in (event.get("output") or {}).get("data") or []:
            source_id = canonical_source_id(
                str(item.get("source_id") or item.get("pmid") or "")
            )
            if (
                not source_id.startswith(expected_prefix)
                or source_id in opened
                or source_id in seen
            ):
                continue
            seen.add(source_id)
            candidate = {"source_id": source_id}
            for key in ("title", "domain", "url", "year", "publicationDate", "source_type"):
                if item.get(key) not in (None, "", []):
                    candidate[key] = item[key]
            candidates.append(candidate)
    # After a protected-page failure, suggest a different domain first. This
    # only orders visible alternatives; the Agent still chooses the action.
    candidates.sort(
        key=lambda row: str(row.get("domain") or _host(str(row.get("url") or "")))
        in failed_domains
    )
    return candidates[: max(1, int(limit))]


def response_language(question: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", question or "") else "en"


def language_instruction(question: str) -> str:
    if response_language(question) == "zh":
        return "The final answer must be written in Chinese because the user asked in Chinese. Tool queries may use English."
    return "The final answer must be written in English because the user asked in English."


def answer_matches_language(question: str, answer: str) -> bool:
    natural_answer = re.sub(r"<[^>]+>", " ", answer or "")
    natural_answer = re.sub(
        r"\b(?:PMID:\d+|WEB:[0-9A-Fa-f]+)(?:#s\d+-c\d+)?\b",
        " ",
        natural_answer,
    )
    if response_language(question) == "zh":
        return len(re.findall(r"[\u4e00-\u9fff]", natural_answer)) >= 12
    # English answers can contain cited titles or identifiers, but should not be
    # predominantly Chinese.
    chinese = len(re.findall(r"[\u4e00-\u9fff]", natural_answer))
    latin = len(re.findall(r"[A-Za-z]", natural_answer))
    return latin >= max(20, chinese * 2)


def _opened_browse_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events
            if event.get("type") == "tool_output"
            and event.get("tool") in {"browse_document", "browse_webpage"}
            and not (event.get("output") or {}).get("failed")]


def _candidate_items(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "tool_output":
            continue
        for item in (event.get("output") or {}).get("data") or []:
            source_id = canonical_source_id(str(item.get("source_id") or item.get("pmid") or ""))
            if source_id:
                result.setdefault(source_id, item)
    return result


def evidence_coverage_gaps(
    question: str,
    events: Iterable[dict[str, Any]],
    route_class: str | None = None,
) -> list[str]:
    """Apply conservative, observable coverage gates before accepting an answer.

    This intentionally checks evidence *classes*, not whether a specific site or
    paper was chosen.  Claim-level entailment remains a separate judge task.
    """
    events = list(events)
    if route_class == "no_tool":
        return []
    browses = _opened_browse_events(events)
    candidates = _candidate_items(events)
    web_outputs = [event for event in browses if event.get("tool") == "browse_webpage"]
    doc_outputs = [event for event in browses if event.get("tool") == "browse_document"]
    gaps: list[str] = []

    if route_class in {"web_only", "hybrid"} and not web_outputs:
        gaps.append("open an authoritative webpage rather than relying on search metadata")
    if route_class in {"pubmed_only", "hybrid"} and not doc_outputs:
        gaps.append("open a directly relevant biomedical document")

    lowered = question.lower()
    tokens = _tokens(question)
    broad_safety = bool(tokens & SAFETY_TERMS) or any(term in question for term in ("无害", "危害", "安全吗", "安全性", "副作用", "健康影响"))
    current_authority = bool(tokens & CURRENT_AUTHORITY_TERMS) or any(term in question for term in ("最新", "当前", "指南", "批准", "警告", "监管", "推荐"))

    opened_urls: list[str] = []
    opened_texts: list[str] = []
    opened_sources: set[str] = set()
    for event in browses:
        output = event.get("output") or {}
        metadata = output.get("document_metadata") or {}
        if metadata.get("url"):
            opened_urls.append(str(metadata["url"]))
        for item in output.get("data") or []:
            source_id = canonical_source_id(str(item.get("source_id") or ""))
            if source_id:
                opened_sources.add(source_id)
            if item.get("url"):
                opened_urls.append(str(item["url"]))
            opened_texts.append(" ".join(str(item.get(key) or "") for key in ("title", "heading", "text")).lower())

    if (broad_safety or current_authority) and not any(is_high_authority_url(url) for url in opened_urls):
        gaps.append("open a regulator, public-health body, guideline producer, or independent evidence-review source")

    if broad_safety:
        human = False
        for event in doc_outputs:
            for item in (event.get("output") or {}).get("data") or []:
                source = canonical_source_id(str(item.get("source_id") or ""))
                candidate = candidates.get(source, {})
                text = " ".join(
                    str(value or "") for value in (
                        item.get("title"), item.get("text"), candidate.get("title"),
                        candidate.get("discovery_text"), " ".join(candidate.get("publication_types") or []),
                    )
                ).lower()
                if any(marker in text for marker in NONHUMAN_MARKERS):
                    continue
                if any(marker in text for marker in HUMAN_EVIDENCE_MARKERS):
                    human = True
                    break
            if human:
                break
        if not human:
            gaps.append("open direct human clinical or epidemiological evidence, preferably a synthesis or comparative study")
        if len(opened_sources) < 2:
            gaps.append("use at least two distinct opened sources for a broad safety or harm conclusion")

    return list(dict.fromkeys(gaps))


def validate_medical_answer(
    question: str,
    answer: str,
    events: Iterable[dict[str, Any]],
    route_class: str | None = None,
    *,
    require_opened_evidence: bool | None = None,
    claim_support_validator: Callable[[str, str, dict[str, str]], bool | str] | None = None,
) -> MedicalAnswerValidation:
    """Apply the shared deployment/evaluation final-answer gate.

    The deterministic checks cover protocol provenance, evidence classes,
    language, and obvious claim/citation mismatches.  A caller may provide an
    independent NLI/LLM validator for true semantic entailment; its rejection
    is folded into the same validation result.
    """
    events = list(events)
    citation_ids = tuple(CITE_RE.findall(answer or ""))
    opened_ids = opened_evidence_ids(events)
    if require_opened_evidence is None:
        require_opened_evidence = route_class != "no_tool" and bool(events)

    reasons: list[str] = []
    missing = sorted(set(citation_ids) - opened_ids)
    if missing:
        reasons.append("citations must be exact chunks opened in this session: " + ", ".join(missing))
    if require_opened_evidence and (not opened_ids or not citation_ids):
        reasons.append("a searched medical answer requires at least one citation from opened evidence")
    if route_class == "no_tool" and citation_ids:
        reasons.append("a no-tool answer must not fabricate or attach unopened citations")

    gaps = evidence_coverage_gaps(question, events, route_class)
    reasons.extend(gaps)
    language_ok = answer_matches_language(question, answer)
    if not language_ok:
        reasons.append(language_instruction(question))
    reasons.extend(deterministic_claim_support_issues(answer, events))

    support_checked = claim_support_validator is not None
    if claim_support_validator is not None:
        evidence = _evidence_by_id(events)
        verdict = claim_support_validator(question, answer, evidence)
        if verdict is not True:
            reasons.append(
                str(verdict) if isinstance(verdict, str) else "independent claim-support validation failed"
            )

    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    return MedicalAnswerValidation(
        accepted=not reasons,
        reasons=tuple(reasons),
        citation_ids=tuple(sorted(set(citation_ids))),
        opened_evidence_ids=tuple(sorted(opened_ids)),
        coverage_gaps=tuple(gaps),
        language_ok=language_ok,
        claim_support_checked=support_checked,
    )


def evidence_ledger(
    question: str,
    events: Iterable[dict[str, Any]],
    route_class: str | None,
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    """Build the hidden heuristic evidence ledger used by validators/rewards.

    This object contains audit-only routing information and explicit evidence
    gaps.  It must never be appended to the policy prompt.  Use
    :func:`public_runtime_state` for observations that are safe to expose to
    the agent.
    """
    events = list(events)
    calls_used = sum(event.get("type") == "tool_call" for event in events)
    remaining = max(0, int(max_tool_calls) - calls_used)
    browses = _opened_browse_events(events)
    has_web = any(event.get("tool") == "browse_webpage" for event in browses)
    has_document = any(event.get("tool") == "browse_document" for event in browses)
    gaps = evidence_coverage_gaps(question, events, route_class)
    opened_urls = [
        str(url)
        for event in browses
        for url in (
            [(event.get("output") or {}).get("document_metadata", {}).get("url")]
            + [item.get("url") for item in (event.get("output") or {}).get("data") or []]
        )
        if url
    ]
    opened_sources = {
        canonical_source_id(str(item.get("source_id") or ""))
        for event in browses
        for item in (event.get("output") or {}).get("data") or []
        if canonical_source_id(str(item.get("source_id") or ""))
    }
    if not gaps:
        guidance = "Required evidence classes are satisfied; answer now unless a material claim remains unsupported."
    elif remaining <= 2:
        guidance = (
            "Only fill the listed evidence gaps using the best unopened relevant candidate; "
            "do not repeat a broad search or reopen an overlapping source."
        )
    else:
        guidance = (
            "Choose the next action that fills one listed evidence gap; avoid repeating "
            "queries or reopening overlapping sources."
        )
    return {
        "max_agent_tool_calls": int(max_tool_calls),
        "calls_used": calls_used,
        "remaining_calls": remaining,
        "route_class": route_class,
        "evidence_slots": {
            "webpage_opened": has_web,
            "authoritative_web_opened": any(
                is_high_authority_url(url) for url in opened_urls
            ),
            "biomedical_document_opened": has_document,
            "distinct_opened_sources": len(opened_sources),
        },
        "remaining_evidence_gaps": gaps,
        "budget_guidance": guidance,
    }


def public_runtime_state(
    events: Iterable[dict[str, Any]],
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    """Return non-prescriptive execution state that is safe for the agent.

    The policy may see its remaining budget and a compact accounting of work
    already performed.  It must infer the evidence need and route from the
    question and opened observations; consequently this state intentionally
    excludes route labels, required evidence classes, remaining gaps, and
    tool recommendations.
    """
    events = list(events)
    calls_used = sum(event.get("type") == "tool_call" for event in events)
    browses = _opened_browse_events(events)
    opened_sources = {
        canonical_source_id(str(item.get("source_id") or ""))
        for event in browses
        for item in (event.get("output") or {}).get("data") or []
        if canonical_source_id(str(item.get("source_id") or ""))
    }
    failed_outputs = sum(
        event.get("type") == "tool_output"
        and bool((event.get("output") or {}).get("failed"))
        for event in events
    )
    duplicate_blocks = sum(
        event.get("type") == "tool_output"
        and bool((event.get("output") or {}).get("duplicate_blocked"))
        for event in events
    )
    return {
        "max_agent_tool_calls": int(max_tool_calls),
        "calls_used": calls_used,
        "remaining_calls": max(0, int(max_tool_calls) - calls_used),
        "successful_browses": len(browses),
        "distinct_opened_sources": len(opened_sources),
        "failed_tool_outputs": failed_outputs,
        "duplicate_blocks": duplicate_blocks,
    }
