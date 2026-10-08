# Adapted public research release; see THIRD_PARTY_NOTICES.md.
import logging
import os
import re
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

import requests
from ...runtime_policy import pubmed_candidate_value_score, rank_pubmed_candidates
from ..cache import cached
from .semantic_scholar_apis import (
    download_paper_details_batch as download_paper_details_from_semantic_scholar_batch,
)
from .utils import call_api_with_retry

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_REQUEST_TIMEOUT = float(os.getenv("PUBMED_REQUEST_TIMEOUT", "30"))
EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUROPE_PMC_REQUEST_TIMEOUT = float(os.getenv("EUROPE_PMC_REQUEST_TIMEOUT", "30"))
logger = logging.getLogger(__name__)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
BIOMEDICAL_ARTICLE_HOSTS = {
    "nejm.org", "sciencedirect.com", "onlinelibrary.wiley.com",
    "ahajournals.org", "jacc.org", "jamanetwork.com", "bmj.com",
    "publications.aap.org", "endocrinepractice.org", "ajmc.com",
    "ard.bmj.com", "akjournals.com", "pubmed.ncbi.nlm.nih.gov",
}


class MedicalDocumentLoadError(RuntimeError):
    """Structured pre-document failure retained across the MCP boundary."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        fetch_attempts: List[Dict[str, str]],
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.fetch_attempts = list(fetch_attempts)


_PUBMED_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "do", "does", "for", "from", "has", "have",
    "how", "in", "into", "is", "it", "its", "may", "of", "on", "or",
    "should", "that", "the", "their", "there", "these", "this", "those",
    "to", "was", "were", "what", "when", "where", "which", "who", "with",
    "would", "according", "available", "current", "currently", "evidence",
    "latest", "literature", "peer-reviewed", "published", "research", "show",
    "studies", "study",
}

_PUBMED_DESIGN_TERMS = {
    "article", "articles", "cohort", "cohorts", "evidence", "literature",
    "meta-analysis", "observational", "paper", "papers", "peer-reviewed",
    "prospective", "randomized", "retrospective", "review", "reviews",
    "study", "studies", "systematic", "trial", "trials",
}


def pubmed_query_candidates(query: str) -> List[str]:
    """Build conservative PubMed fallbacks for a natural-language query.

    The original query is always attempted first.  Fallbacks only remove
    question framing and low-information prose; biomedical terms, numbers and
    comparison words are preserved.  These are backend requests and therefore
    still count as one model-visible tool call.
    """
    normalized = " ".join(query.replace("’", "'").replace("–", "-").split())
    if not normalized:
        return []

    candidates = [normalized]
    # Remove common instruction framing without encoding any disease-specific
    # behavior. NCBI Automatic Term Mapping still handles the resulting terms.
    reframed = re.sub(
        r"^(?:according to\s+)?(?:the\s+)?(?:current|latest)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    reframed = re.sub(
        r"^(?:what|which|how|when|where|who|does|do|is|are|can|should)\b[^:;?]{0,45}?\b(?:evidence|studies|literature|research)\b(?:\s+(?:show|suggest|support|report))?\s*(?:that|about|for|on)?\s*",
        "",
        reframed,
        flags=re.IGNORECASE,
    )
    reframed = reframed.strip(" ?.,:;")
    if reframed and reframed.casefold() != normalized.casefold():
        candidates.append(reframed)

    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", normalized)
    compact = [token for token in tokens if token.casefold() not in _PUBMED_QUERY_STOPWORDS]
    # A compact fallback is most helpful for verbose questions. Keep enough
    # terms to preserve population/intervention/comparator/outcome semantics.
    if len(tokens) >= 8 and len(compact) >= 3:
        compact_query = " ".join(compact[:32])
        if all(compact_query.casefold() != item.casefold() for item in candidates):
            candidates.append(compact_query)

    # Teacher-generated queries are often already compact but over-constrained
    # by several study-design words. A final fallback removes only those generic
    # design filters, keeping all clinical concepts intact.
    broad = [token for token in tokens if token.casefold() not in _PUBMED_DESIGN_TERMS]
    if len(broad) >= 3 and len(broad) < len(tokens):
        broad_query = " ".join(broad[:32])
        if all(broad_query.casefold() != item.casefold() for item in candidates):
            candidates.append(broad_query)

    # Compact model-generated searches can still over-constrain PubMed by
    # combining every requested outcome and duration term.  If earlier
    # rewrites produced fewer than three distinct attempts, progressively
    # relax to the first six informative concepts.  Original-query ranking is
    # still applied to the returned papers, so broad discovery does not decide
    # which candidate the Agent opens.
    if len(candidates) < 3 and len(compact) > 6:
        relaxed_query = " ".join(compact[:6])
        if all(relaxed_query.casefold() != item.casefold() for item in candidates):
            candidates.append(relaxed_query)
    return candidates[:3]


def _ordered_pubmed_discovery_queries(
    focused_query: str,
    original_question: str | None = None,
) -> List[str]:
    """Order bounded PubMed requests without letting verbose prose lead discovery.

    The original question remains the semantic anchor used by the downstream
    public-metadata and MedCPT rerankers.  It is not automatically sent as the
    first Entrez query: long natural-language strings can match broad adjacent
    topics and monopolize a fixed candidate pool before the focused variants
    are considered.
    """
    focused = pubmed_query_candidates(focused_query)
    if not focused:
        return []

    token_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", focused[0]))
    ordered: List[str] = []

    def add(candidate: str) -> None:
        if candidate and all(
            candidate.casefold() != existing.casefold() for existing in ordered
        ):
            ordered.append(candidate)

    # For a verbose Agent query, let its conservative compact/broadened forms
    # retrieve first.  For an already compact query, preserve the exact query
    # first so historical short-query behavior is unchanged.
    if token_count > 12 and len(focused) > 1:
        for candidate in focused[1:]:
            add(candidate)
        add(focused[0])
    else:
        for candidate in focused:
            add(candidate)

    # Only use an original-question rewrite to fill an otherwise short bundle.
    # The unmodified original question is already used by both rerankers and
    # need not consume a discovery request merely to provide an anchor.
    if original_question and len(ordered) < 3:
        original_candidates = pubmed_query_candidates(original_question)
        preferred = original_candidates[1:] or original_candidates[:1]
        for candidate in preferred:
            add(candidate)
            if len(ordered) >= 3:
                break
    return ordered[:3]


def _round_robin_unique_ids(
    query_id_lists: List[List[str]],
    limit: int,
) -> List[str]:
    """Fuse PubMed result lists fairly so one query cannot fill the pool."""
    fused: List[str] = []
    seen = set()
    max_depth = max((len(values) for values in query_id_lists), default=0)
    for rank in range(max_depth):
        for values in query_id_lists:
            if rank >= len(values):
                continue
            pmid = values[rank]
            if pmid in seen:
                continue
            seen.add(pmid)
            fused.append(pmid)
            if len(fused) >= limit:
                return fused
    return fused


def _ncbi_common_params() -> Dict[str, str]:
    """Return the identification parameters recommended by NCBI."""
    params = {
        "tool": os.getenv("NCBI_TOOL", "dr-tulu-medical-agent"),
        "email": os.getenv("NCBI_EMAIL", "anonymous@example.com"),
    }
    if api_key := os.getenv("NCBI_API_KEY"):
        params["api_key"] = api_key
    return params


def _get_xml(url: str, params: Dict) -> ElementTree.Element:
    def request_once() -> ElementTree.Element:
        response = requests.get(url, params=params, timeout=PUBMED_REQUEST_TIMEOUT)
        response.raise_for_status()
        return ElementTree.fromstring(response.content)

    # NCBI may return 429 during bursty Teacher collection. Retrying here keeps
    # transport throttling out of the model-visible tool policy; batch callers
    # should still serialize PubMed work when no NCBI API key is configured.
    return call_api_with_retry(request_once)


def extract_all_text(tag: ElementTree.Element) -> str:
    """
    Sometimes tag.text will produce a None value if there's rich text
    inside the tag. For example,

    In this paper, https://pubmed.ncbi.nlm.nih.gov/39355906/, the returned
    title data is the following:

    <ArticleTitle><i>LRP1</i> Repression by SNAIL Results in ECM Remodeling
    in Genetic Risk for Vascular Diseases.</ArticleTitle>

    And tag.text will return None.

    This function will extract all text from the tag, including rich text.
    """
    return " ".join([_.strip() for _ in tag.itertext()])


@cached(ttl=86400 * 10)
def search_pubmed_with_keywords(keywords: str, offset: int = 0, limit: int = 10):
    search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": keywords,
        "retmax": limit,
        "retstart": offset,
        "usehistory": "n",
        "sort": "relevance",
        **_ncbi_common_params(),
    }
    root = _get_xml(search_url, params)
    id_list = [id_elem.text for id_elem in root.findall("./IdList/Id")]
    #  (root.find("./Count").text, root.find("./RetStart").text, root.find("./RetMax").text)
    return {
        "ids": id_list,
        "count": root.find("./Count").text,
        "offset": root.find("./RetStart").text,
        "limit": root.find("./RetMax").text,
        "next": int(root.find("./RetStart").text) + int(root.find("./RetMax").text),
    }


@cached(ttl=86400 * 10)
def fetch_pubmed_details(id_list):
    if not id_list:
        return []

    fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
        **_ncbi_common_params(),
    }
    papers = _get_xml(fetch_url, params)

    paper_data_list = []
    for paper in papers.findall("./PubmedArticle"):
        article = paper.find(".//Article")
        pmid = paper.find(".//PMID").text
        title = (
            extract_all_text(article.find(".//ArticleTitle"))
            if article.find(".//ArticleTitle") is not None
            else ""
        )
        abstract = (
            "\n".join(
                [
                    extract_all_text(abstract_text)
                    for abstract_text in article.findall(".//Abstract/AbstractText")
                ]
            )
            if article.find(".//Abstract") is not None
            else None
        )
        abstract = []
        if article.find(".//Abstract") is not None:
            for abstract_text in article.findall(".//Abstract/AbstractText"):
                if abstract_text.attrib.get("Label"):
                    abstract.append(f"{abstract_text.attrib['Label']}")
                abstract.append(extract_all_text(abstract_text))
        abstract = "\n".join(abstract)

        authors = [
            {
                "name": f"{author.find('./LastName').text}, {author.find('./ForeName').text}"
            }
            for author in article.findall(".//Author")
            if author.find("./LastName") is not None
            and author.find("./ForeName") is not None
        ]
        year = article.find(".//Journal/JournalIssue/PubDate/Year")
        venue = (
            article.find(".//Journal/Title").text
            if article.find(".//Journal/Title") is not None
            else None
        )
        article_dates = article.findall(".//ArticleDate")

        publication_date = None
        if article_dates:
            # Grab the first ArticleDate's Year element. Adjust as necessary for Month/Day.
            article_date_year = article_dates[0].find("Year")
            publication_date = (
                article_date_year.text if article_date_year is not None else None
            )

        doi = None
        pmcid = None
        for article_id in paper.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text
            elif article_id.attrib.get("IdType") == "pmc":
                pmcid = article_id.text

        publication_types = [
            node.text
            for node in article.findall(".//PublicationTypeList/PublicationType")
            if node.text
        ]
        normalized_publication_types = " ".join(publication_types).casefold()
        if "randomized controlled trial" in normalized_publication_types:
            study_design = "randomized_controlled_trial"
        elif "clinical trial" in normalized_publication_types:
            study_design = "clinical_trial"
        elif "meta-analysis" in normalized_publication_types:
            study_design = "meta_analysis"
        elif "systematic review" in normalized_publication_types:
            study_design = "systematic_review"
        elif "review" in normalized_publication_types:
            study_design = "review"
        else:
            study_design = "unspecified"

        paper_data = {
            # Stable, source-qualified identifier used by the medical agent layer.
            "source_id": f"PMID:{pmid}",
            "source": "PubMed",
            "source_type": "journal_article",
            "pmid": pmid,
            "paperId": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "externalIds": {"PubMed": pmid},
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "text": abstract,
            "year": year.text if year is not None else None,
            "venue": venue,
            "journal": venue,
            "doi": doi,
            "pmcid": pmcid,
            "abstract_available": bool(abstract.strip()),
            # A missing PMCID here is not proof that Europe PMC has no full text.
            # Keep the state tri-valued instead of incorrectly labelling it false.
            "full_text_hint": "pmc" if pmcid else "unknown",
            "study_design": study_design,
            "publication_types": publication_types,
            "publicationDate": publication_date,
        }
        paper_data_list.append(paper_data)
    return paper_data_list


def fetch_semantic_scholar_details(paper_data: List[Dict]):

    paper_ids = [f'PMID:{paper["externalIds"]["PubMed"]}' for paper in paper_data]

    try:
        results = download_paper_details_from_semantic_scholar_batch(paper_ids)
        # print(results)
        # print(paper_data)
        for idx in range(len(paper_data)):
            semantic_scholar_data = results[idx]
            for key in semantic_scholar_data.keys():
                if key not in paper_data[idx]:
                    paper_data[idx][key] = semantic_scholar_data[key]
        # print(paper_data)
    except Exception as exc:
        for paper in paper_data:
            paper.update({"citationCount": None})
        logger.warning("Error fetching PubMed citation counts from Semantic Scholar: %s", exc)

    # for paper in paper_data:
    #     paper_id = paper["externalIds"]["PubMed"]
    #     try:
    #         semantic_scholar_data = download_paper_details_from_semantic_scholar(f"PMID:{paper_id}")
    #         # We prioritize the data from PubMed
    #         for key in semantic_scholar_data.keys():
    #             if key not in paper:
    #                 paper[key] = semantic_scholar_data[key]
    #     except:
    #         paper.update({"citationCount": None})
    #     time.sleep(0.2)  # Add a delay to avoid rate limiting

    return paper_data


def rerank_pubmed_candidates_v28(
    query: str,
    paper_data: List[Dict],
    *,
    original_question: str | None = None,
    backend=None,
) -> List[Dict]:
    """Rerank PubMed candidates with MedCPT without changing their schema.

    PubMed's relevance ordering and the deterministic runtime policy remain the
    broad-recall stage.  In V28 mode, MedCPT then reranks a small leading pool
    using title and abstract text.  Any model/load/inference failure falls back
    to the input order so retrieval infrastructure cannot break Search.
    """
    rows = [dict(row) for row in paper_data]
    if len(rows) < 2:
        return rows
    if os.getenv("MEDGAP_V28_PUBMED_CANDIDATE_RERANK", "true").lower() != "true":
        return rows

    rerank_limit = max(
        2, int(os.getenv("MEDGAP_V28_PUBMED_CANDIDATE_RERANK_LIMIT", "10"))
    )
    head, tail = rows[:rerank_limit], rows[rerank_limit:]
    passages = [
        "\n".join(
            value
            for value in (
                str(row.get("title") or "").strip(),
                str(row.get("abstract") or row.get("text") or "").strip(),
            )
            if value
        )
        for row in head
    ]
    try:
        if backend is None:
            from .medical_passage_retriever_v28 import get_v28_backend

            backend = get_v28_backend()
        focused_scores, diagnostics = backend.cross_scores(query, passages)
        original_scores = focused_scores
        if original_question and original_question.casefold() != query.casefold():
            original_scores, original_diagnostics = backend.cross_scores(
                original_question, passages
            )
            diagnostics = {
                **diagnostics,
                "elapsed_ms": float(diagnostics.get("elapsed_ms") or 0.0)
                + float(original_diagnostics.get("elapsed_ms") or 0.0),
            }
        scores = [
            0.6 * float(focused) + 0.4 * float(original)
            for focused, original in zip(focused_scores, original_scores)
        ]
        if len(scores) != len(head):
            raise ValueError("MedCPT candidate reranker returned the wrong score count")
        medcpt_order = sorted(
            range(len(head)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )
        medcpt_rank = {
            index: rank for rank, index in enumerate(medcpt_order, start=1)
        }
        value_scores = [
            0.6 * pubmed_candidate_value_score(query, row)
            + 0.4 * pubmed_candidate_value_score(original_question or query, row)
            for row in head
        ]
        value_order = sorted(
            range(len(head)),
            key=lambda index: (value_scores[index], -index),
            reverse=True,
        )
        value_rank = {
            index: rank for rank, index in enumerate(value_order, start=1)
        }
        rrf_k = max(1.0, float(os.getenv("MEDGAP_PUBMED_RRF_K", "20")))
        value_weight = max(
            0.0, float(os.getenv("MEDGAP_PUBMED_VALUE_WEIGHT", "1.1"))
        )
        fused_scores = [
            1.0 / (rrf_k + medcpt_rank[index])
            + value_weight / (rrf_k + value_rank[index])
            for index in range(len(head))
        ]
        ranked = [
            row
            for _, _, row in sorted(
                (
                    (fused_scores[index], -index, row)
                    for index, row in enumerate(head)
                ),
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
        ]
        logger.info(
            "V28 MedCPT reranked %d PubMed candidates in %.3f ms",
            len(head),
            float(diagnostics.get("elapsed_ms") or 0.0),
        )
        logger.info(
            "PubMed fused candidate order: %s",
            [str(row.get("source_id") or row.get("pmid") or "") for row in ranked],
        )
        return ranked + tail
    except Exception as exc:
        logger.warning("V28 PubMed candidate rerank failed; retaining prior order: %s", exc)
        return rows


def search_pubmed(
    keywords: str,
    limit: int = 10,
    offset: int = 0,
    include_citation_count: Optional[bool] = None,
    original_question: str | None = None,
):
    if not keywords.strip():
        raise ValueError("PubMed query must not be empty.")
    if not 1 <= limit <= 100:
        raise ValueError("PubMed limit must be between 1 and 100.")
    if offset < 0:
        raise ValueError("PubMed offset must be non-negative.")

    focused_candidates = pubmed_query_candidates(keywords)
    query_candidates = _ordered_pubmed_discovery_queries(
        keywords,
        original_question=original_question,
    )
    if not query_candidates:
        raise ValueError("PubMed query must not be empty.")

    attempted_queries = []
    searchStat = None
    ids: List[str] = []
    seen_ids = set()
    query_id_lists: List[List[str]] = []
    v28_top_up = (
        offset == 0
        and os.getenv("MEDGAP_PASSAGE_RETRIEVAL_MODE", "bm25").strip().lower() == "v28"
        and os.getenv("MEDGAP_V28_PUBMED_QUERY_TOPUP", "true").lower() == "true"
    )
    v37_multiquery_fusion = (
        offset == 0
        and (
            bool(original_question)
            or os.getenv("MEDGAP_V37_PUBMED_MULTIQUERY_FUSION", "false")
            .strip().lower() in {"1", "true", "yes"}
        )
    )
    candidate_pool_limit = min(
        30,
        max(
            limit,
            int(os.getenv("MEDGAP_V37_PUBMED_CANDIDATE_POOL", str(limit * 3))),
        ),
    )
    # Legacy mode uses later rewrites only after a genuine zero-result response.
    # V37 deliberately fuses all bounded first-page variants; pagination stays
    # tied to the original query in both modes.
    for candidate_index, candidate in enumerate(
        query_candidates if offset == 0 else query_candidates[:1]
    ):
        attempted_queries.append(candidate)
        candidate_stat = search_pubmed_with_keywords(
            candidate,
            offset=offset,
            limit=(limit if not v37_multiquery_fusion else candidate_pool_limit),
        )
        if searchStat is None or (not ids and candidate_stat["ids"]):
            searchStat = candidate_stat
        candidate_ids = list(candidate_stat["ids"])
        query_id_lists.append(candidate_ids)
        if v37_multiquery_fusion:
            ids = _round_robin_unique_ids(query_id_lists, candidate_pool_limit)
            seen_ids = set(ids)
        else:
            for pmid in candidate_ids:
                if pmid not in seen_ids:
                    seen_ids.add(pmid)
                    ids.append(pmid)
        if ids and not v37_multiquery_fusion and (not v28_top_up or len(ids) >= limit):
            break
        # Multi-query fusion deliberately executes the complete bounded bundle.
        # Round-robin fusion, rather than early truncation, gives every query a
        # chance to contribute candidates before public metadata and MedCPT rank.
    assert searchStat is not None
    ids = ids[: candidate_pool_limit if v37_multiquery_fusion else limit]

    paper_data = fetch_pubmed_details(ids)
    if include_citation_count is None:
        include_citation_count = (
            os.getenv("PUBMED_ENRICH_SEMANTIC_SCHOLAR", "false").lower() == "true"
        )
    if include_citation_count and paper_data:
        paper_data = call_api_with_retry(fetch_semantic_scholar_details, paper_data)
    else:
        for paper in paper_data:
            paper.setdefault("citationCount", None)

    # Keep candidate ordering identical for MCP, local deployment, evaluation,
    # and rollout workers.  The model still chooses what to open.
    paper_data = rank_pubmed_candidates(
        keywords, paper_data, original_question=original_question
    )
    if os.getenv("MEDGAP_PASSAGE_RETRIEVAL_MODE", "bm25").strip().lower() == "v28":
        paper_data = rerank_pubmed_candidates_v28(
            keywords, paper_data, original_question=original_question
        )
    paper_data = paper_data[:limit]

    return {
        "query": keywords,
        "backend_queries": attempted_queries,
        "backend_fallback_used": len(attempted_queries) > 1,
        "backend_multiquery_fusion": v37_multiquery_fusion,
        "anchored_query_bundle": bool(original_question),
        "candidate_pool_size": len(ids),
        "total": max(int(searchStat["count"]), len(paper_data)),
        "offset": int(searchStat["offset"]),
        "next": int(searchStat["next"]),
        "data": paper_data,
    }


def normalize_pmid(source_id: str) -> str:
    value = source_id.strip()
    if value.upper().startswith("PMID:"):
        value = value.split(":", 1)[1].strip()
    if not value.isdigit():
        raise ValueError("source_id must be a PMID such as PMID:38914124")
    return value


def doi_from_url(url: str) -> Optional[str]:
    """Extract a DOI embedded in a publisher URL without contacting the site."""
    decoded = unquote(str(url or ""))
    match = DOI_RE.search(decoded)
    if not match:
        return None
    return match.group(0).rstrip(".,;)]}").casefold()


@cached(ttl=86400 * 30)
def resolve_doi_to_pmid(doi: str) -> Optional[str]:
    """Resolve a DOI through free NCBI metadata and verify the returned record."""
    normalized = str(doi or "").strip().casefold()
    if not DOI_RE.fullmatch(normalized):
        raise ValueError("invalid DOI")
    result = search_pubmed_with_keywords(f'"{normalized}"[AID]', limit=3)
    for paper in fetch_pubmed_details(result.get("ids") or []):
        if str(paper.get("doi") or "").strip().casefold() == normalized:
            return normalize_pmid(str(paper.get("pmid") or paper.get("paperId") or ""))
    return None


@cached(ttl=86400 * 30)
def resolve_web_article_to_pmid(url: str, title: str = "") -> Optional[str]:
    """Resolve a blocked article page to a verified PubMed record.

    DOI matching is exact.  Title fallback is deliberately conservative so an
    unrelated PubMed result cannot silently replace the selected webpage.
    """
    if doi := doi_from_url(url):
        resolved = resolve_doi_to_pmid(doi)
        if resolved:
            return resolved
    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    article_like = any(
        host == candidate or host.endswith("." + candidate)
        for candidate in BIOMEDICAL_ARTICLE_HOSTS
    ) or any(marker in parsed.path.casefold() for marker in ("/article/", "/doi/"))
    if not article_like:
        return None
    normalized_title = " ".join(str(title or "").split())
    title_tokens = set(re.findall(r"[a-z0-9]+", normalized_title.casefold()))
    if len(title_tokens) < 5:
        return None
    result = search_pubmed_with_keywords(f'"{normalized_title}"[Title]', limit=3)
    for paper in fetch_pubmed_details(result.get("ids") or []):
        candidate_tokens = set(
            re.findall(r"[a-z0-9]+", str(paper.get("title") or "").casefold())
        )
        similarity = len(title_tokens & candidate_tokens) / max(
            1, len(title_tokens | candidate_tokens)
        )
        if similarity >= 0.8:
            return normalize_pmid(str(paper.get("pmid") or paper.get("paperId") or ""))
    return None


@cached(ttl=86400 * 30)
def resolve_pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Resolve a PMID to a PMC open-full-text identifier when available."""
    pmid = normalize_pmid(pmid)
    root = _get_xml(
        f"{PUBMED_BASE_URL}/elink.fcgi",
        {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "linkname": "pubmed_pmc",
            **_ncbi_common_params(),
        },
    )
    linked_id = root.findtext(".//LinkSetDb/Link/Id")
    return f"PMC{linked_id}" if linked_id else None


@cached(ttl=86400 * 30)
def fetch_pmc_xml(pmcid: str) -> bytes:
    value = pmcid.strip().upper()
    if value.startswith("PMC"):
        value = value[3:]
    if not value.isdigit():
        raise ValueError("pmcid must look like PMC1234567")
    params = {
        "db": "pmc",
        "id": value,
        "retmode": "xml",
        **_ncbi_common_params(),
    }
    def request_once() -> bytes:
        response = requests.get(
            f"{PUBMED_BASE_URL}/efetch.fcgi",
            params=params,
            timeout=PUBMED_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        if b"<article" not in response.content:
            raise ValueError(f"PMC returned no article XML for PMC{value}")
        return response.content

    return call_api_with_retry(request_once)


@cached(ttl=86400 * 30)
def resolve_pmid_to_europe_pmcid(pmid: str) -> Optional[str]:
    """Resolve a PMID through Europe PMC without changing model-visible IDs."""
    pmid = normalize_pmid(pmid)

    def request_once() -> Optional[str]:
        response = requests.get(
            f"{EUROPE_PMC_BASE_URL}/search",
            params={
                "query": f"EXT_ID:{pmid} AND SRC:MED",
                "resultType": "core",
                "format": "json",
                "pageSize": 1,
            },
            timeout=EUROPE_PMC_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        value = str(results[0].get("pmcid") or "").strip().upper()
        return value if re.fullmatch(r"PMC\d+", value) else None

    return call_api_with_retry(request_once)


@cached(ttl=86400 * 30)
def resolve_pmid_to_open_pdf_url(pmid: str) -> Optional[str]:
    """Resolve a legal Europe PMC open-access PDF without changing PMID IDs."""
    pmid = normalize_pmid(pmid)

    def request_once() -> Optional[str]:
        response = requests.get(
            f"{EUROPE_PMC_BASE_URL}/search",
            params={
                "query": f"EXT_ID:{pmid} AND SRC:MED",
                "resultType": "core",
                "format": "json",
                "pageSize": 1,
            },
            timeout=EUROPE_PMC_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        urls = results[0].get("fullTextUrlList", {}).get("fullTextUrl", []) or []
        for item in urls:
            if str(item.get("documentStyle") or "").lower() != "pdf":
                continue
            value = str(item.get("url") or "").strip()
            if value.startswith("https://"):
                return value
        return None

    return call_api_with_retry(request_once)


@cached(ttl=86400 * 30)
def fetch_europe_pmc_xml(pmcid: str) -> bytes:
    value = str(pmcid or "").strip().upper()
    if not re.fullmatch(r"PMC\d+", value):
        raise ValueError("pmcid must look like PMC1234567")

    def request_once() -> bytes:
        response = requests.get(
            f"{EUROPE_PMC_BASE_URL}/{value}/fullTextXML",
            timeout=EUROPE_PMC_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        if b"<article" not in response.content:
            raise ValueError(f"Europe PMC returned no article XML for {value}")
        return response.content

    return call_api_with_retry(request_once)


@cached(ttl=86400 * 30)
def load_medical_document(
    source_id: str, parser_version: str = "jats_multi_abstract_v3_europepmc"
) -> Dict:
    """Load normalized PMC full text, falling back to the PubMed abstract."""
    from .medical_document_parser import parse_medical_xml

    pmid = normalize_pmid(source_id)
    fetch_attempts: List[Dict[str, str]] = []
    try:
        papers = fetch_pubmed_details([pmid])
    except Exception as exc:
        fetch_attempts.append(
            {
                "stage": "pubmed_record",
                "status": "error",
                "error_type": type(exc).__name__,
            }
        )
        raise MedicalDocumentLoadError(
            f"PubMed record fetch failed for PMID:{pmid}",
            error_code="paper_document_load_failed",
            fetch_attempts=fetch_attempts,
        ) from exc
    if not papers:
        fetch_attempts.append({"stage": "pubmed_record", "status": "not_available"})
        raise MedicalDocumentLoadError(
            f"PubMed returned no record for PMID:{pmid}",
            error_code="pubmed_record_not_found",
            fetch_attempts=fetch_attempts,
        )
    fetch_attempts.append({"stage": "pubmed_record", "status": "success"})
    paper = papers[0]
    pubmed_metadata = {
        "title": paper.get("title") or "",
        "publication_types": list(paper.get("publication_types") or []),
        "publication_date": paper.get("publicationDate") or "",
        "year": paper.get("year"),
        "journal": paper.get("journal") or paper.get("venue") or "",
        "doi": paper.get("doi"),
        "abstract_available": bool(str(paper.get("abstract") or "").strip()),
        "study_design": paper.get("study_design") or "unspecified",
    }
    try:
        pmcid = resolve_pmid_to_pmcid(pmid)
    except Exception as exc:
        # NCBI link-resolution failure must not prevent the independent Europe
        # PMC fallback from resolving the same PMID.
        pmcid = None
        fetch_attempts.append(
            {
                "stage": "ncbi_pmc_resolve",
                "status": "error",
                "error_type": type(exc).__name__,
            }
        )
        logger.warning("NCBI PMC PMID resolution failed for PMID:%s: %s", pmid, exc)
    else:
        fetch_attempts.append(
            {
                "stage": "ncbi_pmc_resolve",
                "status": "success" if pmcid else "not_available",
            }
        )
    if pmcid:
        try:
            document = parse_medical_xml(fetch_pmc_xml(pmcid))
            fetch_attempts.append({"stage": "ncbi_pmc_fetch", "status": "success"})
            document["metadata"].update(
                {
                    "source_id": f"PMID:{pmid}",
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                    "content_level": "pmc_full_text",
                    "abstract_only": False,
                    "full_text_available": True,
                    "evidence_scope": "full_text",
                    "fetch_method": "ncbi_pmc_full_text",
                    "fetch_attempts": list(fetch_attempts),
                    **pubmed_metadata,
                }
            )
            return document
        except Exception as exc:
            fetch_attempts.append(
                {
                    "stage": "ncbi_pmc_fetch",
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            )
            logger.warning("PMC full-text load failed for %s: %s", pmcid, exc)

    europe_pmcid = pmcid
    if not europe_pmcid:
        try:
            europe_pmcid = resolve_pmid_to_europe_pmcid(pmid)
        except Exception as exc:
            fetch_attempts.append(
                {
                    "stage": "europe_pmc_resolve",
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            )
            logger.warning("Europe PMC PMID resolution failed for PMID:%s: %s", pmid, exc)
        else:
            fetch_attempts.append(
                {
                    "stage": "europe_pmc_resolve",
                    "status": "success" if europe_pmcid else "not_available",
                }
            )
    if europe_pmcid:
        try:
            document = parse_medical_xml(fetch_europe_pmc_xml(europe_pmcid))
            fetch_attempts.append({"stage": "europe_pmc_fetch", "status": "success"})
            document["metadata"].update(
                {
                    "source_id": f"PMID:{pmid}",
                    "pmid": pmid,
                    "pmcid": europe_pmcid,
                    "url": f"https://europepmc.org/articles/{europe_pmcid}",
                    "content_level": "europe_pmc_full_text",
                    "abstract_only": False,
                    "full_text_available": True,
                    "evidence_scope": "full_text",
                    "fetch_method": "europe_pmc_full_text",
                    "fetch_attempts": list(fetch_attempts),
                    **pubmed_metadata,
                }
            )
            return document
        except Exception as exc:
            fetch_attempts.append(
                {
                    "stage": "europe_pmc_fetch",
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            )
            logger.warning("Europe PMC full-text load failed for %s: %s", europe_pmcid, exc)

    if (
        os.getenv("MEDGAP_PASSAGE_RETRIEVAL_MODE", "bm25").strip().lower() == "v28"
        and os.getenv("MEDGAP_MINERU_BASE_URL")
    ):
        try:
            pdf_url = resolve_pmid_to_open_pdf_url(pmid)
            if pdf_url:
                from .medical_document_parser import parse_medical_markdown
                from .mineru_client import get_mineru_client

                markdown, mineru_metadata = get_mineru_client().parse_url(pdf_url)
                document = parse_medical_markdown(markdown)
                document["metadata"].update(
                    {
                        "source_id": f"PMID:{pmid}",
                        "pmid": pmid,
                        "pmcid": europe_pmcid,
                        "url": pdf_url,
                        "content_level": "mineru_open_pdf_full_text",
                        "abstract_only": False,
                        "full_text_available": True,
                        "evidence_scope": "full_text",
                        "fetch_method": "europe_pmc_pdf_mineru",
                        "fetch_attempts": [
                            *fetch_attempts,
                            {"stage": "mineru_open_pdf", "status": "success"},
                        ],
                        **mineru_metadata,
                        **pubmed_metadata,
                    }
                )
                return document
        except Exception as exc:
            fetch_attempts.append(
                {
                    "stage": "mineru_open_pdf",
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            )
            logger.warning("MinerU open-PDF load failed for PMID:%s: %s", pmid, exc)

    abstract = paper.get("abstract") or ""
    fetch_attempts.append(
        {
            "stage": "pubmed_abstract",
            "status": "success" if abstract.strip() else "not_available",
        }
    )
    document = {
        "title": paper.get("title") or "",
        "text": abstract,
        "sections": [{"heading": "Abstract", "text": abstract}] if abstract else [],
        "metadata": {
            "source_format": "pubmed_xml",
            "section_count": 1 if abstract else 0,
            "source_id": f"PMID:{pmid}",
            "pmid": pmid,
            "pmcid": None,
            "url": paper.get("url"),
            "content_level": "pubmed_abstract",
            "abstract_only": True,
            "full_text_available": False,
            "evidence_scope": "abstract_only",
            "fetch_method": "pubmed_abstract",
            "fetch_attempts": list(fetch_attempts),
            "canonical_document_available": bool(abstract.strip()),
            **pubmed_metadata,
        },
    }
    return document
