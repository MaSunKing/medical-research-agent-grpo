# Adapted public research release; see THIRD_PARTY_NOTICES.md.
import json
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

# Load the repository-local development secrets deterministically. Explicit
# process variables (for example the server secret launcher) retain priority
# because python-dotenv defaults to override=False.
dotenv.load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_API_KEY_FALLBACK = os.getenv("SERPER_API_KEY_FALLBACK")
SERPER_API_KEY_FALLBACK_2 = os.getenv("SERPER_API_KEY_FALLBACK_2")
SERPER_API_KEY_FALLBACK_3 = os.getenv("SERPER_API_KEY_FALLBACK_3")
TIMEOUT = int(os.getenv("API_TIMEOUT", 10))

# A key that has returned an authentication/quota error is skipped for the
# remainder of the process.  Only an in-memory marker is stored; keys are never
# written to logs or cache metadata.
_DISABLED_KEYS: set[str] = set()
_KEY_STATE_LOCK = Lock()


def _available_api_keys(explicit_api_key: str | None = None) -> list[str]:
    """Return primary-then-fallback keys without exposing or duplicating them."""
    if explicit_api_key:
        candidates = [explicit_api_key]
    else:
        # Preserve the historical unsuffixed first fallback, then accept
        # numbered fallbacks in order. Supporting a bounded sequence keeps
        # configuration simple while avoiding a code change for every key.
        names = ["SERPER_API_KEY", "SERPER_API_KEY_FALLBACK"]
        names.extend(f"SERPER_API_KEY_FALLBACK_{index}" for index in range(2, 21))
        candidates = [os.getenv(name) for name in names]
    unique = []
    for key in candidates:
        if key and key not in unique:
            unique.append(key)
    if not unique:
        raise ValueError(
            "SERPER_API_KEY is not set. Optionally set SERPER_API_KEY_FALLBACK "
            "and numbered fallbacks for automatic quota failover."
        )
    with _KEY_STATE_LOCK:
        enabled = [key for key in unique if key not in _DISABLED_KEYS]
    if not enabled:
        raise RuntimeError("All configured Serper API keys are unavailable or exhausted.")
    return enabled


def _is_key_failure(response: requests.Response) -> bool:
    """Identify auth/quota responses that justify switching API keys."""
    if response.status_code in {401, 403, 429}:
        return True
    body = response.text.lower()
    return response.status_code == 400 and any(
        marker in body for marker in ("credit", "quota", "rate limit", "exhaust")
    )


def _post_with_key_fallback(
    url: str,
    payload: str,
    api_key: str | None,
    operation: str,
) -> requests.Response:
    """Use the primary key until auth/quota failure, then try the fallback."""
    failures = []
    keys = _available_api_keys(api_key)
    for index, key in enumerate(keys, start=1):
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            # DNS/connect/read failures are endpoint- or network-specific, not
            # credential-specific. Rotating through every configured key turns
            # one 10-second scrape timeout into N x 10 seconds without changing
            # the request or its likelihood of success. Preserve fallback keys
            # for genuine authentication/quota failures only.
            raise RuntimeError(f"Serper {operation} network error: {exc}") from exc

        if response.status_code == 200:
            return response
        if _is_key_failure(response):
            with _KEY_STATE_LOCK:
                _DISABLED_KEYS.add(key)
            failures.append(f"key_{index}: HTTP {response.status_code} auth/quota failure")
            continue
        raise RuntimeError(
            f"Serper {operation} failed with HTTP {response.status_code}: {response.text}"
        )

    raise RuntimeError(f"Serper {operation} failed after key fallback: {'; '.join(failures)}")


class KnowledgeGraph(TypedDict, total=False):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Optional[Dict[str, str]]


class Sitelink(TypedDict):
    title: str
    link: str


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]]
    attributes: Optional[Dict[str, str]]
    date: Optional[str]


class PeopleAlsoAsk(TypedDict):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(TypedDict):
    query: str


class SearchResponse(TypedDict, total=False):
    searchParameters: Dict[str, Union[str, int, bool]]
    knowledgeGraph: Optional[KnowledgeGraph]
    organic: List[SearchResult]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]]
    relatedSearches: Optional[List[RelatedSearch]]


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: Union[int, str]
    citedBy: int


class ScholarResponse(TypedDict):
    searchParameters: Dict[str, Union[str, int, bool]]
    organic: List[ScholarResult]


class WebpageContentResponse(TypedDict, total=False):
    url: str
    text: str
    markdown: str
    metadata: Dict[str, Union[str, int, bool]]
    credits: int


@cached()
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",  # Can be "search", "places", "news", "images"
    api_key: str = None,
) -> SearchResponse:
    """
    Search using Serper.dev API for general web search.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
        gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
        hl: Host language of user interface (default: en)
        search_type: Type of search to perform (default: "search")
                    Options: "search", "places", "news", "images"
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        SearchResponse containing:
        - searchParameters: Dict with search metadata
        - knowledgeGraph: Optional knowledge graph information
        - organic: List of organic search results
        - peopleAlsoAsk: Optional list of related questions
        - relatedSearches: Optional list of related search queries
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query, "num": num_results, "gl": gl, "hl": hl, "type": search_type})

    try:
        response = _post_with_key_fallback(url, payload, api_key, "search")
        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper search: {str(e)}")


@cached()
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
) -> ScholarResponse:
    """
    Search academic papers using Serper.dev Scholar API.

    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        ScholarResponse containing:
        - organic: List of academic paper results with:
            - title: Paper title
            - link: URL to the paper
            - publicationInfo: Author and publication details
            - snippet: Brief excerpt from the paper
            - year: Publication year
            - citedBy: Number of citations
    """
    url = "https://google.serper.dev/scholar"

    payload = json.dumps({"q": query, "num": num_results})

    try:
        response = _post_with_key_fallback(url, payload, api_key, "scholar search")
        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper scholar search: {str(e)}")


@cached()
def fetch_webpage_content(
    url: str,
    include_markdown: bool = True,
    api_key: str = None,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Args:
        url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        WebpageContentResponse containing:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
    """
    scrape_url = "https://scrape.serper.dev"

    payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

    try:
        response = _post_with_key_fallback(scrape_url, payload, api_key, "webpage fetch")
        data = response.json()
        data["url"] = url
        return data

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage content: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Regular search example
    try:
        results = search_serper("apple inc", num_results=5)
        print("Regular Search Results:")
        print(f"Found {len(results.get('organic', []))} results")
        if "knowledgeGraph" in results:
            print(f"Knowledge Graph: {results['knowledgeGraph']['title']}")
        print()
    except Exception as e:
        print(f"Search error: {e}")

    # Scholar search example
    try:
        scholar_results = search_serper_scholar(
            "attention is all you need", num_results=5
        )
        print("Scholar Search Results:")
        print(f"Found {len(scholar_results.get('organic', []))} academic papers")
        for paper in scholar_results.get("organic", [])[:2]:
            print(
                f"- {paper['title']} ({paper['year']}) - Cited by: {paper['citedBy']}"
            )
        print()
    except Exception as e:
        print(f"Scholar search error: {e}")
