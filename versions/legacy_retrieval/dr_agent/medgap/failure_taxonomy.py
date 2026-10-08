# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Controlled failure classes shared by rollout runtime and hidden reward."""

from __future__ import annotations

from typing import Any, Mapping


FAILURE_TYPES = {
    "search_no_results",
    "environment_failure",
    "policy_invalid",
    "budget_violation",
    "final_answer_reserve",
}

_NO_RESULTS = ("no results found", "no search results", "zero results")
_ENVIRONMENT = (
    "timeout", "timed out", "network error", "connection reset",
    "temporarily unavailable", "http 429", "http 500", "http 502",
    "http 503", "http 504", "scraping failed", "no readable webpage content",
    "no webpage content returned", "failed to fetch webpage", "could not fetch webpage",
)


def classify_error_text(error: str) -> str:
    value = str(error or "").casefold()
    if any(cue in value for cue in _NO_RESULTS):
        return "search_no_results"
    if any(cue in value for cue in _ENVIRONMENT):
        return "environment_failure"
    if "max_total_tool_calls_exceeded" in value:
        return "budget_violation"
    if "tool_use_budget_reserved_for_final_answer" in value:
        return "final_answer_reserve"
    return "policy_invalid"


def classify_failed_output(output: Mapping[str, Any]) -> str | None:
    if not output.get("failed"):
        return None
    explicit = str(output.get("failure_type") or "").strip().casefold()
    if explicit in FAILURE_TYPES:
        return explicit
    return classify_error_text(str(output.get("error") or ""))
