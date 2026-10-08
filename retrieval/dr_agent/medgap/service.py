# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""End-to-end hidden reward service for live MedGap-GRPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from medgap_policy_markup import (
    parse_policy_owned_final_answer,
)
from medgap_final_repair import repair_policy_owned_final_answer

from dr_agent.medical_source_metadata import (
    AUTHORITY_TYPES,
    HUMAN_EVIDENCE_TYPES,
    STUDY_DESIGNS,
    infer_medical_source_metadata,
)

from .reward import (
    DecisionCredit,
    HiddenEvidenceLedger,
    source_satisfies_requirements,
)
from .failure_taxonomy import classify_failed_output
from .schemas import EvidenceChunk, EvidenceRubric, SourceMetadata
from .verifier import (
    QwenMaxVerifier,
    TERMINAL_BEHAVIOR_FALLBACK_PROMPT_VERSION,
    VerifierError,
    VerifierOutputError,
    citation_independent_evidence_order,
    citation_independent_semantic_answer,
)
from .terminal_reward import (
    linked_search_browse_credit,
    positive_reward_assessments_complete,
    terminal_behavior_reward,
    terminal_positive_reward,
)


_CALL_RE = re.compile(
    r"<call_tool\b(?P<attrs>[^>]*)>(?P<body>.*?)</call_tool>",
    re.DOTALL | re.IGNORECASE,
)
_OUTPUT_RE = re.compile(r"<tool_output\b[^>]*>(?P<body>.*?)</tool_output>", re.DOTALL | re.IGNORECASE)
_CITATION_RE = re.compile(r'<cite\s+id=["\'](?P<id>[^"\']+)["\']', re.IGNORECASE)
_CITATION_BLOCK_RE = re.compile(
    r'<cite\s+id=["\'](?P<id>[^"\']+)["\'][^>]*>(?P<body>.*?)</cite>',
    re.IGNORECASE | re.DOTALL,
)
_NAME_RE = re.compile(r"\bname\s*=\s*['\"](?P<name>[^'\"]+)['\"]", re.IGNORECASE)
_BASE_SOURCE_ID_RE = re.compile(r"^(?:PMID:\d+|WEB:[0-9A-Fa-f]+)$")
_CHUNK_SOURCE_ID_RE = re.compile(r"^(?:PMID:\d+|WEB:[0-9A-Fa-f]+)#s\d+-c\d+$")


def _base_source_id(evidence_id: str) -> str:
    return str(evidence_id).split("#", 1)[0]


def _citation_pointer_diagnostics(
    actual_citations: set[str], opened_ids: set[str]
) -> dict[str, Any]:
    opened_bases = {_base_source_id(value) for value in opened_ids}
    exact_opened = {value for value in actual_citations if value in opened_ids}
    base_opened = {
        value
        for value in actual_citations
        if _BASE_SOURCE_ID_RE.fullmatch(value) and value in opened_bases
    }
    source_unopened = actual_citations - exact_opened - base_opened
    return {
        "opened_bases": opened_bases,
        "exact_opened": exact_opened,
        "base_opened": base_opened,
        "source_unopened": source_unopened,
        "pointer_exact": bool(actual_citations) and not (actual_citations - opened_ids),
        "source_opened": bool(actual_citations) and not source_unopened,
    }


def _citation_resolution_score(claim: str, evidence: str) -> tuple[float, int]:
    """Score one claim against verifier-approved opened evidence.

    This is deliberately only a pointer resolver, not an entailment judge.  It
    may choose among chunks that the frozen verifier has already marked as
    supporting the answer, but it can never promote a new or unopened chunk.
    Numeric claims are only mapped to chunks containing the same numbers.
    """

    claim_tokens = {
        value.casefold()
        for value in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", claim or "")
    }
    evidence_tokens = {
        value.casefold()
        for value in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", evidence or "")
    }
    claim_numbers = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?", claim or ""))
    evidence_numbers = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?", evidence or ""))
    if claim_numbers and not claim_numbers.issubset(evidence_numbers):
        return (-1.0, 0)
    overlap = len(claim_tokens & evidence_tokens)
    return (overlap / max(1, len(claim_tokens)), overlap)


def _slot_support_type(item: Any) -> str:
    """Normalize cached V4 judgments and V5 judgments to one contract."""
    explicit = getattr(item, "support_type", None)
    if explicit in {"direct_evidence", "opened_evidence_limitation", "unsupported"}:
        return explicit
    return "direct_evidence" if item.supported_by_opened_evidence else "unsupported"


def _resolve_final_citations(
    answer_text: str,
    *,
    opened_chunks: list[EvidenceChunk],
    verifier_supported_ids: set[str],
) -> dict[str, Any]:
    """Repair citation pointers using only verifier-supported opened chunks.

    The model's citation remains an auditable suggestion.  A wrong pointer is
    replaced only when the cited claim can be mapped to a real opened chunk.
    Ambiguous or unsupported claims are left unchanged and reported rather
    than being force-mapped to convenient evidence.
    """

    evidence_by_id = {chunk.evidence_id: chunk.text for chunk in opened_chunks}
    eligible_ids = sorted(set(evidence_by_id) & set(verifier_supported_ids))
    repairs: list[dict[str, str]] = []
    removals: list[dict[str, str]] = []
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw_id = match.group("id").strip()
        prefix = (answer_text or "")[: match.start()]
        preceding = re.split(r"(?<=[.!?])\s+|\n+", prefix)[-1][-800:]
        claim = f"{preceding} {match.group('body')}".strip()
        if raw_id in eligible_ids:
            return match.group(0)

        candidates: list[tuple[float, int, str]] = []
        for evidence_id in eligible_ids:
            score, overlap = _citation_resolution_score(
                re.sub(r"<[^>]+>", " ", claim), evidence_by_id[evidence_id]
            )
            if score >= 0:
                candidates.append((score, overlap, evidence_id))
        candidates.sort(reverse=True)
        if not candidates:
            unresolved.append(raw_id)
            removals.append({"raw_id": raw_id, "reason": "no_supported_opened_match"})
            return match.group("body")
        best_score, best_overlap, replacement = candidates[0]
        tied = len(candidates) > 1 and candidates[1][:2] == candidates[0][:2]
        # One verifier-approved chunk is unambiguous.  With several possible
        # chunks, require lexical evidence and a unique best candidate.
        if len(candidates) > 1 and (best_overlap == 0 or tied):
            unresolved.append(raw_id)
            removals.append({"raw_id": raw_id, "reason": "ambiguous_supported_match"})
            return match.group("body")
        repairs.append({"raw_id": raw_id, "resolved_id": replacement})
        return f'<cite id="{replacement}">{claim}</cite>'

    raw_ids = [match.group("id").strip() for match in _CITATION_BLOCK_RE.finditer(answer_text or "")]
    resolved = _CITATION_BLOCK_RE.sub(replace, answer_text or "")
    if not raw_ids and eligible_ids:
        # The semantic verifier has already established which opened chunks
        # support the answer. Add provenance without asking the policy to copy
        # opaque chunk IDs perfectly. This never creates support: eligible_ids
        # is the strict intersection of verifier support and Runtime-opened
        # evidence.
        suffix = " ".join(
            f'<cite id="{evidence_id}">supporting opened evidence</cite>'
            for evidence_id in eligible_ids
        )
        resolved = f"{resolved.rstrip()}\n\nEvidence: {suffix}".strip()
        repairs.extend(
            {"raw_id": "", "resolved_id": evidence_id, "repair_type": "missing_citation"}
            for evidence_id in eligible_ids
        )
    resolved_ids = [
        match.group("id").strip() for match in _CITATION_BLOCK_RE.finditer(resolved)
    ]
    return {
        "raw_final": answer_text,
        "resolved_final": resolved,
        "raw_citation_ids": sorted(set(raw_ids)),
        "resolved_final_citation_ids": sorted(set(resolved_ids)),
        "citation_repaired": bool(repairs),
        "citation_repairs": repairs,
        "citation_removals": removals,
        "citation_resolver_unresolved_ids": sorted(set(unresolved)),
        "citation_resolver_success": bool(resolved_ids) and not unresolved and all(
            value in eligible_ids for value in resolved_ids
        ),
    }


def _trusted_receipt_for_call(
    receipts: list[dict[str, Any]],
    *,
    decision_index: int,
    tool_name: str,
) -> dict[str, Any] | None:
    """Return the Runtime-authenticated observation for one policy Browse call."""

    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        if receipt.get("schema_version") != "medgap_evidence_delivery_v1":
            continue
        if receipt.get("observation_delivered") is not True:
            continue
        if int(receipt.get("call_index") or -1) != decision_index + 1:
            continue
        if str(receipt.get("tool_name") or "") != tool_name:
            continue
        output = receipt.get("trusted_output")
        if not isinstance(output, dict) or output.get("failed") or output.get("error"):
            continue
        return receipt
    return None
@dataclass(frozen=True)
class RolloutRewardResult:
    trajectory_reward: float
    trajectory_reward_observed: bool
    final_answer_reward: float
    final_answer_reward_observed: bool
    terminal_behavior_penalty: float
    terminal_behavior_penalty_observed: bool
    terminal_behavior_bonus: float
    terminal_behavior_bonus_observed: bool
    local_returns: list[float]
    private_audit: dict[str, Any]


def rubric_from_ground_truth(value: Any) -> EvidenceRubric:
    """Extract a versioned hidden rubric from a transformed dataset value."""

    if isinstance(value, EvidenceRubric):
        return value
    if isinstance(value, list) and len(value) == 1:
        return rubric_from_ground_truth(value[0])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("MedGap ground_truth must contain JSON rubric data") from exc
    if not isinstance(value, dict):
        raise ValueError("MedGap ground_truth must be a rubric object")
    if "medgap_rubric" in value:
        value = value["medgap_rubric"]
    return EvidenceRubric.model_validate(value)


def _parse_output(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"failed": True, "error": "invalid_tool_output_json", "data": []}
    return value if isinstance(value, dict) else {"data": value}


def _paired_calls(
    transcript: str,
) -> list[tuple[str, str, dict[str, Any] | None]]:
    outputs = list(_OUTPUT_RE.finditer(transcript))
    calls = [
        match
        for match in _CALL_RE.finditer(transcript)
        if not any(output.start() <= match.start() < output.end() for output in outputs)
    ]
    result = []
    for index, call in enumerate(calls):
        next_call_start = calls[index + 1].start() if index + 1 < len(calls) else len(transcript)
        output = next(
            (candidate for candidate in outputs if call.end() <= candidate.start() < next_call_start),
            None,
        )
        name = _NAME_RE.search(call.group("attrs"))
        result.append(
            (
                name.group("name") if name else "",
                call.group("body").strip(),
                _parse_output(output.group("body")) if output else None,
            )
        )
    return result


def _candidate_source_ids(output: dict[str, Any] | None) -> list[str]:
    """Return normalized candidate IDs emitted by a Search observation."""

    if not isinstance(output, dict):
        return []
    values: set[str] = set()
    for field in ("data", "recommended_unopened_candidates"):
        items = output.get(field) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source_id = _base_source_id(str(item.get("source_id") or "").strip())
            if _BASE_SOURCE_ID_RE.fullmatch(source_id):
                values.add(source_id)
    return sorted(values)


def _browse_call_source_id(call_body: str) -> str | None:
    """Extract the exact PMID/WEB candidate selected by a Browse call."""

    match = re.search(r"(?:PMID:\d+|WEB:[0-9A-Fa-f]+)", call_body or "")
    return _base_source_id(match.group(0)) if match else None


def _failure_type(output: dict[str, Any]) -> str | None:
    """Return a controlled failure class, with backward-compatible inference."""
    return classify_failed_output(output)


_ENVIRONMENT_NEUTRAL_REASONS = {
    "environment_failure_neutral",
    "search_no_results_neutral",
    "final_answer_reserve_neutral",
}
_POLICY_FAILURE_PENALTIES = {
    "policy_invalid_action": -0.25,
    "unknown_research_tool": -0.25,
    "browse_without_verifiable_evidence": -0.10,
    "duplicate_action": -0.10,
    "budget_violation": -0.10,
    "post_reserve_tool_attempt": -0.10,
    "tool_call_after_coverage_complete": -0.10,
    "unnecessary_tool_for_no_tool_question": -0.10,
}

# These failures are fully established by policy-authored syntax plus trusted
# controller/rubric state.  They do not need a backend tool observation.  All
# other negative reasons remain fail-closed and require an exact atomic Runtime
# observation before they may create a policy gradient.
_DETERMINISTIC_POLICY_FAILURE_REASONS = {
    "unknown_research_tool",
    "post_reserve_tool_attempt",
    "tool_call_after_coverage_complete",
    "unnecessary_tool_for_no_tool_question",
}


def _parse_atomic_runtime_output(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """Parse one Runtime-owned atomic observation without blaming the policy.

    Invalid or missing Runtime diagnostics are an observability failure, not a
    policy action failure.  They therefore return ``None`` and are neutralized
    by the caller instead of being converted to ``policy_invalid``.
    """

    if not isinstance(raw, str) or not raw.strip():
        return None, "unobserved_missing_atomic_output"
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None, "unobserved_invalid_atomic_json"
    if not isinstance(value, dict):
        return None, "unobserved_nonobject_atomic_json"
    return value, "atomic_observed"


def _runtime_tool_outputs(
    runtime_diagnostics: list[dict[str, Any]],
    *,
    policy_tools: list[str],
) -> tuple[
    list[dict[str, Any] | None],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Bind Runtime-authenticated atomic observations to exact policy calls.

    V65 incorrectly parsed the cumulative ``tool_output_snapshot`` as though
    it were one JSON document.  V67 consumes only the new per-call
    ``atomic_tool_output`` field and requires an exact one-based ``call_index``
    and ``tool_name`` match.  Missing, malformed, duplicate, or mismatched
    diagnostics remain unobserved and can never create a policy penalty.
    """

    outputs: list[dict[str, Any] | None] = [None] * len(policy_tools)
    bindings: list[dict[str, Any]] = [
        {
            "decision_index": index,
            "tool_name": tool,
            "call_index": index + 1,
            "call_id": None,
            "status": "unobserved_no_checkpoint",
        }
        for index, tool in enumerate(policy_tools)
    ]
    bound_indexes: set[int] = set()
    invalidated_indexes: set[int] = set()
    ignored_status_counts: dict[str, int] = {}

    def ignore(status: str) -> None:
        ignored_status_counts[status] = ignored_status_counts.get(status, 0) + 1

    for item in runtime_diagnostics:
        if not isinstance(item, dict):
            ignore("ignored_nonobject_diagnostic")
            continue
        if item.get("diagnostic_type") != "runtime_final_checkpoint":
            continue
        if item.get("checkpoint_kind") != "atomic_tool_observation":
            ignore("ignored_nonatomic_checkpoint")
            continue
        call_index = item.get("call_index")
        if isinstance(call_index, bool) or not isinstance(call_index, int):
            ignore("ignored_missing_call_index")
            continue
        decision_index = call_index - 1
        if not 0 <= decision_index < len(policy_tools):
            ignore("ignored_out_of_range_call_index")
            continue
        if decision_index in bound_indexes or decision_index in invalidated_indexes:
            outputs[decision_index] = None
            invalidated_indexes.add(decision_index)
            bindings[decision_index] = {
                **bindings[decision_index],
                "call_id": item.get("call_id"),
                "status": "unobserved_duplicate_call_index",
            }
            ignore("duplicate_call_index")
            continue

        tool_name = str(item.get("tool_name") or "")
        if tool_name != policy_tools[decision_index]:
            invalidated_indexes.add(decision_index)
            bindings[decision_index] = {
                **bindings[decision_index],
                "call_id": item.get("call_id"),
                "runtime_tool_name": tool_name,
                "status": "unobserved_tool_name_mismatch",
            }
            ignore("tool_name_mismatch")
            continue

        output, status = _parse_atomic_runtime_output(
            item.get("atomic_tool_output")
        )
        bindings[decision_index] = {
            **bindings[decision_index],
            "call_id": item.get("call_id"),
            "runtime_tool_name": tool_name,
            "status": status,
        }
        if output is None:
            invalidated_indexes.add(decision_index)
            ignore(status)
            continue
        outputs[decision_index] = output
        bound_indexes.add(decision_index)

    status_counts: dict[str, int] = {}
    for binding in bindings:
        status = str(binding["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "mode": "atomic_call_index_tool_name_v67",
        "total_policy_tool_decisions": len(policy_tools),
        "atomic_observed_decisions": status_counts.get("atomic_observed", 0),
        "unobserved_decisions": len(policy_tools)
        - status_counts.get("atomic_observed", 0),
        "status_counts": status_counts,
        "ignored_checkpoint_status_counts": ignored_status_counts,
        "cumulative_snapshot_parsing_disabled": True,
    }
    return outputs, bindings, summary


def _policy_failure_subtype(error: str, reasons: set[str]) -> str | None:
    """Return an auditable subtype without changing the controlled taxonomy."""

    value = str(error or "").casefold()
    if "requires a web source id" in value or "requires a pmid source id" in value:
        return "wrong_source_id_type"
    if "unsupported" in value and ("argument" in value or "parameter" in value):
        return "unsupported_arguments"
    if "exact_action_already_attempted" in value or "duplicate_action" in reasons:
        return "duplicate_exact_action"
    if "tool_use_budget_reserved_for_final_answer" in value:
        return "tool_after_finalization_reserve"
    if "malformed" in value or "policy_invalid_action" in reasons:
        return "malformed_or_invalid_tool_call"
    if "budget_violation" in reasons:
        return "tool_budget_violation"
    if "browse_without_verifiable_evidence" in reasons:
        return "browse_without_verifiable_evidence"
    if "tool_call_after_coverage_complete" in reasons:
        return "tool_after_coverage_complete"
    if "unnecessary_tool_for_no_tool_question" in reasons:
        return "unnecessary_tool_call"
    if "unknown_research_tool" in reasons:
        return "unknown_research_tool"
    return None


def _apply_failure_aware_local_credit(
    decisions: list[dict[str, Any]],
    positive_returns: list[float],
) -> tuple[list[float], dict[str, int]]:
    """Overlay small action-local negatives while keeping environment failures neutral."""

    if len(decisions) != len(positive_returns):
        raise ValueError("failure-aware credit must align with policy decisions")
    returns = list(float(value) for value in positive_returns)
    counts = {
        "positive": 0,
        "policy_negative": 0,
        "atomic_runtime_negative": 0,
        "deterministic_policy_negative": 0,
        "unobserved_negative_neutralized": 0,
        "environment_masked": 0,
        "neutral_no_gain": 0,
    }
    for index, decision in enumerate(decisions):
        reasons = {str(value) for value in decision.get("reasons") or []}
        environment_masked = bool(reasons & _ENVIRONMENT_NEUTRAL_REASONS)
        penalty = min(
            (_POLICY_FAILURE_PENALTIES[reason] for reason in reasons if reason in _POLICY_FAILURE_PENALTIES),
            default=0.0,
        )
        if environment_masked:
            # Runtime/network failures are not evidence about policy quality.
            returns[index] = 0.0
            outcome = "environment_masked"
            counts["environment_masked"] += 1
        elif penalty < 0.0:
            runtime_status = str(
                decision.get("runtime_observation_status") or "unobserved"
            )
            atomic_runtime_proof = runtime_status == "atomic_observed"
            deterministic_policy_proof = bool(reasons) and reasons.issubset(
                _DETERMINISTIC_POLICY_FAILURE_REASONS
            )
            if atomic_runtime_proof:
                # Backend-derived failures are safe only when the Runtime
                # observation is bound to this exact call index and tool name.
                returns[index] = penalty
                outcome = "policy_controllable_negative"
                negative_proof_source = "atomic_runtime_observation"
                negative_requires_atomic_binding = True
                counts["policy_negative"] += 1
                counts["atomic_runtime_negative"] += 1
            elif deterministic_policy_proof:
                # These failures are provable from the policy action plus
                # trusted controller/rubric state; no backend observation is
                # logically required.
                returns[index] = penalty
                outcome = "policy_controllable_negative"
                negative_proof_source = "deterministic_policy_state"
                negative_requires_atomic_binding = False
                counts["policy_negative"] += 1
                counts["deterministic_policy_negative"] += 1
            else:
                # An unbound or ambiguous backend failure is observability
                # missingness, not evidence of a bad policy action.
                returns[index] = 0.0
                outcome = "unobserved_negative_neutralized"
                negative_proof_source = "unobserved"
                negative_requires_atomic_binding = True
                counts["unobserved_negative_neutralized"] += 1
        elif returns[index] > 0.0:
            outcome = "marginal_evidence_positive"
            negative_proof_source = "not_applicable"
            negative_requires_atomic_binding = False
            counts["positive"] += 1
        else:
            returns[index] = 0.0
            outcome = "neutral_no_gain"
            negative_proof_source = "not_applicable"
            negative_requires_atomic_binding = False
            counts["neutral_no_gain"] += 1
        if environment_masked:
            negative_proof_source = "environment_failure_neutral"
            negative_requires_atomic_binding = False
        decision["environment_masked"] = environment_masked
        decision["failure_aware_outcome"] = outcome
        decision["failure_aware_penalty"] = float(min(0.0, penalty))
        decision["negative_proof_source"] = negative_proof_source
        decision["negative_requires_atomic_binding"] = bool(
            negative_requires_atomic_binding
        )
        decision["negative_proof_verified"] = bool(
            outcome == "policy_controllable_negative"
        )
        decision["failure_subtype"] = _policy_failure_subtype(
            str(decision.get("failure_error") or ""), reasons
        )
    return returns, counts


def _annotate_all_browse_provenance(decisions: list[dict[str, Any]]) -> None:
    """Attach source-exact provenance even when a Browse earned zero gain."""

    matching_search = {
        "browse_document": "pubmed_search",
        "browse_webpage": "medical_web_search",
    }
    for browse_index, decision in enumerate(decisions):
        expected = matching_search.get(str(decision.get("tool") or ""))
        if expected is None:
            continue
        source_id = str(decision.get("source_candidate_id") or "")
        matches = [
            index
            for index in range(browse_index)
            if str(decisions[index].get("tool") or "") == expected
            and source_id
            in {
                str(value)
                for value in decisions[index].get("discovered_candidate_ids") or []
            }
        ]
        if len(matches) == 1:
            status = "exact"
            search_index: int | None = matches[0]
        elif matches:
            status = "ambiguous"
            search_index = None
        else:
            status = "unavailable"
            search_index = None
        decision["candidate_provenance_status"] = status
        decision["source_exact_search_decision_index"] = search_index


def _language_matches(question: str, answer: str) -> bool:
    # Protocol tags and opaque citation IDs are not natural-language content.
    # Counting the letters in ``<cite id="PMID:...">`` previously made a
    # Chinese answer look English even when its prose contained no English.
    natural_answer = re.sub(r"<[^>]+>", " ", answer or "")
    natural_answer = re.sub(
        r"\b(?:PMID:\d+|WEB:[0-9A-Fa-f]+)(?:#s\d+-c\d+)?\b",
        " ",
        natural_answer,
    )
    question_cjk = len(re.findall(r"[\u3400-\u9fff]", question))
    answer_cjk = len(re.findall(r"[\u3400-\u9fff]", natural_answer))
    if question_cjk:
        return answer_cjk >= 4
    latin = len(re.findall(r"[A-Za-z]", natural_answer))
    return latin >= max(8, answer_cjk)


def _chunks_from_output(output: dict[str, Any]) -> list[EvidenceChunk]:
    data = output.get("data") or []
    if not isinstance(data, list):
        return []
    document = output.get("document_metadata") or {}
    chunks = []
    for item in data:
        if not isinstance(item, dict) or not item.get("source_id") or not item.get("text"):
            continue
        source_id = str(document.get("source_id") or str(item["source_id"]).split("#", 1)[0])
        publication_types = item.get("publication_types") or document.get("publication_types") or []
        if isinstance(publication_types, str):
            publication_types = [publication_types]
        title = str(item.get("title") or output.get("title") or "")
        url = str(item.get("url") or document.get("url") or "")
        evidence_text = str(item.get("text") or "")
        inferred = infer_medical_source_metadata(
            title=title,
            url=url,
            publication_types=publication_types,
            evidence_text=evidence_text,
        )
        explicit_authority = item.get("authority_type") or document.get("authority_type")
        authority_type = (
            explicit_authority
            if explicit_authority in AUTHORITY_TYPES and explicit_authority != "none"
            else inferred["authority_type"]
        )

        explicit_design = item.get("study_design") or document.get("study_design")
        study_design = (
            explicit_design
            if explicit_design in STUDY_DESIGNS and explicit_design != "unknown"
            else inferred["study_design"]
        )
        explicit_human_types = item.get("human_evidence_types") or document.get("human_evidence_types")
        human_types = (
            list(explicit_human_types)
            if isinstance(explicit_human_types, list)
            else list(inferred["human_evidence_types"])
        )
        explicit_is_human = item.get("is_human", document.get("is_human"))
        is_human = bool(explicit_is_human) or bool(inferred["is_human"])
        chunks.append(
            EvidenceChunk(
                evidence_id=str(item["source_id"]),
                text=str(item["text"]),
                source=SourceMetadata(
                    source_id=source_id,
                    source_type=str(
                        item.get("source_type")
                        or document.get("content_level")
                        or document.get("source_format")
                        or ("webpage" if source_id.startswith("WEB:") else "biomedical_document")
                    ),
                    title=title,
                    url=url,
                    publication_types=[str(value) for value in publication_types],
                    publication_date=str(item.get("publication_date") or document.get("publication_date") or ""),
                    authority_type=authority_type,
                    study_design=study_design,
                    is_human=is_human,
                    human_evidence_types=sorted(set(human_types) & HUMAN_EVIDENCE_TYPES),
                ),
            )
        )
    return chunks


def _browse_support_for_slot(
    *,
    slot_id: str,
    verification: Any,
    source: SourceMetadata,
    rubric: EvidenceRubric,
    support_threshold: float,
    relevance_threshold: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Replay the exact ledger gates for one evidence/slot pair.

    This is audit-only: it does not alter coverage or reward.
    """

    reasons: list[str] = []
    slot = next((item for item in rubric.slots if item.slot_id == slot_id), None)
    support = next(
        (item for item in verification.slot_support if item.slot_id == slot_id),
        None,
    )
    metadata_requirement_met = bool(
        slot is not None and source_satisfies_requirements(slot.source_requirements, source)
    )
    if verification.major_contradiction:
        reasons.append("major_contradiction")
    if verification.evidence_relevance < relevance_threshold:
        reasons.append("evidence_below_relevance_threshold")
    if support is None:
        reasons.append("slot_not_assessed")
    else:
        if support.verdict != "supported":
            reasons.append(f"verdict_{support.verdict}")
        if support.confidence < support_threshold:
            reasons.append("confidence_below_support_threshold")
        if not support.source_requirement_met:
            reasons.append("semantic_source_requirement_not_met")
    if not metadata_requirement_met:
        reasons.append("metadata_source_requirement_not_met")
    detail = {
        "browse_evidence_relevance": verification.evidence_relevance,
        "browse_major_contradiction": verification.major_contradiction,
        "browse_verdict": support.verdict if support is not None else None,
        "browse_confidence": support.confidence if support is not None else None,
        "browse_source_requirement_met": (
            support.source_requirement_met if support is not None else None
        ),
        "metadata_source_requirement_met": metadata_requirement_met,
    }
    return not reasons, reasons, detail


def _browse_final_verifier_consistency(
    *,
    final_verification: Any | None,
    browse_verifications: dict[str, tuple[Any, SourceMetadata]],
    rubric: EvidenceRubric,
    support_threshold: float,
    relevance_threshold: float,
    current_covered_slots: set[str] | None = None,
) -> dict[str, Any]:
    """Compare final-supported citation pairs with their earlier Browse gates."""

    covered_slots = set(current_covered_slots or ())
    if final_verification is None:
        return {
            "status": "not_available",
            "credit_source": credit_source,
            "final_supported_evidence_pairs": 0,
            "browse_supported_pairs": 0,
            "disagreement_count": 0,
            "disagreements": [],
            "counterfactual_new_slots": [],
            "counterfactual_coverage": len(covered_slots) / max(1, len(rubric.slots)),
        }

    final_pairs = 0
    browse_supported_pairs = 0
    disagreements: list[dict[str, Any]] = []
    counterfactual_slots = set(covered_slots)
    for assessment in final_verification.slot_assessments:
        if not (assessment.addressed and assessment.supported_by_opened_evidence):
            continue
        evidence_ids = sorted(set(assessment.citation_ids))
        final_pairs += len(evidence_ids)
        assessment_details = []
        assessment_supported = False
        relevance_only_supported = False
        for evidence_id in evidence_ids:
            browse_entry = browse_verifications.get(evidence_id)
            if browse_entry is None:
                assessment_details.append({
                    "slot_id": assessment.slot_id,
                    "evidence_id": evidence_id,
                    "reason_codes": ["browse_verification_not_found"],
                })
                continue
            browse_verification, source = browse_entry
            passed, reasons, detail = _browse_support_for_slot(
                slot_id=assessment.slot_id,
                verification=browse_verification,
                source=source,
                rubric=rubric,
                support_threshold=support_threshold,
                relevance_threshold=relevance_threshold,
            )
            if passed:
                browse_supported_pairs += 1
                assessment_supported = True
            else:
                # A supported Browse verdict blocked only by the global
                # relevance threshold is a credible verifier inconsistency.
                # Do not promote final-verifier claims that the Browse verifier
                # itself called partial/unsupported; this prevents an
                # "evidence gap" sentence from becoming positive slot credit.
                if (
                    detail.get("browse_verdict") == "supported"
                    and detail.get("browse_confidence", 0.0) >= support_threshold
                    and detail.get("browse_source_requirement_met") is True
                    and detail.get("metadata_source_requirement_met") is True
                    and set(reasons) == {"evidence_below_relevance_threshold"}
                ):
                    relevance_only_supported = True
                assessment_details.append({
                    "slot_id": assessment.slot_id,
                    "evidence_id": evidence_id,
                    "reason_codes": reasons,
                    **detail,
                })
        # Citations attached to one final slot are a collective evidence set.
        # If any cited chunk already passed the Browse gate, do not count its
        # siblings as separate contradictions. Otherwise emit one assessment-
        # level disagreement with the individual diagnostics preserved.
        if not assessment_supported and assessment_details:
            if relevance_only_supported and assessment.slot_id in {
                slot.slot_id for slot in rubric.slots
            }:
                counterfactual_slots.add(assessment.slot_id)
            first = assessment_details[0]
            disagreements.append({
                "slot_id": assessment.slot_id,
                "evidence_id": first.get("evidence_id") if len(evidence_ids) == 1 else None,
                "evidence_ids": evidence_ids,
                "reason_codes": sorted({
                    reason
                    for detail in assessment_details
                    for reason in detail.get("reason_codes", [])
                }),
                "citation_diagnostics": assessment_details,
            })
    return {
        "status": "checked",
        "final_supported_evidence_pairs": final_pairs,
        "browse_supported_pairs": browse_supported_pairs,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "counterfactual_new_slots": sorted(counterfactual_slots - covered_slots),
        "counterfactual_coverage": len(counterfactual_slots) / max(1, len(rubric.slots)),
    }


def _terminal_trajectory_attribution(
    *,
    final_verification: Any | None,
    evidence_origins: dict[str, int],
    limitation_weight: float = 0.5,
    credit_source: str = "final_utilization",
) -> dict[str, Any]:
    """Attribute terminally verified slots to the Browse actions that delivered them.

    The frozen judge may name only evidence IDs that were actually opened.  This
    helper adds a second deterministic boundary: an ID receives action credit
    only when a trusted Browse receipt maps it to a concrete decision index.
    One slot contributes at most one unit of direct credit (or the configured
    fractional limitation credit).  Evidence-availability credit is marginal:
    the earliest trusted Browse action that makes a slot available receives the
    whole slot credit, while later supporting Browse actions receive zero.  The
    legacy final-utilization mode retains its historical equal-share behavior.
    """

    if not 0.0 <= limitation_weight <= 1.0:
        raise ValueError("limitation_weight must be in [0, 1]")
    if credit_source not in {"final_utilization", "evidence_availability"}:
        raise ValueError(
            "credit_source must be final_utilization or evidence_availability"
        )
    if final_verification is None:
        return {
            "status": "not_available",
            "decision_rewards": {},
            "decision_slots": {},
            "direct_supported_slots": [],
            "limitation_supported_slots": [],
            "supported_slots": [],
            "retrieval_available_slots": [],
            "final_utilized_slots": [],
            "unattributed_slots": [],
            "marginal_slot_origins": {},
            "duplicate_or_late_supports": [],
            "slot_credit_lineage": [],
            "weighted_slot_credit": 0.0,
        }
    if credit_source == "final_utilization" and (
        not final_verification.answer_present
        or not final_verification.medical_safety_ok
        or final_verification.unsupported_strong_claim
    ):
        return {
            "status": "final_content_gate_failed",
            "credit_source": credit_source,
            "decision_rewards": {},
            "decision_slots": {},
            "direct_supported_slots": [],
            "limitation_supported_slots": [],
            "supported_slots": [],
            "retrieval_available_slots": [],
            "final_utilized_slots": [],
            "unattributed_slots": [],
            "marginal_slot_origins": {},
            "duplicate_or_late_supports": [],
            "slot_credit_lineage": [],
            "weighted_slot_credit": 0.0,
        }

    decision_rewards: dict[int, float] = {}
    decision_slots: dict[int, set[str]] = {}
    direct_slots: set[str] = set()
    limitation_slots: set[str] = set()
    unattributed_slots: list[dict[str, Any]] = []
    marginal_slot_origins: dict[str, int] = {}
    duplicate_or_late_supports: list[dict[str, Any]] = []
    slot_credit_lineage: list[dict[str, Any]] = []
    weighted_slot_credit = 0.0
    final_utilized_slots: set[str] = set()
    for assessment in final_verification.slot_assessments:
        final_utilized = bool(
            assessment.addressed and assessment.supported_by_opened_evidence
        )
        if final_utilized:
            final_utilized_slots.add(assessment.slot_id)

        if credit_source == "evidence_availability":
            support_type = str(assessment.evidence_available)
            evidence_ids = list(assessment.available_evidence_ids)
            if support_type == "none":
                continue
        else:
            if not final_utilized:
                continue
            support_type = _slot_support_type(assessment)
            evidence_ids = list(assessment.citation_ids)
        if support_type == "direct_evidence":
            slot_weight = 1.0
            direct_slots.add(assessment.slot_id)
        elif support_type == "opened_evidence_limitation":
            if not evidence_ids:
                continue
            slot_weight = limitation_weight
            limitation_slots.add(assessment.slot_id)
        else:
            continue

        origin_indexes = sorted({
            evidence_origins[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_origins
        })
        if not origin_indexes:
            unattributed_slots.append({
                "slot_id": assessment.slot_id,
                "support_type": support_type,
                "evidence_ids": sorted(set(evidence_ids)),
                "reason": "no_opened_evidence_receipt_origin",
            })
            continue
        weighted_slot_credit += slot_weight
        if credit_source == "evidence_availability":
            # A required slot becomes available at the first trusted Browse
            # origin selected by the terminal verifier.  Later origins may add
            # redundant support, but they do not create another marginal gain.
            marginal_origin = origin_indexes[0]
            marginal_slot_origins[assessment.slot_id] = marginal_origin
            credited_evidence_ids = sorted(
                evidence_id
                for evidence_id in set(evidence_ids)
                if evidence_origins.get(evidence_id) == marginal_origin
            )
            decision_rewards[marginal_origin] = (
                decision_rewards.get(marginal_origin, 0.0) + slot_weight
            )
            decision_slots.setdefault(marginal_origin, set()).add(
                assessment.slot_id
            )
            if len(origin_indexes) > 1:
                duplicate_or_late_supports.append({
                    "slot_id": assessment.slot_id,
                    "credited_decision_index": marginal_origin,
                    "zero_credit_decision_indexes": origin_indexes[1:],
                    "reason": "slot_already_available",
                })
            slot_credit_lineage.append({
                "slot_id": assessment.slot_id,
                "support_type": support_type,
                "available": True,
                "utilized": final_utilized,
                "evidence_ids": sorted(set(evidence_ids)),
                "credited_evidence_ids": credited_evidence_ids,
                "browse_decision_index": marginal_origin,
                "raw_gain": slot_weight,
            })
        else:
            share = slot_weight / len(origin_indexes)
            for decision_index in origin_indexes:
                decision_rewards[decision_index] = (
                    decision_rewards.get(decision_index, 0.0) + share
                )
                decision_slots.setdefault(decision_index, set()).add(
                    assessment.slot_id
                )

    return {
        "status": "checked",
        "credit_source": credit_source,
        "decision_rewards": {
            str(index): value for index, value in sorted(decision_rewards.items())
        },
        "decision_slots": {
            str(index): sorted(slots) for index, slots in sorted(decision_slots.items())
        },
        "direct_supported_slots": sorted(direct_slots),
        "limitation_supported_slots": sorted(limitation_slots),
        "supported_slots": sorted(direct_slots | limitation_slots),
        "retrieval_available_slots": sorted(direct_slots | limitation_slots),
        "final_utilized_slots": sorted(final_utilized_slots),
        "unattributed_slots": unattributed_slots,
        "marginal_slot_origins": {
            slot_id: index
            for slot_id, index in sorted(marginal_slot_origins.items())
        },
        "duplicate_or_late_supports": duplicate_or_late_supports,
        "slot_credit_lineage": slot_credit_lineage,
        "weighted_slot_credit": weighted_slot_credit,
    }


class MedGapRewardService:
    """Scores a transcript while keeping all evidence-gap state private."""

    def __init__(
        self,
        verifier: QwenMaxVerifier,
        *,
        gamma: float = 0.9,
        support_threshold: float = 0.8,
        relevance_threshold: float = 0.5,
        verifier_scope: str | None = None,
        limitation_action_credit: float = 0.5,
        missed_supported_slot_weight: float = 0.0,
        unjustified_abstention_weight: float = 0.0,
        supported_slot_utilization_bonus_weight: float = 0.10,
        natural_final_bonus_weight: float = 0.03,
        separate_credit_channels: bool = False,
        conserve_chain_credit: bool = False,
        search_credit_share: float = 0.30,
        browse_credit_share: float = 0.70,
        exact_candidate_provenance: bool = False,
        failure_aware_step_local: bool = False,
        private_trace_path: Path | None = None,
        max_total_tool_calls: int = 6,
    ) -> None:
        self.verifier = verifier
        self.gamma = gamma
        self.support_threshold = support_threshold
        self.relevance_threshold = relevance_threshold
        self.verifier_scope = str(
            verifier_scope
            or os.getenv("MEDGAP_VERIFIER_SCOPE", "per_chunk_and_final")
        ).strip().lower()
        if self.verifier_scope not in {"per_chunk_and_final", "terminal_trajectory"}:
            raise ValueError(
                "verifier_scope must be per_chunk_and_final or terminal_trajectory"
            )
        if not 0.0 <= limitation_action_credit <= 1.0:
            raise ValueError("limitation_action_credit must be in [0, 1]")
        self.limitation_action_credit = limitation_action_credit
        if not 0.0 <= missed_supported_slot_weight <= 1.0:
            raise ValueError("missed_supported_slot_weight must be in [0, 1]")
        if not 0.0 <= unjustified_abstention_weight <= 1.0:
            raise ValueError("unjustified_abstention_weight must be in [0, 1]")
        self.missed_supported_slot_weight = missed_supported_slot_weight
        self.unjustified_abstention_weight = unjustified_abstention_weight
        if not 0.0 <= supported_slot_utilization_bonus_weight <= 1.0:
            raise ValueError("supported_slot_utilization_bonus_weight must be in [0, 1]")
        if not 0.0 <= natural_final_bonus_weight <= 1.0:
            raise ValueError("natural_final_bonus_weight must be in [0, 1]")
        self.supported_slot_utilization_bonus_weight = (
            supported_slot_utilization_bonus_weight
        )
        self.natural_final_bonus_weight = natural_final_bonus_weight
        self.separate_credit_channels = bool(separate_credit_channels)
        self.conserve_chain_credit = bool(conserve_chain_credit)
        for name, value in (
            ("search_credit_share", search_credit_share),
            ("browse_credit_share", browse_credit_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.conserve_chain_credit and abs(
            (search_credit_share + browse_credit_share) - 1.0
        ) > 1e-8:
            raise ValueError(
                "search_credit_share + browse_credit_share must equal 1"
            )
        self.search_credit_share = float(search_credit_share)
        self.browse_credit_share = float(browse_credit_share)
        self.exact_candidate_provenance = bool(exact_candidate_provenance)
        self.failure_aware_step_local = bool(failure_aware_step_local)
        self.private_trace_path = private_trace_path
        self.max_total_tool_calls = max_total_tool_calls
        self._trace_lock = threading.Lock()
        self._verification_lock = threading.Lock()
        self._verification_attempts = 0
        self._verification_failures = 0
        self._failure_threshold = float(os.getenv("MEDGAP_VERIFIER_FAILURE_THRESHOLD", "0.05"))
        self._failure_min_requests = int(os.getenv("MEDGAP_VERIFIER_FAILURE_MIN_REQUESTS", "20"))

    def _record_verification(self, *, failed: bool) -> None:
        if hasattr(self.verifier, "metrics_snapshot"):
            metrics = self.verifier.metrics_snapshot()
            api_requests = int(metrics.get("requests", 0))
            api_failures = int(metrics.get("failures", 0))
            rate = api_failures / max(1, api_requests)
            if api_requests >= self._failure_min_requests and rate > self._failure_threshold:
                raise RuntimeError(
                    f"MedGap verifier API failure rate {rate:.1%} exceeds "
                    f"{self._failure_threshold:.1%}; aborting training"
                )
            return
        # Lightweight test doubles may not expose provider metrics. This
        # fallback is not used by the production QwenMaxVerifier.
        with self._verification_lock:
            self._verification_attempts += 1
            self._verification_failures += int(failed)
            rate = self._verification_failures / self._verification_attempts
            if self._verification_attempts >= self._failure_min_requests and rate > self._failure_threshold:
                raise RuntimeError(
                    f"MedGap verifier failure rate {rate:.1%} exceeds {self._failure_threshold:.1%}; aborting training"
                )

    def score_rollout(
        self,
        *,
        transcript: str,
        rubric: EvidenceRubric,
        trusted_evidence_receipts: list[dict[str, Any]] | None = None,
        runtime_diagnostics: list[dict[str, Any]] | None = None,
    ) -> RolloutRewardResult:
        ledger = HiddenEvidenceLedger(
            rubric,
            support_threshold=self.support_threshold,
            relevance_threshold=self.relevance_threshold,
        )
        decisions = []
        rewards = []
        opened_chunks: list[EvidenceChunk] = []
        evidence_origins: dict[str, int] = {}
        browse_verifications: dict[str, tuple[Any, SourceMetadata]] = {}
        trusted_receipts = list(trusted_evidence_receipts or [])
        runtime_diagnostics = list(runtime_diagnostics or [])
        paired_policy_calls = _paired_calls(transcript)
        if self.failure_aware_step_local:
            (
                runtime_outputs,
                runtime_bindings,
                runtime_binding_summary,
            ) = _runtime_tool_outputs(
                runtime_diagnostics,
                policy_tools=[tool for tool, _body, _output in paired_policy_calls],
            )
        else:
            runtime_outputs = [None] * len(paired_policy_calls)
            runtime_bindings = [
                {
                    "decision_index": index,
                    "tool_name": tool,
                    "call_index": index + 1,
                    "call_id": None,
                    "status": "disabled",
                }
                for index, (tool, _body, _output) in enumerate(paired_policy_calls)
            ]
            runtime_binding_summary = {
                "mode": "disabled",
                "total_policy_tool_decisions": len(paired_policy_calls),
                "atomic_observed_decisions": 0,
                "unobserved_decisions": 0,
                "status_counts": {"disabled": len(paired_policy_calls)},
                "ignored_checkpoint_status_counts": {},
                "cumulative_snapshot_parsing_disabled": True,
            }
        require_trusted_browse_delivery = trusted_evidence_receipts is not None
        trusted_receipts_used: list[dict[str, Any]] = []
        finalization_locked = False
        for decision_index, (tool, call_body, output) in enumerate(
            paired_policy_calls
        ):
            verification_audit = []
            runtime_output = runtime_outputs[decision_index]
            runtime_binding = runtime_bindings[decision_index]
            if tool in {"browse_document", "browse_webpage"} and require_trusted_browse_delivery:
                receipt = _trusted_receipt_for_call(
                    trusted_receipts,
                    decision_index=decision_index,
                    tool_name=tool,
                )
                if receipt is None:
                    # A model-authored or truncated <tool_output> is never
                    # proof of evidence delivery.  A Runtime-authenticated
                    # failed observation may still classify the policy action
                    # (wrong source type, unsupported args, duplicate, etc.),
                    # but it can never register chunks or earn positive credit.
                    output = (
                        runtime_output
                        if isinstance(runtime_output, dict)
                        and runtime_output.get("failed")
                        else None
                    )
                else:
                    output = dict(receipt["trusted_output"])
                    trusted_receipts_used.append({
                        "call_id": str(receipt.get("call_id") or ""),
                        "call_index": int(receipt.get("call_index") or 0),
                        "tool_name": tool,
                        "source_id": str(receipt.get("source_id") or ""),
                        "citation_eligible_ids": list(
                            receipt.get("citation_eligible_ids") or []
                        ),
                    })
            elif self.failure_aware_step_local:
                # Search results and failed observations are safe to classify
                # only from an exact atomic Runtime binding.  An unobserved
                # binding is neutral; never fall back to transcript text that
                # could contain a policy-simulated tool output.
                output = runtime_output
            if finalization_locked:
                credit = ledger.score_tool_decision(
                    tool=tool,
                    post_reserve_tool_attempt=True,
                )
            elif output is None:
                credit = ledger.score_tool_decision(tool=tool, environment_failure=True)
            else:
                failure_type = _failure_type(output)
                chunks = _chunks_from_output(output) if tool in {"browse_document", "browse_webpage"} else []
                duplicate = bool(output.get("duplicate_blocked"))
                environment_failure = failure_type == "environment_failure"
                search_no_results = failure_type == "search_no_results"
                budget_violation = failure_type == "budget_violation"
                final_answer_reserve = failure_type == "final_answer_reserve"
                policy_invalid = failure_type == "policy_invalid"
                if tool in {"browse_document", "browse_webpage"} and not any(
                    (
                        rubric.no_tool_expected, duplicate, environment_failure,
                        policy_invalid, budget_violation, final_answer_reserve,
                    )
                ):
                    opened_chunks.extend(chunks)
                    for chunk in chunks:
                        # Both paper and web Browse actions enter the same
                        # provenance map.  The terminal judge cannot create an
                        # origin; it can only select from these opened IDs.
                        evidence_origins.setdefault(chunk.evidence_id, decision_index)
                    if self.verifier_scope == "terminal_trajectory":
                        verification_audit.extend(
                            {
                                "evidence_id": chunk.evidence_id,
                                "status": "deferred_to_terminal_trajectory_judge",
                                "originating_decision_index": decision_index,
                            }
                            for chunk in chunks
                        )
                        credit = (
                            DecisionCredit(
                                0.0,
                                reasons=("evidence_delivery_deferred_to_terminal_judge",),
                            )
                            if chunks
                            else ledger.score_tool_decision(
                                tool=tool, environment_failure=True
                            )
                        )
                    else:
                        verified = []
                        for chunk in chunks:
                            try:
                                envelope = self.verifier.verify(rubric, chunk)
                            except VerifierOutputError as exc:
                                self._record_verification(failed=False)
                                verification_audit.append(
                                    {
                                        "evidence_id": chunk.evidence_id,
                                        "status": "verifier_output_invalid",
                                        "error": str(exc),
                                    }
                                )
                                continue
                            except VerifierError as exc:
                                self._record_verification(failed=True)
                                verification_audit.append(
                                    {
                                        "evidence_id": chunk.evidence_id,
                                        "status": "verifier_failed",
                                        "error": str(exc),
                                    }
                                )
                                continue
                            self._record_verification(failed=False)
                            verified.append(
                                (chunk.evidence_id, envelope.verification, chunk.source)
                            )
                            browse_verifications[chunk.evidence_id] = (
                                envelope.verification,
                                chunk.source,
                            )
                            verification_audit.append(
                                {
                                    "evidence_id": chunk.evidence_id,
                                    "status": "verified",
                                    "cache_key": envelope.cache_key,
                                    "from_cache": envelope.from_cache,
                                    "verification": envelope.verification.model_dump(
                                        mode="json"
                                    ),
                                }
                            )
                        credit = (
                            ledger.score_tool_decision(
                                tool=tool, environment_failure=True
                            )
                            if chunks and not verified
                            else ledger.score_browse_decision(
                                tool=tool, verified_evidence=verified
                            )
                        )
                else:
                    credit = ledger.score_tool_decision(
                        tool=tool,
                        duplicate=duplicate,
                        policy_invalid=policy_invalid,
                        environment_failure=environment_failure,
                        search_no_results=search_no_results,
                        budget_violation=budget_violation,
                        final_answer_reserve=final_answer_reserve,
                    )
                if final_answer_reserve:
                    finalization_locked = True
            rewards.append(credit.reward)
            decisions.append(
                {
                    "decision_index": decision_index,
                    "tool": tool,
                    "local_reward": credit.reward,
                    "newly_covered_slots": list(credit.newly_covered_slots),
                    "reasons": list(credit.reasons),
                    "failure_type": (
                        _failure_type(output)
                        if isinstance(output, dict)
                        else "environment_failure"
                    ),
                    "failure_error": (
                        str(output.get("error") or "")
                        if isinstance(output, dict)
                        else str(runtime_binding.get("status") or "unobserved")
                    ),
                    "runtime_observation_status": str(
                        runtime_binding.get("status") or "unobserved"
                    ),
                    "runtime_observation_call_id": runtime_binding.get("call_id"),
                    "hidden_verification": verification_audit,
                    "source_candidate_id": (
                        _browse_call_source_id(call_body)
                        if tool in {"browse_document", "browse_webpage"}
                        else None
                    ),
                    "discovered_candidate_ids": (
                        _candidate_source_ids(output)
                        if tool in {"pubmed_search", "medical_web_search"}
                        else []
                    ),
                }
            )

        raw_answer_parse = parse_policy_owned_final_answer(transcript)
        answer_repair = repair_policy_owned_final_answer(transcript)
        final_transcript = answer_repair.text
        answer_parse = parse_policy_owned_final_answer(final_transcript)
        answer_text = answer_parse.answer.body.strip() if answer_parse.answer else ""
        calls_used = len(decisions)
        actual_citations = set(_CITATION_RE.findall(answer_text))
        opened_ids = {chunk.evidence_id for chunk in opened_chunks}
        unopened_citations = actual_citations - opened_ids
        citation_pointers = _citation_pointer_diagnostics(actual_citations, opened_ids)
        source_unopened_citations = set(citation_pointers["source_unopened"])
        citation_policy_violation = bool(actual_citations) and (
            rubric.no_tool_expected or bool(unopened_citations)
        )
        source_citation_policy_violation = bool(actual_citations) and (
            rubric.no_tool_expected or bool(source_unopened_citations)
        )
        final_verification = None
        final_answer_reward_observed = True
        terminal_behavior_penalty = 0.0
        terminal_behavior_penalty_observed = True
        terminal_behavior = None
        terminal_behavior_bonus = 0.0
        terminal_behavior_bonus_observed = True
        terminal_positive = None
        natural_final_detected = False
        hard_final_recovery = False
        terminal_behavior_verification = None
        terminal_behavior_fallback_allowed = False
        terminal_behavior_verification_audit = {
            "status": "not_required",
            "source": None,
        }
        language_ok = _language_matches(rubric.question, answer_text) if answer_text else False
        if answer_parse.malformed:
            final_answer_reward = 0.0
            final_answer_audit = {
                "mode": "no_tool_semantic" if rubric.no_tool_expected else "search",
                "answer_present": False,
                "status": "malformed",
                "reason": answer_parse.reason,
            }
        elif rubric.no_tool_expected:
            if not answer_text or actual_citations:
                final_answer_reward = 0.0
                final_answer_audit = {
                    "mode": "no_tool_semantic", "answer_present": bool(answer_text),
                    "actual_citations": sorted(actual_citations), "status": "deterministic_gate_failed",
                }
            else:
                try:
                    no_tool_verification = self.verifier.verify_no_tool_answer(rubric, answer_text)
                    self._record_verification(failed=False)
                    final_answer_reward = float(
                        calls_used == 0
                        and no_tool_verification.answered_question
                        and no_tool_verification.factually_correct
                        and no_tool_verification.stable_evidence_literacy_topic
                        and no_tool_verification.medical_safety_ok
                        and no_tool_verification.no_fabricated_citation
                    )
                    final_answer_audit = no_tool_verification.model_dump(mode="json")
                    final_answer_audit["language_ok"] = language_ok
                    final_answer_audit["language_compliance_passed"] = bool(
                        language_ok and no_tool_verification.language_matches
                    )
                    final_answer_audit["selected_final_reward"] = final_answer_reward
                except VerifierOutputError as exc:
                    self._record_verification(failed=False)
                    final_answer_reward = 0.0
                    final_answer_reward_observed = False
                    final_answer_audit = {
                        "status": "verifier_output_invalid",
                        "error": str(exc),
                        "terminal_reward_observed": False,
                        "environment_neutral": True,
                    }
                except VerifierError as exc:
                    self._record_verification(failed=True)
                    final_answer_reward = 0.0
                    final_answer_reward_observed = False
                    final_answer_audit = {
                        "status": "verifier_failed",
                        "error": str(exc),
                        "terminal_reward_observed": False,
                        "environment_neutral": True,
                    }
        elif not answer_text or not opened_chunks:
            final_answer_reward = 0.0
            missing_reason = (
                "answer_missing"
                if not answer_text
                else "answer_present_but_no_opened_evidence"
            )
            final_answer_audit = {
                "mode": "search",
                "answer_present": bool(answer_text),
                "status": missing_reason,
                "missing_reason": missing_reason,
            }
        else:
            try:
                citation_gate_passed = bool(actual_citations) and not unopened_citations
                citation_source_opened = bool(citation_pointers["source_opened"])
                citation_error_codes = []
                if not actual_citations:
                    citation_error_codes.append("missing_citation")
                if unopened_citations:
                    citation_error_codes.append("unopened_or_invalid_citation_id")
                citation_error_details = []
                for citation_id in sorted(unopened_citations):
                    if _BASE_SOURCE_ID_RE.fullmatch(citation_id):
                        error_type = "base_id_not_citable"
                    elif _CHUNK_SOURCE_ID_RE.fullmatch(citation_id):
                        error_type = "unopened_chunk_id"
                    else:
                        error_type = "malformed_or_unknown_id"
                    citation_error_details.append({
                        "citation_id": citation_id,
                        "error_type": error_type,
                    })
                semantic_answer = citation_independent_semantic_answer(answer_text)
                semantic_chunks = citation_independent_evidence_order(opened_chunks)
                final_verification = self.verifier.verify_final_answer(
                    rubric, semantic_answer, semantic_chunks
                )
                self._record_verification(failed=False)
                direct_supported = [
                    item
                    for item in final_verification.slot_assessments
                    if item.addressed
                    and item.supported_by_opened_evidence
                    and _slot_support_type(item) == "direct_evidence"
                    and bool(item.citation_ids)
                ]
                limitation_assessed = [
                    item
                    for item in final_verification.slot_assessments
                    if item.addressed
                    and _slot_support_type(item) == "opened_evidence_limitation"
                ]
                limitation_supported = [
                    item
                    for item in limitation_assessed
                    if item.addressed
                    and item.supported_by_opened_evidence
                    and bool(item.citation_ids)
                ]
                credited_assessments = direct_supported + limitation_supported
                direct_weight = float(os.getenv("MEDGAP_DIRECT_EVIDENCE_WEIGHT", "1.0"))
                limitation_weight = float(
                    os.getenv("MEDGAP_OPENED_LIMITATION_WEIGHT", "0.5")
                )
                weighted_supported = (
                    direct_weight * len(direct_supported)
                    + limitation_weight * len(limitation_supported)
                )
                verifier_citations_valid = all(
                    set(item.citation_ids).issubset(actual_citations)
                    and (_slot_support_type(item) != "direct_evidence" or bool(item.citation_ids))
                    for item in final_verification.slot_assessments
                )
                verifier_citations_source_valid = all(
                    all(
                        evidence_id in actual_citations
                        or _base_source_id(evidence_id) in actual_citations
                        for evidence_id in item.citation_ids
                    )
                    and (_slot_support_type(item) != "direct_evidence" or bool(item.citation_ids))
                    for item in final_verification.slot_assessments
                )
                verifier_reported_support_ids = {
                    evidence_id
                    for item in credited_assessments
                    for evidence_id in item.citation_ids
                }
                # The verifier is a semantic judge, not an authority to mint
                # provenance.  Resolver eligibility is the strict intersection
                # of verifier support and chunks actually opened in this trace.
                verifier_supported_ids = verifier_reported_support_ids & opened_ids
                resolved_citations_by_slot = {
                    item.slot_id: sorted(set(item.citation_ids) & opened_ids)
                    for item in credited_assessments
                }
                resolved_citations_by_slot = {
                    slot_id: values
                    for slot_id, values in resolved_citations_by_slot.items()
                    if values
                }
                resolved_citation_ids = sorted({
                    evidence_id
                    for values in resolved_citations_by_slot.values()
                    for evidence_id in values
                })
                supported_slots_have_grounding = all(
                    bool(set(item.citation_ids) & opened_ids)
                    for item in credited_assessments
                )
                citation_coverage_valid = bool(actual_citations) and all(
                    citation_id in verifier_supported_ids
                    or any(
                        _base_source_id(evidence_id) == citation_id
                        for evidence_id in verifier_supported_ids
                    )
                    for citation_id in actual_citations
                )
                # Citation correctness remains a deterministic property of the
                # model-authored IDs.  The semantic Judge only selected immutable
                # OPENED_EVIDENCE array positions; it never minted an ID or quote.
                final_verification.citations_grounded = bool(
                    citation_gate_passed
                    and verifier_citations_valid
                    and citation_coverage_valid
                )
                base = weighted_supported / max(1, len(rubric.slots))
                content_semantic_score = (
                    base
                    if final_verification.answer_present
                    and final_verification.medical_safety_ok
                    and not final_verification.unsupported_strong_claim
                    else 0.0
                )
                strict_final_reward = (
                    content_semantic_score
                    if citation_gate_passed
                    and final_verification.citations_grounded
                    and verifier_citations_valid
                    else 0.0
                )
                source_entailment_gate_passed = bool(
                    citation_source_opened
                    and verifier_citations_source_valid
                    and citation_coverage_valid
                    and direct_supported
                    and content_semantic_score > 0
                )
                source_entailment_final_reward = (
                    content_semantic_score if source_entailment_gate_passed else 0.0
                )
                content_grounded_gate_passed = bool(
                    credited_assessments
                    and supported_slots_have_grounding
                    and resolved_citation_ids
                    and content_semantic_score > 0
                )
                content_grounded_final_reward = (
                    content_semantic_score if content_grounded_gate_passed else 0.0
                )
                citation_gate_mode = os.getenv(
                    "MEDGAP_CITATION_GATE_MODE", "strict_exact_chunk"
                ).strip().lower()
                if citation_gate_mode not in {
                    "strict_exact_chunk", "source_entailment", "content_grounded"
                }:
                    raise ValueError(
                        "MEDGAP_CITATION_GATE_MODE must be strict_exact_chunk, "
                        "source_entailment, or content_grounded"
                    )
                citation_resolution = (
                    _resolve_final_citations(
                        answer_text,
                        opened_chunks=opened_chunks,
                        verifier_supported_ids=verifier_supported_ids,
                    )
                    if citation_gate_mode == "content_grounded"
                    else {
                        "raw_final": answer_text,
                        "resolved_final": answer_text,
                        "raw_citation_ids": sorted(actual_citations),
                        "resolved_final_citation_ids": sorted(actual_citations),
                        "citation_repaired": False,
                        "citation_repairs": [],
                        "citation_removals": [],
                        "citation_resolver_unresolved_ids": [],
                        "citation_resolver_success": False,
                    }
                )
                if citation_gate_mode == "content_grounded":
                    final_answer_reward = content_grounded_final_reward
                elif citation_gate_mode == "source_entailment":
                    final_answer_reward = source_entailment_final_reward
                else:
                    final_answer_reward = strict_final_reward
                final_answer_audit = final_verification.model_dump(mode="json")
                if not citation_gate_passed and not language_ok:
                    status = "semantic_evaluated_with_citation_and_language_error"
                elif not citation_gate_passed:
                    status = "semantic_evaluated_with_citation_error"
                elif not language_ok:
                    status = "semantic_evaluated_with_language_error"
                else:
                    status = "semantic_evaluated"
                final_answer_audit["status"] = status
                final_answer_audit["actual_citations"] = sorted(actual_citations)
                final_answer_audit["valid_opened_citations"] = sorted(actual_citations & opened_ids)
                final_answer_audit["unopened_citations"] = sorted(unopened_citations)
                final_answer_audit["citation_error_codes"] = citation_error_codes
                final_answer_audit["citation_error_details"] = citation_error_details
                final_answer_audit["language_ok"] = language_ok
                final_answer_audit["language_compliance_passed"] = language_ok
                final_answer_audit["language_compliance_score"] = float(language_ok)
                final_answer_audit["language_affects_content_reward"] = False
                final_answer_audit["verifier_citations_valid"] = verifier_citations_valid
                final_answer_audit["verifier_citations_source_valid"] = (
                    verifier_citations_source_valid
                )
                final_answer_audit["citation_gate_passed"] = citation_gate_passed
                final_answer_audit["citation_source_opened"] = citation_source_opened
                final_answer_audit["citation_claim_supported"] = (
                    verifier_citations_source_valid and citation_coverage_valid
                )
                final_answer_audit["citation_coverage_valid"] = citation_coverage_valid
                final_answer_audit["citation_pointer_exact"] = citation_pointers[
                    "pointer_exact"
                ]
                final_answer_audit["source_opened_base_citations"] = sorted(
                    citation_pointers["base_opened"]
                )
                final_answer_audit["source_unopened_citations"] = sorted(
                    source_unopened_citations
                )
                final_answer_audit["source_entailment_gate_passed"] = (
                    source_entailment_gate_passed
                )
                final_answer_audit["source_entailment_status"] = (
                    "semantic_evaluated"
                    if source_entailment_gate_passed
                    else "deterministic_gate_failed"
                )
                final_answer_audit["citation_gate_mode"] = citation_gate_mode
                final_answer_audit["selected_gate_passed"] = (
                    content_grounded_gate_passed
                    if citation_gate_mode == "content_grounded"
                    else (
                        source_entailment_gate_passed
                        if citation_gate_mode == "source_entailment"
                        else citation_gate_passed
                    )
                )
                final_answer_audit["content_semantic_score"] = content_semantic_score
                final_answer_audit["direct_evidence_weight"] = direct_weight
                final_answer_audit["opened_limitation_weight"] = limitation_weight
                final_answer_audit["weighted_supported_slots"] = weighted_supported
                final_answer_audit["strict_final_reward"] = strict_final_reward
                final_answer_audit["source_entailment_final_reward"] = (
                    source_entailment_final_reward
                )
                final_answer_audit["content_grounded_gate_passed"] = (
                    content_grounded_gate_passed
                )
                final_answer_audit["content_grounded_final_reward"] = (
                    content_grounded_final_reward
                )
                final_answer_audit["direct_supported_slots"] = [
                    item.slot_id for item in direct_supported
                ]
                final_answer_audit["opened_evidence_limitation_slots"] = [
                    item.slot_id for item in limitation_assessed
                ]
                final_answer_audit["credited_opened_evidence_limitation_slots"] = [
                    item.slot_id for item in limitation_supported
                ]
                final_answer_audit["resolved_citations_by_slot"] = (
                    resolved_citations_by_slot
                )
                final_answer_audit["resolved_citation_ids"] = resolved_citation_ids
                final_answer_audit["model_citation_compliance_passed"] = (
                    citation_gate_passed
                    and final_verification.citations_grounded
                    and verifier_citations_valid
                )
                final_answer_audit["selected_final_reward"] = final_answer_reward
                final_answer_audit.update(citation_resolution)
                final_answer_audit["citation_ids_auto_repaired"] = bool(
                    citation_resolution["citation_repaired"]
                )
                final_answer_audit["quote_normalization_repairs"] = sum(
                    int(reference.quote_normalization_repaired)
                    for item in final_verification.slot_assessments
                    for reference in item.supporting_evidence
                )
                final_answer_audit["quote_validation_failed_slots"] = [
                    item.slot_id
                    for item in final_verification.slot_assessments
                    if "[deterministic_quote_validation_failed]" in item.rationale
                ]
            except VerifierOutputError as exc:
                self._record_verification(failed=False)
                final_answer_reward = 0.0
                final_answer_reward_observed = False
                final_answer_audit = {
                    "status": "verifier_output_invalid",
                    "error": str(exc),
                    "terminal_reward_observed": False,
                    "environment_neutral": True,
                }
            except VerifierError as exc:
                self._record_verification(failed=True)
                final_answer_reward = 0.0
                final_answer_reward_observed = False
                final_answer_audit = {
                    "status": "verifier_failed",
                    "error": str(exc),
                    "terminal_reward_observed": False,
                    "environment_neutral": True,
                }

        if final_verification is not None:
            terminal_behavior_verification = final_verification
            terminal_behavior_verification_audit = {
                "status": "observed",
                "source": "final_answer_verification",
            }
        elif not final_answer_reward_observed:
            terminal_behavior_penalty_observed = False
            terminal_behavior_fallback_allowed = bool(
                final_answer_audit.get("status") == "verifier_output_invalid"
            )
            terminal_behavior_verification_audit = {
                "status": "unobserved",
                "source": "final_answer_verification",
                "verifier_status": final_answer_audit.get("status"),
                "error": final_answer_audit.get("error"),
                "environment_neutral": True,
            }

        # Missing or malformed Finals still need the same single terminal
        # judgment when real opened evidence exists. This exposes evidence the
        # policy found but failed to use, without adding a second judge call to
        # ordinary successful trajectories.
        if (
            not rubric.no_tool_expected
            and opened_chunks
            and final_verification is None
            and final_answer_reward_observed
        ):
            try:
                final_verification = self.verifier.verify_final_answer(
                    rubric,
                    citation_independent_semantic_answer(answer_text),
                    citation_independent_evidence_order(opened_chunks),
                )
                self._record_verification(failed=False)
                terminal_behavior_verification = final_verification
                terminal_behavior_verification_audit = {
                    "status": "observed",
                    "source": "missing_or_malformed_final_fallback",
                }
            except VerifierOutputError as exc:
                self._record_verification(failed=False)
                terminal_behavior_penalty_observed = False
                terminal_behavior_fallback_allowed = True
                terminal_behavior_verification_audit = {
                    "status": "verifier_output_invalid",
                    "source": "missing_or_malformed_final_fallback",
                    "error": str(exc),
                    "environment_neutral": True,
                }
            except VerifierError as exc:
                self._record_verification(failed=True)
                terminal_behavior_penalty_observed = False
                terminal_behavior_verification_audit = {
                    "status": "verifier_failed",
                    "source": "missing_or_malformed_final_fallback",
                    "error": str(exc),
                    "environment_neutral": True,
                }

        # The full Final verifier has a broad output contract.  If its response
        # is structurally invalid (but the provider is healthy), recover only
        # the two semantic primitives required by the terminal penalty.  The
        # full Final reward remains unobserved/environment-neutral, and this
        # fallback never supplies citation, safety, content, or local-credit
        # judgments.
        if (
            not rubric.no_tool_expected
            and opened_chunks
            and terminal_behavior_verification is None
            and terminal_behavior_fallback_allowed
        ):
            primary_verification_audit = dict(
                terminal_behavior_verification_audit
            )
            try:
                terminal_behavior_verification = (
                    self.verifier.verify_terminal_behavior_fallback(
                        rubric,
                        citation_independent_semantic_answer(answer_text),
                        citation_independent_evidence_order(opened_chunks),
                    )
                )
                self._record_verification(failed=False)
                terminal_behavior_penalty_observed = True
                terminal_behavior_verification_audit = {
                    "status": "observed",
                    "source": "terminal_semantic_fallback_after_output_invalid",
                    "primary_verification": primary_verification_audit,
                    "fallback_prompt_version": (
                        TERMINAL_BEHAVIOR_FALLBACK_PROMPT_VERSION
                    ),
                }
            except VerifierOutputError as exc:
                self._record_verification(failed=False)
                terminal_behavior_penalty_observed = False
                terminal_behavior_verification_audit = {
                    "status": "verifier_output_invalid",
                    "source": "terminal_semantic_fallback_after_output_invalid",
                    "primary_verification": primary_verification_audit,
                    "error": str(exc),
                    "environment_neutral": True,
                }
            except VerifierError as exc:
                self._record_verification(failed=True)
                terminal_behavior_penalty_observed = False
                terminal_behavior_verification_audit = {
                    "status": "verifier_failed",
                    "source": "terminal_semantic_fallback_after_output_invalid",
                    "primary_verification": primary_verification_audit,
                    "error": str(exc),
                    "environment_neutral": True,
                }

        if terminal_behavior_verification is not None:
            terminal_behavior = terminal_behavior_reward(
                terminal_behavior_verification.slot_assessments,
                missed_supported_slot_weight=self.missed_supported_slot_weight,
                unjustified_abstention_weight=self.unjustified_abstention_weight,
            )
            terminal_behavior_penalty = terminal_behavior.total_penalty

        # Positive utilization/natural-Final credit requires the complete Final
        # semantic verification contract. The narrow terminal fallback lacks
        # addressed/support fields by design and is therefore unobserved for
        # positive shaping, even when it successfully supplies legacy labels.
        positive_assessments_complete = bool(
            final_verification is not None
            and positive_reward_assessments_complete(
                final_verification.slot_assessments
            )
        )
        positive_bonus_applicable = bool(
            not rubric.no_tool_expected and opened_chunks
        )
        terminal_behavior_bonus_observed = bool(
            not positive_bonus_applicable or positive_assessments_complete
        )
        if positive_assessments_complete:
            hard_final_recovery = any(
                bool(item.get("hard_finalization_triggered"))
                or item.get("phase") == "hard_finalization_recovery"
                or item.get("diagnostic_type") == "hard_finalization_recovery"
                for item in runtime_diagnostics
                if isinstance(item, dict)
            )
            natural_final_detected = bool(
                raw_answer_parse.answer is not None
                and not raw_answer_parse.malformed
                and not answer_repair.repaired
                and not hard_final_recovery
            )
            answer_safe = bool(
                getattr(final_verification, "answer_present", False)
                and getattr(final_verification, "medical_safety_ok", False)
                and not getattr(
                    final_verification,
                    "unsupported_strong_claim",
                    True,
                )
            )
            terminal_positive = terminal_positive_reward(
                final_verification.slot_assessments,
                supported_slot_utilization_weight=(
                    self.supported_slot_utilization_bonus_weight
                ),
                natural_final_weight=self.natural_final_bonus_weight,
                natural_final_detected=natural_final_detected,
                answer_safe=answer_safe,
            )
            terminal_behavior_bonus = terminal_positive.total_bonus
        if terminal_behavior_verification is None and not final_answer_reward_observed:
            terminal_behavior_penalty_observed = False

        terminal_attribution = _terminal_trajectory_attribution(
            final_verification=final_verification,
            evidence_origins=evidence_origins,
            limitation_weight=self.limitation_action_credit,
            credit_source=(
                "evidence_availability"
                if self.separate_credit_channels
                else "final_utilization"
            ),
        )
        if self.verifier_scope == "terminal_trajectory":
            attributed_rewards = {
                int(index): float(value)
                for index, value in terminal_attribution["decision_rewards"].items()
            }
            attributed_slots = {
                int(index): list(values)
                for index, values in terminal_attribution["decision_slots"].items()
            }
            for index, decision in enumerate(decisions):
                if "evidence_delivery_deferred_to_terminal_judge" not in set(
                    decision["reasons"]
                ):
                    continue
                rewards[index] = attributed_rewards.get(index, 0.0)
                decision["local_reward"] = rewards[index]
                decision["newly_covered_slots"] = attributed_slots.get(index, [])
                decision["reasons"] = [
                    (
                        "terminal_attributed_evidence_gain"
                        if rewards[index] > 0
                        else "terminal_judge_no_attributed_evidence"
                    )
                ]
            verifier_consistency = {
                "status": "not_applicable_terminal_trajectory_scope",
                "reason": "per_chunk_browse_verification_disabled",
            }
            covered_slots = set(terminal_attribution["supported_slots"])
            coverage = len(covered_slots) / max(1, len(rubric.slots))
            weighted_coverage = float(
                terminal_attribution["weighted_slot_credit"]
            ) / max(1, len(rubric.slots))
        else:
            verifier_consistency = _browse_final_verifier_consistency(
                final_verification=final_verification,
                browse_verifications=browse_verifications,
                rubric=rubric,
                support_threshold=self.support_threshold,
                relevance_threshold=self.relevance_threshold,
                current_covered_slots=ledger.covered_slots,
            )
            covered_slots = set(ledger.covered_slots)
            coverage = ledger.coverage
            weighted_coverage = ledger.coverage

        if self.failure_aware_step_local:
            _annotate_all_browse_provenance(decisions)
        linked_credit = linked_search_browse_credit(
            [str(decision["tool"]) for decision in decisions],
            [list(decision["reasons"]) for decision in decisions],
            rewards,
            gamma=self.gamma,
            slot_count=max(1, len(rubric.slots)),
            conserve_chain_credit=self.conserve_chain_credit,
            search_share=self.search_credit_share,
            browse_share=self.browse_credit_share,
            discovered_candidate_ids=[
                list(decision["discovered_candidate_ids"])
                for decision in decisions
            ],
            browse_source_ids=[
                decision["source_candidate_id"] for decision in decisions
            ],
            require_exact_provenance=self.exact_candidate_provenance,
        )
        if self.failure_aware_step_local:
            returns, failure_aware_counts = _apply_failure_aware_local_credit(
                decisions,
                list(linked_credit.local_returns),
            )
        else:
            returns = list(linked_credit.local_returns)
            failure_aware_counts = {}
        for decision, value in zip(decisions, returns):
            decision["local_return"] = value
        provenance_by_browse = {
            int(browse): {
                "search_decision_index": search,
                "source_candidate_id": source_id,
                "provenance_status": status,
            }
            for (
                (search, browse, _gain, _search_credit, _browse_credit),
                (_, source_id, status),
            ) in zip(linked_credit.chain_credits, linked_credit.provenance)
        }
        credit_lineage = []
        required_slot_count = max(1, len(rubric.slots))
        for item in terminal_attribution.get("slot_credit_lineage", []):
            browse_index = int(item["browse_decision_index"])
            provenance = provenance_by_browse.get(
                browse_index,
                {
                    "search_decision_index": None,
                    "source_candidate_id": decisions[browse_index].get(
                        "source_candidate_id"
                    ),
                    "provenance_status": "unavailable",
                },
            )
            weighted_credit = float(item["raw_gain"]) / required_slot_count
            exact = provenance["provenance_status"] == "exact"
            credit_lineage.append({
                **item,
                **provenance,
                "search_credit": (
                    self.search_credit_share * weighted_credit if exact else 0.0
                ),
                "browse_credit": (
                    self.browse_credit_share * weighted_credit
                    if exact
                    else weighted_credit
                ),
                "weighted_credit": weighted_credit,
            })
        tool_call_limit_reached = calls_used >= self.max_total_tool_calls
        final_answer_present = answer_parse.answer is not None and bool(answer_text)
        budget_terminated_without_final = bool(
            tool_call_limit_reached and not final_answer_present
        )
        coverage_complete = len(covered_slots) == len(rubric.slots)
        budget_penalty = -0.25 if tool_call_limit_reached and not coverage_complete else 0.0
        trajectory_reward = (
            (1.0 if calls_used == 0 else 0.0)
            if rubric.no_tool_expected
            else weighted_coverage + budget_penalty
        )
        trajectory_reward_observed = not (
            self.verifier_scope == "terminal_trajectory"
            and bool(opened_chunks)
            and final_verification is None
        )
        audit = {
            "question_id": rubric.question_id,
            "rubric_version": rubric.rubric_version,
            "steps": decisions,
            "local_credit_version": (
                "failure_aware_atomic_source_exact_step_local_v67"
                if self.failure_aware_step_local
                else "source_exact_conserved_availability_search_browse_v59"
                if self.exact_candidate_provenance
                else "conserved_availability_search_browse_v57"
                if self.separate_credit_channels and self.conserve_chain_credit
                else "nearest_matching_search_browse_v55"
            ),
            "local_credit_source": (
                "new_required_evidence_availability"
                if self.separate_credit_channels
                else "final_utilization"
            ),
            "search_browse_credit_links": [
                {"search_decision_index": search, "browse_decision_index": browse}
                for search, browse in linked_credit.links
            ],
            "search_browse_chain_credits": [
                {
                    "search_decision_index": search,
                    "browse_decision_index": browse,
                    "total_evidence_gain": gain,
                    "search_credit": search_credit,
                    "browse_credit": browse_credit,
                    "source_candidate_id": source_id,
                    "provenance_status": provenance_status,
                }
                for (
                    (search, browse, gain, search_credit, browse_credit),
                    (_, source_id, provenance_status),
                ) in zip(linked_credit.chain_credits, linked_credit.provenance)
            ],
            "search_browse_credit_conserved": bool(
                not self.conserve_chain_credit
                or abs(
                    sum(linked_credit.local_returns)
                    - sum(item[2] for item in linked_credit.chain_credits)
                ) <= 1e-7
            ),
            "failure_aware_credit_counts": failure_aware_counts,
            "runtime_observation_binding": runtime_binding_summary,
            "failure_aware_negative_scope": (
                "offending_decision_span_only"
                if self.failure_aware_step_local
                else "disabled"
            ),
            "environment_failure_policy": (
                "neutral_masked" if self.failure_aware_step_local else "legacy"
            ),
            "credit_lineage": credit_lineage,
            "coverage": coverage,
            "weighted_coverage": weighted_coverage,
            "covered_slots": sorted(covered_slots),
            "required_slots": [slot.slot_id for slot in rubric.slots],
            "calls_used": calls_used,
            "max_total_tool_calls": self.max_total_tool_calls,
            # Legacy alias retained for historical comparison.  The explicit
            # fields below distinguish merely reaching the tool-call limit
            # from actually ending without a Final answer.
            "budget_exhausted": tool_call_limit_reached,
            "tool_call_limit_reached": tool_call_limit_reached,
            "budget_terminated_without_final": budget_terminated_without_final,
            "budget_termination_reason": (
                "tool_call_limit_and_no_final"
                if budget_terminated_without_final
                else ("tool_call_limit_reached" if tool_call_limit_reached else None)
            ),
            "coverage_at_exhaustion": (
                coverage if tool_call_limit_reached else None
            ),
            "budget_penalty": budget_penalty,
            "citation_policy_violation": citation_policy_violation,
            "source_citation_policy_violation": source_citation_policy_violation,
            "final_answer_parse": {
                "answer_present": answer_parse.answer is not None,
                "malformed": answer_parse.malformed,
                "reason": answer_parse.reason,
            },
            "final_answer_repair": {
                "raw_malformed": raw_answer_parse.malformed,
                "raw_reason": raw_answer_parse.reason,
                "syntax_repaired": answer_repair.repaired,
                "repair_type": answer_repair.repair_type,
            },
            "unopened_citations": sorted(unopened_citations),
            "hidden_reward_only": True,
            "trajectory_reward_observed": trajectory_reward_observed,
            "final_answer_reward": final_answer_reward,
            "final_answer_reward_observed": final_answer_reward_observed,
            "terminal_reward_version": "medgap_terminal_positive_v55",
            "missed_supported_slot_weight": self.missed_supported_slot_weight,
            "unjustified_abstention_weight": self.unjustified_abstention_weight,
            "missed_supported_slots": (
                terminal_behavior.missed_supported_slots if terminal_behavior else 0
            ),
            "unjustified_abstention": bool(
                terminal_behavior.unjustified_abstention if terminal_behavior else False
            ),
            "terminal_behavior_penalty": terminal_behavior_penalty,
            "terminal_behavior_penalty_observed": terminal_behavior_penalty_observed,
            "supported_slot_utilization_bonus_weight": (
                self.supported_slot_utilization_bonus_weight
            ),
            "natural_final_bonus_weight": self.natural_final_bonus_weight,
            "available_supported_slots": (
                terminal_positive.available_supported_slots if terminal_positive else 0
            ),
            "utilized_supported_slots": (
                terminal_positive.utilized_supported_slots if terminal_positive else 0
            ),
            "supported_slot_utilization_ratio": (
                terminal_positive.supported_slot_utilization_ratio
                if terminal_positive
                else 0.0
            ),
            "supported_slot_utilization_bonus": (
                terminal_positive.supported_slot_utilization_bonus
                if terminal_positive
                else 0.0
            ),
            "natural_final_eligible": bool(
                terminal_positive.natural_final_eligible if terminal_positive else False
            ),
            "natural_final_detected": bool(natural_final_detected),
            "hard_final_recovery": bool(hard_final_recovery),
            "natural_final_bonus": (
                terminal_positive.natural_final_bonus if terminal_positive else 0.0
            ),
            "terminal_behavior_bonus": terminal_behavior_bonus,
            "terminal_behavior_bonus_observed": terminal_behavior_bonus_observed,
            "terminal_behavior_verification": terminal_behavior_verification_audit,
            "final_answer_verification": final_answer_audit,
            "content_grounded_pass": bool(
                final_answer_audit.get("content_grounded_gate_passed", False)
            ),
            "strict_exact_citation_pass": bool(
                final_answer_audit.get("citation_gate_passed", False)
            ),
            "source_entailment_pass": bool(
                final_answer_audit.get("source_entailment_gate_passed", False)
            ),
            "model_citation_compliance_pass": bool(
                final_answer_audit.get("model_citation_compliance_passed", False)
            ),
            "citation_resolver_success": bool(
                final_answer_audit.get("citation_resolver_success", False)
            ),
            "evidence_delivery": {
                "trusted_channel_enabled": require_trusted_browse_delivery,
                "receipts_received": len(trusted_receipts),
                "receipts_used": len(trusted_receipts_used),
                "registered_citation_eligible_ids": sorted({
                    evidence_id
                    for receipt in trusted_receipts_used
                    for evidence_id in receipt["citation_eligible_ids"]
                }),
                "receipts": trusted_receipts_used,
            },
            "browse_final_verifier_consistency": verifier_consistency,
            "verifier_scope": self.verifier_scope,
            "terminal_trajectory_attribution": terminal_attribution,
        }
        self._write_private_trace(audit)
        return RolloutRewardResult(
            trajectory_reward=trajectory_reward,
            trajectory_reward_observed=trajectory_reward_observed,
            final_answer_reward=final_answer_reward,
            final_answer_reward_observed=final_answer_reward_observed,
            terminal_behavior_penalty=terminal_behavior_penalty,
            terminal_behavior_penalty_observed=terminal_behavior_penalty_observed,
            terminal_behavior_bonus=terminal_behavior_bonus,
            terminal_behavior_bonus_observed=terminal_behavior_bonus_observed,
            local_returns=returns,
            private_audit=audit,
        )

    def _write_private_trace(self, audit: dict[str, Any]) -> None:
        if self.private_trace_path is None:
            return
        with self._trace_lock:
            self.private_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.private_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
