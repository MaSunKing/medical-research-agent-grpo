# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Deterministic terminal labels for MedGap reward shaping.

V55 keeps the historical penalty function for replay compatibility, but its
training contract sets both penalty weights to zero.  The new signal is a
bounded positive bonus for using directly supported slots in the Final, plus a
small conditional bonus for a naturally emitted (non-runtime-recovered) Final.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


DIRECT_EVIDENCE = "direct_evidence"
ABSTAINED = "abstained"
OMITTED = "omitted"


@dataclass(frozen=True)
class TerminalBehaviorReward:
    missed_supported_slots: int
    unjustified_abstention: bool
    missed_supported_slot_penalty: float
    unjustified_abstention_penalty: float
    total_penalty: float


@dataclass(frozen=True)
class TerminalPositiveReward:
    available_supported_slots: int
    utilized_supported_slots: int
    supported_slot_utilization_ratio: float
    supported_slot_utilization_bonus: float
    natural_final_eligible: bool
    natural_final_bonus: float
    total_bonus: float


@dataclass(frozen=True)
class LinkedSearchBrowseCredit:
    local_returns: tuple[float, ...]
    links: tuple[tuple[int, int], ...]
    chain_credits: tuple[tuple[int | None, int, float, float, float], ...] = ()
    provenance: tuple[tuple[int, str | None, str], ...] = ()


_POSITIVE_ASSESSMENT_FIELDS = (
    "evidence_available",
    "addressed",
    "supported_by_opened_evidence",
    "support_type",
)


def positive_reward_assessments_complete(
    assessments: Sequence[Any],
) -> bool:
    """Return whether assessments carry the full positive-credit contract.

    Terminal-behavior fallback assessments intentionally expose only
    evidence availability and disposition. They are sufficient for the legacy
    negative labels, but never for citation-independent utilization credit.
    """

    return all(
        all(hasattr(item, field) for field in _POSITIVE_ASSESSMENT_FIELDS)
        for item in assessments
    )


def derive_terminal_labels(
    *, evidence_available: str, final_disposition: str
) -> dict[str, bool]:
    direct = evidence_available == DIRECT_EVIDENCE
    abstained = final_disposition == ABSTAINED
    omitted = final_disposition == OMITTED
    return {
        "missed_supported_slot": direct and (abstained or omitted),
        "unjustified_abstention": direct and abstained,
        "abstention_justified": abstained and not direct,
    }


def terminal_behavior_reward(
    assessments: Sequence[Any],
    *,
    missed_supported_slot_weight: float,
    unjustified_abstention_weight: float,
) -> TerminalBehaviorReward:
    if not 0.0 <= missed_supported_slot_weight <= 1.0:
        raise ValueError("missed_supported_slot_weight must be in [0, 1]")
    if not 0.0 <= unjustified_abstention_weight <= 1.0:
        raise ValueError("unjustified_abstention_weight must be in [0, 1]")
    labels = [
        derive_terminal_labels(
            evidence_available=str(item.evidence_available),
            final_disposition=str(item.final_disposition),
        )
        for item in assessments
    ]
    slot_count = max(1, len(labels))
    missed = sum(int(item["missed_supported_slot"]) for item in labels)
    unjustified = any(item["unjustified_abstention"] for item in labels)
    missed_penalty = missed_supported_slot_weight * missed / slot_count
    abstention_penalty = unjustified_abstention_weight if unjustified else 0.0
    return TerminalBehaviorReward(
        missed_supported_slots=missed,
        unjustified_abstention=unjustified,
        missed_supported_slot_penalty=missed_penalty,
        unjustified_abstention_penalty=abstention_penalty,
        total_penalty=missed_penalty + abstention_penalty,
    )


def terminal_positive_reward(
    assessments: Sequence[Any],
    *,
    supported_slot_utilization_weight: float,
    natural_final_weight: float,
    natural_final_detected: bool,
    answer_safe: bool,
    minimum_natural_final_utilization: float = 0.75,
) -> TerminalPositiveReward:
    """Return citation-independent, positive-only Final shaping.

    A slot is available only when the frozen Judge says opened evidence
    directly supports it.  It is utilized only when the Final addressed that
    slot with direct opened-evidence support.  Unsupported or unavailable
    slots receive neither reward nor penalty.  Citation correctness remains a
    separate deterministic gate and is deliberately not inferred here.
    """

    if not positive_reward_assessments_complete(assessments):
        raise ValueError(
            "terminal positive reward requires complete Final semantic assessments"
        )

    for name, value in (
        ("supported_slot_utilization_weight", supported_slot_utilization_weight),
        ("natural_final_weight", natural_final_weight),
        ("minimum_natural_final_utilization", minimum_natural_final_utilization),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    available = 0
    utilized = 0
    for item in assessments:
        evidence_available = str(item.evidence_available)
        if evidence_available != DIRECT_EVIDENCE:
            continue
        available += 1
        if (
            bool(item.addressed)
            and bool(item.supported_by_opened_evidence)
            and str(item.support_type) == DIRECT_EVIDENCE
        ):
            utilized += 1

    utilization_ratio = utilized / available if available else 0.0
    utilization_bonus = (
        supported_slot_utilization_weight * utilization_ratio
        if available
        else 0.0
    )
    natural_eligible = bool(
        natural_final_detected
        and answer_safe
        and available > 0
        and utilization_ratio >= minimum_natural_final_utilization
    )
    natural_bonus = natural_final_weight if natural_eligible else 0.0
    return TerminalPositiveReward(
        available_supported_slots=available,
        utilized_supported_slots=utilized,
        supported_slot_utilization_ratio=utilization_ratio,
        supported_slot_utilization_bonus=utilization_bonus,
        natural_final_eligible=natural_eligible,
        natural_final_bonus=natural_bonus,
        total_bonus=utilization_bonus + natural_bonus,
    )


def linked_search_browse_credit(
    tools: Sequence[str],
    reasons: Sequence[Sequence[str]],
    rewards: Sequence[float],
    *,
    gamma: float,
    slot_count: int,
    clip: float = 1.0,
    conserve_chain_credit: bool = False,
    search_share: float = 0.30,
    browse_share: float = 0.70,
    discovered_candidate_ids: Sequence[Sequence[str]] | None = None,
    browse_source_ids: Sequence[str | None] | None = None,
    require_exact_provenance: bool = False,
) -> LinkedSearchBrowseCredit:
    """Assign delayed Browse gain to its exact candidate-producing Search.

    V59 uses source-exact provenance when requested: a Browse is linked only
    when exactly one earlier compatible Search returned the browsed PMID/WEB
    source. If provenance is unavailable or ambiguous, the Browse keeps the
    complete gain and Search gets zero; credit is never guessed and never
    lost. The historical nearest-search behavior remains available only for
    replay compatibility when ``require_exact_provenance`` is false.
    """

    if not (len(tools) == len(reasons) == len(rewards)):
        raise ValueError("tools, reasons, and rewards must have equal length")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if slot_count < 1:
        raise ValueError("slot_count must be positive")
    if clip <= 0:
        raise ValueError("clip must be positive")
    if discovered_candidate_ids is not None and len(discovered_candidate_ids) != len(tools):
        raise ValueError("discovered_candidate_ids must align with tools")
    if browse_source_ids is not None and len(browse_source_ids) != len(tools):
        raise ValueError("browse_source_ids must align with tools")
    if require_exact_provenance and (
        discovered_candidate_ids is None or browse_source_ids is None
    ):
        raise ValueError("exact provenance requires candidate and Browse source IDs")

    if conserve_chain_credit:
        for name, value in (
            ("search_share", search_share),
            ("browse_share", browse_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if abs((search_share + browse_share) - 1.0) > 1e-8:
            raise ValueError("search_share + browse_share must equal 1")

        # V57 treats an evidence gain as one conserved unit of credit.  A
        # matched Search->Browse chain shares that unit; it is never copied to
        # both decisions.  A Browse without a matching discovery Search keeps
        # the whole gain.  Zero/duplicate-gain decisions remain exactly zero.
        values = [0.0 for _ in rewards]
        matching_search = {
            "browse_document": "pubmed_search",
            "browse_webpage": "medical_web_search",
        }
        links: list[tuple[int, int]] = []
        chain_credits: list[tuple[int | None, int, float, float, float]] = []
        provenance: list[tuple[int, str | None, str]] = []
        for browse_index, (tool, reward) in enumerate(zip(tools, rewards)):
            expected_search = matching_search.get(str(tool))
            raw_gain = max(0.0, float(reward)) / slot_count
            gain = min(clip, raw_gain)
            if expected_search is None or gain <= 0.0:
                continue
            browse_source_id = None
            if browse_source_ids is not None:
                browse_source_id = str(browse_source_ids[browse_index] or "") or None
            if require_exact_provenance:
                matching_indices = [
                    index
                    for index in range(0, browse_index)
                    if str(tools[index]) == expected_search
                    and "candidate_discovery_only" in set(reasons[index])
                    and browse_source_id is not None
                    and browse_source_id
                    in {str(value) for value in discovered_candidate_ids[index]}
                ]
                if len(matching_indices) == 1:
                    search_index = matching_indices[0]
                    provenance_status = "exact"
                elif matching_indices:
                    # The source ID proves that each Search discovered the
                    # candidate, but not which repeated discovery caused the
                    # later Browse. Do not guess between them.
                    search_index = None
                    provenance_status = "ambiguous"
                else:
                    search_index = None
                    provenance_status = "unavailable"
            else:
                search_index = next(
                    (
                        index
                        for index in range(browse_index - 1, -1, -1)
                        if str(tools[index]) == expected_search
                        and "candidate_discovery_only" in set(reasons[index])
                    ),
                    None,
                )
                provenance_status = "nearest_legacy" if search_index is not None else "unavailable"
            if search_index is None:
                search_credit = 0.0
                browse_credit = gain
            else:
                search_credit = search_share * gain
                browse_credit = browse_share * gain
                values[search_index] += search_credit
                links.append((search_index, browse_index))
            values[browse_index] += browse_credit
            chain_credits.append(
                (
                    search_index,
                    browse_index,
                    gain,
                    search_credit,
                    browse_credit,
                )
            )
            provenance.append((browse_index, browse_source_id, provenance_status))

        values = [max(-clip, min(clip, value)) for value in values]
        if abs(sum(values) - sum(item[2] for item in chain_credits)) > 1e-7:
            raise RuntimeError("conserved Search/Browse credit does not sum to evidence gain")
        return LinkedSearchBrowseCredit(
            local_returns=tuple(values),
            links=tuple(links),
            chain_credits=tuple(chain_credits),
            provenance=tuple(provenance),
        )

    values = [
        max(-clip, min(clip, float(reward) / slot_count))
        for reward in rewards
    ]
    for index, item_reasons in enumerate(reasons):
        if "candidate_discovery_only" in set(item_reasons):
            values[index] = 0.0

    matching_search = {
        "browse_document": "pubmed_search",
        "browse_webpage": "medical_web_search",
    }
    links: list[tuple[int, int]] = []
    for browse_index, (tool, reward) in enumerate(zip(tools, rewards)):
        expected_search = matching_search.get(str(tool))
        if expected_search is None or float(reward) <= 0.0:
            continue
        search_index = next(
            (
                index
                for index in range(browse_index - 1, -1, -1)
                if str(tools[index]) == expected_search
                and "candidate_discovery_only" in set(reasons[index])
            ),
            None,
        )
        if search_index is None:
            continue
        delayed = gamma * float(reward) / slot_count
        values[search_index] = max(
            -clip,
            min(clip, values[search_index] + delayed),
        )
        links.append((search_index, browse_index))

    return LinkedSearchBrowseCredit(
        local_returns=tuple(values),
        links=tuple(links),
    )
