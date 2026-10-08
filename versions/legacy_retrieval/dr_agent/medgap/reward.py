# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Hidden evidence ledger and bounded decision-span credit calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .schemas import EvidenceRubric, SemanticVerification, SourceMetadata


@dataclass(frozen=True)
class DecisionCredit:
    reward: float
    newly_covered_slots: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass
class HiddenEvidenceLedger:
    """Reward-only state. Never serialize this into a policy observation."""

    rubric: EvidenceRubric
    support_threshold: float = 0.80
    relevance_threshold: float = 0.50
    covered_by: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.covered_by = {slot.slot_id: set() for slot in self.rubric.slots}

    @property
    def covered_slots(self) -> set[str]:
        return {slot_id for slot_id, evidence_ids in self.covered_by.items() if evidence_ids}

    @property
    def complete(self) -> bool:
        return self.rubric.no_tool_expected or len(self.covered_slots) == len(self.rubric.slots)

    @property
    def coverage(self) -> float:
        if self.rubric.no_tool_expected:
            return 1.0
        return len(self.covered_slots) / max(1, len(self.rubric.slots))

    def score_tool_decision(
        self,
        *,
        tool: str,
        evidence_id: str | None = None,
        verification: SemanticVerification | None = None,
        duplicate: bool = False,
        policy_invalid: bool = False,
        environment_failure: bool = False,
        search_no_results: bool = False,
        budget_violation: bool = False,
        final_answer_reserve: bool = False,
        post_reserve_tool_attempt: bool = False,
    ) -> DecisionCredit:
        was_complete = self.complete
        if post_reserve_tool_attempt:
            return DecisionCredit(-0.25, reasons=("post_reserve_tool_attempt",))
        if environment_failure:
            return DecisionCredit(0.0, reasons=("environment_failure_neutral",))
        if search_no_results:
            return DecisionCredit(0.0, reasons=("search_no_results_neutral",))
        if final_answer_reserve:
            return DecisionCredit(0.0, reasons=("final_answer_reserve_neutral",))
        if duplicate:
            return DecisionCredit(-0.5, reasons=("duplicate_action",))
        if budget_violation:
            return DecisionCredit(-0.25, reasons=("budget_violation",))
        if policy_invalid:
            return DecisionCredit(-0.25, reasons=("policy_invalid_action",))
        if self.rubric.no_tool_expected:
            return DecisionCredit(-0.25, reasons=("unnecessary_tool_for_no_tool_question",))
        if was_complete:
            return DecisionCredit(-0.25, reasons=("tool_call_after_coverage_complete",))
        if tool in {"pubmed_search", "medical_web_search"}:
            return DecisionCredit(0.0, reasons=("candidate_discovery_only",))
        if tool not in {"browse_document", "browse_webpage"}:
            return DecisionCredit(-0.25, reasons=("unknown_research_tool",))
        if verification is None or not evidence_id:
            return DecisionCredit(-0.25, reasons=("browse_without_verifiable_evidence",))

        return self.score_browse_decision(
            tool=tool,
            verified_evidence=[(evidence_id, verification)],
        )

    def score_browse_decision(
        self,
        *,
        tool: str,
        verified_evidence: Iterable[
            tuple[str, SemanticVerification]
            | tuple[str, SemanticVerification, SourceMetadata]
        ],
    ) -> DecisionCredit:
        """Score all focused chunks returned by one Browse action as one decision."""
        if tool not in {"browse_document", "browse_webpage"}:
            raise ValueError("score_browse_decision requires a browse tool")
        if self.complete:
            return DecisionCredit(-0.25, reasons=("tool_call_after_coverage_complete",))

        new_slots = []
        saw_evidence = False
        saw_irrelevant = False
        saw_source_mismatch = False
        saw_contradiction = False
        for item in verified_evidence:
            evidence_id, verification = item[:2]
            source = item[2] if len(item) == 3 else None
            saw_evidence = True
            if verification.major_contradiction:
                # A negative or null medical result is not itself a contradiction.
                # If the verifier finds an actual internal mismatch, ignore that
                # chunk for coverage instead of assigning direction-dependent reward.
                saw_contradiction = True
                continue
            if verification.evidence_relevance < self.relevance_threshold:
                saw_irrelevant = True
                continue
            for support in verification.slot_support:
                slot = next((value for value in self.rubric.slots if value.slot_id == support.slot_id), None)
                metadata_requirement_met = (
                    source_satisfies_requirements(slot.source_requirements, source)
                    if slot is not None and source is not None
                    else True
                )
                if support.source_requirement_met and not metadata_requirement_met:
                    saw_source_mismatch = True
                if (
                    support.verdict == "supported"
                    and support.confidence >= self.support_threshold
                    and support.source_requirement_met
                    and metadata_requirement_met
                    and support.slot_id in self.covered_by
                    and not self.covered_by[support.slot_id]
                ):
                    self.covered_by[support.slot_id].add(evidence_id)
                    new_slots.append(support.slot_id)
        if not saw_evidence:
            return DecisionCredit(-0.25, reasons=("browse_without_verifiable_evidence",))
        if new_slots:
            return DecisionCredit(
                float(len(new_slots)),
                newly_covered_slots=tuple(sorted(new_slots)),
                reasons=("marginal_evidence_gain",),
            )
        if saw_source_mismatch:
            return DecisionCredit(0.0, reasons=("source_metadata_requirement_not_met",))
        if saw_irrelevant:
            return DecisionCredit(0.0, reasons=("evidence_below_relevance_threshold",))
        if saw_contradiction:
            return DecisionCredit(0.0, reasons=("contradictory_chunk_ignored",))
        return DecisionCredit(0.0, reasons=("no_new_evidence_slot",))


def source_satisfies_requirements(
    requirements: Iterable[str],
    source: SourceMetadata,
) -> bool:
    """Deterministically enforce source-type requirements that metadata can prove.

    Semantic constraints such as population matching remain the verifier's job;
    hard provenance constraints (guideline, regulatory, randomized, systematic
    review, and human clinical evidence) require corroborating metadata.
    """

    requirements = list(requirements)
    if len(requirements) != 1:
        return False
    requirement = requirements[0].casefold()
    human_types = set(source.human_evidence_types)
    exact_checks = {
        "official current clinical guideline": source.authority_type == "guideline",
        "official public-health guidance": source.authority_type == "public_health",
        "official regulatory source": source.authority_type == "regulatory",
        "randomized human evidence": source.is_human and source.study_design == "randomized_trial",
        "systematic review or meta-analysis": source.study_design in {"systematic_review", "meta_analysis"},
        "diagnostic-accuracy human evidence": source.is_human and source.study_design == "diagnostic_accuracy",
        "comparative human evidence": source.is_human and "comparative" in human_types,
        "prognostic human evidence": source.is_human and "prognostic" in human_types,
        "pharmacoepidemiologic human evidence": source.is_human and "pharmacoepidemiologic" in human_types,
        "reported human safety outcomes": source.is_human and "safety" in human_types,
        "population-specific human evidence": source.is_human,
        "human mechanistic evidence": source.is_human and "mechanistic" in human_types,
        "primary trial report or protocol": source.is_human and source.study_design in {
            "randomized_trial", "clinical_trial", "protocol"
        },
        "primary methodological or reporting standard": (
            source.authority_type == "methodological_standard"
            or source.study_design == "methodological_standard"
        ),
    }
    return exact_checks.get(requirement, False)


def local_return_to_go(
    rewards: Iterable[float],
    *,
    gamma: float = 0.9,
    slot_count: int = 1,
    clip: float = 1.0,
) -> list[float]:
    """Compute bounded local returns so Search receives delayed Browse credit."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if slot_count < 1:
        raise ValueError("slot_count must be positive")
    values = list(float(value) for value in rewards)
    result = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = values[index] + gamma * running
        normalized = running / slot_count
        result[index] = max(-clip, min(clip, normalized))
    return result
