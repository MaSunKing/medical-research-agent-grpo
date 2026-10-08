# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Versioned schemas used by the hidden MedGap reward service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ControlledRequirement = Literal[
    "official current clinical guideline",
    "official regulatory source",
    "official public-health guidance",
    "randomized human evidence",
    "comparative human evidence",
    "systematic review or meta-analysis",
    "diagnostic-accuracy human evidence",
    "prognostic human evidence",
    "pharmacoepidemiologic human evidence",
    "reported human safety outcomes",
    "population-specific human evidence",
    "human mechanistic evidence",
    "primary trial report or protocol",
    "primary methodological or reporting standard",
]


class EvidenceSlot(StrictModel):
    slot_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    description: str = Field(min_length=3, max_length=500)
    source_requirements: list[ControlledRequirement] = Field(min_length=1, max_length=1)


class EvidenceRubric(StrictModel):
    question_id: str = Field(min_length=1, max_length=160)
    rubric_version: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=3, max_length=4000)
    slots: list[EvidenceSlot] = Field(default_factory=list, max_length=8)
    no_tool_expected: bool = False

    @model_validator(mode="after")
    def validate_slot_ids(self) -> "EvidenceRubric":
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("rubric slot_id values must be unique")
        if self.no_tool_expected and self.slots:
            raise ValueError("a no-tool rubric cannot require evidence slots")
        if not self.no_tool_expected and not self.slots:
            raise ValueError("a search rubric must contain at least one evidence slot")
        return self


class SourceMetadata(StrictModel):
    source_id: str = Field(min_length=1, max_length=300)
    source_type: str = Field(min_length=1, max_length=100)
    title: str = Field(default="", max_length=1000)
    url: str = Field(default="", max_length=3000)
    publication_types: list[str] = Field(default_factory=list, max_length=30)
    publication_date: str = Field(default="", max_length=80)
    authority_type: Literal[
        "none", "guideline", "public_health", "regulatory", "methodological_standard"
    ] = "none"
    study_design: Literal[
        "unknown", "randomized_trial", "clinical_trial", "protocol", "systematic_review",
        "meta_analysis", "observational", "diagnostic_accuracy", "methodological_standard"
    ] = "unknown"
    is_human: bool = False
    human_evidence_types: list[
        Literal["comparative", "prognostic", "pharmacoepidemiologic", "safety", "mechanistic"]
    ] = Field(default_factory=list, max_length=5)


class EvidenceChunk(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=30000)
    source: SourceMetadata


class SlotSupport(StrictModel):
    slot_id: str = Field(min_length=1, max_length=80)
    verdict: Literal["supported", "partial", "unsupported"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_quote: str = Field(default="", max_length=2000)
    source_requirement_met: bool
    rationale: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def supported_requires_quote(self) -> "SlotSupport":
        if self.verdict == "supported" and not self.supporting_quote.strip():
            raise ValueError("supported verdict requires a supporting_quote")
        return self


class SemanticVerification(StrictModel):
    slot_support: list[SlotSupport] = Field(max_length=8)
    evidence_relevance: float = Field(ge=0.0, le=1.0)
    major_contradiction: bool
    contradiction_rationale: str = Field(default="", max_length=800)


class FinalEvidenceReference(StrictModel):
    """Judge-selected provenance with a verbatim, mechanically checkable quote."""

    evidence_id: str = Field(min_length=1, max_length=300)
    supporting_quote: str = Field(min_length=1, max_length=1200)
    quote_normalization_repaired: bool = False


class FinalAnswerSemanticSlotAssessment(StrictModel):
    """Judge-owned semantics with provenance expressed only as input indexes.

    The Judge never copies an opaque evidence ID or source quote.  Runtime code
    binds these indexes back to the immutable OPENED_EVIDENCE array.
    """

    slot_id: str = Field(min_length=1, max_length=80)
    evidence_available: Literal[
        "direct_evidence", "opened_evidence_limitation", "none"
    ]
    final_disposition: Literal[
        "substantive_answer", "qualified_answer", "abstained", "omitted"
    ]
    support_type: Literal[
        "direct_evidence", "opened_evidence_limitation", "unsupported"
    ]
    available_evidence_indices: list[int] = Field(default_factory=list, max_length=20)
    answer_support_evidence_indices: list[int] = Field(
        default_factory=list, max_length=20
    )
    rationale: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_semantic_consistency(self) -> "FinalAnswerSemanticSlotAssessment":
        available = self.available_evidence_indices
        support = self.answer_support_evidence_indices
        if any(index < 0 for index in available + support):
            raise ValueError("evidence indexes must be non-negative")
        if len(available) != len(set(available)) or len(support) != len(set(support)):
            raise ValueError("evidence indexes must be unique")
        if self.evidence_available == "none" and available:
            raise ValueError("evidence_available=none requires no available indexes")
        if self.evidence_available != "none" and not available:
            raise ValueError("available evidence requires at least one evidence index")
        if self.support_type == "unsupported" and support:
            raise ValueError("unsupported answer requires no support indexes")
        if self.support_type != "unsupported" and not support:
            raise ValueError("supported answer requires at least one support index")
        if not set(support).issubset(available):
            raise ValueError("answer support indexes must be a subset of available indexes")
        expected_availability = {
            "direct_evidence": "direct_evidence",
            "opened_evidence_limitation": "opened_evidence_limitation",
        }.get(self.support_type)
        if expected_availability and self.evidence_available != expected_availability:
            raise ValueError("support_type contradicts evidence_available")
        if self.final_disposition in {"abstained", "omitted"}:
            if self.support_type != "unsupported" or support:
                raise ValueError("abstained or omitted Final cannot receive answer support")
        return self


class FinalAnswerSemanticVerification(StrictModel):
    """Minimal full-answer contract returned by the terminal Judge."""

    answer_present: bool
    medical_safety_ok: bool
    unsupported_strong_claim: bool
    slot_assessments: list[FinalAnswerSemanticSlotAssessment] = Field(max_length=8)


# V54.4 Judge-owned state.  Every older compatibility field is derived from
# this single enum in deterministic runtime code.
MinimalSemanticState = Literal[
    "direct_supported_substantive",
    "direct_supported_qualified",
    "direct_unsupported_substantive",
    "direct_unsupported_qualified",
    "direct_abstained",
    "direct_omitted",
    "limitation_supported_substantive",
    "limitation_supported_qualified",
    "limitation_unsupported_substantive",
    "limitation_unsupported_qualified",
    "limitation_abstained",
    "limitation_omitted",
    "none_unsupported_substantive",
    "none_unsupported_qualified",
    "none_abstained",
    "none_omitted",
]


class MinimalSemanticSlotAssessment(StrictModel):
    semantic_state: MinimalSemanticState
    # Required on purpose.  An empty list is legal only for a none_* state.
    evidence_indices: list[int] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_minimal_state(self) -> "MinimalSemanticSlotAssessment":
        indexes = self.evidence_indices
        if any(index < 0 for index in indexes):
            raise ValueError("evidence indexes must be non-negative")
        if len(indexes) != len(set(indexes)):
            raise ValueError("evidence indexes must be unique")
        no_evidence = self.semantic_state.startswith("none_")
        if no_evidence and indexes:
            raise ValueError("none_* state requires an empty evidence index list")
        if not no_evidence and not indexes:
            raise ValueError("non-none state requires at least one evidence index")
        return self


class MinimalSemanticVerification(StrictModel):
    answer_present: bool
    medical_safety_ok: bool
    unsupported_strong_claim: bool
    # Dynamic required slot keys are supplied in the request JSON Schema.
    slot_assessments: dict[str, MinimalSemanticSlotAssessment]


class FinalAnswerSlotAssessment(StrictModel):
    slot_id: str = Field(min_length=1, max_length=80)
    evidence_available: Literal[
        "direct_evidence", "opened_evidence_limitation", "none"
    ]
    final_disposition: Literal[
        "substantive_answer", "qualified_answer", "abstained", "omitted"
    ]
    addressed: bool
    supported_by_opened_evidence: bool
    support_type: Literal[
        "direct_evidence", "opened_evidence_limitation", "unsupported"
    ]
    citation_ids: list[str] = Field(default_factory=list, max_length=20)
    available_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    supporting_evidence: list[FinalEvidenceReference] = Field(
        default_factory=list, max_length=20
    )
    rationale: str = Field(min_length=1, max_length=800)


class FinalAnswerVerification(StrictModel):
    answer_present: bool
    citations_grounded: bool
    medical_safety_ok: bool
    unsupported_strong_claim: bool
    slot_assessments: list[FinalAnswerSlotAssessment] = Field(max_length=8)


class TerminalBehaviorSlotAssessment(StrictModel):
    """Minimal semantic primitives used only when the full Final judge is invalid."""

    slot_id: str = Field(min_length=1, max_length=80)
    evidence_available: Literal[
        "direct_evidence", "opened_evidence_limitation", "none"
    ]
    final_disposition: Literal[
        "substantive_answer", "qualified_answer", "abstained", "omitted"
    ]
    available_evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    supporting_evidence: list[FinalEvidenceReference] = Field(
        default_factory=list, max_length=20
    )
    rationale: str = Field(min_length=1, max_length=800)


class TerminalBehaviorVerification(StrictModel):
    """Quote-backed fallback judgment; reward labels remain deterministic code."""

    slot_assessments: list[TerminalBehaviorSlotAssessment] = Field(max_length=8)


class TerminalBehaviorSemanticSlotAssessment(StrictModel):
    """Quote-free terminal fallback result, bound by evidence array index."""

    slot_id: str = Field(min_length=1, max_length=80)
    evidence_available: Literal[
        "direct_evidence", "opened_evidence_limitation", "none"
    ]
    final_disposition: Literal[
        "substantive_answer", "qualified_answer", "abstained", "omitted"
    ]
    available_evidence_indices: list[int] = Field(default_factory=list, max_length=20)
    rationale: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def validate_semantic_consistency(
        self,
    ) -> "TerminalBehaviorSemanticSlotAssessment":
        indexes = self.available_evidence_indices
        if any(index < 0 for index in indexes):
            raise ValueError("evidence indexes must be non-negative")
        if len(indexes) != len(set(indexes)):
            raise ValueError("evidence indexes must be unique")
        if self.evidence_available == "none" and indexes:
            raise ValueError("evidence_available=none requires no available indexes")
        if self.evidence_available != "none" and not indexes:
            raise ValueError("available evidence requires at least one evidence index")
        return self


class TerminalBehaviorSemanticVerification(StrictModel):
    slot_assessments: list[TerminalBehaviorSemanticSlotAssessment] = Field(max_length=8)


class NoToolAnswerVerification(StrictModel):
    answered_question: bool
    factually_correct: bool
    stable_evidence_literacy_topic: bool
    medical_safety_ok: bool
    language_matches: bool
    no_fabricated_citation: bool
    rationale: str = Field(min_length=1, max_length=1000)
