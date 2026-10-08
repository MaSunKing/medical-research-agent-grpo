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


class FinalAnswerSlotAssessment(StrictModel):
    slot_id: str = Field(min_length=1, max_length=80)
    addressed: bool
    supported_by_opened_evidence: bool
    support_type: Literal[
        "direct_evidence", "opened_evidence_limitation", "unsupported"
    ]
    citation_ids: list[str] = Field(default_factory=list, max_length=20)
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


class NoToolAnswerVerification(StrictModel):
    answered_question: bool
    factually_correct: bool
    stable_evidence_literacy_topic: bool
    medical_safety_ok: bool
    language_matches: bool
    no_fabricated_citation: bool
    rationale: str = Field(min_length=1, max_length=1000)
