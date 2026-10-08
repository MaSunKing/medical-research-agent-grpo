# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Hidden reward components for Step-aware MedGap-GRPO.

Nothing from this package should be appended to the policy conversation.  The
agent-facing runtime lives in :mod:`dr_agent.medical_runtime`; this package is
only for rubric construction, semantic verification, and reward accounting.
"""

from .reward import HiddenEvidenceLedger, local_return_to_go
from .schemas import (
    EvidenceChunk,
    EvidenceRubric,
    EvidenceSlot,
    SemanticVerification,
    SourceMetadata,
)
from .verifier import QwenMaxVerifier, VerifierConfig, VerificationEnvelope
from .service import MedGapRewardService, RolloutRewardResult, rubric_from_ground_truth

__all__ = [
    "EvidenceChunk",
    "EvidenceRubric",
    "EvidenceSlot",
    "HiddenEvidenceLedger",
    "QwenMaxVerifier",
    "SemanticVerification",
    "SourceMetadata",
    "VerificationEnvelope",
    "VerifierConfig",
    "local_return_to_go",
    "MedGapRewardService",
    "RolloutRewardResult",
    "rubric_from_ground_truth",
]
