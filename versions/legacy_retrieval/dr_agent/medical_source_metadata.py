# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Deterministic source metadata shared by MCP browse tools and MedGap reward."""

from __future__ import annotations

import re
from typing import Iterable


AUTHORITY_TYPES = {"none", "guideline", "public_health", "regulatory", "methodological_standard"}
STUDY_DESIGNS = {
    "unknown", "randomized_trial", "clinical_trial", "protocol", "systematic_review",
    "meta_analysis", "observational", "diagnostic_accuracy", "methodological_standard",
}
HUMAN_EVIDENCE_TYPES = {"comparative", "prognostic", "pharmacoepidemiologic", "safety", "mechanistic"}


def _contains(text: str, *terms: str) -> bool:
    """Match controlled words/phrases without substring false positives."""
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])", text)
        for term in terms
    )


def infer_medical_source_metadata(
    *,
    title: str = "",
    url: str = "",
    publication_types: Iterable[str] = (),
    evidence_text: str = "",
) -> dict[str, object]:
    """Infer conservative controlled labels when an upstream source lacks them.

    Document identity is inferred from title, URL, and publication types;
    evidence-content labels such as safety are inferred from the actual focused
    chunks.  Explicit metadata supplied by a backend should take precedence.
    """

    types = [str(value) for value in publication_types]
    identity = " ".join([title, url, *types]).casefold()
    content = " ".join([identity, evidence_text]).casefold()

    non_human = _contains(
        content,
        "mouse", "mice", "murine", "rat", "rats", "rodent", "rodents",
        "animal", "animals", "animal model", "animal models", "in vitro",
        "cell line", "cell lines", "preclinical", "rabbit", "rabbits",
    )
    human_signal = _contains(
        content,
        "participant", "participants", "patient", "patients", "adult", "adults",
        "child", "children", "woman", "women", "man", "men", "human subject",
        "human subjects", "human study", "human studies", "clinical study",
        "clinical studies", "clinical trial", "clinical trials", "cohort study",
        "cohort studies",
    )
    normalized_types = {value.casefold().strip() for value in types}
    pubmed_human_design = bool(normalized_types & {
        "randomized controlled trial", "clinical trial", "controlled clinical trial",
        "observational study", "comparative study",
    })

    # Document type outranks organization: a WHO clinical guideline is still
    # a guideline, not merely a generic public-health webpage.
    # Reporting/methodological standards outrank the generic word "guideline".
    if _contains(
        identity,
        "reporting guideline", "methodological standard", "consensus statement",
        "consort", "strobe statement", "prisma statement",
    ):
        authority_type = "methodological_standard"
    elif any(term in identity for term in (
        "clinical guideline", "practice guideline", "treatment guideline",
        "diagnostic guideline", "/guideline", "/guidelines", " guideline ",
        "nice.org.uk/guidance", "kdigo.org",
    )):
        authority_type = "guideline"
    elif any(term in identity for term in (
        "fda.gov", "ema.europa.eu", "nmpa.gov.cn", "drug label", "product label",
        "boxed warning", "regulatory assessment",
    )):
        authority_type = "regulatory"
    elif any(term in identity for term in (
        "who.int", "cdc.gov", "ecdc.europa.eu", "public health", "health advisory",
    )):
        authority_type = "public_health"
    else:
        authority_type = "none"

    if authority_type == "methodological_standard":
        study_design = "methodological_standard"
    elif _contains(identity, "systematic review"):
        study_design = "systematic_review"
    elif _contains(identity, "meta-analysis", "meta analysis"):
        study_design = "meta_analysis"
    elif "randomized controlled trial" in normalized_types or _contains(
        content, "randomized", "randomised", "randomly assigned", "random allocation"
    ):
        study_design = "randomized_trial"
    elif _contains(identity, "protocol"):
        study_design = "protocol"
    elif _contains(identity, "clinical trial", "controlled trial"):
        study_design = "clinical_trial"
    elif any(term in content for term in ("diagnostic accuracy", "sensitivity and specificity")):
        study_design = "diagnostic_accuracy"
    elif any(term in identity for term in ("cohort", "case-control", "case control", "observational", "registry")):
        study_design = "observational"
    else:
        study_design = "unknown"

    is_human = not non_human and (human_signal or pubmed_human_design)

    human_types: set[str] = set()
    if (study_design == "randomized_trial" and is_human) or _contains(
        content, "comparative", "versus", "compared with", "case-control", "case control"
    ):
        human_types.add("comparative")
    if _contains(content, *(
        "prognostic", "prognosis", "survival prediction", "risk prediction", "predictor of",
        "predict progression", "prediction of progression", "progression risk", "risk model",
        "c-statistic", "c statistic", "external validation", "prognostic factor",
    )):
        human_types.add("prognostic")
    if _contains(content, *(
        "pharmacoepidemi", "claims database", "pharmacovigilance", "drug utilization",
    )) or "pharmacoepidemi" in content:
        human_types.add("pharmacoepidemiologic")
    if _contains(content, *(
        "safety", "adverse event", "adverse effect", "serious adverse", "harm", "toxicity",
    )):
        human_types.add("safety")
    if _contains(content, *(
        "mechanism", "mechanistic", "pharmacodynamic", "biomarker",
    )):
        human_types.add("mechanistic")

    return {
        "authority_type": authority_type,
        "study_design": study_design,
        "is_human": is_human,
        "human_evidence_types": sorted(human_types),
    }
