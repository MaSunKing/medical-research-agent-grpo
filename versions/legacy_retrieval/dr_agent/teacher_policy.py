# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Versioned policy helpers for live medical Teacher collection."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml


PROMPT_VERSION = "medical_tool_calling_v2"
POLICY_VERSION = "medical_teacher_policy_v2_1"

_HARD_MUST_SEARCH_PATTERNS = (
    r"\b(current|currently|latest|guideline|recommendation|recommended)\b",
    r"\b(approved|approval|label(?:ing)?|boxed warning|safety alert|recall)\b",
    r"\b(regulator|regulatory|fda|ema|who|cdc|uspstf|contraindication|restriction)\b",
    r"(当前|最新|指南|推荐|批准|说明书|黑框警告|安全警示|监管)",
)

_EVIDENCE_LITERACY_PATTERNS = (
    r"\b(relative risk reduction|absolute risk reduction)\b",
    r"\bfalse[- ]positive\b.*\b(low[- ]prevalence|prevalence)\b",
    r"\bobservational association\b.*\b(caus|establish)\b",
    r"\bconfounding by indication\b",
    r"\b(noninferiority|non-inferiority)\b.*\b(equivalence|trial)\b",
    r"\bsubgroup (finding|analysis|analyses)\b",
    r"\bconfidence interval\b.*\bp-value\b",
    r"\bstatistically significant\b.*\bclinically important\b",
    r"\bimmortal[- ]time bias\b",
    r"\bstopping a randomized trial early\b",
    r"\bspectrum bias\b",
    r"\bverification bias\b",
    r"\bsurrogate endpoint",
    r"\bintention-to-treat\b.*\bper-protocol\b",
    r"\bnumber needed to treat\b",
    r"\bheterogeneity\b.*\bmeta-analysis\b",
    r"\bpublication bias\b",
    r"\bcompeting[- ]risk bias\b",
    r"\bcalibration\b.*\bdiscrimination\b",
    r"\bmissing data\b.*\bcomplete-case\b",
    r"\bwhat does the abbreviation\b",
)

_EMPIRICAL_MEDICAL_PATTERNS = (
    r"\b(effective|effectiveness|efficacy|benefit|harms?|safe|safety|adverse|toxicity)\b",
    r"\b(accuracy|accurately|sensitivity|specificity|predict|prognos|risk|associated|association)\b",
    r"\b(incidence|prevalence|mortality|survival|hospitali[sz]ation|recurrence|remission|outcome)\b",
    r"\b(trial|randomi[sz]ed|meta-analysis|systematic review|cohort|compared? with|compare|versus)\b",
    r"\b(treatment|therapy|medication|drug|vaccine|screening|diagnos|biomarker|genetic)\b",
    r"\b(who should|when should|which patients|how common|how strongly|how well)\b",
    r"(诊断|准确率|敏感度|特异度|临床试验|治疗|药物|疫苗|筛查|预后|风险|疗效|伤害|发生率|死亡率)",
)


def must_search(question: str) -> bool:
    """Return whether the medical question requires retrieved evidence."""
    text = " ".join(question.split()).lower()
    if any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _HARD_MUST_SEARCH_PATTERNS
    ):
        return True
    if any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _EVIDENCE_LITERACY_PATTERNS
    ):
        return False
    if any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _EMPIRICAL_MEDICAL_PATTERNS
    ):
        return True
    # This runtime is a medical Research Agent.  Bias safely toward retrieval
    # for any question that is not explicitly recognized as stable evidence
    # literacy or a simple abbreviation definition.
    return True


def load_teacher_prompt(workspace: Path) -> dict[str, str]:
    path = workspace / "agent" / "dr_agent" / "shared_prompts" / f"{PROMPT_VERSION}.yaml"
    raw = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    prompt = str(parsed["system_prompt"]).strip()
    return {
        "prompt_version": PROMPT_VERSION,
        "policy_version": POLICY_VERSION,
        "system_prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
