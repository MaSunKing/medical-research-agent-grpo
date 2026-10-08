# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Validation and finalization for Codex medical Teacher batches."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .trajectory_logger import append_trajectory


DECISION_SCHEMA_VERSION = "medical_teacher_decision_v1"
ALLOWED_TOOLS = {
    "pubmed_search",
    "browse_document",
    "medical_web_search",
    "browse_webpage",
}
BROWSE_TOOLS = {"browse_document", "browse_webpage"}
V1_MAX_AGENT_TOOL_CALLS = 6


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        decision = json.loads(line)
        if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
            raise ValueError(f"Decision line {line_number}: unexpected schema_version")
        seed_id = str(decision.get("seed_id") or "").strip()
        if not seed_id:
            raise ValueError(f"Decision line {line_number}: seed_id is required")
        if seed_id in decisions:
            raise ValueError(f"Decision line {line_number}: duplicate seed_id {seed_id}")
        if decision.get("decision") not in {"accepted", "rejected"}:
            raise ValueError(
                f"Decision line {line_number}: decision must be accepted or rejected"
            )
        decisions[seed_id] = decision
    return decisions


def load_sessions(
    directory: Path, pattern: str = "[0-9][0-9]_*.json"
) -> dict[str, tuple[Path, dict[str, Any]]]:
    sessions: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob(pattern)):
        session = json.loads(path.read_text(encoding="utf-8"))
        seed_id = str(session.get("seed_id") or "").strip()
        if not seed_id:
            raise ValueError(f"Session {path}: seed_id is required")
        if seed_id in sessions:
            raise ValueError(f"Duplicate session seed_id {seed_id}")
        sessions[seed_id] = (path, session)
    return sessions


def _validate_accepted_decision(
    decision: dict[str, Any], session: dict[str, Any]
) -> tuple[str, list[str]]:
    final_answer = str(decision.get("final_answer") or "").strip()
    if not final_answer:
        raise ValueError(f"{session['seed_id']}: accepted decision requires final_answer")
    if decision.get("question") and decision["question"] != session.get("question"):
        raise ValueError(f"{session['seed_id']}: decision question does not match session")

    tool_calls = [
        item for item in session.get("events", []) if item.get("type") == "tool_call"
    ]
    unexpected_tools = {item.get("tool") for item in tool_calls} - ALLOWED_TOOLS
    if unexpected_tools:
        raise ValueError(
            f"{session['seed_id']}: unexpected tools {sorted(unexpected_tools)}"
        )
    if len(tool_calls) > V1_MAX_AGENT_TOOL_CALLS:
        raise ValueError(
            f"{session['seed_id']}: {len(tool_calls)} Agent Tool Calls exceed "
            f"Medical Agent V1 limit {V1_MAX_AGENT_TOOL_CALLS}"
        )
    rationales = decision.get("rationales")
    if not isinstance(rationales, list) or not all(
        isinstance(item, str) and item.strip() for item in rationales
    ):
        raise ValueError(
            f"{session['seed_id']}: rationales must be a non-empty string list"
        )
    if len(rationales) != len(tool_calls):
        raise ValueError(
            f"{session['seed_id']}: {len(rationales)} rationales for "
            f"{len(tool_calls)} tool calls"
        )

    citation_ids = re.findall(r'<cite id="([^"]+)">', final_answer)
    no_tool = decision.get("no_tool") is True
    if no_tool:
        if tool_calls:
            raise ValueError(f"{session['seed_id']}: no-tool decision has tool calls")
        if session.get("must_search") is not False:
            raise ValueError(f"{session['seed_id']}: no-tool is forbidden by must_search")
        if session.get("metrics", {}).get("no_tool") is not True:
            raise ValueError(f"{session['seed_id']}: session was not finished as no-tool")
        if citation_ids:
            raise ValueError(f"{session['seed_id']}: no-tool answer must not contain citations")
        claim_checks = decision.get("no_tool_claim_checks")
        if not isinstance(claim_checks, list) or not claim_checks:
            raise ValueError(f"{session['seed_id']}: no_tool_claim_checks are required")
        for index, audit in enumerate(claim_checks, 1):
            if not isinstance(audit, dict) or not str(audit.get("claim") or "").strip():
                raise ValueError(f"{session['seed_id']}: no-tool claim {index} is missing")
            if audit.get("supported") is not True:
                raise ValueError(f"{session['seed_id']}: no-tool claim {index} is not supported")
            if not str(audit.get("explanation") or "").strip():
                raise ValueError(f"{session['seed_id']}: no-tool claim {index} lacks explanation")
        checks = decision.get("quality_checks", {})
        if checks.get("no_tool_policy_validated") is not True:
            raise ValueError(f"{session['seed_id']}: no_tool_policy_validated must be true")
        if checks.get("completed") is not True or checks.get("evidence_sufficient") is not True:
            raise ValueError(f"{session['seed_id']}: completed no-tool answer is not sufficient")
        return final_answer, []

    if not citation_ids:
        raise ValueError(f"{session['seed_id']}: final_answer has no citations")
    successful_browse_outputs = []
    for item in session.get("events", []):
        if item.get("type") != "tool_output" or item.get("tool") not in BROWSE_TOOLS:
            continue
        output = item.get("output")
        if not isinstance(output, dict):
            continue
        if output.get("failed") or output.get("error"):
            continue
        if not isinstance(output.get("data"), list) or not output["data"]:
            continue
        successful_browse_outputs.append(item)
    browse_outputs = json.dumps(successful_browse_outputs, ensure_ascii=False)
    missing = [citation_id for citation_id in citation_ids if citation_id not in browse_outputs]
    if missing:
        raise ValueError(
            f"{session['seed_id']}: citations are not present in successful browse outputs: {missing}"
        )
    claim_support = decision.get("claim_support")
    strict_claim_support = session.get("prompt_version") == "medical_tool_calling_v2"
    if strict_claim_support or claim_support is not None:
        if not isinstance(claim_support, list) or not claim_support:
            raise ValueError(f"{session['seed_id']}: atomic claim_support is required")
        mapped_citations: set[str] = set()
        for index, audit in enumerate(claim_support, 1):
            if not isinstance(audit, dict) or not str(audit.get("claim") or "").strip():
                raise ValueError(f"{session['seed_id']}: claim_support {index} has no claim")
            ids = audit.get("citation_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError(
                    f"{session['seed_id']}: claim_support {index} has no citations"
                )
            if audit.get("supported") is not True:
                raise ValueError(
                    f"{session['seed_id']}: claim_support {index} is not supported"
                )
            invalid_ids = [citation_id for citation_id in ids if citation_id not in browse_outputs]
            if invalid_ids:
                raise ValueError(
                    f"{session['seed_id']}: claim_support {index} has invalid citations "
                    f"{invalid_ids}"
                )
            mapped_citations.update(ids)
        if set(citation_ids) != mapped_citations:
            raise ValueError(
                f"{session['seed_id']}: final-answer citations and claim_support citations differ"
            )
        checks = decision.get("quality_checks", {})
        if checks.get("claim_support_validated") is not True:
            raise ValueError(
                f"{session['seed_id']}: claim_support_validated must be true"
            )
        if checks.get("completed") is not True:
            raise ValueError(f"{session['seed_id']}: completed must be true")
        if checks.get("evidence_sufficient") is not True:
            raise ValueError(f"{session['seed_id']}: evidence_sufficient must be true")
    return final_answer, [item.strip() for item in rationales]


def finalize_batch(
    *,
    sessions_dir: Path,
    decisions_path: Path,
    output_path: Path,
    rejections_path: Path,
    batch_id: str,
    model: str = "codex_desktop_runtime",
    teacher_interface: str = "codex_desktop",
    teacher_model: str = "runtime_not_exposed",
    prompt_version: str = "medical_tool_calling_v1",
    tool_schema_version: str = "medical_tools_v1",
    expect_cases: int | None = None,
    session_pattern: str = "[0-9][0-9]_*.json",
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if rejections_path.exists():
        raise FileExistsError(f"Refusing to overwrite {rejections_path}")

    sessions = load_sessions(sessions_dir, session_pattern)
    decisions = load_decisions(decisions_path)
    if expect_cases is not None and len(sessions) != expect_cases:
        raise ValueError(f"Expected {expect_cases} sessions, found {len(sessions)}")
    missing_decisions = sorted(set(sessions) - set(decisions))
    extra_decisions = sorted(set(decisions) - set(sessions))
    if missing_decisions or extra_decisions:
        raise ValueError(
            f"Decision/session mismatch; missing={missing_decisions}, extra={extra_decisions}"
        )

    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for seed_id, (session_path, session) in sessions.items():
        decision = decisions[seed_id]
        if decision["decision"] == "rejected":
            reason = str(decision.get("reason") or "").strip()
            if not reason:
                raise ValueError(f"{seed_id}: rejected decision requires reason")
            rejected.append(
                {
                    "schema_version": DECISION_SCHEMA_VERSION,
                    "seed_id": seed_id,
                    "source_id": session.get("source_id"),
                    "question": session.get("question"),
                    "reason": reason,
                    "evidence_session": str(session_path.resolve()),
                    "quality_status": "rejected_after_evidence_audit",
                    "quality_checks": decision.get("quality_checks", {}),
                }
            )
            continue

        final_answer, rationales = _validate_accepted_decision(decision, session)
        trajectory = []
        rationale_index = 0
        session_events = session.get("events", [])
        for event_index, item in enumerate(session_events):
            if item.get("type") == "tool_call":
                previous = session_events[event_index - 1] if event_index else None
                if not previous or previous.get("type") != "assistant_message":
                    trajectory.append(
                        {
                            "type": "assistant_message",
                            "content": rationales[rationale_index],
                        }
                    )
                rationale_index += 1
            trajectory.append(item)
        trajectory.append({"type": "assistant_message", "content": final_answer})
        prepared.append(
            {
                "session": session,
                "session_path": session_path,
                "trajectory": trajectory,
                "final_answer": final_answer,
                "decision": decision,
                "tool_call_count": rationale_index,
            }
        )

    # Validation above completes before either output file is created.
    for item in prepared:
        session = item["session"]
        append_trajectory(
            output_path,
            question=session["question"],
            model=model,
            prompt_version=prompt_version,
            tool_schema_version=tool_schema_version,
            trajectory=item["trajectory"],
            final_answer=item["final_answer"],
            metadata={
                "purpose": "medical_teacher_batch",
                "batch_id": batch_id,
                "source_seed_id": session["seed_id"],
                "source_id": session.get("source_id"),
                "teacher_interface": teacher_interface,
                "teacher_model": teacher_model,
                "environment_mode": "live",
                "evidence_session": str(item["session_path"].resolve()),
                "tool_call_count": item["tool_call_count"],
                "deployment_policy": "medical_agent_v1_16k_6calls",
                "max_agent_tool_calls": V1_MAX_AGENT_TOOL_CALLS,
                "evidence_metrics": {
                    **session.get("metrics", {}),
                    "completed": item["decision"].get("quality_checks", {}).get(
                        "completed", True
                    ),
                    "evidence_sufficient": item["decision"].get(
                        "quality_checks", {}
                    ).get("evidence_sufficient", True),
                },
                "quality_checks": item["decision"].get("quality_checks", {}),
                "quality_status": "accepted_after_evidence_audit",
            },
        )
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejected),
        encoding="utf-8",
    )
    return {
        "batch_id": batch_id,
        "sessions": len(sessions),
        "accepted": len(prepared),
        "rejected": len(rejected),
        "output": str(output_path.resolve()),
        "rejections": str(rejections_path.resolve()),
    }
