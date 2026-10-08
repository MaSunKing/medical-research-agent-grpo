# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Shared Medical Research Agent runtime controller.

This is the single policy surface used by local deployment, evaluation, and
future rollout workers.  It never chooses a tool for the model; it only blocks
wasteful actions, normalizes observations, tracks evidence/budget state, and
validates a proposed final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .anchored_query import anchored_execution_arguments
from .runtime_policy import (
    MedicalAnswerValidation,
    duplicate_action_reason,
    evidence_ledger,
    normalized_tool_output,
    rank_pubmed_candidates,
    rank_web_candidates,
    public_runtime_state,
    tool_call_arguments,
    unopened_candidates_for_tool,
    validate_medical_answer,
)


@dataclass
class MedicalRuntimeController:
    question: str
    route_class: str | None = None
    max_tool_calls: int = 6
    claim_support_validator: Callable[[str, str, dict[str, str]], bool | str] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    duplicate_blocks: int = 0
    finalizer_rejections: int = 0
    evidence_delivery_receipts: list[dict[str, Any]] = field(default_factory=list)

    def arguments_from_parsed_call(
        self,
        tool: str,
        content: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return tool_call_arguments(tool, content, parameters)

    def execution_arguments(
        self, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach the private original-question anchor only at execution time."""

        return anchored_execution_arguments(tool, arguments, self.question)

    @property
    def calls_used(self) -> int:
        return sum(event.get("type") == "tool_call" for event in self.events)

    def prepare_action(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Record a visible action and return a structured block reason, if any."""
        reason = duplicate_action_reason(tool, arguments, self.events)
        call_id = f"local:{self.calls_used + 1}"
        self.events.append({
            "type": "tool_call",
            "tool": tool,
            "arguments": dict(arguments),
            "call_id": call_id,
        })
        if not reason:
            return None
        self.duplicate_blocks += 1
        return {
            "error": reason,
            "failed": True,
            "duplicate_blocked": True,
            "recommended_unopened_candidates": unopened_candidates_for_tool(
                tool, self.events
            ),
        }

    def record_output(self, tool: str, output: Any) -> dict[str, Any]:
        """Normalize, consistently rank, record, and annotate a tool observation."""
        raw = normalized_tool_output(output)
        data = raw.get("data")
        if isinstance(data, list):
            raw = dict(raw)
            if tool == "pubmed_search":
                query = str((self.events[-1].get("arguments") or {}).get("query") or "")
                # The MCP PubMed backend already performs deterministic
                # evidence-value ranking followed by MedCPT rank fusion.  Do
                # not erase that fused order in the shared controller.  Older
                # or test backends without backend_queries still receive the
                # deterministic compatibility ranking here.
                if "backend_queries" not in raw:
                    raw["data"] = rank_pubmed_candidates(query, data)
            elif tool == "medical_web_search":
                query = str((self.events[-1].get("arguments") or {}).get("query") or "")
                if "routed_queries" not in raw:
                    raw["data"] = rank_web_candidates(query, data)
        if (
            tool == "browse_webpage"
            and (
                raw.get("failed")
                or raw.get("error")
                or not (raw.get("data") or [])
            )
        ):
            # Do not hard-route to a replacement. Expose a small list of
            # already-discovered alternatives so the Agent can choose one.
            raw = dict(raw)
            raw["recommended_unopened_candidates"] = unopened_candidates_for_tool(
                tool,
                [
                    *self.events,
                    {"type": "tool_output", "tool": tool, "output": raw},
                ],
            )
            if raw.get("retryable") is True:
                raw["message"] = (
                    "This webpage had a transient read failure. At most one retry is "
                    "allowed with a different focused query; otherwise choose a different "
                    "unopened candidate."
                )
            else:
                raw["message"] = (
                    "This webpage could not be read. Do not retry this source in the "
                    "current trajectory; choose a different unopened candidate."
                )
        call_id = str(self.events[-1].get("call_id") or "") if self.events else ""
        self.events.append({
            "type": "tool_output",
            "tool": tool,
            "output": raw,
            "call_id": call_id,
            "observation_delivered": False,
        })
        raw["runtime_state"] = self.public_state()
        return raw

    def record_blocked_output(self, tool: str, output: dict[str, Any]) -> dict[str, Any]:
        raw = dict(output)
        self.events.append({"type": "tool_output", "tool": tool, "output": raw})
        raw["runtime_state"] = self.public_state()
        return raw

    def confirm_last_output_delivery(self, tool: str) -> dict[str, Any] | None:
        """Confirm that one trusted observation was appended atomically.

        Only successful Browse results create citation eligibility. Search
        candidates, blocked calls, and model-authored markup never enter this
        Runtime-owned ledger.
        """

        event = next(
            (
                value
                for value in reversed(self.events)
                if value.get("type") == "tool_output" and value.get("tool") == tool
            ),
            None,
        )
        if event is None or event.get("observation_delivered") is True:
            return None
        event["observation_delivered"] = True
        if tool not in {"browse_document", "browse_webpage"}:
            return None
        output = event.get("output") or {}
        if output.get("failed") or output.get("error"):
            return None
        ids = [
            str(item.get("source_id") or "").strip()
            for item in output.get("data") or []
            if isinstance(item, dict)
            and str(item.get("source_id") or "").strip()
            and str(item.get("text") or "").strip()
        ]
        if not ids:
            return None
        receipt = {
            "schema_version": "medgap_evidence_delivery_v1",
            "call_id": str(event.get("call_id") or ""),
            "tool_name": tool,
            "observation_delivered": True,
            "citation_eligible_ids": list(dict.fromkeys(ids)),
        }
        self.evidence_delivery_receipts.append(receipt)
        return receipt

    def public_state(self) -> dict[str, Any]:
        """State safe to append to policy observations."""
        return public_runtime_state(self.events, self.max_tool_calls)

    def reward_ledger(self) -> dict[str, Any]:
        """Hidden audit/reward state; never append this to the policy prompt."""
        ledger = evidence_ledger(
            self.question,
            self.events,
            self.route_class,
            self.max_tool_calls,
        )
        ledger["evidence_delivery"] = {
            "receipts": list(self.evidence_delivery_receipts),
            "registered_citation_eligible_ids": sorted({
                evidence_id
                for receipt in self.evidence_delivery_receipts
                for evidence_id in receipt["citation_eligible_ids"]
            }),
        }
        return ledger

    def ledger(self) -> dict[str, Any]:
        """Backward-compatible alias for the public, non-prescriptive state."""
        return self.public_state()

    def validate_answer(self, answer: str) -> MedicalAnswerValidation:
        verdict = validate_medical_answer(
            self.question,
            answer,
            self.events,
            self.route_class,
            require_opened_evidence=self.calls_used > 0,
            claim_support_validator=self.claim_support_validator,
        )
        if not verdict.accepted:
            self.finalizer_rejections += 1
        return verdict
