"""No live model/API: exact no-progress replay and one-correction boundaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from trainable_agent.contracts import browse_context_signature, browse_no_progress_warning, validate_action
from trainable_agent.context import ContextBuilder, OfflineCodec
from trainable_agent.engine import Engine, initial_state
from trainable_agent.storage import Store


ACTION = dict(action="browse", requirement_ids=["R1"], source_id="PMID:1", reading_focus="Trial dates")


def fixture(status="source_unreadable", added=False):
    s = initial_state("fixture", "Find trial dates.", "process", "final", "tools")
    s["requirements"] = {"R1": "Find trial dates."}
    s["assessments"] = {"R1": dict(support="missing", evidence_ids=[], gap="Trial dates.", conflict="")}
    s["candidates"] = [dict(source_id="PMID:1", title="Study", preview="Abstract.", channel="paper", url="")]
    feedback = dict(status=status, retryable=False, new_evidence_ids=["E1"] if added else [],
                    browse_context_sha256=browse_context_signature(ACTION, s))
    s["history"] = [dict(turn=1, action=copy.deepcopy(ACTION), feedback=feedback)]
    return s


class GuardTests(unittest.TestCase):
    def test_advise_identical_unreadable_empty_or_no_new_ok(self):
        for status in ("source_unreadable", "empty", "ok"):
            with self.subTest(status=status):
                if status == "source_unreadable":
                    with self.assertRaisesRegex(ValueError, "nonretryable_source_failure"):
                        validate_action(ACTION, fixture(status))
                else:
                    self.assertEqual(validate_action(ACTION, fixture(status)), ACTION)
                self.assertIn("advisory", browse_no_progress_warning(ACTION, fixture(status)))

    def test_case_whitespace_and_other_requirement_do_not_reset(self):
        s = fixture(); s["requirements"]["R2"] = "Verify dates."
        a = dict(ACTION, reading_focus=" TRIAL   dates ", requirement_ids=["R2"])
        self.assertEqual(validate_action(a, s)["reading_focus"], "TRIAL   dates")
        self.assertIsNotNone(browse_no_progress_warning(a, s))

    def test_same_source_success_with_new_evidence_can_continue(self):
        self.assertEqual(validate_action(ACTION, fixture("ok", added=True)), ACTION)
        self.assertIsNone(browse_no_progress_warning(ACTION, fixture("ok", added=True)))

    def test_different_focus_or_changed_source_preview_can_continue(self):
        self.assertEqual(validate_action(dict(ACTION, reading_focus="Safety outcomes"), fixture())["action"], "browse")
        s = fixture(); s["candidates"][0]["preview"] = "New actual preview."
        self.assertEqual(validate_action(ACTION, s), ACTION)

    def test_new_same_source_evidence_can_change_reading_conditions(self):
        s = fixture(); s["evidence"]["E1"] = dict(source_id="PMID:1", chunk_id="c1", text="New original text.")
        self.assertEqual(validate_action(ACTION, s), ACTION)

    def test_other_source_evidence_search_or_state_does_not_reset(self):
        s = fixture(); s["evidence"]["E1"] = dict(source_id="PMID:2", chunk_id="c1", text="Other text.")
        s["assessments"]["R1"]["gap"] = "Different self-reported gap."
        s["history"].append(dict(action=dict(action="search", query="Different search"), feedback=dict(status="ok")))
        with self.assertRaisesRegex(ValueError, "nonretryable_source_failure"):
            validate_action(ACTION, s)
        self.assertIsNotNone(browse_no_progress_warning(ACTION, s))

    def test_retryable_environment_failure_preserves_one_retry_limit(self):
        s = fixture("environment_failure"); s["history"][0]["feedback"]["retryable"] = True
        self.assertEqual(validate_action(ACTION, s), ACTION)
        s["history"].append(copy.deepcopy(s["history"][0]))
        with self.assertRaisesRegex(ValueError, "temporary_failure_retry_limit"):
            validate_action(ACTION, s)

    def test_legacy_receipt_without_identity_is_not_rewritten_or_guessed(self):
        s = fixture(); del s["history"][0]["feedback"]["browse_context_sha256"]
        original = copy.deepcopy(s); self.assertEqual(validate_action(ACTION, s), ACTION)
        self.assertEqual(s, original)

    def test_q15_replay_shape_current_focus_failure_rejected_without_rewriting(self):
        s = fixture(); s["history"].insert(0, dict(action=copy.deepcopy(ACTION),
            feedback=dict(status="ok", new_evidence_ids=["E1", "E2"])))
        original = copy.deepcopy(s)
        with self.assertRaisesRegex(ValueError, "nonretryable_source_failure"):
            validate_action(ACTION, s)
        self.assertIsNotNone(browse_no_progress_warning(ACTION, s))
        self.assertEqual(s, original)


class CorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def run_correction(self, repeat):
        class Model:
            def __init__(self): self.requests = []
            async def generate(self, request):
                self.requests.append(copy.deepcopy(request))
                value = dict(ACTION, source_id="UNKNOWN") if len(self.requests) == 1 or repeat else dict(action="stop", reason="diminishing_returns")
                return dict(text=json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"trajectory.sqlite")
            try:
                model = Model(); s = fixture(); budget = copy.deepcopy(s["budget"])
                engine = Engine(store, model, model, None, ContextBuilder(OfflineCodec(), 30000), None)
                result = await engine.decision(dict(session=s)); after = result["session"]
                self.assertEqual(after["budget"], budget)
                self.assertEqual(len(model.requests), 2)
                self.assertIn("unknown_source_alias", model.requests[1]["messages"][-1]["content"])
                self.assertEqual(after["termination"], None if repeat else "policy_stop:diminishing_returns")
                self.assertEqual(after["next_node"], "decision" if repeat else "prefinal")
            finally: store.db.close()

    async def test_rejected_tool_action_can_be_corrected_to_honest_stop(self):
        await self.run_correction(False)

    async def test_two_source_rejections_return_to_decision_not_normal_stop(self):
        await self.run_correction(True)

    async def test_execute_persists_marker_and_specific_feedback(self):
        class Tools:
            async def execute(self, args): return dict(status="source_unreadable", error_code="no_evidence_bearing_chunks")
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"trajectory.sqlite")
            try:
                s = fixture(); s["history"] = []; s["action"] = copy.deepcopy(ACTION)
                engine = Engine(store, None, None, Tools(), ContextBuilder(OfflineCodec(), 30000), None)
                after = (await engine.execute(dict(session=s)))["session"]
                self.assertEqual(after["feedback"]["browse_context_sha256"], browse_context_signature(ACTION, s))
                self.assertIn("does not establish", after["feedback"]["message"])
                self.assertNotIn("content_status", after["feedback"])
                self.assertEqual(after["budget"]["charged"], s["budget"]["charged"]+1)
                with self.assertRaisesRegex(ValueError, "nonretryable_source_failure"):
                    validate_action(ACTION, after)
            finally: store.db.close()

    async def test_repeated_no_new_browse_runs_until_normal_budget(self):
        class Model:
            def __init__(self): self.calls = 0
            async def generate(self, request):
                self.calls += 1
                return dict(text=json.dumps(ACTION))
        class Tools:
            def __init__(self): self.calls = 0
            async def execute(self, args):
                self.calls += 1
                return dict(status="source_unreadable", error_code="no_evidence_bearing_chunks",
                    backend_return=dict(metadata=dict(novelty_status="no_new_passages"),
                        document_metadata=dict(alternate_fetch_method="pubmed_abstract")))
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"trajectory.sqlite")
            try:
                model, tools, s = Model(), Tools(), fixture()
                s["history"][0]["feedback"]["content_status"] = "no_new_passages"
                engine = Engine(store, model, model, tools, ContextBuilder(OfflineCodec(), 30000), None)
                for _ in range(s["budget"]["max_charged"]):
                    s = (await engine.decision(dict(session=s)))["session"]
                    self.assertEqual(s["next_node"], "execute")
                    s = (await engine.execute(dict(session=s)))["session"]
                    self.assertEqual(s["feedback"]["content_status"], "no_new_passages")
                    self.assertIn("not article full text", s["feedback"]["content_scope_message"])
                    self.assertIn("advisory", s["feedback"]["browse_repeat_warning"])
                    self.assertFalse(s["feedback"]["budget_exempt"])
                s = (await engine.decision(dict(session=s)))["session"]
                self.assertEqual(s["termination"], "runtime_budget_exhausted")
                self.assertEqual(s["next_node"], "prefinal")
                self.assertEqual(model.calls, 6)
                self.assertEqual(tools.calls, 6)
                self.assertEqual(s["budget"]["actual"], 6)
                self.assertEqual(s["budget"]["charged"], 6)
                self.assertEqual(s["budget"]["exemptions"], 0)
                self.assertFalse(s.get("blocked_sources"))
                self.assertFalse(s["evidence"])
            finally: store.db.close()
