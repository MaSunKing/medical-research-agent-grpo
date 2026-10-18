"""Synthetic offline callbacks: no live model, API or source text rewriting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from trainable_agent.engine import Engine, initial_state
from trainable_agent.contracts import validate_action
from trainable_agent.context import ContextBuilder, OfflineCodec
from trainable_agent.search_cache import visible_candidates
from trainable_agent.storage import Store
from test_agent import ScriptModel, ScriptTools, CHECK, SEARCH, BROWSE, UPDATE, STOP, OPENED, rank


FOUND8 = dict(status="ok", candidates=[dict(source_id=f"PMID:{i}", channel="paper",
    title=f"Source {i}", preview=f"Original abstract {i}", url=f"https://example.invalid/{i}") for i in range(1, 9)])
NO_NEW = dict(status="source_unreadable", error_code="no_evidence_bearing_chunks",
              backend_return=dict(metadata=dict(novelty_status="no_new_passages")))


class CacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "trajectory.sqlite")
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.store.db.close)

    def engine(self, tools, policy=None):
        return Engine(self.store, policy, ScriptModel([]), tools, ContextBuilder(OfflineCodec(), 100000), rank)

    def state(self):
        s = initial_state("cache", "Question", "p", "f", "t", process_only=True, max_charged=8)
        s["requirements"] = {"R1": "Effect", "R2": "Safety"}
        return s

    async def execute(self, engine, s, action):
        s = copy.deepcopy(s)
        s["action"] = copy.deepcopy(action)
        return (await engine.execute(dict(session=s)))["session"]

    async def test_eight_candidates_partial_last_exhausted_removed_other_source_browsable(self):
        second = dict(status="ok", chunks=[dict(source_id="PMID:2", chunk_id="second", text="Another observation.")])
        tools = ScriptTools([FOUND8, OPENED, NO_NEW, second])
        model = ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, SEARCH, BROWSE, SEARCH,
                             {**BROWSE, "source_id": "B"}, {"updates": []}, STOP])
        result = await self.engine(tools, model).run(self.state())
        # First repeat: original A..H, but B..H unread before partially read A.
        after_partial = json.loads(model.requests[5]["messages"][1]["content"])
        self.assertEqual([c["source_id"] for c in after_partial["candidates"]], list("BCDEFGHA"))
        self.assertEqual(after_partial["candidates"][-1]["read_status"], "partially_read")
        # Explicit no-new scope: only B..H on the next identical cache lookup.
        after_exhaustion = json.loads(model.requests[7]["messages"][1]["content"])
        self.assertEqual([c["source_id"] for c in after_exhaustion["candidates"]], list("BCDEFGH"))
        self.assertTrue(after_exhaustion["feedback"]["cache_hit"])
        self.assertFalse(after_exhaustion["feedback"]["network_executed"])
        self.assertEqual(len(tools.calls), 4)  # one Search, three Browse, not three Search
        self.assertEqual(result["budget"]["charged"], 6)
        self.assertEqual(result["budget"]["actual"], 6)  # actions include two local cache hits
        self.assertEqual(result["decision_budget"]["rejected"], 0)
        self.assertEqual(result["evidence"]["E1"]["text"], OPENED["chunks"][0]["text"])
        self.assertEqual(result["evidence"]["E2"]["source_id"], "PMID:2")
        self.assertEqual(len(result["candidate_history"]), 8)
        self.assertEqual(result["run_status"], "process_complete")

    async def test_repeat_only_naturally_exhausts_normal_budget_not_rejection_budget(self):
        model = ScriptModel([CHECK] + [SEARCH] * 6)
        tools = ScriptTools([FOUND8])
        s = self.state(); s["budget"].update(max_charged=6, max_actual=11)
        result = await self.engine(tools, model).run(s)
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(len(self.store.events("cache", "search_cache_hit")), 5)
        self.assertEqual(result["termination"], "runtime_budget_exhausted")
        self.assertEqual(result["budget"]["charged"], 6)
        self.assertEqual(result["decision_budget"]["rejected"], 0)
        self.assertEqual(result["evidence"], {})

    async def test_search_r1_browse_r2_exhaustion_filters_current_candidates(self):
        tools = ScriptTools([FOUND8, NO_NEW])
        engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, {**BROWSE, "source_id": "PMID:1", "requirement_ids": ["R2"]})
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        self.assertEqual(len(s["candidate_history"]), 8)
        # The independent R1 search scope is not permanently banned.
        s = await self.execute(engine, s, SEARCH)
        self.assertIn("PMID:1", [c["source_id"] for c in s["candidates"]])

    async def test_duplicate_state_rejected_then_corrected_without_silent_loss(self):
        bad = '{"updates":[{"requirement_id":"R1","requirement_id":"R2","support":"partial","evidence_ids":["E1"],"gap":"Unresolved","conflict":""}]}'
        model = ScriptModel([bad, {"updates": []}])
        engine = self.engine(ScriptTools([]), model)
        s = self.state()
        result = (await engine.state({"session": s}))["session"]
        events = self.store.events("cache", "generation")
        self.assertEqual([e["data"]["accepted"] for e in events], [False, True])
        self.assertIn("duplicate_json_key:requirement_id", events[0]["data"]["error"])
        self.assertEqual(result["assessments"], s["assessments"])
        self.assertEqual(result["next_node"], "decision")

    async def test_empty_result_is_cached_and_clears_previous_query_candidates(self):
        tools = ScriptTools([FOUND8, {"status": "empty"}]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        other = {**SEARCH, "query": "other empty query"}
        s = await self.execute(engine, s, other)
        s = await self.execute(engine, s, other)
        self.assertEqual(s["candidates"], [])
        self.assertEqual(len(tools.calls), 2)
        self.assertTrue(s["feedback"]["cache_hit"])

    async def test_cache_does_not_mix_later_queries_or_rerank(self):
        newer = dict(status="ok", candidates=[dict(FOUND8["candidates"][0], source_id="PMID:99")])
        tools = ScriptTools([FOUND8, newer]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, {**SEARCH, "query": "other query"})
        s = await self.execute(engine, s, SEARCH)
        self.assertEqual([c["source_id"] for c in s["candidates"]], [f"PMID:{i}" for i in range(1, 9)])
        self.assertIn("PMID:99", s["candidate_history"])

    async def test_roundtrip_and_old_checkpoint_restore_original_receipt(self):
        tools = ScriptTools([FOUND8]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        for legacy in (False, True):
            restored = json.loads(json.dumps(s))
            if legacy: restored.pop("search_cache")
            restored = await self.execute(engine, restored, SEARCH)
            self.assertEqual(len(restored["candidates"]), 8)
            self.assertTrue(restored["feedback"]["cache_hit"])
        self.assertEqual(len(tools.calls), 1)

    async def test_missing_legacy_receipt_never_fabricates_or_requeries(self):
        tools = ScriptTools([]); engine = self.engine(tools); s = self.state()
        s["history"] = [dict(action=SEARCH, feedback=dict(status="ok"))]
        with self.assertRaisesRegex(ValueError, "prior_search_cache_receipt_missing"):
            await self.execute(engine, s, SEARCH)
        self.assertEqual(tools.calls, [])

    async def test_timeout_preserved_but_content_failure_filtered_without_claiming_exhaustion(self):
        for receipt in (dict(status="environment_failure", retryable=True),
                        dict(status="source_unreadable", error_code="no_evidence_bearing_chunks")):
            s = self.state(); s["run_id"] = receipt["status"]
            tools = ScriptTools([FOUND8, receipt]); engine = self.engine(tools)
            s = await self.execute(engine, s, SEARCH)
            s = await self.execute(engine, s, BROWSE)
            s = await self.execute(engine, s, SEARCH)
            self.assertEqual("PMID:1" in [c["source_id"] for c in s["candidates"]],
                             receipt["status"] == "environment_failure")
            self.assertNotEqual(s["history"][1]["feedback"].get("content_status"), "no_new_passages")

    async def test_exhaustion_scope_does_not_hide_new_query_requirement_preview_or_windows(self):
        tools = ScriptTools([FOUND8, OPENED, NO_NEW]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, BROWSE)
        s = await self.execute(engine, s, BROWSE)
        original = FOUND8["candidates"]
        candidates, _ = visible_candidates(s, original, SEARCH)
        self.assertNotIn("PMID:1", [c["source_id"] for c in candidates])
        for action in ({**SEARCH, "query": "different useful query"}, {**SEARCH, "requirement_ids": ["R2"]}):
            candidates, _ = visible_candidates(s, original, action)
            self.assertIn("PMID:1", [c["source_id"] for c in candidates])
        changed = copy.deepcopy(original); changed[0]["preview"] += " Updated route"
        self.assertEqual(len(visible_candidates(s, changed, SEARCH)[0]), 8)
        altered = copy.deepcopy(s); altered["evidence"]["E2"] = dict(OPENED["chunks"][0], text="New window.")
        self.assertEqual(len(visible_candidates(altered, original, SEARCH)[0]), 8)

    async def test_blocked_removed_without_recycling_alias_or_evidence(self):
        tools = ScriptTools([FOUND8]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s["blocked_sources"]["PMID:1"] = "confirmed_source_access_barrier"
        s = await self.execute(engine, s, SEARCH)
        self.assertEqual([c["source_id"] for c in s["candidates"]], [f"PMID:{i}" for i in range(2, 9)])
        self.assertEqual(s["source_aliases"]["PMID:1"], "A")
        self.assertEqual(s["source_aliases"]["PMID:2"], "B")

    async def test_case_whitespace_requirement_changes_hit_cache_but_channel_change_fetches(self):
        tools = ScriptTools([FOUND8, dict(status="empty")]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, {**SEARCH, "query": "  TREATMENT\t effectiveness ", "requirement_ids": ["R2"]})
        self.assertTrue(s["feedback"]["cache_hit"])
        self.assertEqual(len(tools.calls), 1)
        s = await self.execute(engine, s, {**SEARCH, "channel": "web"})
        self.assertNotIn("cache_hit", s["feedback"])
        self.assertEqual(len(tools.calls), 2)

    async def test_changed_tool_identity_never_uses_old_receipt(self):
        tools = ScriptTools([FOUND8]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s["identities"]["tools"] = "different frozen tool version"
        with self.assertRaisesRegex(ValueError, "search_cache_tool_identity_mismatch"):
            await self.execute(engine, s, SEARCH)
        self.assertEqual(len(tools.calls), 1)

    async def test_unreadable_current_focus_removed_and_cache_does_not_restore(self):
        failed = dict(status="source_unreadable", retryable=False, error_code="no_evidence_bearing_chunks")
        tools = ScriptTools([FOUND8, failed]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, BROWSE)
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        self.assertTrue(s["feedback"]["removed_candidates"])
        self.assertIn("PMID:1", s["candidate_history"])
        s = await self.execute(engine, s, SEARCH)
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        self.assertEqual(len(tools.calls), 2)
        changed, _ = visible_candidates(s, FOUND8["candidates"], dict(SEARCH, query="different focused search"))
        self.assertIn("PMID:1", [c["source_id"] for c in changed])

    async def test_temporary_failure_one_retry_then_removed(self):
        failed = dict(status="environment_failure", retryable=True, error_code="retrieval_transport_failure")
        tools = ScriptTools([FOUND8, failed, failed]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, BROWSE)
        self.assertIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        validate_action(BROWSE, s)
        s = await self.execute(engine, s, BROWSE)
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        s = await self.execute(engine, s, SEARCH)
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])

    async def test_temporary_failure_success_preserves_remaining_passages(self):
        failed = dict(status="environment_failure", retryable=True)
        tools = ScriptTools([FOUND8, failed, OPENED]); engine = self.engine(tools)
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, BROWSE)
        s = await self.execute(engine, s, BROWSE)
        validate_action(BROWSE, s)
        self.assertIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        self.assertEqual(len(s["evidence"]), 1)

    async def test_nonretryable_environment_and_unknown_timeout_removed(self):
        for index, failure in enumerate((dict(status="environment_failure", retryable=False),
                dict(status="environment_failure", retryable=True, error_code="retrieval_operation_timeout"))):
            s = self.state(); s["run_id"] = "terminal" + str(index)
            engine = self.engine(ScriptTools([FOUND8, failure]))
            s = await self.execute(engine, s, SEARCH)
            s = await self.execute(engine, s, BROWSE)
            self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])

    async def test_verified_pdf_same_task_resume_stays_available_once(self):
        failure = dict(status="environment_failure", retryable=True, error_code="retrieval_operation_timeout",
                       execution=dict(outcome="unknown", phase="worker_operation", receipt_id="a"*64,
                                      resume_mode="resume_same_task"))
        engine = self.engine(ScriptTools([FOUND8, failure, failure]))
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, BROWSE)
        validate_action(BROWSE, s)
        s = await self.execute(engine, s, BROWSE)
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])

    async def test_model_sees_filtered_cache_then_chooses_another_source(self):
        failed = dict(status="source_unreadable", retryable=False, error_code="no_evidence_bearing_chunks")
        opened = dict(status="ok", chunks=[dict(source_id="PMID:2", chunk_id="new", text="Actual scripted fixture text.")])
        tools = ScriptTools([FOUND8, failed, opened])
        model = ScriptModel([CHECK, SEARCH, BROWSE, SEARCH,
                             dict(BROWSE, source_id="B"), UPDATE, STOP])
        result = await self.engine(tools, model).run(self.state())
        visible = json.loads(model.requests[4]["messages"][1]["content"])
        self.assertNotIn("A", [c["source_id"] for c in visible["candidates"]])
        self.assertIn("B", [c["source_id"] for c in visible["candidates"]])
        self.assertTrue(visible["feedback"]["cache_hit"])
        self.assertEqual(len(tools.calls), 3)
        self.assertEqual(result["decision_budget"]["rejected"], 0)
        self.assertEqual(result["evidence"]["E1"]["source_id"], "PMID:2")
        self.assertEqual(result["run_status"], "process_complete")

    async def test_changed_preview_reopens_failed_scope_and_checkpoint_keeps_filter(self):
        failed = dict(status="source_unreadable", retryable=False, error_code="no_evidence_bearing_chunks")
        engine = self.engine(ScriptTools([FOUND8, failed]))
        s = await self.execute(engine, self.state(), SEARCH)
        s = await self.execute(engine, s, BROWSE)
        s = json.loads(json.dumps(s))
        s = await self.execute(engine, s, SEARCH)
        self.assertNotIn("PMID:1", [c["source_id"] for c in s["candidates"]])
        changed = copy.deepcopy(FOUND8["candidates"])
        changed[0]["preview"] += " Real changed available content"
        shown, _ = visible_candidates(s, changed, SEARCH)
        self.assertIn("PMID:1", [c["source_id"] for c in shown])
        s["candidates"] = shown
        validate_action(BROWSE, s)


if __name__ == "__main__":
    unittest.main()
