import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

from trainable_agent.context import ContextBuilder, OfflineCodec, ContextOverflow
from trainable_agent.contracts import validate_action, validate_delta
from trainable_agent.engine import Engine, initial_state
from trainable_agent.storage import Store, digest
from trainable_agent.tools import create_server, MCPTools, normalize
from trainable_agent.training import sft_samples, group_advantages, require_grpo_capture, clipped_surrogate


def js(value):
    return {"text": json.dumps(value), "grpo_capture_ready": False}


class ScriptModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    async def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        item = self.outputs.pop(0)
        return {"text": item} if isinstance(item, str) else js(item)


class ScriptTools:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def execute(self, args):
        self.calls.append(copy.deepcopy(args))
        return self.outputs.pop(0)


async def rank(question, rows):
    return [float(len(rows)-i) for i in range(len(rows))]


def candidate(sid="PMID:1", preview="trial abstract"):
    return dict(source_id=sid, channel="paper", title="Study", preview=preview, url="")


CHECK = {"requirements": ["Assess effectiveness."]}
SEARCH = dict(action="search", requirement_ids=["R1"], channel="paper", query="treatment effectiveness")
BROWSE = dict(action="browse", requirement_ids=["R1"], source_id="PMID:1", reading_focus="effectiveness")
STOP = dict(action="stop", reason="sufficient")
UPDATE = dict(updates=[dict(requirement_id="R1", support="partial", evidence_ids=["E1"],
                            gap="Long-term outcomes not available.", conflict="")])
FOUND = dict(status="ok", candidates=[candidate()])
OPENED = dict(status="ok", chunks=[dict(source_id="PMID:1", chunk_id="PMID:1:c1", text="Observed outcome.", title="Study")])


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name)/"records.sqlite")
        self.ctx = ContextBuilder(OfflineCodec(), 100000)
        self.state = initial_state("q1", "Does treatment work?", "p1", "f1", "t1")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def engine(self, p, tools, f=None):
        return Engine(self.store, p, f or ScriptModel(["<answer>Evidence remains incomplete.</answer>"]), tools, self.ctx, rank)

    async def test_full_graph_and_exact_prefinal(self):
        p = ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, STOP])
        tools = ScriptTools([FOUND, OPENED])
        result = await self.engine(p, tools, ScriptModel(["<answer>Supported outcome [E1].</answer>"])).run(self.state)
        self.assertEqual(len(tools.calls), 2)
        self.assertEqual([x["stage"] for x in p.requests], ["checklist", "decision", "decision", "state", "decision"])
        self.assertEqual(result["prefinal"]["evidence_snapshot_sha256"], digest(result["evidence"]))
        self.assertIn("R1", result["prefinal"]["unresolved"])
        self.assertEqual(result["final"], "<answer>Supported outcome [E1].</answer>")

    async def test_process_only_never_calls_final(self):
        self.state["process_only"] = True
        f = ScriptModel([])
        result = await self.engine(ScriptModel([CHECK, STOP]), ScriptTools([]), f).run(self.state)
        self.assertEqual(f.requests, [])
        self.assertIsNone(result["final"])

    async def test_browse_only_garbage_skips_state_and_archives_original(self):
        self.state["process_only"] = True
        raw = dict(status="ok", chunks=[dict(source_id="PMID:1", chunk_id="PMID:1:c1",
                   text="Accept all cookies\nHome | About us | Contact\n", title="Original title")])
        policy = ScriptModel([CHECK, SEARCH, BROWSE, STOP])
        result = await self.engine(policy, ScriptTools([FOUND, raw])).run(self.state)
        self.assertEqual(result["evidence"], {})
        self.assertEqual(result["assessments"]["R1"]["support"], "missing")
        self.assertNotIn("state", [r["stage"] for r in policy.requests])
        self.assertEqual(result["budget"]["actual"], 2)
        self.assertEqual(result["budget"]["charged"], 2)
        self.assertEqual(self.store.events("q1", "tool_raw")[-1]["data"]["observation"], raw)
        self.assertIn("no_usable_body_after_rule_cleanup", policy.requests[-1]["messages"][-1]["content"])

    async def test_browse_cleanup_enters_state_with_provenance(self):
        self.state["process_only"] = True
        raw = dict(status="ok", chunks=[dict(source_id="PMID:1", chunk_id="PMID:1:c1",
                   text="Accept all cookies\nNo long-term benefit was established.\n", title="Trial")])
        result = await self.engine(ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, STOP]),
                                   ScriptTools([FOUND, raw])).run(self.state)
        e = result["evidence"]["E1"]
        self.assertEqual(e["text"], "No long-term benefit was established.\n")
        self.assertEqual(e["chunk_id"], "PMID:1:c1")
        self.assertTrue(e["cleaning"]["removed_spans"])
        self.assertEqual(result["prefinal"]["opened_evidence"]["E1"], e)

    async def test_invalid_action_corrected_once_no_execution(self):
        p = ScriptModel([CHECK, {**BROWSE, "source_id": "PMID:999"}, STOP])
        tools = ScriptTools([])
        await self.engine(p, tools).run(self.state)
        rows = self.store.events("q1", "generation")
        self.assertFalse(rows[1]["data"]["accepted"])
        self.assertEqual(tools.calls, [])
        self.assertIn("source_not_in_visible", p.requests[2]["messages"][-1]["content"])

    async def test_two_invalid_actions_stop_without_final(self):
        f = ScriptModel([])
        result = await self.engine(ScriptModel([CHECK, {}, {}]), ScriptTools([]), f).run(self.state)
        self.assertEqual(result["termination"], "model_contract_failure")
        self.assertEqual(f.requests, [])

    async def test_failures_no_state_call_and_five_exemptions(self):
        outputs = [dict(status="environment_failure", retryable=True)]*6
        tools = ScriptTools(outputs)
        result = await self.engine(ScriptModel([CHECK]+[{**SEARCH, "query": "failure scope " + str(i)} for i in range(6)]+[STOP]), tools).run(self.state)
        self.assertEqual(result["budget"]["actual"], 6)
        self.assertEqual(result["budget"]["exemptions"], 5)
        self.assertEqual(result["budget"]["charged"], 1)
        self.assertEqual(result["evidence"], {})

    async def test_six_charged_calls_force_stop(self):
        searches = [{**SEARCH, "query": "effectiveness scope " + str(i)} for i in range(6)]
        result = await self.engine(ScriptModel([CHECK]+searches), ScriptTools([dict(status="empty")]*6)).run(self.state)
        self.assertEqual(result["termination"], "runtime_budget_exhausted")
        self.assertEqual(result["budget"]["actual"], 6)

    async def test_repeat_search_cache_then_browse_is_recorded_without_second_network_search(self):
        self.state["process_only"] = True
        policy = ScriptModel([CHECK, SEARCH, SEARCH, BROWSE, UPDATE, STOP])
        tools = ScriptTools([FOUND, OPENED])
        result = await self.engine(policy, tools).run(self.state)
        self.assertEqual(len(tools.calls), 2)
        self.assertEqual(len(result["evidence"]), 1)
        records = self.store.events("q1", "generation")
        rejected = [r for r in records if not r["data"]["accepted"]]
        self.assertEqual(len(rejected), 0)
        repaired = [r for r in records if r["data"]["attempt"] == 1]
        self.assertEqual(repaired, [])
        self.assertEqual(len(self.store.events("q1", "search_cache_hit")), 1)

    async def test_eleven_actual_calls_force_stop(self):
        result = await self.engine(ScriptModel([CHECK]+[{**SEARCH, "query": "failure scope " + str(i)} for i in range(11)]), ScriptTools([dict(status="environment_failure")]*11)).run(self.state)
        self.assertEqual(result["budget"]["actual"], 11)
        self.assertEqual(result["budget"]["charged"], 6)

    async def test_configurable_three_exemption_baseline(self):
        self.state["budget"].update(max_exemptions=3, max_actual=9)
        result = await self.engine(ScriptModel([CHECK]+[{**SEARCH, "query": "failure scope " + str(i)} for i in range(9)]), ScriptTools([dict(status="environment_failure")]*9)).run(self.state)
        self.assertEqual(result["budget"]["exemptions"], 3)
        self.assertEqual(result["budget"]["actual"], 9)

    async def test_access_barrier_shared_pool_and_source_blocklist(self):
        blocked = dict(status="ok", chunks=[dict(source_id="PMID:1", chunk_id="PMID:1:c1",
                      title="Source", text="Verify you are human\nSign in to continue\n")])
        p = ScriptModel([CHECK]+[{**SEARCH, "query": "failure scope " + str(i)} for i in range(4)]+[BROWSE, BROWSE, STOP])
        tools = ScriptTools([dict(status="environment_failure")]*3+[FOUND, blocked])
        result = await self.engine(p, tools).run(self.state)
        self.assertEqual(len(tools.calls), 5)
        self.assertEqual(result["budget"]["exemptions"], 4)
        self.assertEqual(result["budget"]["charged"], 1)
        self.assertIn("PMID:1", result["blocked_sources"])
        self.assertIn("source_access_blocked_choose_another", p.requests[-1]["messages"][-1]["content"])
        self.assertEqual(result["evidence"], {})

    def test_bad_budget_configuration_rejected(self):
        with self.assertRaises(ValueError):
            initial_state("r", "question", "p", "f", "t", max_exemptions=-1)

    async def test_repeated_browse_no_second_state(self):
        p = ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, BROWSE, STOP])
        result = await self.engine(p, ScriptTools([FOUND, OPENED, OPENED])).run(self.state)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual([r["stage"] for r in p.requests].count("state"), 1)

    async def test_empty_delta_allowed(self):
        p = ScriptModel([CHECK, SEARCH, BROWSE, {"updates": []}, STOP])
        result = await self.engine(p, ScriptTools([FOUND, OPENED])).run(self.state)
        self.assertEqual(result["assessments"]["R1"]["support"], "missing")

    async def test_resume_finished_no_model_or_tool_calls(self):
        await self.engine(ScriptModel([CHECK, STOP]), ScriptTools([])).run(self.state)
        p, tools, f = ScriptModel([]), ScriptTools([]), ScriptModel([])
        await self.engine(p, tools, f).run(self.state)
        self.assertEqual(p.requests+f.requests+tools.calls, [])

    async def test_resume_at_tool_boundary_no_repeat(self):
        engine = self.engine(ScriptModel([CHECK, SEARCH]), ScriptTools([FOUND]))
        s = (await engine.checklist({"session": self.state}))["session"]
        s = (await engine.decision({"session": s}))["session"]
        # Crash after tool commit, before checkpoint: repeated node reuses receipt.
        await engine.execute({"session": s})
        self.store.checkpoint(s)
        p, tools = ScriptModel([STOP]), ScriptTools([])
        await self.engine(p, tools).run(self.state)
        self.assertEqual(tools.calls, [])

    async def test_resume_identity_mismatch(self):
        self.store.checkpoint(self.state)
        changed = copy.deepcopy(self.state)
        changed["identities"]["process"] = "other"
        with self.assertRaisesRegex(ValueError, "identity_mismatch"):
            await self.engine(ScriptModel([]), ScriptTools([])).run(changed)

    async def test_pending_receipt_not_retried(self):
        async def broken():
            raise RuntimeError("process interruption")
        with self.assertRaises(RuntimeError):
            await self.store.once("pending", broken)
        with self.assertRaisesRegex(RuntimeError, "manual_reconciliation"):
            await self.store.once("pending", broken)

    async def test_candidate_latest_then_history(self):
        engine = self.engine(ScriptModel([]), ScriptTools([]))
        await engine.update_candidates(self.state, [candidate(f"PMID:{i}") for i in range(1, 11)])
        self.assertEqual(len(self.state["candidates"]), 8)
        await engine.update_candidates(self.state, [candidate(f"PMID:{i}") for i in range(11, 21)])
        self.assertEqual([c["source_id"] for c in self.state["candidates"][:4]], [f"PMID:{i}" for i in range(11,15)])
        self.assertEqual([c["source_id"] for c in self.state["candidates"][4:]], [f"PMID:{i}" for i in range(1,5)])

    async def test_candidates_empty_previews_excluded(self):
        await self.engine(ScriptModel([]), ScriptTools([])).update_candidates(self.state, [candidate(preview="")])
        self.assertEqual(self.state["candidates"], [])

    async def test_mcp_in_memory_real_protocol(self):
        from fastmcp import Client
        class Bridge:
            async def request(self, op, args):
                if op == "search":
                    return {"data": [dict(source_id="PMID:1", title="Study", abstract="Summary")]}
                if op == "rank_history":
                    return [0.5]*len(args["texts"])
                return {"data": [dict(source_id="PMID:1:c1", text="Original passage")]}
        async with Client(create_server(Bridge())) as client:
            tools = MCPTools(client)
            found = await tools.execute({"question": "question", "action": SEARCH})
            opened = await tools.execute({"question": "question", "action": BROWSE, "channel": "paper"})
            scores = await tools.rank("question", found["candidates"])
        self.assertEqual(found["candidates"][0]["source_id"], "PMID:1")
        self.assertEqual(opened["chunks"][0]["chunk_id"], "PMID:1:c1")
        self.assertEqual(scores, [0.5])

    async def test_citation_unknown_repair(self):
        f = ScriptModel(["<answer>Claim [E999].</answer>", "<answer>Claim [E1].</answer>"])
        result = await self.engine(ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, STOP]), ScriptTools([FOUND, OPENED]), f).run(self.state)
        self.assertEqual(len(f.requests), 2)
        self.assertIn("[E1]", result["final"])

    def test_previews_cannot_support_state(self):
        self.state["requirements"] = {"R1": "effectiveness"}
        with self.assertRaisesRegex(ValueError, "evidence_not_opened"):
            validate_delta(UPDATE, self.state)

    def test_support_requires_evidence(self):
        self.state["requirements"] = {"R1": "effectiveness"}
        row = copy.deepcopy(UPDATE)
        row["updates"][0]["evidence_ids"] = []
        with self.assertRaisesRegex(ValueError, "support_requires"):
            validate_delta(row, self.state)

    def test_unknown_fields_forbidden(self):
        self.state["requirements"] = {"R1": "effectiveness"}
        with self.assertRaises(ValueError):
            validate_action({**SEARCH, "remaining": 99}, self.state)

    def test_source_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "source_mismatch"):
            normalize({"data": [dict(source_id="PMID:2:c1", text="text")]}, "paper", "PMID:1")

    def test_context_overflow_fail_closed(self):
        ctx = ContextBuilder(OfflineCodec(), 10)
        with self.assertRaises(ContextOverflow):
            ctx.build("checklist", self.state)

    def test_offline_tokens_not_trainable(self):
        with self.assertRaisesRegex(ValueError, "exact_deployed"):
            list(sft_samples(self.store, "q1", OfflineCodec()))

    async def test_sft_mask_and_target_boundary(self):
        class ExactFixture(OfflineCodec):
            # Synthetic fixture, not a claim of Qwen tokenizer validation.
            exact = True
        codec = ExactFixture()
        self.ctx = ContextBuilder(codec, 100000)
        await self.engine(ScriptModel([CHECK, STOP]), ScriptTools([])).run(self.state)
        rows = list(sft_samples(self.store, "q1", codec))
        self.assertEqual([r["stage"] for r in rows], ["checklist", "decision"])
        for row in rows:
            n = len(codec.encode(row["messages"]))
            self.assertEqual(row["labels"][:n], [-100]*n)
            self.assertEqual(row["labels"][n:], codec.target(row["target"]))

    async def test_compressed_request_archive_and_sft_reuse(self):
        from test_context import fixture
        class ExactFixture(OfflineCodec):
            exact = True  # Synthetic fixture only.
        codec = ExactFixture()
        state = fixture()
        # OfflineCodec counts bytes, not model tokens; allow the expanded prompt
        # while retaining insufficient room for the long raw evidence fixture.
        context = ContextBuilder(codec, 6000)
        compressor = ScriptModel([{"passage_ids": ["S0", "S1"]}])
        compressor.identity = "frozen-fixture-selector"
        engine = Engine(self.store, ScriptModel([{"updates":[]}]), ScriptModel([]), ScriptTools([]), context, rank,
                        compressor=compressor, compression_context_tokens=100000)
        result = await engine.generate("state", state, lambda text: validate_delta(json.loads(text), state))
        self.assertEqual(result, {"updates":[]})
        archives = self.store.events("ctx", "context_archive")
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0]["data"]["payload"]["opened_evidence"], state["evidence"])
        record = self.store.events("ctx", "generation")[0]["data"]
        self.assertNotIn("raw_context_payload", record["request"])
        rows = list(sft_samples(self.store, "ctx", codec))
        self.assertEqual(rows[0]["messages"], record["request"]["messages"])
        n = len(record["request"]["prompt_token_ids"])
        self.assertEqual(rows[0]["labels"][:n], [-100]*n)
        self.assertTrue(any(r['data'].get('projected_chunk') for r in self.store.events('ctx','scoped_compression')))
        self.assertEqual([row["stage"] for row in rows], ["state"])

    async def test_semantic_compression_requires_provider_on_overflow(self):
        from test_context import fixture
        with self.assertRaisesRegex(ContextOverflow, "semantic_compression_provider_required"):
            await ContextBuilder(OfflineCodec(), 4500).prepare("state", fixture())

    async def test_semantic_selector_format_repair_and_replay_cache(self):
        from test_context import fixture
        from trainable_agent.compression import SemanticSelector
        compressor = ScriptModel([{"passage_ids": ["S999"]}, {"passage_ids": ["S0", "S1"]}])
        compressor.identity = "frozen-fixture-selector"
        selector = SemanticSelector(compressor, OfflineCodec(), self.store, "card", 100000)
        chunk = fixture()["evidence"]["E1"]
        one = await selector.select(chunk, {"question": "Benefits and harms?"})
        two = await selector.select(chunk, {"question": "Benefits and harms?"})
        self.assertEqual(one, two)
        self.assertEqual(len(compressor.requests), 2)
        self.assertIn("Validation feedback", compressor.requests[1]["messages"][-1]["content"])

    async def test_semantic_selector_empty_is_not_deleted_evidence(self):
        from test_context import fixture
        from trainable_agent.compression import SemanticSelector
        compressor = ScriptModel([{"passage_ids": []}])
        compressor.identity = "frozen-fixture-selector"
        selector = SemanticSelector(compressor, OfflineCodec(), self.store, "empty-card", 100000)
        state = fixture()
        original = copy.deepcopy(state)
        self.assertIsNone(await selector.select(state["evidence"]["E1"], {"question": state["question"]}))
        self.assertEqual(state, original)
        self.assertEqual(len(compressor.requests), 1)

    async def test_grpo_without_behavior_capture_rejected(self):
        await self.engine(ScriptModel([CHECK, STOP]), ScriptTools([])).run(self.state)
        with self.assertRaisesRegex(ValueError, "behavior_token"):
            require_grpo_capture([r["data"] for r in self.store.events("q1", "generation")])

    def test_group_advantages_and_clipped_surrogate(self):
        self.assertEqual(group_advantages([1., 1.]), [0., 0.])
        a = group_advantages([0., 1.])
        self.assertAlmostEqual(sum(a), 0.)
        self.assertAlmostEqual(clipped_surrogate([0.], [0.], 1., [True]), -1.)

    def test_run_lease_prevents_concurrent_writer(self):
        owner = self.store.acquire("q1")
        with self.assertRaisesRegex(RuntimeError, "manual_lease"):
            self.store.acquire("q1")
        self.store.release("q1", owner)
        owner = self.store.acquire("q1")
        self.store.release("q1", owner)

    def test_source_failure_not_environment_exemption(self):
        obs = normalize({"failed": True, "error": "blocked"}, "web", "WEB:1")
        self.assertEqual(obs["status"], "source_unreadable")
        obs = normalize({"failed": True, "environment_failure": True, "error": "timeout"}, "web")
        self.assertEqual(obs["status"], "environment_failure")

    async def test_delta_preserves_unchanged_state(self):
        engine = self.engine(ScriptModel([CHECK, SEARCH, BROWSE, UPDATE, STOP]), ScriptTools([FOUND, OPENED]))
        s = copy.deepcopy(self.state)
        s.update(requirements={"R1": "Effectiveness", "R2": "Safety"}, next_node="state")
        s["assessments"] = {"R1": dict(support="missing", evidence_ids=[], gap="Need evidence", conflict=""),
                            "R2": dict(support="missing", evidence_ids=[], gap="Need safety", conflict="")}
        s["evidence"] = {"E1": OPENED["chunks"][0]}
        s["new_evidence_ids"] = ["E1"]
        engine.process = ScriptModel([UPDATE])
        result = (await engine.state({"session": s}))["session"]
        self.assertEqual(result["assessments"]["R2"], s["assessments"]["R2"])
        self.assertEqual(result["assessments"]["R1"]["support"], "partial")


if __name__ == "__main__":
    unittest.main()
