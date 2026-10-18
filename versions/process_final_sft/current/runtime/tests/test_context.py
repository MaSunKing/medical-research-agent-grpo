import copy
import json
import unittest

from trainable_agent.compression import card, validate_card
from trainable_agent.context import ContextBuilder, ContextOverflow, OfflineCodec, CardSelectionRequired
from trainable_agent.engine import initial_state
from trainable_agent.context_processing.aliases import project as alias_project


def fixture():
    state = initial_state("ctx", "Does exercise reduce pain in adults?", "p", "f", "t")
    state["requirements"] = {"R1": "Benefits and harms of exercise in adults."}
    state["assessments"] = {"R1": dict(support="partial", evidence_ids=["E1"], gap="Long-term harm.", conflict="")}
    text = "Exercise reduced pain in adults. Long-term harm was not measured. " + " ".join(
        f"Background item {i} describes a separate historical observation." for i in range(80))
    state["evidence"] = {"E1": dict(source_id="PMID:1", chunk_id="PMID:1:c1", title="Trial", text=text),
                         "E2": dict(source_id="WEB:x", chunk_id="WEB:x:c1", title="New", text="New evidence.")}
    state["new_evidence_ids"] = ["E2"]
    state["history"] = [{"turn": i, "query": "earlier query"} for i in range(12)]
    state["candidates"] = [dict(source_id="PMID:1", preview="Candidate preview", title="Trial")]
    return state


def prepared(builder, stage, state):
    # Stub selected IDs ONLY; this test does not claim real semantic model quality.
    cards = {}
    while True:
        try:
            return builder.build(stage, state, cards=cards)
        except CardSelectionRequired as need:
            cards[need.key] = card(need.chunk, ["S0", "S1"])


class ContextTests(unittest.TestCase):
    def test_card_exact_quotes_and_ids(self):
        original = fixture()["evidence"]["E1"]
        projected = card(original, ["S0", "S1"])
        validate_card(projected, original)
        self.assertIn("not measured", projected["text"])
        self.assertEqual(projected["chunk_id"], original["chunk_id"])
        self.assertFalse(projected["evidence_card"]["semantic_verified"])

    def test_modified_quote_rejected(self):
        original = fixture()["evidence"]["E1"]
        projected = card(original, ["S0"])
        projected["text"] += " Invented improvement."
        with self.assertRaises(ValueError):
            validate_card(projected, original)

    def test_modified_source_rejected(self):
        original = fixture()["evidence"]["E1"]
        projected = card(original, ["S0"])
        changed = {**original, "text": original["text"] + "Changed."}
        with self.assertRaises(ValueError):
            validate_card(projected, changed)

    def test_raw_fits_unchanged(self):
        state = fixture()
        req = ContextBuilder(OfflineCodec(), 100000).build("state", state)
        payload = json.loads(req["messages"][-1]["content"])
        self.assertEqual(payload["opened_evidence"], alias_project({"opened_evidence": state["evidence"]}, state)["opened_evidence"])
        self.assertFalse(req["context_audit"]["evidence_compressed"])

    def test_overflow_cards_keeps_latest_state_candidates_and_raw_archive(self):
        state = fixture()
        before = copy.deepcopy(state)
        # Byte-count fixture budget, not the unchanged 10240-token live budget.
        builder = ContextBuilder(OfflineCodec(), 6000)
        req = prepared(builder, "state", state)
        payload = json.loads(req["messages"][-1]["content"])
        self.assertTrue(req["context_audit"]["evidence_compressed"])
        self.assertEqual(payload["opened_evidence"]["E2"], alias_project({"opened_evidence":state["evidence"]},state)["opened_evidence"]["E2"])
        self.assertEqual(payload["state"], state["assessments"])
        self.assertNotIn("candidates",payload)
        self.assertEqual(set(payload["opened_evidence"]), set(state["evidence"]))
        self.assertEqual(state, before)
        self.assertEqual(req["raw_context_payload"]["opened_evidence"], state["evidence"])
        self.assertLessEqual(len(req["prompt_token_ids"]) + req["max_new_tokens"], builder.limit)
        self.assertEqual(req, prepared(builder, "state", state))

    def test_latest_only_overflow_fails(self):
        state = fixture()
        state["new_evidence_ids"] = list(state["evidence"])
        with self.assertRaises(ContextOverflow):
            ContextBuilder(OfflineCodec(), 4500).build("state", state)

    def test_disable_cards_fails_closed(self):
        with self.assertRaises(ContextOverflow):
            ContextBuilder(OfflineCodec(), 4500, evidence_cards=False).build("state", fixture())

    def test_final_uses_same_projection_without_modifying_prefinal(self):
        state = fixture()
        state["prefinal"] = dict(question=state["question"], checklist=state["requirements"],
                                 state=state["assessments"], opened_evidence=state["evidence"],
                                 compact_history=state["history"], stop_reason="budget", unresolved=state["assessments"])
        original = copy.deepcopy(state["prefinal"])
        req = prepared(ContextBuilder(OfflineCodec(), 6000), "final", state)
        self.assertTrue(req["context_audit"]["evidence_compressed"])
        self.assertEqual(state["prefinal"], original)

    def test_long_sentence_is_not_cut(self):
        original = dict(source_id="PMID:1", chunk_id="c1", text="A" * 5000)
        self.assertEqual(card(original, ["S0"])["text"], original["text"])

    def test_lexical_fallback_not_available(self):
        with self.assertRaises(CardSelectionRequired):
            ContextBuilder(OfflineCodec(), 4500).build("state", fixture())

    def test_unknown_and_duplicate_passage_ids_rejected(self):
        for ids in (["S999"], ["S0", "S0"]):
            with self.assertRaises(ValueError):
                card(fixture()["evidence"]["E1"], ids)


if __name__ == "__main__":
    unittest.main()
