"""Check public excerpt integrity; no model or semantic scoring."""
import hashlib
import json
from pathlib import Path
import re
import unittest

CASE = Path(__file__).resolve().parent / "trajectory_demo"


def read(name):
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TrajectoryExcerptTests(unittest.TestCase):
    def test_source_completion_hashes(self):
        expected = read("provenance.json")["model_completion_sha256_from_source_log"]
        for name, key in (("checklist.json", "checklist_init"),
                          ("evidence_state.json", "state_update")):
            compact = json.dumps(read(name), ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(digest(compact), expected[key])
        for name, key in (("search.json", "search_decision"),
                          ("browse.json", "browse_decision")):
            self.assertEqual(digest(read(name)["completion"]), expected[key])
        final = (CASE / "final_answer.md").read_text(encoding="utf-8").removesuffix("\n")
        self.assertEqual(len(final), read("provenance.json")["final_source_chars"])
        self.assertEqual(digest(final), expected["final"])
        self.assertEqual(digest(read("provenance.json")["final_ready_completion"]),
                         expected["final_ready_decision"])

    def test_display_is_not_scored_training_data(self):
        reward = read("reward_breakdown.json")
        self.assertEqual(reward["status"], "not_scored")
        for key in ("checklist", "search", "browse_source_focus", "evidence_gain",
                    "state", "final_completeness", "final_fidelity", "final_citation", "stop"):
            self.assertIsNone(reward[key])
        self.assertFalse(reward["reward_export_authorized"])
        self.assertEqual(reward["judge_api_calls"], 0)
        self.assertEqual(reward["optimizer_updates"], 0)
        self.assertFalse(read("browse.json")["chunk_text_included"])
        final = (CASE / "final_answer.md").read_text(encoding="utf-8")
        citations = re.findall(r'<cite id="([^"]+)">', final)
        observed = read("browse.json")["chunk_ids_referenced_in_generated_outputs"]
        self.assertEqual(len(citations), 4)
        self.assertTrue(set(citations).issubset(observed))

    def test_document_links_and_public_scope(self):
        root = CASE.parents[1]
        for doc in (root / "README.md", root / "docs/code_navigation.md", CASE / "README.md"):
            for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
                if "://" not in target and not target.startswith("#"):
                    self.assertTrue((doc.parent / target.split("#")[0]).exists(), target)
        for path in CASE.iterdir():
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/msai/", text)
            self.assertNotIn("172.21.26.100", text)
            self.assertNotIn("C:\\Users\\", text)
            self.assertNotRegex(text, r"sk-[A-Za-z0-9_-]{16,}")


if __name__ == "__main__":
    unittest.main()
