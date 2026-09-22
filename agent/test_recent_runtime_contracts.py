import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "retrieval"))

from evidence_contract_v14 import evidence_payload
from final_repetition_guard import repetition_event


def load_table_signature():
    path = ROOT / "retrieval/dr_agent/mcp_backend/apis/table_integrity.py"
    spec = importlib.util.spec_from_file_location("public_table_integrity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.table_signature


table_signature = load_table_signature()


class RecentRuntimeContracts(unittest.TestCase):
    def test_statistical_prose_is_not_a_table(self):
        text = (
            "The trial enrolled 200 patients. "
            "Treatment reduced adverse events (RR 0.80; 95% CI 0.65-0.99)."
        )
        result = table_signature(text)
        self.assertEqual(result["structure_kind"], "prose")
        self.assertTrue(result["prose_sentence_override"])

    def test_complete_table_is_structural_evidence(self):
        text = (
            "Outcome | Events | Sample size | RR (95% CI)\n"
            "Diabetic ketoacidosis | 56 | 14974 | 3.54 (1.27-9.87)"
        )
        self.assertEqual(table_signature(text)["structure_kind"], "table_like")

    def test_structure_fields_are_mandatory_at_runtime_handoff(self):
        with self.assertRaisesRegex(ValueError, "missing opened evidence structure fields"):
            evidence_payload({"data": [{"source_id": "E1", "text": "finding"}]})

    def test_repeated_long_final_unit_is_stopped_not_rewritten(self):
        unit = "This evidence-grounded paragraph is intentionally long enough to represent a material answer unit without changing its text."
        event = repetition_event(f"<answer>{unit}\n\n{unit}</answer>")
        self.assertIsNotNone(event)
        self.assertEqual(event["stop_reason"], "abnormal_repetition")
        self.assertFalse(event["protocol_valid"])


if __name__ == "__main__":
    unittest.main()
