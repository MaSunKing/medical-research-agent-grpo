import copy
import unittest
from pathlib import Path
import tempfile
from trainable_agent.compression import card
from trainable_agent.context import OfflineCodec
from trainable_agent.context_processing.fallback import ScopedFallback
from trainable_agent.storage import Store
from trainable_agent.context_processing.deterministic_context import project_evidence, validate_projection


class TableContextTests(unittest.TestCase):
    def test_exact_receipts_archived_and_partial_scope_visible(self):
        original = {"E1": {"source_id": "PMID:1", "chunk_id": "table:1:window:2",
            "text": "Treatment A | Treatment B | 12% | 8%",
            "metadata": {"year": 2024, "abstract_only": False,
                "window_identity": {"source_spans": [{"start_char": 50, "end_char": 91}]},
                "window_identity_sha256": "a" * 64,
                "table_window": {"version": "exact_table_rows_v1", "row_indices": [2, 3],
                    "complete_table": False, "complete_row_groups": True,
                    "all_columns_retained": True, "all_footnotes_retained": True,
                    "source_spans": [{"start_char": 50, "end_char": 91}]}}}}
        saved = copy.deepcopy(original)
        projected, audit = project_evidence(original)
        self.assertEqual(original, saved)
        self.assertEqual(projected["E1"]["text"], original["E1"]["text"])
        metadata = projected["E1"]["metadata"]
        self.assertEqual(metadata["year"], 2024)
        self.assertFalse(metadata["table_window"]["complete_table"])
        self.assertEqual(metadata["table_window"]["selected_row_count"], 2)
        self.assertNotIn("window_identity", metadata)
        self.assertNotIn("source_spans", metadata["table_window"])
        self.assertEqual(audit["table_provenance_omitted_ids"], ["E1"])
        validate_projection(projected, original)
        projected["E1"]["metadata"]["table_window"]["complete_table"] = True
        with self.assertRaisesRegex(ValueError, "identity_changed"):
            validate_projection(projected, original)

    def test_unknown_metadata_is_preserved(self):
        original = {"E1": {"source_id": "PMID:1", "chunk_id": "c1", "text": "12% vs 8%",
                    "metadata": {"table_window": {"version": "unknown", "meaning": "do not erase"}}}}
        projected, _ = project_evidence(original)
        self.assertEqual(projected, original)


class TableFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_sentence_selection_cannot_drop_table_dependencies(self):
        chunk = {"text": "Header A B. Result 12% 8%. Footnote defines denominator.",
                 "source_id": "PMID:1", "chunk_id": "table-window-1",
                 "metadata": {"table_window": {"version": "exact_table_rows_v1"}}}
        with self.assertRaisesRegex(ValueError, "complete_row_window"):
            card(chunk, ["S1"])
        class Provider:
            identity = 'unused-test-provider'
            async def generate(self, request):
                raise AssertionError('model must not rewrite protected table window')
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / 'audit.sqlite')
            try:
                fallback = ScopedFallback(Provider(), OfflineCodec(), store, 'test', 10240)
                for mode in ('compress', 'clean'):
                    self.assertIsNone(await fallback.transform(chunk, {}, 10, mode))
                self.assertEqual(fallback.budget.calls, 0)
                self.assertEqual(len(store.events('test', 'scoped_compression')), 2)
            finally:
                store.db.close()
