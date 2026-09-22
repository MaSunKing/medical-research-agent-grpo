"""CPU tests for self-contained SFT paths in the public source export."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from shared_interface import template_options
from train_tc2 import complete_checkpoints, loss, prune_checkpoints


class ExportedSftContracts(unittest.TestCase):
    def test_stage_thinking_modes_are_stable(self):
        for name in ("checklist_init", "state_update", "evidence_card"):
            self.assertFalse(template_options(name)["enable_thinking"])
        self.assertTrue(template_options("final")["enable_thinking"])

    def test_weighted_chunked_loss_matches_cross_entropy(self):
        torch.manual_seed(2)

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(13, 5)
                self.lm_head = torch.nn.Linear(5, 13)
                self.config = SimpleNamespace(vocab_size=13)

            def model(self, input_ids, use_cache=False):
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))

            def get_base_model(self):
                return self

        model = Base()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        actual = loss(model, ids, 3, 0.25)
        expected = torch.nn.functional.cross_entropy(
            model.lm_head(model.emb(ids))[:, 2:-1].reshape(-1, 13),
            ids[:, 3:].reshape(-1),
        ) * 0.25
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())

    def test_only_complete_checkpoints_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(5):
                checkpoint = root / f"checkpoint-{step:08d}"
                checkpoint.mkdir()
                for name in (
                    "COMPLETE",
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "state.pt",
                ):
                    (checkpoint / name).write_text("stub")
            (root / ".saving-00000005").mkdir()
            (root / "checkpoint-00000006").mkdir()
            prune_checkpoints(root)
            self.assertEqual(
                [path.name for path in complete_checkpoints(root)],
                [f"checkpoint-{step:08d}" for step in (2, 3, 4)],
            )
            self.assertTrue((root / ".saving-00000005").exists())


if __name__ == "__main__":
    unittest.main()
