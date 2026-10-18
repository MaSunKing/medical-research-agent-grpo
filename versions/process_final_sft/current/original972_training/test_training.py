"""CPU/toy tests only. Does not load Qwen, call APIs or run actual training."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from contract import batch_order, lr_scale, safe_child, save, sha
from training_core import checkpoint_complete, masked_loss


class Contracts(unittest.TestCase):
    def test_steps_last_batch(self):
        batches = [list(range(10575))[i:i+8] for i in range(0, 10575, 8)]
        self.assertEqual(len(batches), 1322)
        self.assertEqual(len(batches[-1]), 7)

    def test_order_independent_of_global_rng(self):
        state = random.getstate()
        first = batch_order(97, 42, 0)
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, batch_order(97, 42, 0))
        self.assertEqual(set(first), set(range(97)))
        self.assertEqual(batch_order(97, 42, 0, 44)[0], 44)

    def test_lr(self):
        self.assertAlmostEqual(lr_scale(0, 1322, 67), 1/67)
        self.assertEqual(lr_scale(66, 1322, 67), 1)
        self.assertEqual(lr_scale(67, 1322, 67), 1)
        self.assertEqual(lr_scale(1322, 1322, 67), 0)

    def test_safe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError): safe_child(d, '../bad')
            self.assertEqual(safe_child(d, 'safe.txt'), Path(d)/'safe.txt')

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a.json'
            save(p, dict(a='医学'))
            self.assertEqual(json.loads(p.read_text(encoding='utf8')), dict(a='医学'))
            self.assertEqual(len(list(Path(d).iterdir())), 1)

    def test_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for n in ('adapter_config.json', 'adapter_model.safetensors', 'state.pt'):
                (p/n).write_bytes(b'toy-not-weights')
            save(p/'COMPLETE.json', dict(identity={'toy':True}, files={n:sha(p/n) for n in
                ('adapter_config.json', 'adapter_model.safetensors', 'state.pt')}))
            checkpoint_complete(p, {'toy':True})
            (p/'state.pt').write_bytes(b'corrupted')
            with self.assertRaises(AssertionError): checkpoint_complete(p, {'toy':True})

    def test_sparse_causal_loss_and_gradients(self):
        import copy
        import torch
        from types import SimpleNamespace
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(31, 9)
                self.lm_head = torch.nn.Linear(9, 31)
            def get_base_model(self): return self
            def model(self, input_ids, use_cache=False):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        torch.manual_seed(13)
        a = Toy()
        b = copy.deepcopy(a)
        tokens = torch.randint(0, 31, (1, 150))
        numerator, mass = masked_loss(a, tokens, [[17, 150, 1.]])
        labels = tokens[:, 1:].clone()
        labels[:, :16] = -100
        logits = b.lm_head(b.embedding(tokens))[:, :-1]
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 31), labels.reshape(-1), ignore_index=-100, reduction='sum')
        self.assertEqual(mass, 133)
        self.assertTrue(torch.allclose(numerator, expected, atol=1e-4, rtol=1e-6))
        (numerator/mass).backward()
        (expected/mass).backward()
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-7, rtol=1e-5))


if __name__ == '__main__': unittest.main()
