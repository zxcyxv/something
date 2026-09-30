"""Behavioral checks for the standalone 512-dimensional Q/K delta variant."""
import unittest

import torch

from . import train_v18 as t


def config(**overrides):
    cfg = dict(t.DEFAULT_CFG, batch_size=2, seq_len=81, num_puzzle_identifiers=1,
               hidden_size=32, num_heads=2, puzzle_emb_ndim=32, loops=10,
               blocks_per_seg=2, amp=False, activation_checkpoint=False)
    cfg.update(overrides)
    return cfg


def batch():
    return dict(inputs=torch.randint(1, 11, (2, 81)),
                labels=torch.randint(2, 11, (2, 81)),
                puzzle_identifiers=torch.zeros(2, dtype=torch.int32))


class V18DeltaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_full_size_shapes_and_defaults(self):
        torch.manual_seed(1)
        cfg = dict(t.DEFAULT_CFG, batch_size=2, seq_len=81, num_puzzle_identifiers=1,
                   blocks_per_seg=1, amp=False, activation_checkpoint=False)
        model = t.LT(cfg).eval()
        layer = model.inner.layers[0]
        self.assertEqual(layer.wq.shape, (8, 64, 512))
        self.assertEqual(layer.wk.shape, (8, 64, 512))
        self.assertEqual(layer.w_sh.shape, (8, 64, 512))
        self.assertFalse(any(n.endswith('.beta') for n in model.state_dict()))
        self.assertEqual(set(t.CFG), set(t.DEFAULT_CFG))
        b = batch()
        with torch.no_grad():
            carry, output = model(model.initial_carry(b), b)
        self.assertEqual(carry.trace.shape, (2, 81, 8, 32, 2, 2))
        self.assertEqual(carry.coupling.shape, (2, 8, 81, 81))
        self.assertEqual(output['logits'].shape, (2, 81, 11))
        self.assertTrue(torch.isfinite(output['logits']).all())

    def test_asymmetry_survives_zero_psi_without_beta(self):
        torch.manual_seed(2)
        model = t.LT(config(psi_zero=True)).eval()
        inner, layer = model.inner, model.inner.layers[0]
        with torch.no_grad():
            layer.theta.zero_()
            h = torch.randn(2, 81, 32)
            q, k = [inner.addr_raw(h, ab) for ab in inner.W_QK(layer)]
            attention = inner.attn_xy(inner._unit(q), inner._unit(k), inner.kernel(layer))
        self.assertGreater(float((attention - attention.transpose(-1, -2)).abs().max()), 0.1)

    def test_mixed_reset_discards_old_memory_and_both_traces(self):
        torch.manual_seed(3)
        model = t.LT(config()).eval()
        b = batch()
        with torch.no_grad():
            old, _ = model(model.initial_carry(b), b)
            old, _ = model(old, b)
            initial = model.initial_carry(b)
            reset = t.replace(old, halted=torch.tensor([True, False]))
            actual, output = model(reset, b)
            clean, clean_output = model(initial, b)
            continuing, continuing_output = model(old, b)
        for name in ('current_hidden', 'trace', 'coupling', 'steps'):
            torch.testing.assert_close(getattr(actual, name)[0], getattr(clean, name)[0], rtol=0, atol=0)
            torch.testing.assert_close(getattr(actual, name)[1], getattr(continuing, name)[1], rtol=0, atol=0)
        torch.testing.assert_close(output['logits'][0], clean_output['logits'][0], rtol=0, atol=0)
        torch.testing.assert_close(output['logits'][1], continuing_output['logits'][1], rtol=0, atol=0)

    def test_original_entrypoint_keeps_shared_projection(self):
        from . import train as original
        model = original.LT(config(address_projection='linear', plastic_select=False))
        layer = model.inner.layers[0]
        self.assertTrue(hasattr(layer, 'wc'))
        self.assertTrue(hasattr(layer, 'beta'))
        self.assertFalse(hasattr(layer, 'wq'))
        self.assertNotEqual(original.model_id_of(original.DEFAULT_CFG), t.model_id_of(t.DEFAULT_CFG))


if __name__ == '__main__':
    unittest.main()
