"""Rule synapse (ModelConfig.rule_syn, LT.rule_core / rule_step), FP64.

1. With exact 2D Fourier characters as keys and queries and one-hot colours as values, the error-driven LTP with the
   sparsity LTD recovers the 4x4 torus toy rules from 3 demonstrations (the standalone result of toy_g/ltd_rule.py).
2. With rule_gain = 0 the model is the band_product model with the same weights (logits and carry identical).
"""
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from lt.model import LT, ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def cfg(**kw):
    base = dict(batch_size=4, seq_len=112, grid=4, num_grids=7, vocab_size=7, puzzle_emb_ndim=0, hidden_size=64,
                num_heads=2, blocks_per_seg=2, loops=1, mlp_inter=64, tag_bound=0.0, bands=4, band_product=True,
                outer_eta=0.0, kernel='torch', amp=False, forward_dtype='float64')
    base.update(kw)
    return ModelConfig(**base)


def fourier_codes(T, grid, P, dtype=torch.float64):
    """[T, 2P] interleaved unit codes: channel p = frequency (p // grid, p % grid) at the local position."""
    t = torch.arange(T)
    r, c = ((t // grid) % grid).to(dtype), (t % grid).to(dtype)
    k1, k2 = (torch.arange(P) // grid).to(dtype), (torch.arange(P) % grid).to(dtype)
    phi = 2 * math.pi * (r[:, None] * k1 + c[:, None] * k2) / grid
    return torch.stack([phi.cos(), phi.sin()], -1).reshape(T, 2 * P) / math.sqrt(P)


class RuleSynapseTest(unittest.TestCase):
    def test_recovers_toy_rules(self):
        torch.manual_seed(0)
        model = LT(cfg(rule_syn=True)).double().eval()
        inst = np.load(ROOT / 'data' / 'gtorus_instances.npz')
        n = 40
        rows = np.random.default_rng(0).choice(len(inst['task']), n, replace=False)
        grids = np.concatenate([np.stack([inst['demo_in'][rows, k], inst['demo_out'][rows, k]], 1) for k in range(3)]
                               + [inst['test_in'][rows][:, None]], 1).reshape(n, 7 * 16).astype(np.int64)  # [n, 112]
        H, D, P, T = 2, 32, 16, 112
        codes = fourier_codes(T, 4, P)[None, None].expand(n, H, T, D)
        v = torch.zeros(n, H, T, D, dtype=torch.float64)
        g = torch.as_tensor(grids)
        for a in range(1, 5):
            v[..., a - 1] = (g == a).to(torch.float64)[:, None, :]
        with torch.no_grad():
            read, w = model.rule_core(codes, codes, v, torch.zeros(n, H, D, P + D, dtype=torch.float64),
                                      torch.float64, inner=300)
        score = read[:, 0, 96:, :4]                                             # query grid, value dims of colours 1..4
        pred = torch.where(score.max(-1).values > 0.5, score.argmax(-1) + 1, 0).numpy()
        exact = (pred == inst['test_out'][rows].reshape(n, 16)).all(1).mean()
        self.assertGreaterEqual(exact, 0.9, f'exact {exact}')
        self.assertEqual(tuple(w.shape), (n, H, D, P + D))

    def test_zero_gain_is_band_product(self):
        torch.manual_seed(0)
        a = LT(cfg(rule_syn=True)).double().eval()
        torch.manual_seed(0)
        b = LT(cfg()).double().eval()
        sd = {k: v for k, v in a.state_dict().items() if not k.startswith('block.rule_')}
        self.assertEqual(set(sd), set(b.state_dict()))
        for k, v in b.state_dict().items():                                      # same draws for the shared weights
            self.assertTrue(torch.equal(v, sd[k]), k)
        with torch.no_grad():
            a.block.rule_gain.zero_()
        batch = dict(inputs=torch.randint(1, 7, (4, 112)), puzzle_identifiers=torch.zeros(4, dtype=torch.long),
                     labels=torch.zeros(4, 112, dtype=torch.long))
        ca, cb = a.initial_carry(batch), b.initial_carry(batch)
        with torch.no_grad():
            ca, la = a(ca, batch)
            cb, lb = b(cb, batch)
        self.assertTrue(torch.allclose(la, lb, atol=1e-12, rtol=0))
        self.assertEqual(tuple(ca.stdp_w.shape), (4, 2, 32, 48))
        self.assertIsNone(cb.stdp_w)


if __name__ == '__main__':
    unittest.main()
