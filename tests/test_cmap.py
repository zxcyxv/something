"""Cognitive-map rule synapse (lt.cmap, lt.cmap_data): numerical checks of the design's propositions before training.
Run: python -m unittest tests.test_cmap"""
import unittest
from pathlib import Path

import numpy as np
import torch

from lt.cmap import FAST, Cmap, CmapConfig, NC, d4_tables, op_t
from lt.cmap_data import CmapEpisodes, op_apply

NPZ = Path(__file__).resolve().parents[1] / 'data' / 'miniarc_pairs.npz'


def small(**kw):
    return CmapConfig(**dict(dict(N=5, D=8, n_tasks=149, d=32, heads=2, base_freqs=1, layers=1, Kc=2, Kg=2, J=2,
                                  steps=3, grad_steps=0, emb=8), **kw))


def transform(batch, k, perm):
    """The same episodes in version (k, perm) applied on top (batch at version 0)."""
    out = dict(batch)
    sig = torch.as_tensor(perm)
    def col(x):
        y = torch.where(x >= 0, sig[x.clamp_min(0)], x)
        return op_t(y, k, (-2, -1))
    out['color'], out['label'] = col(batch['color']), col(batch['label'])
    swap = ((k % 4) % 2 == 1) != (k >= 4)
    out['size'] = batch['size'].flip(-1) if swap else batch['size']
    out['op'] = torch.full_like(batch['op'], k)
    out['perm'] = sig.expand_as(batch['perm']).clone()
    return out


class Code(unittest.TestCase):
    def test_d4_tables(self):
        mats, inv, mul = d4_tables(5)
        for a in range(8):
            self.assertTrue((mats[a] @ mats[inv[a]] == np.eye(2)).all())
            for b in range(8):
                self.assertTrue((mats[a] @ mats[b] == mats[mul[a][b]]).all())

    def test_code_permutation(self):
        torch.manual_seed(0)
        m = Cmap(small()).double()
        x = torch.randn(50, 2, dtype=torch.float64) * 3
        for k in range(8):
            Mx = x @ m.d4[k].double().T
            torch.testing.assert_close(m.code(Mx), m.code(x)[:, m.code_perm[k]], rtol=1e-10, atol=1e-10)


@unittest.skipUnless(NPZ.exists(), 'data/miniarc_pairs.npz missing')
class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = CmapEpisodes(NPZ, max_demos=8)
        cls.batch = next(cls.data.eval_batches(3))

    def test_data_versions(self):
        e0 = self.data.episode(5, self.data.train_pairs[5], self.data.test_pairs[5][0], 0)
        for v in (1, 7, 30):
            e = self.data.episode(5, self.data.train_pairs[5], self.data.test_pairs[5][0], v)
            k, perm = self.data.version(5, v)
            want = np.where(e0['color'] >= 0, perm[np.clip(e0['color'], 0, None)], -1)
            np.testing.assert_array_equal(e['color'], op_apply(want, k))
            self.assertEqual(int(perm[0]), 0)

    def test_equivariance(self):
        """Proposition 3 + group averaging: the version (k, perm) of an episode gives the transformed prediction."""
        torch.manual_seed(1)
        m = Cmap(small()).double().eval()
        with torch.no_grad():
            m.task_emb.normal_(0, 1.0)
            m.dec.weight.normal_(0, 0.1)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            l0, s0, _, _ = m(self.batch, train=False)
            for k in (1, 4, 6):
                perm = np.concatenate([[0], 1 + rng.permutation(9)])
                lk, sk, _, _ = m(transform(self.batch, k, perm), train=False)
                want = op_t(l0, k, (1, 2))[..., torch.as_tensor(np.argsort(perm))]
                torch.testing.assert_close(lk, want, rtol=1e-7, atol=1e-7)
                swap = ((k % 4) % 2 == 1) != (k >= 4)
                torch.testing.assert_close(sk, s0.flip(-1) if swap else s0, rtol=1e-7, atol=1e-7)


class Solver(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.m = Cmap(small()).double()
        B, P, O, Fn, C2 = 2, 9, 25, self.m.F, self.m.C2
        g = torch.Generator().manual_seed(3)
        self.M = torch.randn(B, P, Fn, NC, C2, generator=g, dtype=torch.float64) / (NC * C2) ** 0.5
        self.Uo = torch.nn.functional.normalize(torch.randn(B, P, O, Fn, C2, generator=g, dtype=torch.float64), dim=-1)
        self.kap = torch.softmax(torch.randn(B, P, O, Fn, NC, generator=g, dtype=torch.float64), -1)
        self.y = torch.rand(B, P, O, NC, generator=g, dtype=torch.float64)
        self.ymask = (torch.rand(B, P, O, generator=g) < 0.5).double()
        self.ymask[:, -1] = 0
        self.ctr = {k: 0.1 * torch.randn((B,) + shp, generator=g, dtype=torch.float64)
                    for k, shp in self.m.shapes.items()}

    def lip(self):
        B = 2
        return dict(v={k: torch.ones((B,) + self.m.shapes[k], dtype=torch.float64) for k in FAST},
                    L=torch.zeros(B, dtype=torch.float64), iters=4)

    def test_steps_decrease_the_objective(self):
        """The fast state enters the read linearly, so the write is one lasso; proximal-gradient steps of size 1 / L
        never increase the penalised objective."""
        m = self.m
        ctr = self.ctr
        st = {k: ctr[k].clone() for k in FAST}
        def objective(st):
            r = m.read(st, self.M, self.Uo, self.kap)
            n = self.ymask.flatten(1).sum(1).clamp_min(1)
            f = 0.5 * ((self.y - r).square() * self.ymask[..., None]).flatten(1).sum(1) / n
            for k in FAST:
                f = f + float(torch.nn.functional.softplus(m.tau[k])) * (st[k] - ctr[k]).abs().flatten(1).sum(1)
            return f

        prev = objective(st)
        first = prev.clone()
        with torch.no_grad():
            for _ in range(60):
                st = m.prox_step(st, ctr, 0, self.M, self.Uo, self.kap, self.y, self.ymask, False)
                cur = objective(st)
                self.assertTrue(bool((cur <= prev * (1 + 1e-9) + 1e-9).all()), (prev, cur))
                prev = cur
        self.assertTrue(bool((prev < 0.5 * first).all()))

    def test_query_output_never_written(self):
        m = self.m
        st = {k: v.clone() for k, v in self.ctr.items() if k != 'beta'}
        Uo2, kap2 = self.Uo.clone(), self.kap.clone()
        Uo2[:, -1] += 5.0
        kap2[:, -1] += 3.0
        a = m.prox_step(st, self.ctr, 0, self.M, self.Uo, self.kap, self.y, self.ymask, False)
        b = m.prox_step(st, self.ctr, 0, self.M, Uo2, kap2, self.y, self.ymask, False)
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
