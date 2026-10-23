"""Coupled model (lt.coupled): the reductions and information paths the design claims. Run: python -m unittest tests.test_coupled"""
import unittest

import torch
import torch.nn.functional as F

from lt.coupled import PAD, VOCAB, Coupled, CoupledConfig, soft


def small(**kw):
    return CoupledConfig(**dict(dict(N=3, D=3, d=16, heads=2, blocks=4, grad_blocks=2, phi=8, forward_dtype='float64'), **kw))


def batch(cfg, groups=2, seed=0):
    """groups x (D demonstration lanes + 1 query lane, the last of its group)."""
    g = torch.Generator().manual_seed(seed)
    L, C = cfg.D + 1, cfg.C
    O, n = C * C, groups * (cfg.D + 1)
    inputs = torch.cat([torch.randint(1, VOCAB, (n, O), generator=g), torch.zeros(n, O, dtype=torch.long)], -1)
    labels = torch.cat([torch.full((n, O), PAD), torch.randint(1, VOCAB, (n, O), generator=g)], -1)
    query = (torch.arange(n) % L) == L - 1
    return dict(inputs=inputs, labels=labels, group=torch.arange(n) // L, demo=~query, query=query)


def model(cfg, seed=1):
    torch.manual_seed(seed)
    m = Coupled(cfg).double().eval()
    with torch.no_grad():
        m.w_fb.weight.normal_(0, 0.5)                          # open the feedback path
        m.lam_raw.fill_(-6.0)
    return m


class Coupling(unittest.TestCase):
    def test_no_demonstrations_is_bands(self):
        """Without demonstration lanes Theta stays 0, so the read, the feedback and the head vanish: the bands model."""
        cfg = small()
        m = model(cfg)
        b = batch(cfg)
        b['demo'][:] = False
        with torch.no_grad():
            a, theta = m(b)
            ref, _ = m(b, head=False)
        self.assertEqual(float(theta.abs().max()), 0.0)
        torch.testing.assert_close(a, ref, rtol=0, atol=0)

    def test_query_answer_never_read(self):
        cfg = small()
        m = model(cfg)
        b = batch(cfg)
        b2 = {k: v.clone() for k, v in b.items()}
        q = b['query']
        b2['labels'][q] = torch.randint(1, VOCAB, b2['labels'][q].shape)
        with torch.no_grad():
            torch.testing.assert_close(m(b)[0][q], m(b2)[0][q], rtol=0, atol=0)

    def test_demonstrations_reach_the_query_only_through_theta(self):
        """Perception never mixes lanes: with the head's read switched off (W_fb = 0, tau = 0) the query does not see the
        demonstrations; with it on, it does."""
        cfg = small()
        m = model(cfg)
        b = batch(cfg)
        b2 = {k: v.clone() for k, v in b.items()}
        q = b['query']
        b2['inputs'][0] = torch.randint(1, VOCAB, b2['inputs'][0].shape)     # demonstration 0 of group 0
        b2['labels'][0] = torch.randint(1, VOCAB, b2['labels'][0].shape)
        with torch.no_grad():
            self.assertFalse(torch.allclose(m(b)[0][q][0], m(b2)[0][q][0]))
            torch.testing.assert_close(m(b)[0][q][1], m(b2)[0][q][1], rtol=0, atol=0)   # other group: untouched
            m.w_fb.weight.zero_()
            m.tau.zero_()
            torch.testing.assert_close(m(b)[0][q], m(b2)[0][q], rtol=0, atol=0)

    def test_leave_one_out(self):
        """A demonstration lane's statistics are those of the other demonstrations; the query's are all of them."""
        cfg = small()
        m = model(cfg)
        g = torch.Generator().manual_seed(3)
        b = batch(cfg, groups=3)
        n, O = len(b['group']), cfg.C ** 2
        phi = F.normalize(torch.randn(n, O, cfg.phi, generator=g, dtype=torch.float64), dim=-1)
        y = F.one_hot(torch.randint(0, VOCAB, (n, O), generator=g), VOCAB).double()
        mk = (torch.rand(n, O, generator=g) < 0.7).double()
        demo = b['demo'].double()
        G, S = m.statistics(phi, y, mk, demo, b['group'], 3)
        for j in range(n):
            keep = demo * (b['group'] == b['group'][j]).double()
            keep[j] = 0
            w = mk * keep[:, None]
            tot = w.sum()
            torch.testing.assert_close(G[j], torch.einsum('lo,lop,loq->pq', w, phi, phi) / tot)
            torch.testing.assert_close(S[j], torch.einsum('lo,loc,lop->cp', w, y, phi) / tot)
            self.assertAlmostEqual(float(torch.diagonal(G[j]).sum()), 1.0, places=12)   # tr G = 1

    def test_unit_step_decreases_the_lasso(self):
        """Step 1 is valid because lambda_max(G) <= tr G = 1: the objective never increases."""
        cfg = small()
        m = model(cfg)
        with torch.no_grad():
            m.lam_raw.fill_(-7.0)
        g = torch.Generator().manual_seed(4)
        phi = F.normalize(torch.randn(50, cfg.phi, generator=g, dtype=torch.float64), dim=-1)
        y = F.one_hot(torch.randint(0, VOCAB, (50,), generator=g), VOCAB).double()
        G, S = phi.T @ phi / 50, y.T @ phi / 50
        lam = float(F.softplus(m.lam_raw))
        obj = lambda th: float(0.5 * ((y - phi @ th.T) ** 2).sum(-1).mean() + lam * th.abs().sum())
        th = torch.zeros(VOCAB, cfg.phi, dtype=torch.float64)
        prev = obj(th)
        with torch.no_grad():
            for i in range(40):
                th = m.step(th, G, S, 0)
                cur = obj(th)
                self.assertLessEqual(cur, prev + 1e-12)
                prev = cur
        self.assertLess(prev, obj(torch.zeros_like(th)))


if __name__ == '__main__':
    unittest.main()
