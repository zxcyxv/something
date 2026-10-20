"""Rule-synapse models: the designed score is the explained-cell count, the closed-form window equals the bands
recursion, the window initialisations hit their target lags, gradients reach the window.
Run: python -m unittest toy_g.test_rule_synapse"""
import unittest

import numpy as np
import torch

from toy_g import data as D
from toy_g import group as G
from toy_g.design_check import bands_window
from toy_g.rule_synapse import BandsRuleModel, RuleSynapseModel, window_synapse


class RuleSynapseTest(unittest.TestCase):
    def test_designed_score_counts_explained_cells(self):
        rng = np.random.default_rng(0)
        m = RuleSynapseModel(mode='designed', d=16, cmap='perm', tau=60.0, beta=1.0)
        I = D.random_inputs(rng, 2)
        t, c = 37, 5
        O = np.stack([G.apply_task(g, t, c) for g in I])
        din = torch.as_tensor(I[None].reshape(1, 2, 16))
        dout = torch.as_tensor(O[None].reshape(1, 2, 16))
        res = m(din, dout, torch.ones(1, 2, dtype=torch.bool), din[:, 0])
        explained = int((O.reshape(2, -1) > 0).sum())
        self.assertAlmostEqual(float(res['scores'][0, t]), explained, places=4)

    def test_closed_form_window_equals_recursion(self):
        torch.manual_seed(0)
        model = BandsRuleModel(schedule='persist', bands=3, init='random', seed=1).double()
        with torch.no_grad():
            model.b.normal_()
        k = torch.randn(3, 1, 5, 4, dtype=torch.float64)
        v = torch.randn(3, 1, 5, 4, dtype=torch.float64)
        z, gamma = model.poles()
        a = torch.complex(model.a[:, 0], model.a[:, 1]).detach().numpy()
        b = torch.complex(model.b[:, 0], model.b[:, 1]).detach().numpy()
        rec = bands_window(z.detach().numpy().astype(complex), a.astype(complex), b.astype(complex),
                           k[:, 0].numpy(), v[:, 0].numpy())
        closed = window_synapse(k, v, lambda D: model.lag(D).double())[0].detach().numpy()
        np.testing.assert_allclose(closed, rec, atol=1e-10)

    def test_window_inits(self):
        for init, target in (('odd', [0, -1, 0, 1, 0]), ('causal', [0, 0, 0, 1, 0]), ('even', [0, 1, 0, 1, 0])):
            w = BandsRuleModel(init=init).window()
            np.testing.assert_allclose([w[D] for D in (-2, -1, 0, 1, 2)], target, atol=1e-5)

    def test_gradients_reach_the_window(self):
        rng = np.random.default_rng(1)
        model = BandsRuleModel(schedule='residual', init='odd')
        I = D.random_inputs(rng, 3)
        O = np.stack([G.apply_task(g, 9, 3) for g in I])
        din = torch.as_tensor(I.reshape(1, 3, 16))
        dout = torch.as_tensor(O.reshape(1, 3, 16))
        res = model(din, dout, torch.ones(1, 3, dtype=torch.bool), din[:, 0])
        res['probs'].log().sum().backward()
        for p in (model.a, model.b, model.nu, model.theta, model.Wk, model.Wv):
            self.assertGreater(float(p.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
