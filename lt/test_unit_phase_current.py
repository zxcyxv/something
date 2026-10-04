"""Model-level checks for dynamic phase Gaussian STDP and both residual norms."""
import math
import unittest
from unittest.mock import patch

import torch

from . import train as t
from .kv_stability import DynamicGaussianPhaseCurrentReadInner
from .test_kv_stability import config


class DynamicGaussianPhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(103)
        self.inner = DynamicGaussianPhaseCurrentReadInner(config())
        self.layer = self.inner.layers[0]

    def test_relation_window_before_token_mean_and_history_independence(self):
        inner = self.inner.double()
        hidden = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64,
                              requires_grad=True) for _ in range(3)]
        pk, pv = inner.phases(self.layer, hidden)
        read, g, ek, ev = inner.memory_step(self.layer, q, k, v, phases=(pk, pv))
        qr, kr = [inner.apply_rope(x, self.layer) for x in (q, k)]
        relation = torch.exp(1j*pv[..., :, None]) * torch.exp(-1j*pk[..., None, :])
        delta = pv[..., :, None] - pk[..., None, :]
        expected = (v[..., :, None] * kr[..., None, :] * relation.imag.sign()
                    * torch.exp(-delta.square())).mean(-3)
        torch.testing.assert_close(g, expected)
        torch.testing.assert_close(read, qr @ expected.transpose(-1, -2))
        other = inner.memory_step(self.layer, q, k, v,
                                  torch.full_like(g, float('nan')),
                                  torch.full_like(k, float('nan')),
                                  torch.full_like(v, float('nan')), phases=(pk, pv))
        torch.testing.assert_close(other[0], read)
        self.assertIsNone(ek); self.assertIsNone(ev)
        # A common carrier can change by observation, but cancels within pairs.
        shift = torch.randn(2, 2, 9, 1, dtype=torch.float64)
        shifted = inner.memory_step(self.layer, q, k, v, phases=(pk+shift, pv+shift))
        torch.testing.assert_close(shifted[0], read)
        read.square().mean().backward()
        for x in (q, k, v, hidden, self.layer.phase_k_proj.weight,
                  self.layer.phase_v_proj.weight, self.layer.theta_k_raw,
                  self.layer.theta_v_raw):
            self.assertTrue(x.grad.isfinite().all())
            self.assertGreater(x.grad.norm().item(), 0.)

    def test_state_dependence_alias_margin_zero_lag_and_no_unit_norm(self):
        inner, layer = self.inner, self.layer
        h = torch.randn(2, 9, 16)
        a, b = inner.phases(layer, h), inner.phases(layer, -h)
        for x, y in zip(a, b):
            self.assertFalse(torch.equal(x, y))
            self.assertLess(float(x.detach().abs().max()), math.pi/2)
        with torch.no_grad():
            layer.theta_k_raw.fill_(-100)
            layer.theta_v_raw.fill_(100)
        pk, pv = inner.phases(layer, h)
        self.assertLess(float((pv[..., :, None]-pk[..., None, :]).detach().max()), math.pi)
        q, k, v = [torch.randn(2, 2, 9, 8) for _ in range(3)]
        same = torch.zeros_like(k)
        read, g, _, _ = inner.memory_step(layer, q, k, v, phases=(same, same))
        self.assertEqual(read.abs().max().item(), 0.)
        self.assertEqual(g.abs().max().item(), 0.)
        # Positive rescaling changes activities but must not change A.
        z_k, z_v = torch.exp(1j*a[0]), torch.exp(1j*a[1])
        raw = (2*z_v[..., :, None]*(3*z_k[..., None, :]).conj()).imag.sign()
        unit = (z_v[..., :, None]*z_k[..., None, :].conj()).imag.sign()
        torch.testing.assert_close(raw, unit)

    def test_forward_two_norm_positions_optimizer_and_reload(self):
        inner, layer = self.inner, self.layer
        with torch.no_grad():
            layer.b_down.weight.normal_(std=.015)
        h, inj = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
        x = h+inner.embed_scale*inj
        heads = lambda y: y.reshape(2, 9, 2, 8).transpose(1, 2)
        q, k, v = [heads(p(x)) for p in (layer.q_proj, layer.k_proj, layer.v_proj)]
        read, _, _, _ = inner.memory_step(layer, q, k, v, phases=inner.phases(layer, x))
        residual = x+layer.out_proj(read.transpose(1, 2).reshape(2, 9, 16))
        norm = lambda y: y.float()*torch.rsqrt(y.float().square().mean(-1, keepdim=True)+1e-5)
        u = norm(residual)
        gate, value = layer.b_gate_up(u).chunk(2, -1)
        expected = norm(u+layer.b_down(.5*gate*value))
        with patch.object(inner, 'phi', wraps=inner.phi) as rms:
            actual, g, ek, ev = inner.block(layer, h, inj, None, None, None, None)
            self.assertEqual(rms.call_count, 2)
            torch.testing.assert_close(rms.call_args_list[0].args[0], residual)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(ek); self.assertIsNone(ev)
        copy = type(inner)(config())
        copy.load_state_dict(inner.state_dict())
        torch.testing.assert_close(copy.block(copy.layers[0], h, inj, g, None, None, None)[0], actual)
        before = layer.phase_k_proj.weight.detach().clone()
        optimizer = torch.optim.Adam(inner.parameters(), lr=1e-3)
        actual[..., 0].square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, layer.phase_k_proj.weight))
        self.assertFalse(t._is_no_decay('inner.layers.0.phase_k_proj.weight', layer.phase_k_proj.weight))
        self.assertTrue(t._is_no_decay('inner.layers.0.theta_k_raw', layer.theta_k_raw))


if __name__ == '__main__':
    unittest.main()
