"""Phase-current operator: complex equivalence, history independence and training."""
import math
import unittest
from unittest.mock import patch
import torch
from . import train as t
from .kv_stability import PhaseCurrentReadInner
from .test_kv_stability import config


class PhaseCurrentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.inner = PhaseCurrentReadInner(config()).double()
        self.layer = self.inner.layers[0]
        self.q, self.k, self.v = [torch.randn(2, 2, 9, 8, dtype=torch.float64,
                                             requires_grad=True) for _ in range(3)]

    def test_complex_outer_and_direct_phase_difference(self):
        inner, layer = self.inner, self.layer
        read, g, _, _ = inner.memory_step(layer, self.q, self.k, self.v)
        qr, kr = [inner.apply_rope(x, layer) for x in (self.q, self.k)]
        pk, pv = inner.phases(layer)
        z_k = kr * torch.exp(1j * pk[None, :, None, :])
        z_v = self.v * torch.exp(1j * pv[None, :, None, :])
        expected = (z_v.transpose(-1, -2) @ z_k.conj()).imag / 9
        delta = pv[:, :, None] - pk[:, None, :]
        direct = torch.einsum('bhti,bhtj,hij->bhij', self.v, kr, delta.sin()) / 9
        torch.testing.assert_close(g, expected)
        torch.testing.assert_close(g, direct)
        torch.testing.assert_close(read, qr @ expected.transpose(-1, -2))
        for carrier in (0.3, 13.7):
            rotation = torch.exp(torch.tensor(1j * carrier, dtype=torch.complex128))
            torch.testing.assert_close(expected, ((z_v*rotation).transpose(-1,-2)
                                                  @ (z_k*rotation).conj()).imag/9)
        read.square().mean().backward()
        for x in (self.q, self.k, self.v, layer.theta_k_raw, layer.theta_v_raw):
            self.assertTrue(x.grad.isfinite().all())
            self.assertGreater(x.grad.norm().item(), 0)

    def test_ignores_history_and_phase_parameters_do_not_advance(self):
        before = [x.detach().clone() for x in self.inner.phases(self.layer)]
        read, g, ek, ev = self.inner.memory_step(self.layer, self.q, self.k, self.v)
        other, matrix, _, _ = self.inner.memory_step(
            self.layer, self.q, self.k, self.v, torch.full_like(g, float('nan')),
            torch.full_like(self.k, float('nan')), torch.full_like(self.v, float('nan')))
        torch.testing.assert_close(read, other)
        torch.testing.assert_close(g, matrix)
        self.assertIsNone(ek); self.assertIsNone(ev)
        for a, b in zip(before, self.inner.phases(self.layer)):
            torch.testing.assert_close(a, b)

    def test_bounded_independent_angles_no_decay_and_reload(self):
        layer = self.layer
        self.assertIsNot(layer.theta_k_raw, layer.theta_v_raw)
        self.assertFalse(torch.equal(layer.theta_k_raw, layer.theta_v_raw))
        for name in ('theta_k_raw', 'theta_v_raw'):
            self.assertTrue(t._is_no_decay('inner.layers.0.' + name, getattr(layer, name)))
        with torch.no_grad():
            layer.theta_k_raw.fill_(20)
            layer.theta_v_raw.fill_(-20)
        for phase in self.inner.phases(layer):
            self.assertTrue((phase.abs() <= math.pi/2).all())
        copy = type(self.inner)(config()).double()
        copy.load_state_dict(self.inner.state_dict())
        torch.testing.assert_close(self.inner.memory_step(layer,self.q,self.k,self.v)[0],
                                   copy.memory_step(copy.layers[0],self.q,self.k,self.v)[0])

    def test_block_two_norms_and_optimizer_update(self):
        inner = PhaseCurrentReadInner(config())
        layer = inner.layers[0]
        h, inj = torch.randn(2,9,16), torch.randn(2,9,16)
        with patch.object(inner, 'phi', wraps=inner.phi) as norm:
            output, _, ek, ev = inner.block(layer,h,inj,None,None,None,None)
            self.assertEqual(norm.call_count, 2)
        optimizer = torch.optim.Adam(inner.parameters(), lr=1e-3)
        before = layer.theta_k_raw.detach().clone()
        output[...,0].square().mean().backward()
        optimizer.step()
        self.assertTrue(output.isfinite().all())
        self.assertFalse(torch.equal(before, layer.theta_k_raw))
        self.assertIsNone(ek); self.assertIsNone(ev)


if __name__ == '__main__':
    unittest.main()
