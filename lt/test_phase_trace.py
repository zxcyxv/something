"""Block-EMA phase traces for the pairangle timing term (2026-10-12).

eta = 1 reproduces the untraced write exactly; the traces follow z~ <- (1 - eta) z~ + eta u,
reset to u for fresh puzzles; the write equals the direct pair sum with |z~|-weighted harmonics
of the traced angles; gradients reach q, k, v and the incoming traces.
"""
import unittest

import torch

from .experiment_free_phase_windows import model_class
from .test_kv_stability import config


def build(eta, modes=2, frame='unrotated', seed=5):
    torch.manual_seed(seed)
    inner = model_class('pairangle', modes=modes, qk_l2=True, write_sum=True, phase_frame=frame,
                        dc_hebbian=True, dc_alpha_init=1.0, phase_trace_eta=eta)(config()).double()
    with torch.no_grad():
        inner.layers[0].stdp_alpha.copy_(torch.tensor([0.7, 1.3], dtype=torch.float64))
    return inner


def unit_pairs(x):
    pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a = torch.atan2(pair[..., 1], pair[..., 0])
    return torch.stack((a.cos(), a.sin()), -1).reshape(x.shape)


class PhaseTraceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.q, self.k, self.v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]

    def test_eta_one_equals_untraced_write(self):
        for frame in ('unrotated', 'rotated'):
            traced, plain = build(1.0, frame=frame), build(None, frame=frame)
            for name, value in plain.state_dict().items():
                torch.testing.assert_close(traced.state_dict()[name], value, rtol=0, atol=0)
            args = [z.clone().requires_grad_(True) for z in (self.q, self.k, self.v)]
            old = [torch.randn_like(self.k), torch.randn_like(self.v)]
            a = traced.memory_step(traced.layers[0], *args, None, *old, None)
            b = plain.memory_step(plain.layers[0], *args)
            for x, y in zip(a[:2], b[:2]):
                torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-11)
            cot = torch.randn_like(a[0])
            ga = torch.autograd.grad((a[0] * cot).sum(), args + [traced.layers[0].stdp_alpha])
            gb = torch.autograd.grad((b[0] * cot).sum(), args + [plain.layers[0].stdp_alpha])
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-9)

    def test_trace_update_and_fresh_reset(self):
        eta = 0.3
        inner = build(eta)
        layer = inner.layers[0]
        k2, v2 = torch.randn_like(self.k), torch.randn_like(self.v)
        unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
        _, _, zk1, zv1 = inner.memory_step(layer, self.q, self.k, self.v)
        torch.testing.assert_close(zk1, unit_pairs(unit(self.k)), rtol=0, atol=1e-14)
        torch.testing.assert_close(zv1, unit_pairs(self.v), rtol=0, atol=1e-14)
        fresh = torch.tensor([True, False])
        _, _, zk2, zv2 = inner.memory_step(layer, self.q, k2, v2, None, zk1, zv1, fresh)
        uk2, uv2 = unit_pairs(unit(k2)), unit_pairs(v2)
        torch.testing.assert_close(zk2[0], uk2[0], rtol=0, atol=1e-14)
        torch.testing.assert_close(zv2[0], uv2[0], rtol=0, atol=1e-14)
        torch.testing.assert_close(zk2[1], (1 - eta) * zk1[1] + eta * uk2[1], rtol=0, atol=1e-14)
        torch.testing.assert_close(zv2[1], (1 - eta) * zv1[1] + eta * uv2[1], rtol=0, atol=1e-14)

    def test_write_matches_direct_pair_sum(self):
        eta = 0.4
        inner = build(eta, modes=2)
        layer = inner.layers[0]
        old_k, old_v = torch.randn_like(self.k) * .5, torch.randn_like(self.v) * .5
        read, G, zk, zv = inner.memory_step(layer, self.q, self.k, self.v, None, old_k, old_v, None)
        unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
        qr, kr = (inner.apply_rope(unit(z), layer) for z in (self.q, self.k))
        P = inner.dh // 2
        def polar(z):
            pair = z.reshape(*z.shape[:-1], P, 2)
            return pair.norm(dim=-1).repeat_interleave(2, -1), torch.atan2(pair[..., 1], pair[..., 0]).repeat_interleave(2, -1)
        rk, ak = polar(zk)
        rv, av = polar(zv)
        c = inner.window_coefficients.double()
        delta = av[..., :, None] - ak[..., None, :]
        W = sum(c[r - 1] * torch.sin(r * delta) for r in range(1, 3)) * rv[..., :, None] * rk[..., None, :]
        alpha = layer.stdp_alpha[None, :, None, None, None]
        G_ref = (self.v[..., :, None] * kr[..., None, :] * (1 + alpha * W)).sum(-3)
        torch.testing.assert_close(G, G_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(read, qr @ G_ref.transpose(-1, -2), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(zk, (1 - eta) * old_k + eta * unit_pairs(unit(self.k)), rtol=0, atol=1e-14)

    def test_gradients_reach_activity_and_incoming_traces(self):
        inner = build(0.25, modes=2)
        layer = inner.layers[0]
        old = [torch.randn_like(self.k) * .5, torch.randn_like(self.v) * .5]
        def f(q, k, v, ek, ev):
            read, _, zk, zv = inner.memory_step(layer, q, k, v, None, ek, ev, None)
            return read, zk, zv
        args = tuple(z.clone().requires_grad_(True) for z in (self.q, self.k, self.v, *old))
        self.assertTrue(torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-6))

    def test_learned_rate_starts_at_fixed_rate_and_trains(self):
        from . import train as t
        torch.manual_seed(5)
        learned = model_class('pairangle', modes=2, qk_l2=True, write_sum=True, phase_frame='unrotated',
                              dc_hebbian=True, dc_alpha_init=1.0, phase_trace_eta=0.3,
                              phase_trace_learn=True)(config()).double()
        fixed = build(0.3)
        with torch.no_grad():
            learned.layers[0].stdp_alpha.copy_(fixed.layers[0].stdp_alpha)
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(learned.state_dict()[name], value, rtol=0, atol=0)
        layer = learned.layers[0]
        self.assertEqual(layer.trace_k_eta_raw.shape, (learned.H, learned.dh // 2))
        old = [torch.randn_like(self.k) * .5, torch.randn_like(self.v) * .5]
        a = learned.memory_step(layer, self.q, self.k, self.v, None, *old, None)
        b = fixed.memory_step(fixed.layers[0], self.q, self.k, self.v, None, *old, None)
        for x, y in zip(a, b):   # the logit init is stored in FP32: eta = 0.3 up to ~1e-8
            torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)
        raws = (layer.trace_k_eta_raw, layer.trace_v_eta_raw)
        def f(rk, rv):
            saved = [layer.trace_k_eta_raw, layer.trace_v_eta_raw]
            del layer.trace_k_eta_raw, layer.trace_v_eta_raw
            layer.trace_k_eta_raw, layer.trace_v_eta_raw = rk, rv
            try:
                return learned.memory_step(layer, self.q, self.k, self.v, None, *old, None)[0]
            finally:
                del layer.trace_k_eta_raw, layer.trace_v_eta_raw
                layer.trace_k_eta_raw, layer.trace_v_eta_raw = saved
        args = tuple(r.detach().clone().requires_grad_(True) + .1 * torch.randn_like(r) for r in raws)
        self.assertTrue(torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-6))
        for name, p in learned.named_parameters():
            if 'trace_' in name:
                self.assertTrue(t._is_no_decay(name, p), name)

    def test_block_carries_traces(self):
        inner = build(0.2)
        layer = inner.layers[0]
        h = torch.randn(2, 9, 16, dtype=torch.float64)
        inj = torch.randn_like(h)
        h1, _, ek1, ev1 = inner.block(layer, h, inj, None, None, None, None)
        h2, _, ek2, ev2 = inner.block(layer, h1, inj, None, ek1, ev1, None)
        self.assertEqual(ek1.shape, (2, 2, 9, 8))
        self.assertFalse(torch.equal(ek1, ek2))
        self.assertTrue(torch.isfinite(h2).all())


if __name__ == '__main__':
    unittest.main()
