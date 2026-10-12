"""Cumulative V-K phase offset (lt/cumulative_phase.py): omega = 0 is the pairangle model, the offset
integrates omega over blocks and resets for fresh puzzles, the write equals the direct pair sum with the
shifted lag, gradients reach omega's parameters and the incoming offset, and the torch and fused paths agree.
"""
import math
import unittest

import torch

from .cumulative_phase import offset_model_class
from .experiment_free_phase_windows import model_class
from .test_kv_stability import config


def build(mode='head', kernel='torch', seed=5, double=True):
    torch.manual_seed(seed)
    inner = offset_model_class(mode, math.pi, kernel, 'float32')(config())
    return inner.double() if double else inner


def set_omega(inner, scale=.3, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in inner.layers:
            layer.omega_proj.copy_(torch.randn(layer.omega_proj.shape, generator=g) * scale)
            layer.omega_bias.copy_(torch.randn(layer.omega_bias.shape, generator=g) * scale)
            layer.stdp_alpha.copy_(torch.tensor([0.7, 1.3]))


class CumulativePhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.q, self.k, self.v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]

    def test_zero_omega_is_the_pairangle_model(self):
        inner = build()
        torch.manual_seed(5)
        base = model_class('pairangle', modes=1, qk_l2=True, write_sum=True, tau_phi=2.0, phase_frame='unrotated',
                           dc_hebbian=True, dc_alpha_init=1.0, boundary_ffn='swiglu')(config()).double()
        for name, value in base.state_dict().items():
            torch.testing.assert_close(inner.state_dict()[name], value, rtol=0, atol=0)
        for model in (inner, base):
            with torch.no_grad():
                model.layers[0].stdp_alpha.copy_(torch.tensor([0.7, 1.3], dtype=torch.float64))
        old = (torch.rand(2, 2, 9, 1, dtype=torch.float64) * 2 - 1) * 3
        with torch.no_grad():
            inner.layers[0].omega_bias.zero_()
        a = inner.memory_step(inner.layers[0], self.q, self.k, self.v)
        b = base.memory_step(base.layers[0], self.q, self.k, self.v)
        for x, y in zip(a[:2], b[:2]):
            torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-11)
        # a carried offset with omega = 0 stays where it was
        _, _, off, _ = inner.memory_step(inner.layers[0], self.q, self.k, self.v, None, old, None, None)
        torch.testing.assert_close(off, old, rtol=0, atol=1e-14)

    def test_offset_integrates_omega_and_resets(self):
        inner = build()
        set_omega(inner)
        layer = inner.layers[0]
        k2, v2 = torch.randn_like(self.k), torch.randn_like(self.v)
        om1, lam1 = inner.phase_offset(layer, self.k, self.v, None, None)
        om2, lam2 = inner.phase_offset(layer, k2, v2, lam1, torch.tensor([True, False]))
        torch.testing.assert_close(lam1, om1, rtol=0, atol=1e-12)
        torch.testing.assert_close(lam2[0], om2[0], rtol=0, atol=1e-12)                  # fresh: lambda = omega
        wrapped = torch.remainder(om1[1] + om2[1] + math.pi, 2 * math.pi) - math.pi
        torch.testing.assert_close(lam2[1], wrapped, rtol=0, atol=1e-12)
        self.assertLess(float(om1.abs().max()), math.pi)

    def test_write_matches_direct_pair_sum(self):
        for mode in ('head', 'pair'):
            inner = build(mode)
            set_omega(inner)
            layer = inner.layers[0]
            old = (torch.rand(2, 2, 9, 1 if mode == 'head' else 4, dtype=torch.float64) * 2 - 1) * 3
            read, G, lam, _ = inner.memory_step(layer, self.q, self.k, self.v, None, old, None, None)
            unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
            qr, kr = (inner.apply_rope(unit(z), layer) for z in (self.q, self.k))
            P = inner.dh // 2
            ang = lambda z: torch.atan2(z.reshape(*z.shape[:-1], P, 2)[..., 1],
                                        z.reshape(*z.shape[:-1], P, 2)[..., 0]).repeat_interleave(2, -1)
            lam = lam.expand(*lam.shape[:-1], P).repeat_interleave(2, -1)
            delta = (ang(self.v) + lam)[..., :, None] - ang(self.k)[..., None, :]
            alpha = layer.stdp_alpha[None, :, None, None, None]
            c = inner.window_coefficients.double()[0]
            G_ref = (self.v[..., :, None] * kr[..., None, :] * (1 + alpha * c * torch.sin(delta))).sum(-3)
            with self.subTest(mode=mode):
                torch.testing.assert_close(G, G_ref, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(read, qr @ G_ref.transpose(-1, -2), rtol=1e-10, atol=1e-10)

    def test_gradients_reach_omega_and_incoming_offset(self):
        inner = build('pair')
        set_omega(inner, .2)
        layer = inner.layers[0]
        old = (torch.rand(2, 2, 9, 4, dtype=torch.float64) * 2 - 1) * 1.5     # away from the wrap at +-pi
        def f(q, k, v, prev, w, c):
            saved = layer.omega_proj, layer.omega_bias
            del layer.omega_proj, layer.omega_bias
            layer.omega_proj, layer.omega_bias = w, c
            try:
                read, _, off, _ = inner.memory_step(layer, q, k, v, None, prev, None, None)
                return read, off
            finally:
                del layer.omega_proj, layer.omega_bias
                layer.omega_proj, layer.omega_bias = saved
        args = tuple(z.detach().clone().requires_grad_(True)
                     for z in (self.q, self.k, self.v, old, layer.omega_proj, layer.omega_bias))
        self.assertTrue(torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-6))

    def test_block_carries_the_offset(self):
        inner = build()
        set_omega(inner)
        layer = inner.layers[0]
        h = torch.randn(2, 9, 16, dtype=torch.float64)
        inj = torch.randn_like(h)
        h1, _, off1, ev = inner.block(layer, h, inj, None, None, None, None)
        h2, _, off2, _ = inner.block(layer, h1, inj, None, off1, None, None)
        self.assertIsNone(ev)
        self.assertEqual(off1.shape, (2, 2, 9, 1))
        self.assertFalse(torch.equal(off1, off2))
        self.assertTrue(torch.isfinite(h2).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
    def test_torch_and_fused_paths_agree(self):
        for mode in ('head', 'pair'):
            models = []
            for kernel in ('torch', 'triton'):
                m = build(mode, kernel, double=False).cuda()
                set_omega(m)
                models.append(m)
            m = models[0]
            q, k, v = (torch.randn(2, m.H, m.pos_u.numel(), m.dh, device='cuda') for _ in range(3))
            width = 1 if mode == 'head' else m.dh // 2
            old = (torch.rand(2, m.H, m.pos_u.numel(), width, device='cuda') * 2 - 1) * 1.5
            fresh = torch.tensor([True, False], device='cuda')
            cot = torch.randn(q.shape, device='cuda')
            outs, grads = [], []
            for model in models:
                layer = model.layers[0]
                args = [z.clone().requires_grad_() for z in (q, k, v, old)]
                read, cur, off, _ = model.memory_step(layer, *args[:3], None, args[3], None, fresh)
                outs.append((read, cur, off))
                loss = (read * cot).sum() + off.sum()
                grads.append(torch.autograd.grad(loss, args + [layer.omega_proj, layer.omega_bias, layer.stdp_alpha]))
            with self.subTest(mode=mode):
                for a, b in zip(outs[1], outs[0]):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                for a, b in zip(grads[1], grads[0]):
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
