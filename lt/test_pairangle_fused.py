"""Fused pairangle attention kernel: forward/backward against the FP64 atan2 reference,
strided (qkv-chunk) inputs, eager/compiled agreement, and the model's torch path."""
import unittest

import torch

from .pairangle_fused import (pairangle_attention, pairangle_attention_offset, pairangle_attention_traced,
                              reference_attention, reference_attention_traced)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
class PairangleFusedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.T, self.D = 2, 4, 81, 104
        self.coef = torch.tensor([2.8159, 0.2438], device='cuda')

    def inputs(self):
        B, H, T, D = self.B, self.H, self.T, self.D
        qkv = torch.randn(B, T, 3 * H * D, device='cuda')
        q, k, v = (x.reshape(B, T, H, D).transpose(1, 2) for x in qkv.chunk(3, -1))
        angles = torch.randn(H, T, D // 2, device='cuda') * 3
        alpha = torch.tensor([1.0, 1.3, -0.7, 0.2], device='cuda')
        return q, k, v * 2, angles, alpha

    def test_matches_fp64_reference(self):
        q, k, v, angles, alpha = self.inputs()
        dr = torch.randn(self.B, self.H, self.T, self.D, device='cuda')
        rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
        for hebbian in (True, False):
            for rotated in (False, True):
                ref = [x.detach().double().requires_grad_() for x in (q, k, v, angles, alpha)]
                r_ref, g_ref = reference_attention(*ref, self.coef.double(), 1e-4, hebbian, rotated)
                (r_ref * dr.double()).sum().backward()
                for bf16, tol in ((False, 2e-6), (True, 2e-2)):
                    x = [t.detach().clone().requires_grad_() for t in (q, k, v, angles, alpha)]
                    r, g = pairangle_attention(*x, self.coef, 1e-4, hebbian, bf16, rotated)
                    (r * dr).sum().backward()
                    with self.subTest(hebbian=hebbian, rotated=rotated, bf16=bf16):
                        self.assertLess(rel(r, r_ref), tol)
                        self.assertLess(rel(g, g_ref), tol)
                        for a, b in zip(x, ref):
                            self.assertLess(rel(a.grad, b.grad), tol)

    def test_compiled_matches_eager(self):
        q, k, v, angles, alpha = (t.requires_grad_() for t in self.inputs())
        def f(q, k, v, angles, alpha):
            r, _ = pairangle_attention(q, k, v, angles, alpha, self.coef, 1e-4, True, True)
            return r.square().sum()
        grads = []
        for fn in (f, torch.compile(f)):
            out = fn(q, k, v, angles, alpha)
            grads.append(torch.autograd.grad(out, (q, k, v, angles, alpha)))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_model_torch_and_triton_paths_agree(self):
        from .experiment_free_phase_windows import model_class
        from .test_kv_stability import config
        kw = dict(modes=2, qk_l2=True, write_sum=True, phase_frame='unrotated', dc_hebbian=True, dc_alpha_init=1.0)
        models = []
        for kernel in ('torch', 'triton'):
            torch.manual_seed(7)
            models.append(model_class('pairangle', kernel=kernel, **kw)(config()).cuda())
        layer_t, layer_f = models[0].layers[0], models[1].layers[0]
        m = models[0]
        q, k, v = (torch.randn(2, m.H, m.pos_u.numel(), m.dh, device='cuda') for _ in range(3))
        with torch.no_grad():
            layer_t.stdp_alpha.copy_(torch.linspace(-1, 2, m.H))
            layer_f.stdp_alpha.copy_(torch.linspace(-1, 2, m.H))
        ref = models[0].memory_step(layer_t, q, k, v)[:2]
        out = models[1].memory_step(layer_f, q, k, v)[:2]
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


    def traces(self):
        B, H, T, D = self.B, self.H, self.T, self.D
        return torch.randn(B, H, T, D, device='cuda') * .6, torch.randn(B, H, T, D, device='cuda') * .6

    def test_traced_matches_fp64_reference(self):
        q, k, v, angles, alpha = self.inputs()
        zk, zv = self.traces()
        coef = self.coef[:1]
        dr = torch.randn(self.B, self.H, self.T, self.D, device='cuda')
        rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
        for hebbian in (True, False):
            ref = [x.detach().double().requires_grad_() for x in (q, k, v, zk, zv, angles, alpha)]
            r_ref, g_ref = reference_attention_traced(*ref, coef.double(), 1e-4, hebbian)
            (r_ref * dr.double()).sum().backward()
            for bf16, tol in ((False, 2e-6), (True, 2e-2)):
                x = [t.detach().clone().requires_grad_() for t in (q, k, v, zk, zv, angles, alpha)]
                r, g = pairangle_attention_traced(*x, coef, 1e-4, hebbian, bf16)
                (r * dr).sum().backward()
                with self.subTest(hebbian=hebbian, bf16=bf16):
                    self.assertLess(rel(r, r_ref), tol)
                    self.assertLess(rel(g, g_ref), tol)
                    for a, b in zip(x, ref):
                        self.assertLess(rel(a.grad, b.grad), tol)

    def test_traced_compiled_matches_eager(self):
        q, k, v, angles, alpha = (t.requires_grad_() for t in self.inputs())
        zk, zv = (t.requires_grad_() for t in self.traces())
        def f(q, k, v, zk, zv, angles, alpha):
            r, _ = pairangle_attention_traced(q, k, v, zk, zv, angles, alpha, self.coef[:1], 1e-4, True, True)
            return r.square().sum()
        grads = []
        for fn in (f, torch.compile(f)):
            out = fn(q, k, v, zk, zv, angles, alpha)
            grads.append(torch.autograd.grad(out, (q, k, v, zk, zv, angles, alpha)))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_model_traced_torch_and_triton_paths_agree(self):
        from .experiment_free_phase_windows import model_class
        from .test_kv_stability import config
        kw = dict(modes=1, qk_l2=True, write_sum=True, phase_frame='unrotated', dc_hebbian=True, dc_alpha_init=1.0,
                  phase_trace_eta=0.3)
        models = []
        for kernel in ('torch', 'triton'):
            torch.manual_seed(7)
            models.append(model_class('pairangle', kernel=kernel, **kw)(config()).cuda())
        m = models[0]
        q, k, v = (torch.randn(2, m.H, m.pos_u.numel(), m.dh, device='cuda') for _ in range(3))
        ek, ev = (torch.randn(2, m.H, m.pos_u.numel(), m.dh, device='cuda') * .5 for _ in range(2))
        fresh = torch.tensor([True, False], device='cuda')
        for model in models:
            with torch.no_grad():
                model.layers[0].stdp_alpha.copy_(torch.linspace(-1, 2, m.H))
        outs, grads = [], []
        cot = torch.randn(q.shape, device='cuda')       # one logical cotangent for both read layouts
        for model in models:
            args = [t.clone().requires_grad_() for t in (q, k, v, ek, ev)]
            read, current, zk, zv = model.memory_step(model.layers[0], *args[:3], None, *args[3:], fresh)
            outs.append((read, current, zk, zv))
            loss = (read * cot).sum() + zk.square().sum() + zv.sum()
            grads.append(torch.autograd.grad(loss, args + [model.layers[0].stdp_alpha]))
        for a, b in zip(outs[1], outs[0]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        for a, b in zip(grads[1], grads[0]):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


    def test_offset_matches_fp64_reference(self):
        q, k, v, angles, alpha = self.inputs()
        lam = torch.randn(self.B, self.H, self.T, self.D // 2, device='cuda') * 2
        dr = torch.randn(self.B, self.H, self.T, self.D, device='cuda')
        rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
        for hebbian in (True, False):
            for rotated in (False, True):
                ref = [x.detach().double().requires_grad_() for x in (q, k, v, angles, alpha, lam)]
                r_ref, g_ref = reference_attention(*ref[:5], self.coef.double(), 1e-4, hebbian, rotated, lam=ref[5])
                (r_ref * dr.double()).sum().backward()
                for bf16, tol in ((False, 2e-6), (True, 2e-2)):
                    x = [t.detach().clone().requires_grad_() for t in (q, k, v, lam, angles, alpha)]
                    r, g = pairangle_attention_offset(*x, self.coef, 1e-4, hebbian, bf16, rotated)
                    (r * dr).sum().backward()
                    with self.subTest(hebbian=hebbian, rotated=rotated, bf16=bf16):
                        self.assertLess(rel(r, r_ref), tol)
                        self.assertLess(rel(g, g_ref), tol)
                        order = (0, 1, 2, 5, 3, 4)                  # x is (q, k, v, lam, angles, alpha)
                        for a, j in zip(x, order):
                            self.assertLess(rel(a.grad, ref[j].grad), tol)
        # zero offset is the plain kernel
        x = [t.detach() for t in (q, k, v, angles, alpha)]
        r0, g0 = pairangle_attention_offset(x[0], x[1], x[2], torch.zeros_like(lam), *x[3:], self.coef, 1e-4)
        r1, g1 = pairangle_attention(*x, self.coef, 1e-4)
        torch.testing.assert_close(r0, r1, rtol=0, atol=0)
        torch.testing.assert_close(g0, g1, rtol=0, atol=0)

    def test_offset_compiled_matches_eager(self):
        q, k, v, angles, alpha = (t.requires_grad_() for t in self.inputs())
        lam = (torch.randn(self.B, self.H, self.T, self.D // 2, device='cuda')).requires_grad_()
        def f(q, k, v, lam, angles, alpha):
            r, _ = pairangle_attention_offset(q, k, v, lam, angles, alpha, self.coef[:1], 1e-4, True, True)
            return r.square().sum()
        grads = []
        for fn in (f, torch.compile(f)):
            out = fn(q, k, v, lam, angles, alpha)
            grads.append(torch.autograd.grad(out, (q, k, v, lam, angles, alpha)))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
