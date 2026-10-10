"""Fused pairangle attention kernel: forward/backward against the FP64 atan2 reference,
strided (qkv-chunk) inputs, eager/compiled agreement, and the model's torch path."""
import unittest

import torch

from .pairangle_fused import pairangle_attention, reference_attention


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


if __name__ == '__main__':
    unittest.main()
