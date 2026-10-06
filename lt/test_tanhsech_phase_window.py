"""The fused tanh*sech pair kernel against direct FP64 pairs, and the window algebra."""
import unittest

import torch

from . import tanhsech_phase_window as ts


class TanhSechWindowTests(unittest.TestCase):
    def test_rational_form_and_shape(self):
        d = torch.linspace(-3.1, 3.1, 4001, dtype=torch.float64, requires_grad=True)
        r = d.exp()
        p = r.square()
        torch.testing.assert_close(ts.window(d), 2 * r * (p - 1) / (p + 1).square(), rtol=0, atol=1e-15)
        auto = torch.autograd.grad(ts.window(d).sum(), d)[0]
        torch.testing.assert_close(ts.derivative(d.detach()), auto, rtol=0, atol=1e-14)
        torch.testing.assert_close(ts.window(-d), -ts.window(d), rtol=0, atol=0)
        zero = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.assertEqual(float(torch.autograd.grad(ts.window(zero).sum(), zero)[0]), 1.0)
        self.assertTrue((ts.window(d.detach()[d > 0]) > 0).all())
        peak = torch.asinh(torch.ones((), dtype=torch.float64))
        self.assertAlmostEqual(float(ts.window(peak)), .5, places=14)

    def test_cpu_entry_is_direct(self):
        k, v, pk, pv = [torch.randn(2, 3, 7, 5, dtype=torch.float64) for _ in range(4)]
        torch.testing.assert_close(ts.write(k, v, pk, pv), ts.direct_write(k, v, pk, pv), rtol=0, atol=0)

    def test_window_scale_multiplies_write_and_read(self):
        from .experiment_free_phase_windows import model_class
        from .test_kv_stability import config
        torch.manual_seed(3)
        one = model_class('tanhsech', True, generator='diagonal')(config()).double()
        torch.manual_seed(3)
        two = model_class('tanhsech', True, generator='diagonal', window_scale=2.)(config()).double()
        with torch.no_grad():
            for m in (one, two):
                m.layers[0].phase_local_gain.normal_(0, .3)
        two.load_state_dict(one.state_dict())
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) for _ in range(3)]
        a = one.memory_step(one.layers[0], q, k, v)
        b = two.memory_step(two.layers[0], q, k, v)
        for x, y in zip(a[:2], b[:2]):
            torch.testing.assert_close(y, 2 * x, rtol=1e-13, atol=1e-13)
        d = torch.linspace(-2, 2, 9, dtype=torch.float64)
        torch.testing.assert_close(two.window(d), 2 * one.window(d), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
    def test_fused_kernel_value_and_all_gradients(self):
        torch.manual_seed(5)
        # Model shape, a ragged shape, and D > 128 (three channel tiles).
        for shape in ((2, 3, 81, 104), (1, 2, 9, 17), (1, 1, 5, 130)):
            raw = [torch.randn(shape, device='cuda', dtype=torch.float64) for _ in range(2)]
            raw += [(torch.rand(shape, device='cuda', dtype=torch.float64) - .5) * torch.pi * .999 for _ in range(2)]
            fp32 = [x.float().requires_grad_() for x in raw]
            fp64 = [x.detach().double().requires_grad_() for x in fp32]
            out = ts.fused_write(*fp32)
            ref = ts.direct_write(*fp64)
            cot = torch.randn_like(ref)
            grads = torch.autograd.grad((out * cot.float()).sum(), fp32)
            refs = torch.autograd.grad((ref * cot).sum(), fp64)
            rel = lambda a, b: float((a.double() - b).norm() / b.norm())
            self.assertLess(rel(out.detach(), ref.detach()), 2e-6, shape)
            for g, r in zip(grads, refs):
                self.assertLess(rel(g, r), 2e-6, shape)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
    def test_compiled_matches_eager(self):
        torch.manual_seed(6)
        args = [torch.randn(2, 2, 81, 104, device='cuda') for _ in range(2)]
        args += [(torch.rand(2, 2, 81, 104, device='cuda') - .5) * 3 for _ in range(2)]
        eager = [x.clone().requires_grad_() for x in args]
        comp = [x.clone().requires_grad_() for x in args]
        fn = torch.compile(lambda *a: ts.fused_write(*a).square().sum(), fullgraph=True)
        a = torch.autograd.grad(ts.fused_write(*eager).square().sum(), eager)
        b = torch.autograd.grad(fn(*comp), comp)
        for x, y in zip(a, b):
            torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
