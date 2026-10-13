"""Fused phase-clock kernel: forward/backward against the FP64 torch reference, strided inputs,
upstream trace gradients, eager/compiled agreement, and the research model's torch vs Triton paths."""
import unittest

import torch

from .phase_clock_fused import phase_clock_attention, reference_phase_clock


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
class PhaseClockFusedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.T, self.D = 2, 4, 81, 104

    def inputs(self, dtype=torch.float32):
        B, H, T, D = self.B, self.H, self.T, self.D
        qkv = torch.randn(B, T, 3 * H * D, device='cuda', dtype=dtype)
        q, k, v = (x.reshape(B, T, H, D).transpose(1, 2) for x in qkv.chunk(3, -1))
        ek, ev = (torch.randn(B, H, T, 2 * D, device='cuda') * .6 for _ in range(2))
        phk, phv = (torch.rand(B, H, T, D // 2, device='cuda') * 6.2 - 3.1 for _ in range(2))
        lk, lv = (torch.rand(B, H, T, D // 2, device='cuda') * .9 + .05 for _ in range(2))
        angles = torch.randn(H, T, D // 2, device='cuda') * 3
        alpha = torch.tensor([1.0, 1.3, -0.7, 0.2], device='cuda')
        return [q, k, v * 2, ek, ev, phk, phv, lk, lv, angles, alpha]

    def cotangents(self):
        B, H, T, D = self.B, self.H, self.T, self.D
        return (torch.randn(B, H, T, D, device='cuda'), torch.randn(B, H, T, 2 * D, device='cuda'),
                torch.randn(B, H, T, 2 * D, device='cuda'))

    def test_matches_fp64_reference(self):
        x0 = self.inputs()
        dr, deko, devo = self.cotangents()
        rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
        ref = [x.detach().double().requires_grad_() for x in x0]
        outs_ref = reference_phase_clock(*ref, 1e-4)
        grads_ref = torch.autograd.grad((outs_ref[0] * dr.double()).sum(), ref, retain_graph=True)
        grads_ref_traced = torch.autograd.grad((outs_ref[0] * dr.double()).sum() + (outs_ref[2] * deko.double()).sum()
                                               + (outs_ref[3] * devo.double()).sum(), ref)
        for bf16, tol in ((False, 3e-6), (True, 2e-2)):
            for traced in (False, True):
                x = [t.detach().clone().requires_grad_() for t in x0]
                outs = phase_clock_attention(*x, 1e-4, bf16)
                loss = (outs[0] * dr).sum()
                if traced:
                    loss = loss + (outs[2] * deko).sum() + (outs[3] * devo).sum()
                loss.backward()
                with self.subTest(bf16=bf16, traced=traced):
                    for a, b in zip(outs, outs_ref):
                        self.assertLess(rel(a, b), tol)
                    expected = grads_ref_traced if traced else grads_ref
                    for a, b, name in zip(x, expected, 'q k v ek ev phk phv lk lv angles alpha'.split()):
                        self.assertLess(rel(a.grad, b), tol, name)

    def test_bf16_inputs_and_zero_retention(self):
        # BF16 projections (as under autocast) and lam = 0 / 1 corners; compare with the FP64 reference.
        x = self.inputs(torch.bfloat16)
        x[7] = torch.where(torch.rand_like(x[7]) < .5, torch.zeros_like(x[7]), torch.ones_like(x[7]))
        x[8] = torch.where(torch.rand_like(x[8]) < .5, torch.zeros_like(x[8]), torch.ones_like(x[8]))
        ref = reference_phase_clock(*(t.double() for t in x), 1e-4)
        outs = phase_clock_attention(*x, 1e-4, False)
        for a, b in zip(outs, ref):
            self.assertLess(((a.double() - b).norm() / b.norm()).item(), 3e-6)
        self.assertEqual(outs[0].dtype, torch.float32)

    def test_compiled_matches_eager(self):
        x = [t.requires_grad_() for t in self.inputs()]
        dr, deko, devo = self.cotangents()

        def f(*args):
            r, _, eko, evo = phase_clock_attention(*args, 1e-4, True)
            return (r * dr).sum() + (eko * deko).sum() + (evo * devo).sum()
        grads = []
        for fn in (f, torch.compile(f)):
            grads.append(torch.autograd.grad(fn(*x), x))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_model_torch_and_triton_paths_agree(self):
        for clock in ('integrate', 'leaky', 'displacement'):
            with self.subTest(clock=clock):
                self._model_paths_agree(clock)

    def _model_paths_agree(self, clock):
        from .research_phase_clock_stdp import model_class, _activate_clock
        from .test_kv_stability import config
        models = []
        for kernel in ('torch', 'triton'):
            torch.manual_seed(7)
            model = model_class(kernel=kernel, feature_precision='float32', clock=clock)(config())
            _activate_clock(model, torch.Generator().manual_seed(3))
            models.append(model.cuda())
        m = models[0]
        B, T = 2, m.pos_u.numel()
        hidden = torch.randn(B, T, m.d, device='cuda')
        q, k, v = (torch.randn(B, m.H, T, m.dh, device='cuda') for _ in range(3))
        phases = torch.rand(B, m.H, T, m.clock_width, device='cuda') * 6 - 3
        ek, ev = (torch.randn(B, m.H, T, 2 * m.dh, device='cuda') * .5 for _ in range(2))
        fresh = torch.tensor([True, False], device='cuda')
        cot = torch.randn(q.shape, device='cuda')
        outs, grads = [], []
        for model in models:
            args = [t.clone().requires_grad_() for t in (q, k, v, phases, ek, ev, hidden)]
            read, phi, zk, zv = model.memory_step(model.layers[0], *args[:3], *args[3:6], fresh, hidden=args[6])
            outs.append((read, phi, zk, zv))
            loss = (read * cot).sum() + phi.sin().sum() + zk.square().sum() + zv.sum()
            names = ['stdp_alpha', 'theta', 'clock_lam_raw']
            if clock != 'displacement':
                names += ['clock_freq.weight', 'clock_dt.weight']
            if clock == 'leaky':
                names += ['clock_omega_raw', 'clock_psi_lam_raw']
            params = [model.layers[0].get_parameter(n) for n in names]
            grads.append(torch.autograd.grad(loss, args + params, allow_unused=True))   # hidden is unused by the displacement clock
        for a, b in zip(outs[1], outs[0]):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
        for a, b in zip(grads[1], grads[0]):
            self.assertEqual(a is None, b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
