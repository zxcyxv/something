"""Fused kernel (lt/kernel.py): forward/backward against the FP64 torch reference, BF16 inputs and retention corners,
compiled vs eager, and the model's torch vs Triton paths. CUDA only."""
import unittest

import torch

from lt.kernel import phase_clock_attention, reference_phase_clock
from lt.model import LT, ModelConfig


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
class KernelTests(unittest.TestCase):
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

    def test_model_paths_agree(self):
        """One segment of the model through the torch path and the fused kernel (FP32 dots): logits, carry and
        parameter gradients agree; the second segment starts from a carried clock and traces."""
        torch.manual_seed(7)
        cfg = dict(batch_size=2, hidden_size=64, num_heads=4, puzzle_emb_ndim=64, blocks_per_seg=3, amp=False,
                   kernel_bf16=False)
        models = []
        for kernel in ('torch', 'triton'):
            torch.manual_seed(7)
            with torch.device('cuda'):
                models.append(LT(ModelConfig(kernel=kernel, **cfg)))
        with torch.no_grad():
            for p in models[0].parameters():
                p.add_(torch.randn_like(p) * 0.2)
        models[1].load_state_dict(models[0].state_dict())
        g = torch.Generator().manual_seed(0)
        batch = dict(inputs=torch.randint(1, 11, (2, 81), generator=g).cuda(),
                     labels=torch.randint(2, 11, (2, 81), generator=g).cuda(),
                     puzzle_identifiers=torch.zeros(2, dtype=torch.int32).cuda())
        outs, grads = [], []
        for m in models:
            carry = m.initial_carry(batch)
            carry, _ = m(carry, batch)
            carry, logits = m(carry, batch)
            logits.square().mean().backward()
            outs.append((logits, carry.clock, carry.key_trace, carry.value_trace))
            grads.append({n: p.grad for n, p in m.named_parameters()})
        for a, b, name in zip(*outs, ('logits', 'clock', 'key trace', 'value trace')):
            # The clock holds unit phasors of near-zero pairs, which amplify the FP32 dot differences of segment 1.
            tol = 1e-3 if name == 'clock' else 1e-4
            torch.testing.assert_close(a, b, rtol=tol, atol=tol, msg=name)
        for n in grads[0]:
            torch.testing.assert_close(grads[1][n], grads[0][n], rtol=1e-3, atol=1e-5, msg=n)


if __name__ == '__main__':
    unittest.main()
