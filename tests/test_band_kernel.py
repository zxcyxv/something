"""Fused band step (lt/band_kernel.py): forward/backward against the FP64 torch reference (strided k~/v views, several
shapes), no upstream trace gradients and traces without gradient, compiled vs eager, and band_product's kernel path
against its torch path in the model. CUDA only."""
import unittest

import torch

from lt.band_kernel import band_product, reference_band_product
from lt.model import LT, ModelConfig

rel = lambda a, b: float((a.detach().double() - b.detach().double()).norm() / b.detach().double().norm())


def inputs(B=3, H=4, T=81, D=104, R=4, seed=0):
    g = torch.Generator('cuda').manual_seed(seed)
    kr = torch.randn(B, T, H, D, device='cuda', generator=g).transpose(1, 2)     # strided, as from the projections
    kr = kr / kr.norm(dim=-1, keepdim=True)
    v = torch.randn(B, T, H, D, device='cuda', generator=g).transpose(1, 2) * 2
    ek, ev = (torch.randn(B, H, T, 2 * R * D, device='cuda', generator=g) * .5 for _ in range(2))
    mag = torch.rand(H, R, device='cuda', generator=g) * .9 + .05
    theta = torch.randn(H, R, device='cuda', generator=g) * 2
    c = torch.randn(2, H, R, device='cuda', generator=g)
    gamma = (1 - mag ** 2).sqrt()
    return [kr, v, ek, ev, torch.stack((mag * theta.cos(), mag * theta.sin(), gamma, gamma * c[0], gamma * c[1]))]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA kernel')
class BandKernel(unittest.TestCase):
    def test_matches_fp64_reference(self):
        """M, the new traces and every input gradient, with and without upstream gradients on the new traces."""
        for shape in ((3, 4, 81, 104, 4), (2, 2, 33, 64, 2), (2, 3, 17, 16, 3)):
            x0 = inputs(*shape)
            B, H, T, D, R = shape
            cot = [torch.randn(B, H, D, D, device='cuda'), torch.randn_like(x0[2]), torch.randn_like(x0[3])]
            ref = [t.detach().double().requires_grad_() for t in x0]
            outs_ref = reference_band_product(*ref)
            for traced in (False, True):
                n = 3 if traced else 1
                g_ref = torch.autograd.grad(sum((o * c.double()).sum() for o, c in zip(outs_ref[:n], cot)), ref,
                                            retain_graph=True)
                x = [t.detach().clone().requires_grad_() for t in x0]
                outs = band_product(*x)
                sum((o * c).sum() for o, c in zip(outs[:n], cot)).backward()
                with self.subTest(shape=shape, traced=traced):
                    for a, b, name in zip(outs, outs_ref, ('M', 'E_K', 'E_V')):
                        self.assertLess(rel(a, b), 3e-6, name)
                    for t, b, name in zip(x, g_ref, ('kr', 'v', 'ek', 'ev', 'coef')):
                        self.assertLess(rel(t.grad, b), 3e-6, name)

    def test_traces_without_gradient(self):
        """First block of a segment: the carried traces need no gradient and only M is used; the other gradients
        equal those of the full computation."""
        x0 = inputs()
        w = torch.randn(3, 4, 104, 104, device='cuda')
        full = [t.detach().clone().requires_grad_() for t in x0]
        (band_product(*full)[0] * w).sum().backward()
        part = [t.detach().clone().requires_grad_(i not in (2, 3)) for i, t in enumerate(x0)]
        (band_product(*part)[0] * w).sum().backward()
        for i in (0, 1, 4):
            torch.testing.assert_close(part[i].grad, full[i].grad, rtol=0, atol=0)
        self.assertIsNone(part[2].grad)

    def test_compiled_matches_eager(self):
        x0 = inputs(seed=1)
        cot = [torch.randn(3, 4, 104, 104, device='cuda'), torch.randn_like(x0[2]), torch.randn_like(x0[3])]

        def f(*args):
            return sum((o * c).sum() for o, c in zip(band_product(*args), cot))
        grads = []
        for fn in (f, torch.compile(f, dynamic=False)):
            x = [t.detach().clone().requires_grad_() for t in x0]
            grads.append(torch.autograd.grad(fn(*x), x))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def test_model_kernel_matches_torch_path(self):
        """band_product in FP32 on CUDA: kernel 'triton' (fused band step) against kernel 'torch' (TraceProduct), same
        weights: memory reads and traces over 6 blocks, then logits and every gradient through two segments."""
        def model(kernel, hebb=True):
            torch.manual_seed(0)
            m = LT(ModelConfig(batch_size=2, hidden_size=416, num_heads=4, puzzle_emb_ndim=416, blocks_per_seg=3,
                               loops=4, amp=False, bands=4, tag_bound=0.0, band_product=True, hebb=hebb,
                               kernel=kernel)).cuda()
            randn = lambda s, seed: torch.randn(s, generator=torch.Generator().manual_seed(seed)).cuda()
            with torch.no_grad():                     # O(1) c, moved poles, a nonzero FFN
                m.block.band_c.copy_(randn(m.block.band_c.shape, 1))
                m.block.band_pole.add_(randn(m.block.band_pole.shape, 2) * .3)
                m.block.ffn_down.weight.copy_(randn(m.block.ffn_down.weight.shape, 3) * .02)
            return m
        for hebb in (True, False):
            mk, mt = model('triton', hebb), model('torch', hebb)
            self.assertTrue(mk.use_band_kernel(torch.device('cuda'), torch.float32))
            self.assertFalse(mt.use_band_kernel(torch.device('cuda'), torch.float32))
            sk = mk.initial_state(2, torch.device('cuda'))
            st = mt.initial_state(2, torch.device('cuda'))
            g = torch.Generator('cuda').manual_seed(3)
            with torch.no_grad():
                for _ in range(6):
                    q, k, v = (torch.randn(2, 4, 81, 104, device='cuda', generator=g) for _ in range(3))
                    rk, sk = mk.memory_step(q, k, v, sk)
                    rt, st = mt.memory_step(q, k, v, st)
                    self.assertLess(rel(rk, rt), 1e-5)
                    self.assertLess(rel(sk[1], st[1]), 1e-6)
                    self.assertLess(rel(sk[2], st[2]), 1e-6)
            gb = torch.Generator().manual_seed(4)
            batch = dict(inputs=torch.randint(1, 11, (2, 81), generator=gb).cuda(),
                         labels=torch.randint(2, 11, (2, 81), generator=gb).cuda(),
                         puzzle_identifiers=torch.zeros(2, dtype=torch.int32).cuda())
            outs = []
            for m in (mk, mt):
                carry = m.initial_carry(batch)
                for _ in range(2):
                    carry, logits = m(carry, batch)
                logits.square().mean().backward()
                outs.append((logits.detach(), {n: p.grad for n, p in m.named_parameters()}))
            with self.subTest(hebb=hebb):
                self.assertLess(rel(outs[0][0], outs[1][0]), 1e-5)
                for n, gt in outs[1][1].items():
                    self.assertLess(rel(outs[0][1][n], gt), 1e-4, n)


if __name__ == '__main__':
    unittest.main()
