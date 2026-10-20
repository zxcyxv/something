"""CPU/FP64 checks of the trace-product band memory (band_product): M = sum_r gamma_r Re(c_r P_r) with
P_r = sum_t E_V,r conj(E_K,r)^T is the per-band EMA of the bands model's S_r (a = conj c, b = c) at rate 1 - |z_r|^2,
equals the EMA bands model when every |z_r|^2 = 1 - eta, and is the pair double sum
M(n) = sum over the cell's pairs (m1, m2 <= n) of sum_r gamma_r^3 Re(c_r z_r^(n-m1) conj(z_r)^(n-m2)) v(m1) k~(m2)^T.
Run: python -m unittest discover tests"""
import math
import tempfile
import unittest

import torch

from lt.model import LT, ModelConfig, TraceProduct
from lt.packed import PackedLT
from lt.train import DEFAULTS, create_optimizers
from tests.test_bands import band_tiny, randn_qkv
from tests.test_model import batch, tiny
from tests.test_train import tiny_run

CPU = torch.device('cpu')


def product_tiny(**kw):
    return tiny(**dict(dict(bands=4, tag_bound=0.0, band_product=True), **kw))


def perturbed(**kw):
    """Product model with the poles moved off their initial periods and O(1) coefficients c."""
    torch.manual_seed(0)
    m = LT(product_tiny(**kw)).double().eval()
    with torch.no_grad():
        m.block.band_pole.add_(torch.randn_like(m.block.band_pole) * 0.3)
        m.block.band_c.normal_()
    return m


def poles(m):
    """Per head and band [H,R] (complex): z, gamma, c."""
    zr, zi, gamma, _, _, br, bi = (x.detach() for x in m.block.bands())
    return torch.complex(zr, zi), gamma, torch.complex(br, bi)


def rope_unit(m):
    """x -> RoPE(x / (||x|| + eps)): q~, k~ from the raw projections."""
    angles = m.rope_angles()
    tables = angles.cos()[None], angles.sin()[None]
    return lambda x: m.rope(x / (x.norm(dim=-1, keepdim=True) + m.config.eps), tables)


def split(t, R, D):
    """Trace slot [B,H,T,2RD] -> complex traces [B,H,T,R,D]."""
    B, H, T, _ = t.shape
    re, im = t.reshape(B, H, T, 2, R, D).unbind(3)
    return torch.complex(re, im)


def per_band(x):
    """[H,R] -> [1,H,1,R,1], broadcast over the traces [B,H,T,R,D]."""
    return x[None, :, None, :, None]


class BandProduct(unittest.TestCase):
    def test_per_band_ema_of_s(self):
        """Perturbed poles, random c: complex reference traces, S_r per band (a = conj c, b = c, D_V,r = z_r E_V,r(n-1))
        and M_ref,r <- |z_r|^2 M_ref,r + (1 - |z_r|^2) S_r give the read q~ (sum_t v k~^T + sum_r M_ref,r)^T and the
        state's traces; there is no W."""
        m = perturbed()
        z, gamma, c = poles(m)
        R, D, B = m.config.bands, m.dh, 2
        rk = rope_unit(m)
        z2 = (z.abs() ** 2)[None, :, :, None, None]
        ek = ev = torch.zeros(B, m.H, m.config.seq_len, R, D, dtype=torch.complex128)
        m_ref = torch.zeros(B, m.H, R, D, D, dtype=torch.float64)
        state = m.initial_state(B, CPU, torch.float64)
        with torch.no_grad():
            for q, k, v in randn_qkv(m, B, 10, 3):
                kr = rk(k)
                ek = per_band(z) * ek + per_band(gamma) * kr.unsqueeze(3)
                dv = per_band(z) * ev
                s_r = (torch.einsum('bhtd,bhtre->bhrde', v, (per_band(c.conj()) * ek).real)
                       + torch.einsum('bhtrd,bhte->bhrde', (per_band(c) * dv).real, kr))
                ev = dv + per_band(gamma) * v.unsqueeze(3)
                m_ref = z2 * m_ref + (1 - z2) * s_r
                read, state = m.memory_step(q, k, v, state)
                torch.testing.assert_close(read, rk(q) @ (v.transpose(-1, -2) @ kr + m_ref.sum(2)).transpose(-1, -2),
                                           rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(split(state[1], R, D), ek, rtol=1e-12, atol=1e-13)
                torch.testing.assert_close(split(state[2], R, D), ev, rtol=1e-12, atol=1e-13)
                self.assertIsNone(state[3])

    def test_equals_ema_bands_when_poles_match_eta(self):
        """Every |z_r|^2 = 1 - eta (nu = log(-log(1 - eta) / 2)), random theta and c, and a = conj(c), b = c in the EMA
        bands model: the same read every block, and its W equals M = sum_r gamma_r Re(c_r P_r) of the product's traces."""
        eta = 0.125
        torch.manual_seed(0)
        ref = LT(band_tiny(outer_eta=eta)).double().eval()
        torch.manual_seed(1)
        m = LT(product_tiny()).double().eval()
        H, R, D, B = m.H, m.config.bands, m.dh, 2
        g = torch.Generator().manual_seed(4)
        theta = torch.randn(H, R, generator=g, dtype=torch.float64) * 2
        c = torch.randn(2, H, R, generator=g, dtype=torch.float64)
        with torch.no_grad():
            m.block.rope_theta.copy_(ref.block.rope_theta)
            for x in (ref, m):
                x.block.band_pole[0].fill_(math.log(-math.log(1 - eta) / 2))
                x.block.band_pole[1].copy_(theta)
            ref.block.band_coef.copy_(torch.stack((c[0], -c[1], c[0], c[1])))
            m.block.band_c.copy_(c)
        z, gamma, cc = poles(m)
        torch.testing.assert_close(z.abs() ** 2, torch.full_like(gamma, 1 - eta))
        s_ref = ref.initial_state(B, CPU, torch.float64)
        s_new = m.initial_state(B, CPU, torch.float64)
        with torch.no_grad():
            for q, k, v in randn_qkv(m, B, 12, 5):
                r_ref, s_ref = ref.memory_step(q, k, v, s_ref)
                r_new, s_new = m.memory_step(q, k, v, s_new)
                torch.testing.assert_close(r_new, r_ref, rtol=1e-11, atol=1e-12)
                p = torch.einsum('bhtrd,bhtre->bhrde', split(s_new[2], R, D), split(s_new[1], R, D).conj())
                mem = (gamma[None, :, :, None, None] * (cc[None, :, :, None, None] * p).real).sum(2)
                torch.testing.assert_close(s_ref[3], mem, rtol=1e-11, atol=1e-12)

    def test_memory_is_pair_double_sum(self):
        self.check_pair_double_sum(hebb=True)

    def test_without_hebb_reads_m(self):
        """hebb=False: read = q~ M^T (no tonic term, nothing but M)."""
        self.check_pair_double_sum(hebb=False)

    def check_pair_double_sum(self, hebb):
        """M(n) = sum_{i, j <= n} sum_r gamma_r^3 Re(c_r z_r^(n-i) conj(z_r)^(n-j)) v(i) k~(j)^T over the cells (value
        of block i, key of block j), compared through the read."""
        m = perturbed(hebb=hebb)
        z, gamma, c = poles(m)
        rk = rope_unit(m)
        B, N = 2, 7
        seq = randn_qkv(m, B, N, 6)
        krs = [rk(k) for _, k, _ in seq]
        state = m.initial_state(B, CPU, torch.float64)
        with torch.no_grad():
            for n, (q, k, v) in enumerate(seq):
                mem = torch.zeros(B, m.H, m.dh, m.dh, dtype=torch.float64)
                for i in range(n + 1):
                    for j in range(n + 1):
                        w = (gamma ** 3 * (c * z ** (n - i) * z.conj() ** (n - j)).real).sum(-1)     # [H]
                        mem = mem + w[None, :, None, None] * (seq[i][2].transpose(-1, -2) @ krs[j])
                read, state = m.memory_step(q, k, v, state)
                tonic = v.transpose(-1, -2) @ krs[n] if hebb else torch.zeros_like(mem)
                torch.testing.assert_close(read, rk(q) @ (tonic + mem).transpose(-1, -2), rtol=1e-11, atol=1e-12)

    def test_summary_area_is_the_gain_of_m(self):
        """band_summary (a = conj c, b = c): the area is M's gain on a constant state, i.e. with the same q, k, v every
        block the read without the tonic term tends to area_h q~ (sum_t v k~^T)^T."""
        m = perturbed(hebb=False, band_periods=(1.5, 2.5, 3.0, 4.0))
        _, area, _, _ = m.block.band_summary()
        rk = rope_unit(m)
        (q, k, v), = randn_qkv(m, 1, 1, 7)
        state = m.initial_state(1, CPU, torch.float64)
        with torch.no_grad():
            for _ in range(300):
                read, state = m.memory_step(q, k, v, state)
        expected = area[None, :, None, None] * (rk(q) @ (v.transpose(-1, -2) @ rk(k)).transpose(-1, -2))
        torch.testing.assert_close(read, expected, rtol=1e-9, atol=1e-11)

    def test_parameters_and_gradients(self):
        """band_c [2,H,R] in band_coef's place, no eta_raw, alpha or W; every other initial weight equals the bands
        model's with the same seed; band_c and band_pole get gradients through two segments; no fused kernel."""
        torch.manual_seed(0)
        ref = LT(band_tiny())
        torch.manual_seed(0)
        m = LT(product_tiny())
        theirs, mine = dict(ref.named_parameters()), dict(m.named_parameters())
        self.assertEqual(set(mine) ^ set(theirs), {'block.band_c', 'block.band_coef'})
        for n, t in mine.items():
            if n in theirs:
                torch.testing.assert_close(t, theirs[n], rtol=0, atol=0, msg=n)
        torch.testing.assert_close(m.init_hidden, ref.init_hidden, rtol=0, atol=0)
        self.assertEqual(tuple(m.block.band_c.shape), (2, m.H, m.config.bands))
        self.assertGreater(float(m.block.band_c.detach().std()), 0.0)
        for n in ('band_coef', 'eta_raw', 'alpha'):
            self.assertFalse(hasattr(m.block, n), n)
        self.assertFalse(LT(product_tiny(kernel='auto')).use_kernel(torch.device('cuda'), torch.float32))
        m = m.double()
        bt = batch(0)
        carry = m.initial_carry(bt)
        for _ in range(2):
            carry, logits = m(carry, bt)
        logits.sum().backward()
        self.assertIsNone(carry.stdp_w)
        for n in ('band_pole', 'band_c'):
            g = getattr(m.block, n).grad
            self.assertIsNotNone(g, n)
            self.assertGreater(float(g.abs().sum()), 0.0, n)

    def test_config_checks(self):
        for kw in (dict(bands=0), dict(outer_eta=0.125), dict(outer_rho=0.5), dict(outer_eta=0.125, outer_eta_learn=True),
                   dict(outer_eta=0.125, outer_eta_learn=True, outer_eta_cap=True), dict(tag_bound=0.14),
                   dict(exp_stdp=True)):
            with self.assertRaises(ValueError, msg=str(kw)):
                product_tiny(**kw)
        for hebb in (True, False):
            cfg = ModelConfig.from_dict(dict(bands=2, band_periods=[2, 6], tag_bound=0.0, band_product=True, hebb=hebb,
                                             kernel='triton'))   # the fused band step (lt/band_kernel.py)
            self.assertTrue(cfg.band_product and not cfg.outer_memory)
        with self.assertRaises(ValueError):                  # without band_product the bands model needs outer_eta > 0
            ModelConfig.from_dict(dict(bands=2, band_periods=[2, 6], tag_bound=0.0))
        with self.assertRaises(ValueError):                  # no fused kernel for the EMA bands model
            band_tiny(kernel='triton')
        with self.assertRaises(ValueError):                  # the packed prototype implements the EMA bands only
            PackedLT(LT(product_tiny()))

    def test_band_kernel_selection(self):
        """The fused band step runs only for band_product on CUDA in FP32 with kernel auto or triton (no CUDA query
        here: CPU, FP64 and kernel 'torch' answer first)."""
        for kw, device, dtype in ((dict(), 'cpu', torch.float32), (dict(), 'cuda', torch.float64),
                                  (dict(kernel='torch'), 'cuda', torch.float32)):
            self.assertFalse(LT(product_tiny(**kw)).use_band_kernel(torch.device(device), dtype), str((kw, device)))
        self.assertFalse(LT(band_tiny(kernel='auto')).use_band_kernel(torch.device('cuda'), torch.float32))
        self.assertTrue(LT(product_tiny(kernel='triton')).use_band_kernel(torch.device('cuda'), torch.float32))

    def test_trace_product_backward(self):
        """TraceProduct's hand-written backward (dY = E_K dM^T, Y dM from E_V dM) equals autograd of the plain
        [D,2RT] @ [2RT,D] product, and its forward is that product bit for bit."""
        g = torch.Generator().manual_seed(8)
        B, H, T, R, D = 2, 3, 5, 4, 6
        ek, ev = (torch.randn(B, H, T, 2, R, D, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(2))
        gr, gi = (torch.randn(1, H, 1, R, 1, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(2))
        dm = torch.randn(B, H, D, D, generator=g, dtype=torch.float64)

        def plain(ek, ev, gr, gi):
            evr, evi = ev.unbind(3)
            y = torch.stack((gr * evr - gi * evi, gr * evi + gi * evr), 3)
            return y.reshape(B, H, -1, D).transpose(-1, -2) @ ek.reshape(B, H, -1, D)
        ref = plain(ek, ev, gr, gi)
        out = TraceProduct.apply(ek, ev, gr, gi)
        self.assertTrue(torch.equal(out, ref))
        for a, b in zip(torch.autograd.grad(out, (ek, ev, gr, gi), dm), torch.autograd.grad(ref, (ek, ev, gr, gi), dm)):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_harness(self):
        """lt.train: band_c and band_pole are in the no-weight-decay group; a short run saves band_c and no W."""
        m = LT(product_tiny())
        adam = create_optimizers(m, DEFAULTS, 1)[0][-1]
        self.assertEqual(adam.param_groups[0]['weight_decay'], 0.0)
        no_decay = {id(p) for p in adam.param_groups[0]['params']}
        self.assertTrue(id(m.block.band_c) in no_decay and id(m.block.band_pole) in no_decay)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(tiny_run(d, 2, bands=2, band_periods=[2.0, 6.0], tag_bound=0.0, band_product=True), 2)
            ck = torch.load(f'{d}/step_2.pt', weights_only=False)
            self.assertIn('block.band_c', ck['raw_model'])
            self.assertNotIn('block.band_coef', ck['raw_model'])
            self.assertIsNone(ck['rank_states'][0]['carry']['stdp_w'])
            self.assertTrue(ck['cfg']['band_product'])


if __name__ == '__main__':
    unittest.main()
