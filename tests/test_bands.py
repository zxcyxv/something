"""CPU/FP64 checks of the complex-pole band window (bands=R): one band reproduces the phase-clock model at c = 0, and
R bands give S(n) = sum over the cell's pairs whose later event is block n of l(D) v k~^T with
l(D) = sum_r gamma_r Re(a_r z_r^D) (D >= 0), sum_r gamma_r Re(b_r z_r^|D|) (D < 0).
Run: python -m unittest discover tests"""
import math
import unittest

import torch

from lt.model import LT, ModelConfig
from tests.test_model import batch, tiny


def band_tiny(**kw):
    return tiny(**dict(dict(bands=4, tag_bound=0.0, outer_eta=0.125), **kw))


def randn_qkv(m, B, n, seed):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(B, m.H, m.config.seq_len, m.dh, generator=g, dtype=torch.float64) for _ in range(3)]
            for _ in range(n)]


class Bands(unittest.TestCase):
    def test_one_band_is_the_clock_model_at_c0(self):
        """z = lam e^{i omegaT}, a = -i (1 - lam) / gamma, b = +i (1 - lam) / gamma and Hebb + W is model B-EMA with
        tag_bound 0 and alpha 1: same W and read every block."""
        torch.manual_seed(0)
        ref = LT(tiny(tag_bound=0.0, outer_eta=0.125)).double().eval()
        m = LT(band_tiny(bands=1, band_periods=(5.0,))).double().eval()
        lam, wt = ref.config.trace_lambda, ref.config.block_omega
        gamma = math.sqrt(1 - lam ** 2)
        with torch.no_grad():
            m.block.rope_theta.copy_(ref.block.rope_theta)
            m.block.band_pole[0].fill_(math.log(-math.log(lam)))
            m.block.band_pole[1].fill_(wt)
            m.block.band_coef.copy_(torch.tensor([0.0, -(1 - lam) / gamma, 0.0, (1 - lam) / gamma], dtype=torch.float64)[:, None, None])
        B = 2
        s_ref = ref.initial_state(B, torch.device('cpu'), torch.float64)
        s_new = m.initial_state(B, torch.device('cpu'), torch.float64)
        with torch.no_grad():
            for q, k, v in randn_qkv(m, B, 12, 1):
                r_ref, s_ref = ref.memory_step(q, k, v, s_ref)
                r_new, s_new = m.memory_step(q, k, v, s_new)
                torch.testing.assert_close(s_new[3], s_ref[3], rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(r_new, r_ref, rtol=1e-11, atol=1e-12)

    def test_write_is_pair_double_sum(self):
        self.check_pair_double_sum(hebb=True)

    def test_single_window_reads_this_blocks_write(self):
        """hebb=False: same W, read = q~ (S + W)^T."""
        self.check_pair_double_sum(hebb=False)

    def check_pair_double_sum(self, hebb):
        torch.manual_seed(0)
        m = LT(band_tiny(hebb=hebb)).double().eval()
        with torch.no_grad():
            m.block.band_pole.add_(torch.randn_like(m.block.band_pole) * 0.3)
            m.block.band_coef.normal_()
        zr, zi, gamma, ar, ai, br, bi = (x.detach() for x in m.block.bands())
        z, a, b = torch.complex(zr, zi), torch.complex(ar, ai), torch.complex(br, bi)
        ell = lambda d: ((gamma * (a * z ** d).real).sum(-1) if d >= 0
                         else (gamma * (b * z ** -d).real).sum(-1))[None, :, None, None]
        angles = m.rope_angles()
        tables = angles.cos()[None], angles.sin()[None]
        unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
        B, eta, N = 2, m.config.outer_eta, 7
        seq = randn_qkv(m, B, N, 2)
        krs = [m.rope(unit(k), tables) for _, k, _ in seq]
        state = m.initial_state(B, torch.device('cpu'), torch.float64)
        w = torch.zeros_like(state[3])
        with torch.no_grad():
            for n, (q, k, v) in enumerate(seq):
                s = ell(0) * (v.transpose(-1, -2) @ krs[n])
                for d in range(1, n + 1):
                    s = s + ell(d) * (v.transpose(-1, -2) @ krs[n - d])              # value after the key
                    s = s + ell(-d) * (seq[n - d][2].transpose(-1, -2) @ krs[n])    # value before the key
                w = (1 - eta) * w + eta * s
                read, state = m.memory_step(q, k, v, state)
                torch.testing.assert_close(state[3], w, rtol=1e-11, atol=1e-12)
                qr = m.rope(unit(q), tables)
                tonic = v.transpose(-1, -2) @ krs[n] if hebb else s
                torch.testing.assert_close(read, qr @ (tonic + w).transpose(-1, -2),
                                           rtol=1e-11, atol=1e-12)

    def test_initial_poles_and_draw_order(self):
        """|z| = e^{-1/p}, theta = 2 pi / p; every weight drawn before the coefficients matches the model without
        bands, and there is no alpha."""
        torch.manual_seed(0)
        ref = LT(tiny(tag_bound=0.0, outer_eta=0.125))
        torch.manual_seed(0)
        m = LT(band_tiny())
        p = torch.tensor(m.config.band_periods)
        zr, zi, gamma = (x.detach().double() for x in m.block.bands()[:3])
        torch.testing.assert_close(torch.complex(zr, zi).abs(), torch.exp(-1 / p.double()).expand_as(zr))
        torch.testing.assert_close(torch.atan2(zi, zr).abs(), (2 * math.pi / p.double()).expand_as(zr))
        theirs = dict(ref.named_parameters())
        for n, t in m.named_parameters():
            if n in theirs:
                torch.testing.assert_close(t, theirs[n], rtol=0, atol=0, msg=n)
        self.assertFalse(hasattr(m.block, 'alpha'))
        self.assertGreater(float(m.block.band_coef.std()), 0.0)

    def test_eta_cap(self):
        m = LT(band_tiny(outer_eta_learn=True, outer_eta_cap=True))
        with torch.no_grad():
            m.block.eta_raw.add_(torch.tensor([3.0, -3.0]))
        m.block.project_eta_(m.config)
        eta = m.block.eta(m.config)
        self.assertAlmostEqual(float(eta[0]), 0.125, places=6)
        self.assertAlmostEqual(float(eta[1]), 1 / (1 + math.exp(3 - math.log(0.125 / 0.875))), places=6)

    def test_gradients(self):
        torch.manual_seed(0)
        m = LT(band_tiny(outer_eta_learn=True, outer_eta_cap=True)).double()
        bt = batch(0)
        carry = m.initial_carry(bt)
        for _ in range(2):
            carry, logits = m(carry, bt)
        logits.sum().backward()
        for n in ('band_pole', 'band_coef', 'eta_raw'):
            g = getattr(m.block, n).grad
            self.assertIsNotNone(g, n)
            self.assertGreater(float(g.abs().sum()), 0.0, n)

    def test_config_checks(self):
        for kw in (dict(tag_bound=0.14), dict(outer_eta=0.0), dict(band_periods=(2.0, 6.0)),
                   dict(outer_eta_cap=True)):
            with self.assertRaises(ValueError, msg=str(kw)):
                band_tiny(**kw)
        ModelConfig.from_dict(dict(bands=2, band_periods=[2, 6], tag_bound=0.0, outer_eta=0.125))


if __name__ == '__main__':
    unittest.main()
