"""CPU/FP64 checks of the exponential pair STDP memory (exp_stdp=True): the online traces give the pair double sum
dW(n) = sum_{s != 0} L(s) v(n or n-s) k~(...)^T with the per-head window L(s) = A+ (1 - lam+) lam+^s (s > 0),
-A- (1 - lam-) lam-^|s| (s < 0), counted at the later block.
Run: python -m unittest discover tests"""
import unittest

import torch

from lt.model import LT
from tests.test_model import batch, tiny


def exp_tiny(**kw):
    return tiny(**dict(dict(exp_stdp=True, tag_bound=0.0, outer_eta=0.125), **kw))


class ExpSTDP(unittest.TestCase):
    def test_write_is_pair_double_sum(self):
        torch.manual_seed(0)
        m = LT(exp_tiny()).double().eval()
        with torch.no_grad():
            m.block.window_raw.add_(torch.randn_like(m.block.window_raw) * 0.5)
        B, T, D, N = 2, m.config.seq_len, m.dh, 5
        g = torch.Generator().manual_seed(1)
        qs, ks, vs = ([torch.randn(B, m.H, T, D, generator=g, dtype=torch.float64) for _ in range(N)] for _ in range(3))
        angles = m.rope_angles()
        tables = angles.cos()[None], angles.sin()[None]
        unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
        krs = [m.rope(unit(k), tables) for k in ks]
        lp, lm, ap, am = (x.detach()[None, :, None, None] for x in m.block.window())
        eta = m.config.outer_eta
        state = m.initial_state(B, torch.device('cpu'), torch.float64)
        w = torch.zeros_like(state[3])
        with torch.no_grad():
            for n in range(N):
                dw = torch.zeros_like(w)
                for s in range(1, n + 1):                       # pairs whose later event is block n
                    dw += ap * (1 - lp) * lp ** s * (vs[n].transpose(-1, -2) @ krs[n - s])     # pre first: LTP
                    dw -= am * (1 - lm) * lm ** s * (vs[n - s].transpose(-1, -2) @ krs[n])     # post first: LTD
                w = (1 - eta) * w + eta * dw
                read, state = m.memory_step(qs[n], ks[n], vs[n], state)
                torch.testing.assert_close(state[3], w, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(read, m.rope(unit(qs[n]), tables) @ w.transpose(-1, -2),
                                           rtol=1e-12, atol=1e-12)

    def test_no_ema_reads_this_blocks_write(self):
        """outer_eta = 0: no W is carried and the read is q~ dW^T."""
        torch.manual_seed(0)
        m = LT(exp_tiny(outer_eta=0.0)).double().eval()
        B, T, D = 2, m.config.seq_len, m.dh
        g = torch.Generator().manual_seed(3)
        angles = m.rope_angles()
        tables = angles.cos()[None], angles.sin()[None]
        unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
        lp, lm, ap, am = (x.detach()[None, :, None, None] for x in m.block.window())
        state = m.initial_state(B, torch.device('cpu'), torch.float64)
        self.assertIsNone(state[3])
        with torch.no_grad():
            for n in range(3):
                q, k, v = (torch.randn(B, m.H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
                kr = m.rope(unit(k), tables)
                xbar, ybar = state[1][..., :D], state[2][..., :D]
                dw = ap * (v.transpose(-1, -2) @ (lp * xbar)) - am * ((lm * ybar).transpose(-1, -2) @ kr)
                read, state = m.memory_step(q, k, v, state)
                torch.testing.assert_close(read, m.rope(unit(q), tables) @ dw.transpose(-1, -2), rtol=1e-12, atol=1e-12)

    def test_hebb_read(self):
        """exp_hebb: read = q~ (sum_t v k~^T + alpha W)^T with W the same EMA of the odd write."""
        torch.manual_seed(0)
        m = LT(exp_tiny(exp_hebb=True)).double().eval()
        with torch.no_grad():
            m.block.alpha.add_(torch.randn_like(m.block.alpha))
        ref = LT(exp_tiny()).double().eval()
        ref.load_state_dict({k: v for k, v in m.state_dict().items() if k != 'block.alpha'})
        B, T, D = 2, m.config.seq_len, m.dh
        g = torch.Generator().manual_seed(4)
        angles = m.rope_angles()
        tables = angles.cos()[None], angles.sin()[None]
        unit = lambda x: x / (x.norm(dim=-1, keepdim=True) + m.config.eps)
        s1 = m.initial_state(B, torch.device('cpu'), torch.float64)
        s2 = ref.initial_state(B, torch.device('cpu'), torch.float64)
        alpha = m.block.alpha.detach()[None, :, None, None]
        with torch.no_grad():
            for _ in range(3):
                q, k, v = (torch.randn(B, m.H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
                read, s1 = m.memory_step(q, k, v, s1)
                _, s2 = ref.memory_step(q, k, v, s2)
                hebb = v.transpose(-1, -2) @ m.rope(unit(k), tables)
                torch.testing.assert_close(read, m.rope(unit(q), tables) @ (hebb + alpha * s2[3]).transpose(-1, -2),
                                           rtol=1e-12, atol=1e-12)

    def test_constant_activity_writes_nothing(self):
        """The LTP and LTD areas are equal (A- lam- = A+ lam+), so after the traces settle a fixed point adds nothing."""
        torch.manual_seed(0)
        m = LT(exp_tiny()).double().eval()
        with torch.no_grad():
            m.block.window_raw.add_(torch.randn_like(m.block.window_raw) * 0.5)
        B, T, D = 2, m.config.seq_len, m.dh
        g = torch.Generator().manual_seed(2)
        q, k, v = (torch.randn(B, m.H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
        state = m.initial_state(B, torch.device('cpu'), torch.float64)
        with torch.no_grad():
            for _ in range(200):
                _, state = m.memory_step(q, k, v, state)
            w = state[3]
            _, state = m.memory_step(q, k, v, state)
        torch.testing.assert_close(state[3], (1 - m.config.outer_eta) * w, rtol=0, atol=1e-10)

    def test_gradients_and_no_alpha(self):
        torch.manual_seed(0)
        m = LT(exp_tiny()).double()
        self.assertFalse(hasattr(m.block, 'alpha'))
        b = batch(0)
        carry, logits = m(m.initial_carry(b), b)
        carry, logits = m(carry, b)
        logits.square().sum().backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(p.grad, n)

    def test_config_checks(self):
        with self.assertRaises(ValueError):
            tiny(exp_stdp=True, tag_bound=0.0, outer_rho=0.5)     # no leaky integral
        with self.assertRaises(ValueError):
            tiny(exp_hebb=True, tag_bound=0.0, outer_eta=0.125)   # needs exp_stdp
        with self.assertRaises(ValueError):
            tiny(exp_stdp=True, outer_eta=0.125)                  # needs tag_bound = 0


if __name__ == '__main__':
    unittest.main()
