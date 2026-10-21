"""CPU/FP64 checks of local_conv (depthwise 3x3 per grid slot in the FFN) and frame_rope (content-defined frames:
read phasors F, rotate q~, k~ by 1 - g + g conj F). Run: python -m unittest discover tests"""
import unittest

import torch
import torch.nn.functional as F

from lt.model import LT
from tests.test_model import tiny

S, G = 3, 4                                    # 4 stacked 3x3 slots


def cfg(**kw):
    return tiny(**dict(dict(grid=S, num_grids=G, seq_len=G * S * S, bands=2, band_periods=(2.0, 6.0), tag_bound=0.0,
                            outer_eta=0.125, puzzle_emb_ndim=0, vocab_size=13), **kw))


def batch(seed, n=2):
    g = torch.Generator().manual_seed(seed)
    T = G * S * S
    return dict(inputs=torch.randint(1, 13, (n, T), generator=g), labels=torch.randint(1, 13, (n, T), generator=g),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32))


def run(m, segs=2, seed=0):
    b = batch(seed)
    carry, out = m.initial_carry(b), []
    for _ in range(segs):
        carry, logits = m(carry, b)
        out.append(logits)
    return torch.stack(out)


def tables_at(m, rows, cols):
    """cos, sin [1, H, T, P] of the model's RoPE angles at arbitrary (row, col) per token."""
    th = m.block.rope_theta
    a = (th[..., 0, None] * rows + th[..., 1, None] * cols).transpose(-1, -2)
    return a.cos()[None], a.sin()[None]


class LocalConv(unittest.TestCase):
    def test_zero_init_changes_nothing(self):
        torch.manual_seed(0)
        ref = LT(cfg()).double().eval()
        torch.manual_seed(0)
        new = LT(cfg(local_conv=True, frame_rope=True)).double().eval()
        with torch.no_grad():
            torch.testing.assert_close(run(new), run(ref), rtol=0, atol=1e-12)

    def test_stays_inside_each_slot(self):
        torch.manual_seed(1)
        m = LT(cfg(local_conv=True)).double()
        I = m.block.ffn_down.in_features
        with torch.no_grad():
            m.block.local_conv_w.normal_()
            x = torch.zeros(1, G, S * S, I, dtype=torch.float64)
            x[:, 1] = torch.randn(1, S * S, I, dtype=torch.float64)
            y = m.local_conv(x.reshape(1, G * S * S, I)).reshape(1, G, S * S, I)
        self.assertEqual(float(y[:, [0, 2, 3]].abs().max()), 0.0)

    def test_matches_direct_3x3_sum(self):
        torch.manual_seed(2)
        m = LT(cfg(local_conv=True)).double()
        I = m.block.ffn_down.in_features
        with torch.no_grad():
            m.block.local_conv_w.normal_()
            x = torch.randn(2, G * S * S, I, dtype=torch.float64)
            y = m.local_conv(x).reshape(2, G, S, S, I)
            xs = F.pad(x.reshape(2, G, S, S, I), (0, 0, 1, 1, 1, 1))
            w = m.block.local_conv_w[:, 0]                                        # [I, 3, 3]
            ref = x.reshape(2, G, S, S, I).clone()
            for dr in range(3):
                for dc in range(3):
                    ref = ref + xs[:, :, dr:dr + S, dc:dc + S] * w[:, dr, dc]
        torch.testing.assert_close(y, ref, rtol=1e-12, atol=1e-12)


class FrameRope(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.m = LT(cfg(frame_rope=True)).double()
        with torch.no_grad():
            self.m.block.rope_theta.uniform_(-1.0, 1.0)
        self.B, self.H, self.T, self.D = 2, self.m.H, self.m.config.seq_len, self.m.dh

    def clock_with(self, cos, sin):
        """Clock slot holding F = cos + i sin (stored as F - 1)."""
        f = torch.stack((cos - 1, sin), -1).reshape(cos.shape[0], self.H, self.T, self.D)
        c = torch.zeros(self.B, self.H, self.T, 3 * self.D + 1, dtype=torch.float64)
        c[..., :self.D] = f
        return c

    def test_rotation_is_relative_to_the_anchor(self):
        """g = 1, F = e^{i theta a_t}: frame_rotate(R(theta p) x) = R(theta (p - a)) x."""
        m = self.m
        g = torch.Generator().manual_seed(4)
        rows, cols = m.pos_row.double(), m.pos_col.double()
        a_rows = torch.randint(0, G * S, (self.T,), generator=g).double()
        a_cols = torch.randint(0, S, (self.T,), generator=g).double()
        with torch.no_grad():
            m.block.frame_gate.fill_(1.0)
            x = torch.randn(self.B, self.H, self.T, self.D, generator=g, dtype=torch.float64)
            clock = self.clock_with(*tables_at(m, a_rows, a_cols))
            got = m.frame_rotate(m.rope(x, tables_at(m, rows, cols)), clock)
            want = m.rope(x, tables_at(m, rows - a_rows, cols - a_cols))
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)

    def test_gate_zero_and_fresh_clock_are_identity(self):
        m = self.m
        x = torch.randn(self.B, self.H, self.T, self.D, dtype=torch.float64)
        zero = torch.zeros(self.B, self.H, self.T, 3 * self.D + 1, dtype=torch.float64)
        with torch.no_grad():
            m.block.frame_gate.fill_(0.7)
            torch.testing.assert_close(m.frame_rotate(x, zero), x, rtol=0, atol=0)    # F = 1
            m.block.frame_gate.zero_()
            clock = self.clock_with(*tables_at(m, m.pos_row.double() * 0 + 2, m.pos_col.double()))
            torch.testing.assert_close(m.frame_rotate(x, clock), x, rtol=0, atol=0)   # g = 0

    def test_read_finds_the_attended_token(self):
        """Only token j has a key: every query with q~ . k~_j > 0 reads F = psi_j (its RoPE phasor)."""
        m = self.m
        g = torch.Generator().manual_seed(5)
        tables = tables_at(m, m.pos_row.double(), m.pos_col.double())
        qr = torch.randn(self.B, self.H, self.T, self.D, generator=g, dtype=torch.float64)
        kr = torch.zeros_like(qr)
        j = 7
        kr[:, :, j] = 1e3 * torch.randn(self.B, self.H, self.D, generator=g, dtype=torch.float64)  # |r| >> the 1e-6
        clock = torch.zeros(self.B, self.H, self.T, 3 * self.D + 1, dtype=torch.float64)
        out = m.frame_read(qr, kr, tables, clock)
        f = (out[..., :self.D].reshape(self.B, self.H, self.T, self.D // 2, 2) + torch.tensor([1.0, 0.0],
                                                                                             dtype=torch.float64))
        psi = torch.stack((tables[0][0, :, j], tables[1][0, :, j]), -1)                  # [H, P, 2]
        pos = (qr * kr[:, :, j:j + 1]).sum(-1) > 0                                       # [B, H, T]
        torch.testing.assert_close(f[pos], psi[None, :, None].expand_as(f)[pos], rtol=1e-6, atol=1e-6)  # eps 1e-6 / |r|
        self.assertEqual(float(out[..., self.D:].abs().max()), 0.0)                      # rest of the clock kept

    def test_gate_receives_gradient(self):
        m = LT(cfg(frame_rope=True)).double()
        with torch.no_grad():
            m.block.rope_theta.uniform_(-1.0, 1.0)
        b = batch(6)
        carry = m.initial_carry(b)
        _, logits = m(carry, b)
        logits.square().sum().backward()
        self.assertGreater(float(m.block.frame_gate.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
