"""rope '1d' (URM/HRM fixed rotary angles on the flattened positions): angle values and the unchanged default.
Run: python -m unittest discover tests"""
import unittest

import torch

from lt.model import LT
from tests.test_model import tiny


class Rope1D(unittest.TestCase):
    def test_angles(self):
        m = LT(tiny(rope='1d', linear_attn=True, tag_bound=0.0, puzzle_emb_mode='prefix'))
        a = m.rope_angles()
        H, T, P = a.shape
        self.assertEqual((H, T, P), (m.H, m.n_pos, m.dh // 2))
        inv = 10000.0 ** (-torch.arange(P, dtype=torch.float64) * 2 / m.dh)
        expect = torch.arange(T, dtype=torch.float64)[:, None] * inv[None, :]
        for h in range(H):
            torch.testing.assert_close(a[h].double(), expect)
        self.assertTrue((a[:, 0] == 0).all())                     # the prefix token is position 0

    def test_default_unchanged(self):
        torch.manual_seed(0)
        a = LT(tiny()).rope_angles()
        torch.manual_seed(0)
        m = LT(tiny(rope='learned2d'))
        torch.testing.assert_close(m.rope_angles(), a, rtol=0, atol=0)
        th = m.block.rope_theta
        manual = (th[..., 0, None] * m.pos_row + th[..., 1, None] * m.pos_col).transpose(-1, -2)
        torch.testing.assert_close(a, manual, rtol=0, atol=0)

    def test_config_check(self):
        with self.assertRaises(ValueError):
            tiny(rope='2d')


if __name__ == '__main__':
    unittest.main()
