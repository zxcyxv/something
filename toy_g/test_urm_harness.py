"""URMForHarness: one call = the original URM_Inner.forward on the carried state, lane replacement as URM.forward.
Run: python -m unittest toy_g.test_urm_harness"""
import unittest

import torch

from lt.train import TrainStep
from lt.urm_vendor.urm import URMCarry
from tests.test_model import tiny
from toy_g.urm_harness import URMForHarness


def cfg(**kw):
    return tiny(**dict(dict(grid=4, seq_len=16, vocab_size=7, loops=2, num_puzzle_identifiers=5), **kw))


def batch(B, seed):
    g = torch.Generator().manual_seed(seed)
    return dict(inputs=torch.randint(1, 7, (B, 16), generator=g), labels=torch.randint(1, 7, (B, 16), generator=g),
                puzzle_identifiers=torch.randint(0, 5, (B,), generator=g))


class URMHarness(unittest.TestCase):
    def test_segments_and_replacement(self):
        torch.manual_seed(0)
        m = URMForHarness(cfg()).eval()
        b1, b2 = batch(2, 1), batch(2, 2)
        c = m.initial_carry(b1)
        c1, l1 = m(c, b1)
        self.assertEqual(c1.steps.tolist(), [1, 1])
        self.assertFalse(c1.halted.any())
        ref, ref_l, _ = m.urm(URMCarry(current_hidden=m.urm.init_hidden.expand(2, 17, -1).clone()), b1)
        torch.testing.assert_close(l1, ref_l.float(), rtol=0, atol=0)
        c2, l2 = m(c1, b2)                                    # second segment: still puzzle b1, carried hidden
        self.assertTrue(torch.equal(c2.data['inputs'], b1['inputs']))
        _, ref_l2, _ = m.urm(ref, b1)
        torch.testing.assert_close(l2, ref_l2.float(), rtol=0, atol=0)
        self.assertTrue(c2.halted.all())
        c3, _ = m(c2, b2)                                     # halted lanes take the new puzzle and restart
        self.assertTrue(torch.equal(c3.data['inputs'], b2['inputs']))
        self.assertEqual(c3.steps.tolist(), [1, 1])

    def test_gradients_reach_the_sparse_rows(self):
        torch.manual_seed(0)
        m = URMForHarness(cfg()).train()
        step = TrainStep(m)
        b = batch(2, 3)
        _, loss, _ = step(m.initial_carry(b), b)
        loss.backward()
        self.assertGreater(m.puzzle_emb.local_weights.grad.abs().sum().item(), 0)
        self.assertGreater(m.urm.layers[0].self_attn.qkv_proj.weight.grad.abs().sum().item(), 0)

    def test_ffn_width_check(self):
        with self.assertRaises(ValueError):
            URMForHarness(cfg(hidden_size=64, mlp_inter=128))
        self.assertEqual(URMForHarness(cfg(hidden_size=64, mlp_inter=256)).urm.layers[0].mlp.down_proj.weight.shape[1], 256)


if __name__ == '__main__':
    unittest.main()
