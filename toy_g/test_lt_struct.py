"""LTStruct: the structure scores read the source cell, the head adds nothing at init, gradients reach the task code.
Run: python -m unittest toy_g.test_lt_struct"""
import unittest

import numpy as np
import torch

from tests.test_model import tiny
from toy_g import group as G
from toy_g.lt_struct import LTStruct, structure_scores
from toy_g.model import StructureModel


def cfg(**kw):
    return tiny(**dict(dict(grid=4, seq_len=16, vocab_size=6, loops=2, num_puzzle_identifiers=5, linear_attn=True,
                            tag_bound=0.0, puzzle_emb_mode='prefix', rope='1d', forward_dtype='float32'), **kw))


class LTStructTest(unittest.TestCase):
    def test_designed_scores_are_the_source_permutation(self):
        A = structure_scores(StructureModel(mode='designed', d=16)).numpy()
        expect = np.zeros((G.NUM_T, 16, 16))
        expect[np.arange(G.NUM_T)[:, None], np.arange(16)[None], G.SRC] = 1
        np.testing.assert_allclose(A, expect, atol=1e-6)

    def test_head_is_silent_at_init_and_learns(self):
        torch.manual_seed(0)
        m = LTStruct(cfg(), d_g=16).train()
        b = dict(inputs=torch.randint(1, 6, (2, 16)), puzzle_identifiers=torch.tensor([1, 3]))
        c = m.initial_carry(b)
        sread_before = m.s_out.weight.abs().sum().item()
        _, logits = m(c, b)
        self.assertEqual(sread_before, 0.0)
        self.assertEqual(tuple(m.struct_A.shape), (2, 17, 17))
        self.assertTrue((m.struct_A[:, 0] == 0).all() and (m.struct_A[:, :, 0] == 0).all())
        logits.square().sum().backward()
        self.assertGreater(m.s_out.weight.grad.abs().sum().item(), 0)
        self.assertGreater(m.puzzle_emb.local_weights.grad[:, G.NUM_T:].abs().sum().item(), 0)    # colour logits


if __name__ == '__main__':
    unittest.main()
