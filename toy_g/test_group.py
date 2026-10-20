"""Checks of the shared toy definitions. Run: python -m unittest toy_g.test_group"""
import unittest

import numpy as np

from toy_g import group as G


class Group(unittest.TestCase):
    def test_sizes_and_permutations(self):
        self.assertEqual((G.NUM_T, G.NUM_C, G.NUM_TASKS), (128, 24, 3072))
        self.assertTrue((np.sort(G.SRC, 1) == np.arange(G.CELLS)).all())
        self.assertEqual(len({tuple(r) for r in G.SRC}), G.NUM_T)          # 128 distinct permutations
        self.assertTrue((G.SRC[G.IDENTITY] == np.arange(G.CELLS)).all())

    def test_src_dest_and_apply(self):
        rng = np.random.default_rng(0)
        grid = rng.integers(0, 5, (G.N, G.N))
        for t in range(G.NUM_T):
            out = G.apply_t(grid, t).reshape(-1)
            self.assertTrue((out == grid.reshape(-1)[G.SRC[t]]).all())
            self.assertTrue((out[G.DEST[t]] == grid.reshape(-1)).all())

    def test_closure_inverse_and_words(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            t1, t2 = rng.integers(0, G.NUM_T, 2)
            t12 = G.compose(int(t1), int(t2))
            grid = rng.integers(0, 5, (G.N, G.N))
            self.assertTrue((G.apply_t(G.apply_t(grid, t2), t1) == G.apply_t(grid, t12)).all())
        for t in range(G.NUM_T):
            self.assertEqual(G.compose(G.inverse(t), t), G.IDENTITY)
            acc = G.IDENTITY
            for name in G.word(t):
                acc = G.compose(G.GENERATORS[name], acc)
            self.assertEqual(acc, t)

    def test_tasks(self):
        self.assertTrue((G.CMAPS[:, 0] == 0).all())
        grid = np.array([[0, 1, 2, 3], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        t, c = 37, 5
        out = G.apply_task(grid, t, c)
        self.assertTrue((out == G.CMAPS[c][G.apply_t(grid, t)]).all())
        self.assertEqual(G.split_task(G.task_index(t, c)), (t, c))


if __name__ == '__main__':
    unittest.main()
