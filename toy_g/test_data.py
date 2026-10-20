"""Checks of the toy data. Run: python -m unittest toy_g.test_data"""
import unittest

import numpy as np

from toy_g import data as D
from toy_g import group as G


class Data(unittest.TestCase):
    def test_split(self):
        s = D.make_split(0)
        parts = [s['train'], s['heldout_combo'], s['heldout_t']]
        self.assertEqual(sum(len(p) for p in parts), G.NUM_TASKS)
        self.assertEqual(len(np.unique(np.concatenate(parts))), G.NUM_TASKS)
        self.assertEqual((len(s['heldout_t']), len(s['heldout_t_elements'])), (384, 16))
        train_t = {G.split_task(int(x))[0] for x in s['train']}
        self.assertFalse(train_t & set(s['heldout_t_elements'].tolist()))
        self.assertEqual({G.decode_t(t)[0] for t in train_t}, set(range(8)))              # every point operation
        self.assertEqual({G.decode_t(t)[1:] for t in train_t}, {(a, b) for a in range(4) for b in range(4)})
        self.assertTrue((s['group'][s['heldout_t']] == 2).all() and (s['group'][s['train']] == 0).all())

    def test_inputs_and_apply(self):
        rng = np.random.default_rng(0)
        x = D.random_inputs(rng, 500, (3, 8))
        nz = (x > 0).reshape(500, -1).sum(1)
        self.assertTrue(((nz >= 3) & (nz <= 8)).all() and x.max() <= 4)
        tasks = rng.integers(0, G.NUM_TASKS, 500)
        y = D.apply_tasks(x, tasks)
        for i in range(0, 500, 37):
            t, c = G.split_task(int(tasks[i]))
            self.assertTrue((y[i] == G.apply_task(x[i], t, c)).all())

    def test_consistency_and_determined(self):
        rng = np.random.default_rng(1)
        b = D.sample_batch(rng, 64, 2, np.arange(G.NUM_TASKS))
        cons = D.consistent_tasks(b['demo_in'], b['demo_out'])
        self.assertTrue(cons[np.arange(64), b['task']].all())                              # the true task is consistent
        self.assertTrue(D.determined(b['demo_in'], b['demo_out'], b['query_in'], b['query_out']).all())
        wrong = b['query_out'].copy()
        wrong[:, 0, 0] = (wrong[:, 0, 0] % 4) + 1                                          # change one cell
        self.assertFalse(D.determined(b['demo_in'], b['demo_out'], b['query_in'], wrong).any())

    def test_sample_batch_shapes(self):
        rng = np.random.default_rng(2)
        s = D.make_split(0)
        b = D.sample_batch(rng, 32, 3, s['train'])
        self.assertEqual(b['demo_in'].shape, (32, 3, 4, 4))
        self.assertTrue(np.isin(b['task'], s['train']).all())


if __name__ == '__main__':
    unittest.main()
