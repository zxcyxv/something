"""Checks of the lt exports of the torus toy (toy_g/export_lt.py) and of toy_g/eval_lt.py's protocol.
Run: CUDA_VISIBLE_DEVICES= python -m unittest toy_g.test_lt_export"""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from lt.data import load_data, load_task_ids
from lt.model import LT, ModelConfig
from lt.train import run_segments
from toy_g import data as D
from toy_g import eval_lt as E
from toy_g import group as G

DATA = Path(__file__).resolve().parents[1] / 'data'


def load(name):
    path = DATA / f'gtorus_{name}.npz'
    if not path.is_file():
        raise unittest.SkipTest(f'{path} is missing: run python -m toy_g.export_lt')
    with np.load(path, allow_pickle=False) as z:
        return path, {k: z[k] for k in z.files}


def grids(x, j):
    """Grid j of samples stacked along rows ([N, 4 m, 4] -> [N, 4, 4])."""
    return x[:, G.N * j:G.N * (j + 1)]


class Exports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = D.load_instances()
        cls.group = cls.inst['group'].astype(np.int64)

    def check_common(self, path, z, train_shape, test_shape):
        self.assertEqual(int(z['num_symbols']), 5)
        self.assertEqual(z['train_inputs'].shape, train_shape)
        self.assertEqual(z['test_inputs'].shape, test_shape)
        for key in ('train_inputs', 'train_labels', 'test_inputs', 'test_labels'):
            self.assertEqual(z[key].dtype, np.uint8)
        self.assertEqual((z['train_inputs'].min(), z['train_inputs'].max()), (1, 5))      # colour + 1, never blank
        self.assertEqual((z['test_inputs'].min(), z['test_inputs'].max()), (1, 5))
        self.assertTrue((z['test_group'] == self.group).all())                             # rows in task order
        self.assertEqual(np.bincount(z['test_group']).tolist(), [2419, 269, 384])
        tr_x, tr_y, te_x, te_y, _, _ = load_data(path, 3072)                               # the harness loader
        self.assertEqual(len(te_x), 3072)
        for a, b in ((tr_x, 'train_inputs'), (tr_y, 'train_labels'), (te_x, 'test_inputs'), (te_y, 'test_labels')):
            self.assertTrue((a == z[b]).all())
        self.assertTrue(Path(path).with_suffix('.json').is_file())

    def check_id(self, name, tasks):
        path, z = load(name)
        self.check_common(path, z, (3 * len(tasks), 4, 4), (3072, 4, 4))
        self.assertEqual((z['train_labels'].min(), z['train_labels'].max()), (1, 5))
        self.assertEqual((z['test_labels'].min(), z['test_labels'].max()), (1, 5))
        # test pairs: the fixed instances' test pairs, identifier = task index
        self.assertTrue((z['test_task'] == np.arange(3072)).all())
        self.assertTrue((z['test_inputs'] - 1 == self.inst['test_in']).all())
        self.assertTrue((z['test_labels'] - 1 == self.inst['test_out']).all())
        # training pairs: exactly the 3 demonstration pairs of every trained task, labels = the task's outputs
        t = z['train_task'].astype(np.int64)
        self.assertTrue((np.bincount(t, minlength=3072)[tasks] == 3).all() and np.isin(t, tasks).all())
        self.assertTrue((z['train_group'] == self.group[t]).all())
        x, y = z['train_inputs'].astype(np.int64) - 1, z['train_labels'].astype(np.int64) - 1
        self.assertTrue((D.apply_tasks(x, t) == y).all())
        rows = sorted((int(a), b.tobytes(), c.tobytes()) for a, b, c in zip(t, x.astype(np.uint8), y.astype(np.uint8)))
        want = sorted((int(task), self.inst['demo_in'][task, j].tobytes(), self.inst['demo_out'][task, j].tobytes())
                      for task in tasks for j in range(3))
        self.assertEqual(rows, want)
        tr_ids, te_ids = load_task_ids(path, 3072)
        self.assertTrue((tr_ids == t).all() and (te_ids == np.arange(3072)).all())

    def test_id_trans(self):
        self.check_id('id_trans', np.arange(3072))

    def test_id_ind(self):
        self.check_id('id_ind', np.flatnonzero(self.group == 0))

    def test_ctx(self):
        path, z = load('ctx')
        meta = json.loads(path.with_suffix('.json').read_text())
        n = meta['train_samples']
        self.check_common(path, z, (n, 28, 4), (3072, 28, 4))
        self.assertNotIn('train_task', z)                       # no identifiers: lt.data finds no task ids
        self.assertEqual(load_task_ids(path, 3072), (None, None))
        for split in ('train', 'test'):                          # labels: 0 on the 6 demonstration grids
            y = z[f'{split}_labels']
            self.assertTrue((y[:, :24] == 0).all())
            self.assertEqual((y[:, 24:].min(), y[:, 24:].max()), (1, 5))
        # test: the fixed instances (3 demonstrations, test input; label = test output)
        x, y = z['test_inputs'].astype(np.int64) - 1, z['test_labels'].astype(np.int64) - 1
        for j in range(3):
            self.assertTrue((grids(x, 2 * j) == self.inst['demo_in'][:, j]).all())
            self.assertTrue((grids(x, 2 * j + 1) == self.inst['demo_out'][:, j]).all())
        self.assertTrue((grids(x, 6) == self.inst['test_in']).all())
        self.assertTrue((grids(y, 6) == self.inst['test_out']).all())
        self.assertTrue((z['test_task_index'] == np.arange(3072)).all())
        # train: group-0 tasks only, demonstrations and query label follow the recorded task
        t = z['train_task_index'].astype(np.int64)
        self.assertTrue((self.group[t] == 0).all() and (z['train_group'] == 0).all())
        self.assertEqual(len(np.unique(t)), 2419)
        x, y = z['train_inputs'].astype(np.int64) - 1, z['train_labels'].astype(np.int64) - 1
        din = np.stack([grids(x, 2 * j) for j in range(3)], 1)
        dout = np.stack([grids(x, 2 * j + 1) for j in range(3)], 1)
        self.assertTrue((D.apply_tasks(din, t) == dout).all())
        self.assertTrue((D.apply_tasks(grids(x, 6), t) == grids(y, 6)).all())
        nz = (x > 0).reshape(n, 7, 16).sum(-1)[:, [0, 2, 4, 6]]                 # input sparsity 3..8 cells
        self.assertTrue(((nz >= 3) & (nz <= 8)).all())
        sub = slice(0, 1000)                                                        # determined (subset: ~2 ms each)
        self.assertTrue(D.determined(din[sub], dout[sub], grids(x, 6)[sub], grids(y, 6)[sub]).all())


class EvalProtocol(unittest.TestCase):
    """eval_lt.predict reproduces lt.train.run_segments per puzzle (tiny random model, CPU)."""
    def run_case(self, name, num_grids, pe_ndim, ignore0, n=200):
        path, z = load(name)
        _, _, x, y, _, _ = load_data(path, n)
        _, ids = load_task_ids(path, n)
        torch.manual_seed(0)
        cfg = ModelConfig(batch_size=64, seq_len=num_grids * 16, grid=4, num_grids=num_grids, vocab_size=7,
                          num_puzzle_identifiers=3072 if pe_ndim else 1, puzzle_emb_ndim=pe_ndim, hidden_size=16,
                          num_heads=2, blocks_per_seg=2, loops=2, mlp_inter=32, linear_attn=True, tag_bound=0.0,
                          puzzle_emb_mode='prefix', amp=False)
        model = LT(cfg)
        if pe_ndim:
            with torch.no_grad():
                model.puzzle_emb.weights.normal_(0.0, 0.3)
        acc, exact, tokens = E.predict(model, x, y, 2, 64, torch.device('cpu'), ids, ignore0)
        sums, count, _ = run_segments(model, x, y, 2, 64, 0, 1, torch.device('cpu'), ids=ids, ignore0=ignore0)
        self.assertEqual(int(count), n)
        self.assertEqual(int(exact[-1].sum()), int(sums[1, -1]))
        self.assertAlmostEqual(float(acc[-1].sum()), float(sums[0, -1]), places=9)
        self.assertAlmostEqual(float(acc[0].sum()), float(sums[0, 0]), places=9)
        colours = E.tokens_to_colours(tokens)
        target = self.target(z, n)
        self.assertTrue(((colours == target).reshape(n, -1).mean(-1) == acc[-1]).all())
        self.assertTrue(((colours == target).reshape(n, -1).all(-1) == exact[-1]).all())
        # identifiers matter for the ID format (the evaluation reads them)
        if pe_ndim:
            acc0, _, _ = E.predict(model, x, y, 2, 64, torch.device('cpu'), np.zeros_like(ids), ignore0)
            self.assertFalse(np.allclose(acc0, acc))

    @staticmethod
    def target(z, n):
        return (z['test_labels'][:n, -4:].astype(np.int64) - 1).reshape(n, 4, 4)

    def test_id(self):
        self.run_case('id_trans', 1, 16, False)

    def test_ctx(self):
        self.run_case('ctx', 7, 0, True)


if __name__ == '__main__':
    unittest.main()
