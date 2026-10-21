"""In-context ARC data (data/prep_arc_ctx.py, lt.data.ctx_episode, lt.data.ArcContextDataset)."""
import unittest
from pathlib import Path

import numpy as np

from lt.data import ArcContextDataset, ctx_canvas, ctx_episode, load_arc_ctx

NPZ = Path(__file__).resolve().parents[1] / 'data' / 'arc_ctx15.npz'


def _store(tasks, S=6, kmax=2, splits=None):
    """Pair store like data/prep_arc_ctx.py from {task: [(input, output), ...]}."""
    cells, off, hw, task = [], [], [], []
    pos = 0
    for t, pairs in enumerate(tasks):
        for x, y in pairs:
            x, y = np.asarray(x, np.uint8), np.asarray(y, np.uint8)
            off.append(pos)
            hw.append((*x.shape, *y.shape))
            task.append(t)
            cells += [x.ravel(), y.ravel()]
            pos += x.size + y.size
    return dict(cells=np.concatenate(cells), off=np.array(off), hw=np.array(hw), task=np.array(task),
                split=np.array(splits or [0] * len(tasks), np.int8), S=S, kmax=kmax)


def _dataset(store, **kw):
    args = dict(seed=3, global_batch_size=4, rank=0, world_size=1, epochs_per_iter=3, start_iter=0, total_iters=2)
    args.update(kw)
    return ArcContextDataset(store, **args)


class CanvasTest(unittest.TestCase):
    def test_eos_row_and_column(self):
        c = ctx_canvas(np.array([[0, 5, 9], [1, 2, 3]]), 5)
        np.testing.assert_array_equal(c[:2, :3], [[1, 6, 10], [2, 3, 4]])
        self.assertTrue((c[2, :4] == 11).all() and (c[:3, 3] == 11).all())
        self.assertTrue((c[3:] == 0).all() and (c[:, 4] == 0).all())

    def test_side_one_less_than_slot(self):
        c = ctx_canvas(np.zeros((3, 3), np.uint8), 4)
        self.assertTrue((c[3] == 11).all() and (c[:, 3] == 11).all() and (c[:3, :3] == 1).all())

    def test_episode_layout(self):
        S, K = 4, 3
        d = [(np.full((2, 2), 1), np.full((1, 3), 2)), (np.full((3, 1), 3), np.full((2, 2), 4))]
        q = (np.full((2, 3), 5), np.full((3, 3), 6))
        x, y = ctx_episode(d, q, S, K)
        self.assertEqual(x.shape, ((2 * K + 2) * S, S))
        slots = lambda a: a.reshape(2 * K + 2, S, S)
        np.testing.assert_array_equal(slots(x)[0], ctx_canvas(d[0][0], S))
        np.testing.assert_array_equal(slots(x)[3], ctx_canvas(d[1][1], S))
        self.assertTrue((slots(x)[4:6] == 0).all())                  # unused demonstration slots
        np.testing.assert_array_equal(slots(x)[-2], ctx_canvas(q[0], S))
        self.assertTrue((slots(x)[-1] == 0).all())                   # the query output is not in the input
        self.assertTrue((slots(y)[:-1] == 0).all())                  # loss on the query output only
        np.testing.assert_array_equal(slots(y)[-1], ctx_canvas(q[1], S))


class DatasetTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        grid = lambda: rng.integers(0, 10, (int(rng.integers(1, 6)), int(rng.integers(1, 6))))
        # task 0: identity (output = input); task 1: arbitrary pairs; task 2: evaluation, never drawn
        self.tasks = [[(g, g) for g in (grid() for _ in range(4))],
                      [(grid(), grid()) for _ in range(5)],
                      [(grid(), grid()) for _ in range(3)]]
        self.store = _store(self.tasks, splits=[0, 0, 1])

    def batches(self, ds):
        return [(it, {k: v.clone() for k, v in b.items()}) for it, b in ds]

    def test_shapes_and_tokens(self):
        it, b = next(iter(_dataset(self.store)))
        S, K = self.store['S'], self.store['kmax']
        self.assertEqual(tuple(b['inputs'].shape), (4, (2 * K + 2) * S * S))
        self.assertTrue(int(b['inputs'].min()) >= 1 and int(b['inputs'].max()) <= 12)
        self.assertTrue((b['puzzle_identifiers'] == 0).all())
        lab = b['labels'].reshape(4, 2 * K + 2, S * S)
        self.assertTrue((lab[:, :-1] == 1).all())                    # raw 0: ignored outside the query output

    def test_epoch_covers_training_tasks_only(self):
        ds = _dataset(self.store, global_batch_size=2, epochs_per_iter=1, total_iters=1)
        self.assertEqual(len(self.batches(ds)), 1)                   # 2 training tasks -> one batch of 2
        self.assertEqual(set(ds.tasks.tolist()), {0, 1})

    def test_one_transform_per_episode(self):
        """Identity task: after any dihedral transform and color permutation every output still equals its input."""
        ds = _dataset(_store(self.tasks[:1]), global_batch_size=1, total_iters=1)
        S, K = ds.S, ds.kmax
        for _, b in self.batches(ds):
            x = b['inputs'].reshape(2 * K + 2, S, S)
            y = b['labels'].reshape(2 * K + 2, S, S)
            for i in range(K):
                if (x[2 * i] > 1).any():
                    self.assertTrue(bool((x[2 * i] == x[2 * i + 1]).all()))
            self.assertTrue(bool((x[-2] == y[-1]).all()))

    def test_query_not_among_demonstrations(self):
        ds = _dataset(self.store, global_batch_size=1, total_iters=1)
        rng = np.random.Generator(np.random.Philox(key=np.array([1, 2], np.uint64)))
        for _ in range(20):
            x, _, _ = ds.episode(1, rng)
            slots = x.reshape(2 * ds.kmax + 2, ds.S, ds.S)
            demo_inputs = {slots[2 * i].tobytes() for i in range(ds.kmax)}
            self.assertNotIn(slots[-2].tobytes(), demo_inputs)

    def test_deterministic_and_resumable(self):
        full = self.batches(_dataset(self.store, global_batch_size=2))
        again = self.batches(_dataset(self.store, global_batch_size=2))
        self.assertEqual(len(full), len(again))
        for (i1, b1), (i2, b2) in zip(full, again):
            self.assertEqual(i1, i2)
            self.assertTrue(all(bool((b1[k] == b2[k]).all()) for k in b1))
        per_iter = len(full) // 2
        resumed = self.batches(_dataset(self.store, global_batch_size=2, start_iter=1, skip_batches=1))
        self.assertEqual(len(resumed), per_iter - 1)
        for (i1, b1), (i2, b2) in zip(full[per_iter + 1:], resumed):
            self.assertEqual(i1, i2)
            self.assertTrue(all(bool((b1[k] == b2[k]).all()) for k in b1))


@unittest.skipUnless(NPZ.is_file(), 'data/arc_ctx15.npz not built')
class PrepTest(unittest.TestCase):
    def test_evaluation_episodes_match_the_store(self):
        store = load_arc_ctx(str(NPZ))
        z = np.load(NPZ)
        ds = _dataset(store)
        S, K = store['S'], store['kmax']
        tests = np.flatnonzero(z['pair_test'])
        for e in (0, len(z['test_inputs']) // 2, len(z['test_inputs']) - 1):
            t = int(z['ctx_test_task'][e])
            self.assertEqual(int(store['split'][t]), 1)
            q = [p for p in tests if int(store['task'][p]) == t]
            demos = [p for p in np.flatnonzero(store['task'] == t) if p not in q]
            hits = [p for p in q if np.array_equal(ctx_episode([ds.grids(d) for d in demos], ds.grids(p), S, K)[1],
                                                   z['test_labels'][e])]
            self.assertTrue(hits)

    def test_splits_disjoint(self):
        z = np.load(NPZ)
        names = z['task_names']
        train = {n.split('/')[-1] for n, s in zip(names, z['task_split']) if s == 0}
        evals = {n.split('/')[-1] for n, s in zip(names, z['task_split']) if s == 1}
        self.assertFalse(train & evals)
        self.assertEqual(len(z['train_inputs']), len(train))


HALF = Path(__file__).resolve().parents[1] / 'data' / 'arc_ctx15_half.npz'


@unittest.skipUnless(HALF.is_file(), 'data/arc_ctx15_half.npz not built')
class PrepHalfTest(unittest.TestCase):
    def test_transductive_half(self):
        z = np.load(HALF)
        mode, ptask, ptest, ptrain = z['task_eval_mode'], z['pair_task'], z['pair_test'], z['pair_trainable']
        self.assertFalse((ptrain & ptest & (mode[ptask] > 0)).any())            # no evaluation test pair in training
        self.assertFalse((ptrain & (mode[ptask] == 2)).any())                   # the inductive half never seen
        self.assertTrue((ptrain[(mode[ptask] == 1) & ~ptest]).all())            # transductive: all its train pairs
        self.assertTrue((ptrain[mode[ptask] == 0]).all())
        self.assertEqual(abs(int((mode == 1).sum()) - int((mode == 2).sum())), 1 if (mode > 0).sum() % 2 else 0)
        ds = _dataset(load_arc_ctx(str(HALF)))
        self.assertEqual(len(ds.tasks), len(z['train_inputs']))
        self.assertEqual(set(mode[ds.tasks].tolist()), {0, 1})
        np.testing.assert_array_equal(z['test_task'], mode[z['ctx_test_task']])


if __name__ == '__main__':
    unittest.main()
