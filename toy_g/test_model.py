"""Checks of the structure-code model and its training/evaluation harness (CPU, tiny).
Run: python -m unittest toy_g.test_model"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from toy_g import data as D
from toy_g import group as G
from toy_g import model as M
from toy_g import train_g as T

torch.set_num_threads(4)


def as_batch(b):
    """numpy sample_batch dict -> model inputs (all demonstrations present)."""
    n, k = b['demo_in'].shape[:2]
    return (torch.as_tensor(b['demo_in']), torch.as_tensor(b['demo_out']), torch.ones(n, k, dtype=torch.bool),
            torch.as_tensor(b['query_in']))


class Designed(unittest.TestCase):
    def test_operators_equal_src_permutations(self):
        m = M.StructureModel('designed')
        W = m.gen_matrices()
        for i, name in enumerate(M.GEN_NAMES):                       # W_gen e_x = e_{DEST[gen, x]}
            self.assertTrue(torch.equal(W[i].argmax(0), torch.as_tensor(G.DEST[G.GENERATORS[name]])))
        U = m.operators()
        self.assertTrue(torch.equal(U, torch.as_tensor(M.perm_matrices(G.SRC))))   # all 128 t
        weights = m.structure()[3]
        self.assertTrue(torch.equal(weights.argmax(-1), torch.as_tensor(G.SRC)))

    def test_exact_on_determined_samples(self):
        rng = np.random.default_rng(0)
        for cmap in ('sinkhorn', 'perm'):
            m = M.StructureModel('designed', cmap=cmap, tau=50.0, beta=20.0)
            for k in (1, 2, 3):
                b = D.sample_batch(rng, 48, k, np.arange(G.NUM_TASKS))
                with torch.no_grad():
                    res = m(*as_batch(b))
                pred = res['probs'].argmax(-1).numpy()
                self.assertTrue((pred == b['query_out'].reshape(48, -1)).all(), f'{cmap} k={k}')
                # the true t is among the candidates the posterior keeps
                lp = res['log_post'].numpy()[np.arange(48), b['task'] // G.NUM_C]
                self.assertTrue((lp > np.log(1e-3) - np.log(G.NUM_T)).all())


class Learned(unittest.TestCase):
    def _loss(self, m, k=2, n=24, seed=0):
        b = D.sample_batch(np.random.default_rng(seed), n, k, np.arange(G.NUM_TASKS))
        return M.nll(m(*as_batch(b))['probs'], torch.as_tensor(b['query_out']))

    def test_gradients_reach_generators(self):
        for mode in ('learned', 'learned_free'):
            for gens in ('orthogonal', 'free'):
                m = M.StructureModel(mode, d=16, generators=gens, seed=1)
                loss = self._loss(m)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                used = [0, 1, 2, 3] if mode == 'learned' else [2, 3]         # learned_free: down/right make the codes
                g = m.gen_raw.grad
                self.assertTrue(torch.isfinite(g).all())
                for i in used:
                    self.assertGreater(float(g[i].abs().sum()), 0, f'{mode} {gens} {M.GEN_NAMES[i]}')
                for name in ('g0', 'log_tau', 'log_beta') + (('op_raw',) if mode == 'learned_free' else ()):
                    grad = getattr(m, name).grad
                    self.assertTrue(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name)

    def test_codes_and_words(self):
        m = M.StructureModel('learned', d=32, seed=2)
        with torch.no_grad():
            W = m.gen_matrices()
            self.assertTrue(torch.allclose(W[0] @ W[0].T, torch.eye(32), atol=1e-5))     # orthogonal
            codes = m.codes()
            g0 = m.g0 / m.g0.norm()
            mp = torch.linalg.matrix_power
            for r in range(4):
                for c in range(4):
                    ref = mp(W[2], r) @ mp(W[3], c) @ g0                                  # W_down^r W_right^c g0
                    self.assertTrue(torch.allclose(codes[4 * r + c], ref, atol=1e-5))
            U = m.operators()
            for t in (0, 1, 17, 64, 100, 127):
                ref = torch.eye(32)
                for name in G.word(G.inverse(t)):                                        # application order
                    ref = W[M.GEN_NAMES.index(name)] @ ref
                self.assertTrue(torch.allclose(U[t], ref, atol=1e-4))

    def test_every_cmap_runs(self):
        for cmap in M.CMAP_METHODS:
            m = M.StructureModel('learned', d=16, cmap=cmap, seed=3)
            loss = self._loss(m, k=3, n=8)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(m.gen_raw.grad).all())


class Sampler(unittest.TestCase):
    def test_checks_match_numpy(self):
        rng = np.random.default_rng(3)
        s = T.TorchSampler(np.arange(G.NUM_TASKS), 'cpu', seed=0)
        for k in (1, 2, 3):
            n = 300
            tasks = rng.integers(0, G.NUM_TASKS, n)
            din = D.random_inputs(rng, n * k).reshape(n, k, 4, 4)
            qin = D.random_inputs(rng, n)
            dout, qout = D.apply_tasks(din, tasks), D.apply_tasks(qin, tasks)
            dout[:20, 0, 1, 1] = (dout[:20, 0, 1, 1] + 1) % 5        # some inconsistent demonstrations
            qout[20:40, 2, 3] = (qout[20:40, 2, 3] + 1) % 5          # some wrong queries
            ft = lambda x: torch.as_tensor(x).reshape(*x.shape[:-2], 16)
            cons =s.consistent(ft(din), ft(dout)).reshape(n, -1).numpy()
            self.assertTrue(np.array_equal(cons, D.consistent_tasks(din, dout)))
            det = s.determined(ft(din), ft(dout), ft(qin), ft(qout)).numpy()
            ref = D.determined(din, dout, qin, qout)
            self.assertTrue(np.array_equal(det, ref))
            self.assertTrue(0 < ref.sum() < n)

    def test_apply_matches_numpy(self):
        rng = np.random.default_rng(4)
        s = T.TorchSampler(np.arange(G.NUM_TASKS), 'cpu', seed=0)
        tasks = rng.integers(0, G.NUM_TASKS, 200)
        grids = D.random_inputs(rng, 200 * 3).reshape(200, 3, 4, 4)
        out = s.apply(torch.as_tensor(grids).reshape(200, 3, 16), torch.as_tensor(tasks)).numpy()
        self.assertTrue(np.array_equal(out, D.apply_tasks(grids, tasks).reshape(200, 3, 16)))

    def test_batches_are_determined_and_from_the_task_set(self):
        split = D.make_split(0)
        s = T.TorchSampler(split['train'], 'cpu', seed=5)
        b = s.batch(96, (1, 2, 3), (3, 8))
        self.assertTrue(np.isin(b['task'].numpy(), split['train']).all())
        self.assertEqual(int(s.counts.sum()), 96)
        self.assertEqual(int(s.counts[torch.as_tensor(split['group']) != 0].sum()), 0)
        nz = (torch.cat([b['demo_in'][b['demo_mask']], b['query_in']]) > 0).sum(1)
        self.assertTrue(((nz >= 3) & (nz <= 8)).all())
        for k in (1, 2, 3):
            sel = (b['k'] == k).numpy()
            self.assertTrue((b['demo_mask'][sel].sum(1) == k).all())
            g = lambda key: b[key][sel].numpy()
            din, dout = g('demo_in')[:, :k].reshape(-1, k, 4, 4), g('demo_out')[:, :k].reshape(-1, k, 4, 4)
            qin, qout = g('query_in').reshape(-1, 4, 4), g('query_out').reshape(-1, 4, 4)
            self.assertTrue(D.determined(din, dout, qin, qout).all())
            self.assertTrue((D.apply_tasks(qin, g('task')) == qout).all())


class Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = D.load_instances()
        g = cls.inst['group']
        cls.idx = np.concatenate([np.flatnonzero(g == i)[:12] for i in range(3)])

    def test_evaluate_designed(self):
        m = M.StructureModel('designed', tau=50.0, beta=20.0)
        res = T.evaluate(m, self.inst, 'cpu', idx=self.idx)
        for k in (1, 2, 3):
            for grp in ('g0', 'g1', 'g2'):
                r = res[f'k{k}'][grp]
                self.assertEqual(r['all']['n'], 12)
                if r['det']['n']:
                    self.assertEqual(r['det']['exact'], 1.0)
        self.assertEqual(res['k3']['all']['det']['n'], 36)
        T.format_eval(res)
        diag = M.diagnostics(m, D.make_split(0)['heldout_t_elements'])
        self.assertEqual(diag['read_exact_heldout_t'], 1.0)
        self.assertLess(max(diag['relation_orbit_err'].values()), 1e-6)
        self.assertEqual((diag['realised_label_exact'], diag['realised_distinct'], diag['heldout_elements_realised']),
                         (1.0, 128, 1.0))
        self.assertEqual({k: v['element'] for k, v in diag['generator_action'].items()}, G.GENERATORS)

    def test_evaluate_and_diagnose_learned(self):
        heldout = D.make_split(0)['heldout_t_elements']
        for mode in ('learned', 'learned_free'):
            m = M.StructureModel(mode, d=16, seed=0)
            res = T.evaluate(m, self.inst, 'cpu', idx=self.idx[::3], ks=(1, 3))
            self.assertEqual(set(res), {'k1', 'k3'})
            diag = M.diagnostics(m, heldout)
            for key in ('read_exact_train_t', 'read_exact_heldout_t', 'code_cos_abs_mean', 'margin_min_train_t'):
                self.assertTrue(np.isfinite(diag[key]), key)
            T.format_diag(diag)

    def test_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            T.main(['--mode', 'learned', '--d', '16', '--steps', '3', '--batch', '16', '--device', 'cpu',
                    '--threads', '4', '--log-every', '1', '--eval-every', '0', '--out', tmp])
            audit = json.loads((Path(tmp) / 'train_tasks.json').read_text())
            self.assertEqual(audit['drawn_by_group'][1:], [0, 0])
            self.assertEqual(audit['drawn_samples'], 48)
            self.assertTrue((Path(tmp) / 'eval.json').exists() and (Path(tmp) / 'final.pt').exists())


if __name__ == '__main__':
    unittest.main()
