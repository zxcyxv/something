"""Build data/gtorus_ctx_pairs.npz: the in-context torus toy (data/gtorus_ctx.npz, 2026-10-20) as a pair store for
lt.data.ArcContextDataset / ctx_packed (data/prep_arc_ctx.py layout, slot S = 5, K = 3).

Every training sample of gtorus_ctx (3 demonstrations + query pair of one group-0 task) becomes one store task of 4
pairs; the evaluation episodes are its 3,072 test samples (3 demonstrations, query = the test pair) with test_task =
the toy's group (0 train task / new input, 1 unseen combination, 2 unseen transform), so lt.train reports the three
separately (task_ids_as_identifier false). Colors: the toy stores color + 1 (1 = background); raw colors here are 0..4.

    python data/prep_gtorus_ctx_pairs.py --out data/gtorus_ctx_pairs.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lt.data import ctx_episode  # noqa: E402

N, K = 4, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/gtorus_ctx.npz')
    ap.add_argument('--out', default='data/gtorus_ctx_pairs.npz')
    args = ap.parse_args()
    z = np.load(args.src)
    S = N + 1

    def grids(x, y):
        g = x.reshape(2 * K + 1, N, N).astype(np.int16) - 1
        q_out = y.reshape(2 * K + 1, N, N)[-1].astype(np.int16) - 1
        pairs = [(g[2 * i], g[2 * i + 1]) for i in range(K)] + [(g[-1], q_out)]
        return [(a.astype(np.uint8), b.astype(np.uint8)) for a, b in pairs]

    tr_x, tr_y = z['train_inputs'], z['train_labels']
    te_x, te_y = z['test_inputs'], z['test_labels']
    n_tr, n_te = len(tr_x), len(te_x)
    cells, off, hw, ptask, ptest, ptrain = [], [], [], [], [], []
    pos = 0
    tr_in, tr_lab, te_in, te_lab = [], [], [], []
    for t in range(n_tr + n_te):
        x, y = (tr_x[t], tr_y[t]) if t < n_tr else (te_x[t - n_tr], te_y[t - n_tr])
        pairs = grids(x, y)
        for j, (a, b) in enumerate(pairs):
            off.append(pos)
            hw.append((N, N, N, N))
            ptask.append(t)
            ptest.append(j == K)
            ptrain.append(t < n_tr)
            cells += [a.ravel(), b.ravel()]
            pos += 2 * N * N
        ex, ey = ctx_episode(pairs[:K], pairs[K], S, K)
        (tr_in if t < n_tr else te_in).append(ex)
        (tr_lab if t < n_tr else te_lab).append(ey)
    split = np.r_[np.zeros(n_tr, np.int8), np.ones(n_te, np.int8)]
    np.savez_compressed(
        args.out,
        pair_cells=np.concatenate(cells), pair_off=np.array(off, np.int64), pair_hw=np.array(hw, np.int16),
        pair_task=np.array(ptask, np.int32), pair_test=np.array(ptest, bool), pair_trainable=np.array(ptrain, bool),
        task_split=split, train_inputs=np.stack(tr_in), train_labels=np.stack(tr_lab),
        test_inputs=np.stack(te_in), test_labels=np.stack(te_lab),
        train_task=np.zeros(n_tr, np.int32), test_task=z['test_group'].astype(np.int32),
        ctx_test_task=np.arange(n_tr, n_tr + n_te, dtype=np.int32),
        ctx_slot=np.int32(S), ctx_kmax=np.int32(K), ctx_max_side=np.int32(N), num_symbols=np.int32(11))
    meta = dict(train_tasks=n_tr, eval_episodes=n_te, slot=S, kmax=K, sample_shape=list(tr_in[0].shape),
                eval_groups={int(g): int((z['test_group'] == g).sum()) for g in np.unique(z['test_group'])})
    Path(args.out).with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
