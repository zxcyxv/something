"""Data of the 4x4 torus toy: random sparse input grids, the task split, fixed task instances and a fresh-sample
generator, with exact consistency checks against all 3,072 tasks.

Conventions (toy_g/group.py): raw colours 0..4 (0 background), task = t * 24 + c, out = apply_task(in, t, c).
Every sample is kept only if its query output is uniquely determined by its demonstrations: all tasks consistent with
the demonstrations give the same query output (so an ideal in-context learner can be exact).

Split (make_split): 16 held-out transformations (2 per point operation, 16 distinct translations, so every point
operation and every translation still occurs in training) with all their colour maps -> 'heldout_t' (384 tasks);
10% of the remaining (t, c) pairs -> 'heldout_combo'; the rest -> 'train'. Evaluation groups: 0 train task (new
inputs), 1 heldout_combo, 2 heldout_t.
"""
from pathlib import Path

import numpy as np

from . import group as G

GROUP_NAMES = ('train', 'heldout_combo', 'heldout_t')


def make_split(seed=0, combo_frac=0.1):
    rng = np.random.default_rng(seed)
    trans = rng.permutation(G.CELLS)
    heldout_t = sorted(G.encode_t(p, *divmod(int(trans[2 * p + j]), G.N)) for p in range(8) for j in range(2))
    rest = [G.task_index(t, c) for t in range(G.NUM_T) if t not in heldout_t for c in range(G.NUM_C)]
    rest = np.array(rest)
    pick = rng.permutation(len(rest))[:int(round(combo_frac * len(rest)))]
    combo = np.sort(rest[pick])
    train = np.setdiff1d(rest, combo)
    ht = np.array([G.task_index(t, c) for t in heldout_t for c in range(G.NUM_C)])
    group = np.zeros(G.NUM_TASKS, dtype=np.int64)
    group[combo], group[ht] = 1, 2
    return dict(train=train, heldout_combo=combo, heldout_t=np.sort(ht), heldout_t_elements=np.array(heldout_t),
                group=group)


def random_inputs(rng, n, sparsity=(3, 8)):
    """[n, 4, 4] grids with a uniform number of coloured cells in [lo, hi] (distinct cells), colours uniform 1..4."""
    lo, hi = sparsity
    counts = rng.integers(lo, hi + 1, n)
    order = np.argsort(rng.random((n, G.CELLS)), axis=1)
    colours = rng.integers(1, G.COLORS + 1, (n, G.CELLS))
    values = np.where(np.arange(G.CELLS)[None, :] < counts[:, None], colours, 0)
    grids = np.zeros((n, G.CELLS), dtype=np.int64)
    np.put_along_axis(grids, order, values, axis=1)
    return grids.reshape(n, G.N, G.N)


def apply_tasks(grids, tasks):
    """grids [n, ..., 4, 4], tasks [n] -> outputs of the same shape (each sample under its own task)."""
    t, c = G.split_task(np.asarray(tasks))
    shape = grids.shape
    flat = grids.reshape(shape[0], -1, G.CELLS)                                              # [n, m, 16]
    rows = np.arange(shape[0])[:, None, None]
    moved = flat[rows, np.arange(flat.shape[1])[None, :, None], G.SRC[t][:, None, :]]
    return G.CMAPS[c][rows, moved].reshape(shape)


def consistent_tasks(demo_in, demo_out):
    """demo_in, demo_out [n, k, 4, 4] -> bool [n, 3072]: tasks that map every demonstration input to its output."""
    n, k = demo_in.shape[:2]
    din, dout = demo_in.reshape(n, k, G.CELLS), demo_out.reshape(n, k, G.CELLS)
    moved = din[:, :, G.SRC]                                                                  # [n, k, 128, 16]
    ok = np.empty((n, G.NUM_T, G.NUM_C), dtype=bool)
    for c in range(G.NUM_C):
        ok[:, :, c] = (G.CMAPS[c][moved] == dout[:, :, None, :]).all(axis=(1, 3))
    return ok.reshape(n, G.NUM_TASKS)


def determined(demo_in, demo_out, query_in, query_out, chunk=256):
    """bool [n]: every task consistent with the demonstrations gives query_out (the query is uniquely determined)."""
    n = len(query_in)
    res = np.empty(n, dtype=bool)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        cons = consistent_tasks(demo_in[s:e], demo_out[s:e]).reshape(e - s, G.NUM_T, G.NUM_C)
        q = query_in[s:e].reshape(e - s, G.CELLS)[:, G.SRC]                                  # [m, 128, 16]
        target = query_out[s:e].reshape(e - s, 1, G.CELLS)
        bad = np.zeros((e - s, G.NUM_T, G.NUM_C), dtype=bool)
        for c in range(G.NUM_C):
            bad[:, :, c] = (G.CMAPS[c][q] != target).any(-1)
        res[s:e] = ~(bad & cons).any(axis=(1, 2))
    return res


def sample_batch(rng, n, k, tasks, sparsity=(3, 8), max_rounds=100):
    """Fresh samples: dict(demo_in [n,k,4,4], demo_out, query_in [n,4,4], query_out, task [n]) with tasks drawn
    uniformly from `tasks`, kept only if the query output is determined by the k demonstrations."""
    tasks = np.asarray(tasks)
    keep = {key: [] for key in ('demo_in', 'demo_out', 'query_in', 'query_out', 'task')}
    have = 0
    for _ in range(max_rounds):
        m = max(2 * (n - have), 64)
        task = tasks[rng.integers(0, len(tasks), m)]
        din = random_inputs(rng, m * k, sparsity).reshape(m, k, G.N, G.N)
        qin = random_inputs(rng, m, sparsity)
        dout, qout = apply_tasks(din, task), apply_tasks(qin, task)
        ok = determined(din, dout, qin, qout)
        for key, val in zip(keep, (din, dout, qin, qout, task)):
            keep[key].append(val[ok])
        have += int(ok.sum())
        if have >= n:
            break
    else:
        raise RuntimeError('could not draw enough determined samples; loosen the sparsity or raise k')
    return {key: np.concatenate(val)[:n] for key, val in keep.items()}


def task_instances(seed=0, n_demo=3, sparsity=(3, 8)):
    """One fixed ARC-like instance per task (all 3,072): n_demo demonstration pairs and one test pair whose output is
    determined by the n_demo demonstrations. Also determined_k [3072, n_demo]: determined by the first k (k = 1..)."""
    rng = np.random.default_rng(seed + 1000)
    got = {}
    need = np.arange(G.NUM_TASKS)
    while len(need):
        task = np.repeat(need, 4)                                    # 4 candidates per task still missing
        m = len(task)
        din = random_inputs(rng, m * n_demo, sparsity).reshape(m, n_demo, G.N, G.N)
        qin = random_inputs(rng, m, sparsity)
        dout, qout = apply_tasks(din, task), apply_tasks(qin, task)
        for i in np.flatnonzero(determined(din, dout, qin, qout)):
            got.setdefault(int(task[i]), (din[i], dout[i], qin[i], qout[i]))
        need = np.array([t for t in need if int(t) not in got], dtype=np.int64)
    out = {key: np.stack([got[t][j] for t in range(G.NUM_TASKS)]).astype(np.uint8)
           for j, key in enumerate(('demo_in', 'demo_out', 'test_in', 'test_out'))}
    det = np.zeros((G.NUM_TASKS, n_demo), dtype=bool)
    for k in range(1, n_demo + 1):
        det[:, k - 1] = determined(out['demo_in'][:, :k].astype(np.int64), out['demo_out'][:, :k].astype(np.int64),
                                   out['test_in'].astype(np.int64), out['test_out'].astype(np.int64))
    out['determined_k'] = det
    out['task'] = np.arange(G.NUM_TASKS)
    return out


INSTANCES = Path(__file__).resolve().parents[1] / 'data' / 'gtorus_instances.npz'


def load_instances(path=INSTANCES):
    """The fixed instances (seed 0, 3 demonstrations, sparsity 3..8) plus the split's group per task, from the shared
    file data/gtorus_instances.npz (written once by `python -m toy_g.data`, so every condition sees the same tasks)."""
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


if __name__ == '__main__':
    inst = task_instances(seed=0, n_demo=3, sparsity=(3, 8))
    split = make_split(seed=0)
    np.savez_compressed(INSTANCES, group=split['group'], heldout_t_elements=split['heldout_t_elements'], **inst)
    det = inst['determined_k']
    print(f'wrote {INSTANCES}: {G.NUM_TASKS} tasks; determined by the first k = 1, 2, 3 demonstrations: '
          f'{det[:, 0].mean():.3f} {det[:, 1].mean():.3f} {det[:, 2].mean():.3f}; '
          f'groups {np.bincount(split["group"]).tolist()}')
