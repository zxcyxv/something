"""Build data/miniarc_ctx.npz: Mini-ARC with the demonstrations in context and no task identifiers (BDH-CQ-style
in-context learning). A sample stacks 2K + 1 grids along rows: K demonstration pairs (input, output), then the query
input; labels are 0 (outside the loss, ignore_label0) on the demonstration grids and the query's output on the last.

Training: every version of every task (8 dihedral x --color-perms colour permutations, as data/prep_miniarc.py), every
demonstration in turn as the query with K other demonstrations of the same version as context (a task with fewer
than K + 1 demonstrations repeats one). Test: the identity version's test input with its first K demonstrations.

With --k 0 every demonstration is used: the context has room for the largest demonstration count (8 pairs, 17 grids)
and shorter contexts are padded with empty grids (raw input 0 = the blank token, label 0 = outside the loss); training
samples use all demonstrations other than the query, test samples all demonstrations.

    python data/prep_miniarc_ctx.py PATH/TO/MINI-ARC/data/MiniARC [--k 2 | --k 0] [--color-perms 4] [--seed 0]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def dihedral(g, k):
    g = np.rot90(g, k % 4)
    return g.T if k >= 4 else g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--color-perms', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'miniarc_ctx.npz'))
    opt = ap.parse_args()
    rng = np.random.default_rng(opt.seed)
    files = sorted(glob.glob(str(Path(opt.src) / '*.json')))
    use_all = opt.k == 0
    K = max(len(json.load(open(f))['train']) for f in files) if use_all else opt.k
    zero = np.zeros((5, 5), dtype=np.int64)
    blank = np.full((5, 5), -1, dtype=np.int64)                  # +1 below -> raw input 0 (blank token)

    def sample(ctx, q_in, q_out):
        ctx = list(ctx) + [(blank, None)] * (K - len(ctx))
        grids = [g for pair in ctx for g in (pair[0], pair[0] if pair[1] is None else pair[1])] + [q_in]
        labels = [zero] * (2 * K) + [q_out + 1]
        return np.concatenate(grids) + 1, np.concatenate(labels)

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for f in files:
        task = json.load(open(f))
        perms = [np.arange(10)] + [np.concatenate([[0], 1 + rng.permutation(9)]) for _ in range(opt.color_perms - 1)]
        for perm in perms:
            for k in range(8):
                demos = [(dihedral(perm[np.array(p['input'])], k), dihedral(perm[np.array(p['output'])], k))
                         for p in task['train']]
                for qi, (q_in, q_out) in enumerate(demos):
                    others = [d for j, d in enumerate(demos) if j != qi]
                    if use_all:
                        ctx = [others[j] for j in rng.permutation(len(others))]
                    else:
                        ctx = [others[j] for j in rng.choice(len(others), K, replace=len(others) < K)]
                    x, y = sample(ctx, q_in, q_out)
                    tr_x.append(x); tr_y.append(y)
        demos = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
        ctx = demos if use_all else [demos[j % len(demos)] for j in range(K)]
        x, y = sample(ctx, np.array(task['test'][0]['input']), np.array(task['test'][0]['output']))
        te_x.append(x); te_y.append(y)
    order = rng.permutation(len(tr_x))
    u8 = lambda a: np.array(a, dtype=np.uint8)
    np.savez_compressed(opt.out, train_inputs=u8(tr_x)[order], train_labels=u8(tr_y)[order], test_inputs=u8(te_x),
                        test_labels=u8(te_y), num_symbols=np.int64(10))
    meta = dict(tasks=len(files), k=K, all_demonstrations=use_all, grids_per_sample=2 * K + 1, train_samples=len(tr_x), test_samples=len(te_x),
                seed=opt.seed, source='github.com/KSB21ST/MINI-ARC data/MiniARC', task_identifiers='none')
    Path(opt.out).with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
