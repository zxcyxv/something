"""Build data/miniarc.npz from the Mini-ARC tasks (Kim et al. 2022; github.com/KSB21ST/MINI-ARC, data/MiniARC/*.json).

Every task (149, all grids 5x5, colors 0..9) is expanded into versions = 8 dihedral transforms x `--color-perms`
permutations of colors 1..9 (0 kept); each version is its own puzzle identifier (HRM-style, transductive).
Training: the demonstration pairs of every version. Test: the test pair of the identity version of every task.
Arrays follow lt/data.py with num_symbols = 10: inputs and labels hold color + 1 (1..10), so no cell is blank.

    python data/prep_miniarc.py PATH/TO/MINI-ARC/data/MiniARC [--color-perms 4] [--seed 0]
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
    ap.add_argument('--color-perms', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'miniarc.npz'))
    opt = ap.parse_args()
    rng = np.random.default_rng(opt.seed)
    files = sorted(glob.glob(str(Path(opt.src) / '*.json')))
    nv = 8 * opt.color_perms
    tr_x, tr_y, tr_t, te_x, te_y, te_t, names = [], [], [], [], [], [], []
    for ti, f in enumerate(files):
        task = json.load(open(f))
        names.append(Path(f).stem)
        perms = [np.arange(10)] + [np.concatenate([[0], 1 + rng.permutation(9)]) for _ in range(opt.color_perms - 1)]
        for ci, perm in enumerate(perms):
            for k in range(8):
                vid = ti * nv + ci * 8 + k
                for p in task['train']:
                    tr_x.append(dihedral(perm[np.array(p['input'])], k))
                    tr_y.append(dihedral(perm[np.array(p['output'])], k))
                    tr_t.append(vid)
        p = task['test'][0]
        te_x.append(np.array(p['input']))
        te_y.append(np.array(p['output']))
        te_t.append(ti * nv)                                  # identity version
    order = rng.permutation(len(tr_x))
    arr = lambda a, o=None: (np.array(a, dtype=np.uint8) + 1)[o] if o is not None else np.array(a, dtype=np.uint8) + 1
    np.savez_compressed(opt.out, train_inputs=arr(tr_x, order), train_labels=arr(tr_y, order),
                        train_task=np.array(tr_t, dtype=np.int32)[order], test_inputs=arr(te_x), test_labels=arr(te_y),
                        test_task=np.array(te_t, dtype=np.int32), num_symbols=np.int64(10))
    meta = dict(tasks=len(files), versions_per_task=nv, identifiers=len(files) * nv, train_pairs=len(tr_x),
                test_pairs=len(te_x), seed=opt.seed, source='github.com/KSB21ST/MINI-ARC data/MiniARC',
                identity_output_equals_input=int(sum((np.array(a) == np.array(b)).all() for a, b in zip(te_x, te_y))))
    Path(opt.out).with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
