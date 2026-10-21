"""Build data/miniarc_pairs.npz: Mini-ARC (149 tasks, grids <= 5 x 5) as a pair store for lt.data.ArcContextDataset /
ctx_packed (data/prep_arc_ctx.py layout), every task transductive as URM/HRM: its train pairs are training pairs
(episodes among them), its test pair is evaluated with all its train pairs as demonstrations and never enters training.
For the A/B of 2026-10-21 (puzzle embedding vs puzzle embedding + rule synapse) at a fraction of ARC's sequence length.

S = 6 (a side-5 grid and its EOS row/column), K = 8 slots (the largest train-pair count, so no demonstration is cut;
ctx_packed drops the unused slots).

    python data/prep_miniarc_pairs.py PATH/TO/MINI-ARC --out data/miniarc_pairs.npz
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lt.data import ctx_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('--out', default='data/miniarc_pairs.npz')
    ap.add_argument('--kmax', type=int, default=8)
    args = ap.parse_args()
    files = sorted(glob.glob(f'{args.repo}/data/MiniARC/*.json'))
    tasks = [(os.path.basename(f)[:-5], json.load(open(f))) for f in files]
    side = max(max(len(g), len(g[0])) for _, t in tasks for p in t['train'] + t['test'] for g in (p['input'], p['output']))
    S, K = side + 1, args.kmax
    pairs = lambda ps: [(np.asarray(p['input'], np.uint8), np.asarray(p['output'], np.uint8)) for p in ps]

    cells, off, hw, ptask, ptest = [], [], [], [], []
    pos = 0
    tr_in, tr_lab, te_in, te_lab, te_task = [], [], [], [], []
    for ti, (_, task) in enumerate(tasks):
        demos, tests = pairs(task['train']), pairs(task['test'])
        for is_test, ps in ((False, demos), (True, tests)):
            for x, y in ps:
                off.append(pos)
                hw.append((*x.shape, *y.shape))
                ptask.append(ti)
                ptest.append(is_test)
                cells += [x.ravel(), y.ravel()]
                pos += x.size + y.size
        x, y = ctx_episode(demos[:-1], demos[-1], S, K)
        tr_in.append(x)
        tr_lab.append(y)
        for q in tests:
            x, y = ctx_episode(demos[:K], q, S, K)
            te_in.append(x)
            te_lab.append(y)
            te_task.append(ti)
    try:
        commit = subprocess.run(['git', '-C', args.repo, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    n = len(tasks)
    np.savez_compressed(
        args.out,
        pair_cells=np.concatenate(cells), pair_off=np.array(off, np.int64), pair_hw=np.array(hw, np.int16),
        pair_task=np.array(ptask, np.int32), pair_test=np.array(ptest, bool), pair_trainable=~np.array(ptest, bool),
        task_names=np.array([t[0] for t in tasks]), task_split=np.ones(n, np.int8),
        train_inputs=np.stack(tr_in), train_labels=np.stack(tr_lab),
        test_inputs=np.stack(te_in), test_labels=np.stack(te_lab), ctx_test_task=np.array(te_task, np.int32),
        ctx_slot=np.int32(S), ctx_kmax=np.int32(K), ctx_max_side=np.int32(side), num_symbols=np.int32(11),
        sources=np.array([f'KSB21ST/MINI-ARC {commit}']))
    meta = dict(tasks=n, pairs=len(off), training_episodes_per_epoch=len(tr_in), eval_episodes=len(te_in), slot=S,
                kmax=K, sample_shape=list(tr_in[0].shape), source=f'KSB21ST/MINI-ARC {commit}')
    Path(args.out).with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
