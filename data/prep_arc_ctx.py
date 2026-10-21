"""Build data/arc_ctx15.npz: ARC tasks whose grids all have sides <= 15, in context, without task identifiers
(2026-10-21, docs/claude_2026-10-21_prior_universality.md section 6).

Training pool (split 0): ARC-AGI-1 training + the ARC-AGI-2 training tasks whose ids are not ARC-AGI-1 tasks.
Evaluation (split 1): ARC-AGI-1 evaluation, tasks never seen in training.

A sample stacks 2 K + 2 canvases of S x S along rows (lt.data.ctx_episode; S = 16 holds a side-15 grid and its EOS
row/column, K = 5 demonstration slots): demonstrations (input, output) ..., unused slots pad, then the query input and
the query output (pad in the input, the output canvas in the label). Raw values as data/prep_arc1.py: 0 pad,
1..10 colors 0..9, 11 EOS; lt.data adds 1 and ignore_label0 drops raw 0, so the loss covers the query output's cells
and EOS row/column only.

Stored:
    pair_cells (uint8, raw colors), pair_off, pair_hw (h_in, w_in, h_out, w_out), pair_task, pair_test (official test
    pair): every pair of every task, read by lt.data.ArcContextDataset, which draws training episodes on the fly
    task_names, task_split
    train_inputs / train_labels: one identity episode per training task (query = its last test pair), used by
        lt.train for the epoch length and the data fingerprint
    test_inputs / test_labels / ctx_test_task: every official test pair of every evaluation task, demonstrations =
        that task's train pairs (the first K), identity version
    pair_trainable: pairs that training episodes may use
    ctx_slot = S, ctx_kmax = K, ctx_max_side, num_symbols = 11, sources

--transductive-half SEED splits the evaluation tasks at random into two halves (task_eval_mode 1 and 2):
    mode 1 (transductive, as URM/HRM): the task's train pairs are training pairs as well (episodes among them; the
           test pairs never enter training), so the model has seen the same information a puzzle embedding gets
    mode 2 (inductive): never seen in training
train_task (zeros) and test_task (the mode of each evaluation episode) then let lt.train report the two halves
separately (task_ids_as_identifier false keeps the model's identifier 0).

--transductive-all puts every evaluation task in mode 1 (URM/HRM: the evaluation tasks' train pairs are training
pairs, their test pairs never enter training), for runs with a puzzle embedding per task (lt.data.ArcContextDataset
num_aug versions).
--evaluate-train-tasks (with --transductive-all) treats the training tasks the same way: every task is transductive
and evaluated on its test pairs, which never enter training (the transductive A/B of 2026-10-21 gains evaluation
pairs); --no-arc2 leaves out the ARC-AGI-2 training tasks.

    python data/prep_arc_ctx.py ARC_AGI_REPO ARC_AGI_2_REPO --out data/arc_ctx15.npz
    python data/prep_arc_ctx.py ARC_AGI_REPO ARC_AGI_2_REPO --transductive-half 0 --out data/arc_ctx15_half.npz
    python data/prep_arc_ctx.py ARC_AGI_REPO ARC_AGI_2_REPO --transductive-all --out data/arc_ctx15_trans.npz
    python data/prep_arc_ctx.py ARC_AGI_REPO ARC_AGI_2_REPO --max-side 10 --transductive-all --evaluate-train-tasks \
        --no-arc2 --out data/arc1_ctx10_trans.npz
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


def load_dir(d):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{d}/*.json'))}


def max_side(task):
    return max(max(len(g), len(g[0])) for p in task['train'] + task['test'] for g in (p['input'], p['output']))


def commit(repo):
    try:
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arc1')
    ap.add_argument('arc2')
    ap.add_argument('--out', default='data/arc_ctx15.npz')
    ap.add_argument('--max-side', type=int, default=15)
    ap.add_argument('--kmax', type=int, default=5)
    ap.add_argument('--transductive-half', type=int, default=None, metavar='SEED')
    ap.add_argument('--transductive-all', action='store_true')
    ap.add_argument('--evaluate-train-tasks', action='store_true')
    ap.add_argument('--no-arc2', action='store_true')
    args = ap.parse_args()
    S, K = args.max_side + 1, args.kmax
    a1_train, a1_eval = load_dir(f'{args.arc1}/data/training'), load_dir(f'{args.arc1}/data/evaluation')
    a2_train = load_dir(f'{args.arc2}/data/training')
    a1_ids = set(a1_train) | set(a1_eval)
    tasks = ([(f'arc1/training/{k}', v, 0) for k, v in a1_train.items()]
             + [(f'arc2/training/{k}', v, 0) for k, v in a2_train.items() if k not in a1_ids and not args.no_arc2]
             + [(f'arc1/evaluation/{k}', v, 1) for k, v in a1_eval.items()])
    tasks = [t for t in tasks if max_side(t[1]) <= args.max_side]
    if args.evaluate_train_tasks:
        if not args.transductive_all:
            raise SystemExit('--evaluate-train-tasks needs --transductive-all')
        tasks = [(n, t, 1) for n, t, _ in tasks]
    mode = np.array([0 if s == 0 else 2 for _, _, s in tasks], np.int8)       # 0 train, 1 transductive, 2 inductive
    if args.transductive_half is not None:
        ev = np.flatnonzero(mode == 2)
        rng = np.random.default_rng(args.transductive_half)
        mode[rng.permutation(ev)[:len(ev) // 2]] = 1
    if args.transductive_all:
        mode[mode == 2] = 1

    cells, off, hw, ptask, ptest, ptrain = [], [], [], [], [], []
    pos = 0
    for ti, (_, task, _) in enumerate(tasks):
        for is_test, pairs in ((False, task['train']), (True, task['test'])):
            for p in pairs:
                x, y = np.asarray(p['input'], np.uint8), np.asarray(p['output'], np.uint8)
                off.append(pos)
                hw.append((*x.shape, *y.shape))
                ptask.append(ti)
                ptest.append(is_test)
                ptrain.append(bool(mode[ti] == 0 or (mode[ti] == 1 and not is_test)))
                cells += [x.ravel(), y.ravel()]
                pos += x.size + y.size

    pairs = lambda task: [(np.asarray(p['input'], np.uint8), np.asarray(p['output'], np.uint8)) for p in task]
    tr_in, tr_lab, te_in, te_lab, te_task, te_mode = [], [], [], [], [], []
    for ti, (_, task, split) in enumerate(tasks):
        demos, tests = pairs(task['train']), pairs(task['test'])
        if mode[ti] == 0:
            x, y = ctx_episode(demos + tests[:-1], tests[-1], S, K)
        elif mode[ti] == 1:
            x, y = ctx_episode(demos[:-1], demos[-1], S, K)
        if mode[ti] in (0, 1):
            tr_in.append(x)
            tr_lab.append(y)
        if split == 1:
            for q in tests:
                x, y = ctx_episode(demos, q, S, K)
                te_in.append(x)
                te_lab.append(y)
                te_task.append(ti)
                te_mode.append(int(mode[ti]))

    split = np.array([t[2] for t in tasks], np.int8)
    seen = {g.tobytes() + bytes(g.shape) for _, task, s in tasks if s == 0 for x, y in pairs(task['train'] + task['test'])
            for g in (x, y)}
    leaked = sum(x.tobytes() + bytes(x.shape) in seen for _, task, s in tasks if s == 1 for x, _ in pairs(task['test']))
    extra = {}
    if args.transductive_half is not None:
        extra = dict(train_task=np.zeros(len(tr_in), np.int32), test_task=np.array(te_mode, np.int32),
                     task_eval_mode=mode)
    np.savez_compressed(
        args.out, **extra,
        pair_cells=np.concatenate(cells), pair_off=np.array(off, np.int64), pair_hw=np.array(hw, np.int16),
        pair_task=np.array(ptask, np.int32), pair_test=np.array(ptest, bool), pair_trainable=np.array(ptrain, bool),
        task_names=np.array([t[0] for t in tasks]), task_split=split,
        train_inputs=np.stack(tr_in), train_labels=np.stack(tr_lab),
        test_inputs=np.stack(te_in), test_labels=np.stack(te_lab), ctx_test_task=np.array(te_task, np.int32),
        ctx_slot=np.int32(S), ctx_kmax=np.int32(K), ctx_max_side=np.int32(args.max_side), num_symbols=np.int32(11),
        sources=np.array([f'fchollet/ARC-AGI {commit(args.arc1)}', f'arcprize/ARC-AGI-2 {commit(args.arc2)}']))
    meta = dict(tasks_train=int((split == 0).sum()), tasks_eval=int((split == 1).sum()), pairs=len(off),
                training_episodes_per_epoch=len(tr_in), eval_transductive_tasks=int((mode == 1).sum()),
                eval_inductive_tasks=int((mode == 2).sum()),
                eval_episodes_by_mode={int(m): int(np.sum(np.array(te_mode) == m)) for m in (1, 2)},
                eval_episodes=len(te_in), slot=S, kmax=K, sample_shape=list(tr_in[0].shape),
                eval_test_inputs_seen_in_training=int(leaked),
                eval_tasks_with_more_than_kmax_demos=int(sum(len(t['train']) > K for _, t, s in tasks if s == 1)))
    Path(args.out).with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
