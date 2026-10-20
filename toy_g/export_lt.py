"""Export the 4x4 torus toy (toy_g) to the formats of the lt harness (python -m lt.train): the existing-lineage
baselines (puzzle embedding, transductive and inductive; demonstrations in context) on the same tasks.

Conventions follow data/prep_miniarc.py and data/prep_miniarc_ctx.py: inputs and labels hold colour + 1 (raw colours
0..4 -> 1..5, num_symbols = 5; lt.data adds 1 more for the tokens, so the vocabulary is 0 pad, 1 blank, 2..6 colours
0..4 and no input cell is blank). Every file has test_group per test row and train_group per training row
(0 train task, 1 held-out (t, c) combination, 2 task of a held-out transformation; toy_g.data.GROUP_NAMES). The test
rows are the fixed instances of data/gtorus_instances.npz in task order (row i = task i = t * 24 + c).

- data/gtorus_id_trans.npz (ID format of data/miniarc.npz, transductive as HRM/URM train): train = the 3 fixed
  demonstration pairs of every task (9,216), train_task = task index = puzzle identifier; test = every task's fixed
  test pair (3,072), test_task.
- data/gtorus_id_ind.npz: the same, training pairs of group-0 tasks only (7,257). The test is all 3,072; the held-out
  identifiers are never trained (their embedding rows keep the zero initialization).
- data/gtorus_ctx.npz (in-context format of data/miniarc_ctx.npz, no identifiers): a sample stacks the 3
  demonstration pairs and the query input along rows (in1, out1, in2, out2, in3, out3, query in: 7 grids, [28, 4]);
  labels are 0 (outside the loss, ignore_label0) on the 6 demonstration grids and the query output + 1 on the query
  grid. Train: fresh group-0 samples from toy_g.data.sample_batch (k = 3, query output determined by the
  demonstrations), chunk i drawn from SeedSequence(seed).spawn(...)[i] (the result does not depend on --workers);
  test: the fixed instances (their 3 demonstrations and test pair). train_task_index / test_task_index record the tasks
  as metadata only: the harness takes identifiers from a train_task key, which this file does not have.

    python -m toy_g.export_lt [--only id_trans id_ind ctx] [--ctx-samples 100000] [--seed 0] [--workers 16]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import data as D
from . import group as G

DATA = Path(__file__).resolve().parents[1] / 'data'
K = 3                                     # demonstrations per instance and per in-context sample
NUM_SYMBOLS = G.COLORS + 1                # raw colours 0..4
CTX_GRIDS = 2 * K + 1


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _group_counts(group):
    return {name: int((group == g).sum()) for g, name in enumerate(D.GROUP_NAMES)}


def _write(out, arrays, meta):
    np.savez_compressed(out, **arrays)
    Path(out).with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'wrote {out}: ' + json.dumps(meta))


def export_id(inst, inductive, seed, out):
    group = inst['group'].astype(np.int64)
    tasks = np.flatnonzero(group == 0) if inductive else np.arange(G.NUM_TASKS)
    tr_x = inst['demo_in'][tasks].reshape(-1, G.N, G.N)               # task-major: the K pairs of a task in a row
    tr_y = inst['demo_out'][tasks].reshape(-1, G.N, G.N)
    tr_t = np.repeat(tasks, K)
    order = np.random.default_rng(seed).permutation(len(tr_x))         # as data/prep_miniarc.py
    u8 = lambda a: (np.asarray(a, dtype=np.uint8) + 1)
    test_task = np.arange(G.NUM_TASKS)
    same_input = (inst['demo_in'] == inst['test_in'][:, None]).reshape(G.NUM_TASKS, K, -1).all(-1).any(-1)
    arrays = dict(train_inputs=u8(tr_x)[order], train_labels=u8(tr_y)[order], train_task=tr_t[order].astype(np.int32),
                  train_group=group[tr_t[order]], test_inputs=u8(inst['test_in']), test_labels=u8(inst['test_out']),
                  test_task=test_task.astype(np.int32), test_group=group[test_task], num_symbols=np.int64(NUM_SYMBOLS))
    meta = dict(format='id (data/prep_miniarc.py)', training='inductive: group-0 tasks only' if inductive else
                'transductive: every task', tasks=G.NUM_TASKS, identifiers=G.NUM_TASKS,
                trained_identifiers=int(len(tasks)), train_pairs=int(len(tr_x)), test_pairs=G.NUM_TASKS,
                num_symbols=NUM_SYMBOLS, test_groups=_group_counts(group[test_task]),
                train_groups=_group_counts(group[tr_t]), test_rows='task order (row i = task i = t * 24 + c)',
                train_order=f'shuffled (default_rng({seed}).permutation)',
                test_input_equals_a_demonstration_input=int(same_input.sum()),
                test_output_equals_input=int((inst['test_in'] == inst['test_out']).reshape(G.NUM_TASKS, -1).all(-1).sum()),
                source='data/gtorus_instances.npz (toy_g.data.task_instances seed 0, make_split seed 0)',
                instances_sha256=_sha256(D.INSTANCES))
    _write(out, arrays, meta)


def ctx_stack(demo_in, demo_out, query_in, query_out):
    """[n,K,4,4] x2, [n,4,4] x2 (raw colours) -> inputs [n, 4 (2K+1), 4] (colour + 1) and labels (0 on the
    demonstration grids, query output + 1 on the query grid), data/prep_miniarc_ctx.py's layout."""
    n = len(query_in)
    grids = [g for j in range(K) for g in (demo_in[:, j], demo_out[:, j])] + [query_in]
    x = np.concatenate(grids, axis=1).astype(np.int64) + 1
    y = np.zeros_like(x)
    y[:, 2 * K * G.N:] = np.asarray(query_out, dtype=np.int64).reshape(n, G.N, G.N) + 1
    return x.astype(np.uint8), y.astype(np.uint8)


def _ctx_chunk(job):
    seed_seq, n, tasks = job
    b = D.sample_batch(np.random.default_rng(seed_seq), n, K, tasks)
    return {key: (v.astype(np.uint8) if key != 'task' else v.astype(np.int32)) for key, v in b.items()}


def _sample_key(din, dout, qin):
    return np.concatenate([din.reshape(len(qin), -1), dout.reshape(len(qin), -1), qin.reshape(len(qin), -1)],
                          1).astype(np.uint8)


def export_ctx(inst, n_samples, seed, chunk, workers, out):
    group = inst['group'].astype(np.int64)
    train_tasks = np.flatnonzero(group == 0)
    sizes = [min(chunk, n_samples - s) for s in range(0, n_samples, chunk)]
    seqs = np.random.SeedSequence(seed).spawn(len(sizes))
    began = time.monotonic()
    jobs = [(sq, n, train_tasks) for sq, n in zip(seqs, sizes)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_ctx_chunk, jobs))
    else:
        parts = [_ctx_chunk(j) for j in jobs]
    seconds = time.monotonic() - began
    b = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    tr_x, tr_y = ctx_stack(b['demo_in'], b['demo_out'], b['query_in'], b['query_out'])
    te_x, te_y = ctx_stack(inst['demo_in'], inst['demo_out'], inst['test_in'], inst['test_out'])
    tr_t = b['task'].astype(np.int64)
    # overlap of the training samples with the fixed test instances (same demonstrations and query input)
    tr_key = _sample_key(b['demo_in'], b['demo_out'], b['query_in'])
    te_key = _sample_key(inst['demo_in'], inst['demo_out'], inst['test_in'])
    te_set = {r.tobytes() for r in te_key}
    overlap = sum(r.tobytes() in te_set for r in tr_key)
    # training queries that are the test pair of their own task (same task, same query input)
    test_pairs = {(t, inst['test_in'][t].tobytes()) for t in range(G.NUM_TASKS)}
    query_overlap = sum((int(t), q.tobytes()) in test_pairs for t, q in zip(tr_t, b['query_in']))
    test_task = np.arange(G.NUM_TASKS)
    per_task = np.bincount(tr_t, minlength=G.NUM_TASKS)[train_tasks]
    arrays = dict(train_inputs=tr_x, train_labels=tr_y, train_task_index=tr_t.astype(np.int32),
                  train_group=group[tr_t], test_inputs=te_x, test_labels=te_y,
                  test_task_index=test_task.astype(np.int32), test_group=group[test_task],
                  num_symbols=np.int64(NUM_SYMBOLS))
    meta = dict(format='in context (data/prep_miniarc_ctx.py)', k=K, grids_per_sample=CTX_GRIDS,
                sample_shape=[CTX_GRIDS * G.N, G.N], train_samples=int(n_samples), test_samples=G.NUM_TASKS,
                num_symbols=NUM_SYMBOLS, task_identifiers='none (train_task_index / test_task_index are metadata)',
                train_tasks='group 0 only (2,419), drawn uniformly per sample',
                train_samples_per_task=dict(min=int(per_task.min()), mean=float(per_task.mean()),
                                            max=int(per_task.max())),
                train_generator=f'toy_g.data.sample_batch(rng, n, k={K}, group-0 tasks, sparsity (3, 8)); chunk i of '
                                f'{chunk} samples uses default_rng(SeedSequence({seed}).spawn({len(sizes)})[i])',
                train_order='as generated (tasks are already drawn at random)',
                generation_seconds=round(seconds, 1), workers=workers,
                train_samples_equal_to_a_test_instance=int(overlap),
                train_queries_equal_to_their_task_test_pair=int(query_overlap),
                test_groups=_group_counts(group[test_task]), test_rows='task order (row i = task i = t * 24 + c)',
                source='data/gtorus_instances.npz (toy_g.data.task_instances seed 0, make_split seed 0)',
                instances_sha256=_sha256(D.INSTANCES), seed=seed)
    _write(out, arrays, meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', default=['id_trans', 'id_ind', 'ctx'], choices=['id_trans', 'id_ind', 'ctx'])
    ap.add_argument('--ctx-samples', type=int, default=100_000)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=str(DATA))
    opt = ap.parse_args(argv)
    inst = D.load_instances()
    out = Path(opt.out_dir)
    if 'id_trans' in opt.only:
        export_id(inst, False, opt.seed, out / 'gtorus_id_trans.npz')
    if 'id_ind' in opt.only:
        export_id(inst, True, opt.seed, out / 'gtorus_id_ind.npz')
    if 'ctx' in opt.only:
        export_ctx(inst, opt.ctx_samples, opt.seed, opt.chunk, opt.workers, out / 'gtorus_ctx.npz')


if __name__ == '__main__':
    main()
