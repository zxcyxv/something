"""dense50 + few-shot data: group-0 tasks keep their 50 fresh pairs (data/gtorus_id_dense50.npz); every group-1/2 task adds
its 3 demonstrations from data/gtorus_instances.npz, repeated `repeat` times so each task is drawn about as often as a
group-0 task (51 vs 50 rows) while seeing only 3 distinct pairs. Test = the instance test pair of every task (as dense50).
Tokens are colour + 1 (the LT files' convention).

    python -m toy_g.export_mixed            # -> data/gtorus_id_dense50_fs3.npz
"""
import argparse
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'


def build(dense, inst, repeat):
    few = np.flatnonzero(inst['group'] > 0)
    din = inst['demo_in'][few].reshape(-1, 4, 4) + 1                   # [n_few * 3, 4, 4]
    dout = inst['demo_out'][few].reshape(-1, 4, 4) + 1
    task = np.repeat(inst['task'][few], 3).astype(np.int32)
    grp = np.repeat(inst['group'][few], 3)
    tile = lambda a: np.tile(a, (repeat,) + (1,) * (a.ndim - 1))
    out = {k: dense[k] for k in ('test_inputs', 'test_labels', 'test_task', 'test_group', 'num_symbols')}
    out['train_inputs'] = np.concatenate([dense['train_inputs'], tile(din).astype(np.uint8)])
    out['train_labels'] = np.concatenate([dense['train_labels'], tile(dout).astype(np.uint8)])
    out['train_task'] = np.concatenate([dense['train_task'], tile(task)])
    out['train_group'] = np.concatenate([dense['train_group'], tile(grp)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeat', type=int, default=17)
    ap.add_argument('--out', default=str(DATA / 'gtorus_id_dense50_fs3.npz'))
    opt = ap.parse_args(argv)
    out = build(np.load(DATA / 'gtorus_id_dense50.npz'), np.load(DATA / 'gtorus_instances.npz'), opt.repeat)
    np.savez_compressed(opt.out, **out)
    rows = np.bincount(out['train_task'], minlength=3072)
    print(f"{opt.out}: train {len(out['train_task'])} rows; rows per task by group "
          f"{[sorted(set(rows[out['test_group'] == g].tolist())) for g in range(3)]}; test {len(out['test_task'])}")


if __name__ == '__main__':
    main()
