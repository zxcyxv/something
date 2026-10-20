"""Fresh in-context toy data at scale: n determined samples (k = 3 demonstrations) from the group-0 tasks, drawn on the
GPU by toy_g.train_g.TorchSampler (the distribution of toy_g.data.sample_batch), in the layout of data/gtorus_ctx.npz
(7 stacked grids, colour + 1, labels only on the query grid). The test rows are copied from data/gtorus_ctx.npz.

    python -m toy_g.make_ctx_fresh [--n 1000000]       # -> data/gtorus_ctx_1m.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .export_lt import ctx_stack
from .train_g import TorchSampler

DATA = Path(__file__).resolve().parents[1] / 'data'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=str(DATA / 'gtorus_ctx_1m.npz'))
    opt = ap.parse_args(argv)
    split = D.make_split(seed=0)
    sampler = TorchSampler(np.sort(split['train']), 'cuda' if torch.cuda.is_available() else 'cpu', seed=opt.seed)
    xs, ys, ts = [], [], []
    for s in range(0, opt.n, 50_000):
        din, dout, qin, qout, task = (x.cpu().numpy() for x in sampler.sample(min(50_000, opt.n - s), 3))
        x, y = ctx_stack(din.reshape(-1, 3, 4, 4), dout.reshape(-1, 3, 4, 4), qin.reshape(-1, 4, 4), qout.reshape(-1, 4, 4))
        xs.append(x); ys.append(y); ts.append(task)
    ref = np.load(DATA / 'gtorus_ctx.npz')
    out = dict(train_inputs=np.concatenate(xs), train_labels=np.concatenate(ys),
               train_task_index=np.concatenate(ts).astype(np.int32), train_group=np.zeros(opt.n, np.int64),
               **{k: ref[k] for k in ('test_inputs', 'test_labels', 'test_task_index', 'test_group', 'num_symbols')})
    assert (split['group'][out['train_task_index']] == 0).all()
    np.savez_compressed(opt.out, **out)
    print(f"{opt.out}: train {len(out['train_inputs'])} samples, shape {out['train_inputs'].shape[1:]}, "
          f"tasks drawn {len(np.unique(out['train_task_index']))}")


if __name__ == '__main__':
    main()
