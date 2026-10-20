"""Independent checks of reported toy results: metrics from raw predictions with Wilson intervals, and leakage checks.

    from toy_g import audit
    audit.report(pred, k=3)                 # pred [3072, 4, 4] predicted test outputs (raw colours) for the instances
    audit.check_training_tasks(tasks)       # every training task must be in group 0
"""
import math

import numpy as np

from . import data as D


def wilson(successes, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def metrics(pred, k=3, determined_only=False, inst=None):
    """Per group: n, exact (all 16 cells), its Wilson 95% interval, and cell accuracy."""
    inst = inst or D.load_instances()
    target = inst['test_out'].astype(np.int64)
    pred = np.asarray(pred).astype(np.int64).reshape(target.shape)
    keep = inst['determined_k'][:, k - 1] if determined_only else np.ones(len(target), dtype=bool)
    exact = (pred == target).reshape(len(target), -1).all(1)
    cell = (pred == target).reshape(len(target), -1).mean(1)
    rows = {}
    for g, name in enumerate(D.GROUP_NAMES):
        sel = (inst['group'] == g) & keep
        n, s = int(sel.sum()), int(exact[sel].sum())
        lo, hi = wilson(s, n)
        rows[name] = dict(n=n, exact=s / max(n, 1), exact_ci=(lo, hi), cell=float(cell[sel].mean()) if n else float('nan'))
    return rows


def report(pred, k=3, determined_only=False):
    rows = metrics(pred, k, determined_only)
    for name, r in rows.items():
        print(f"{name:14s} n={r['n']:5d} exact {100 * r['exact']:6.2f}% "
              f"[{100 * r['exact_ci'][0]:.1f}, {100 * r['exact_ci'][1]:.1f}]  cell {100 * r['cell']:6.2f}%")
    return rows


def predict_structure_model(path, k=3, device='cuda', batch=512, inst=None):
    """Predicted test outputs [3072, 4, 4] of a toy_g.train_g checkpoint (final.pt) from the first k demonstrations.
    Only the model's forward is reused; batching, grouping and scoring are this module's own."""
    import torch
    from .model import StructureModel
    inst = inst or D.load_instances()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    c = ck['config']
    model = StructureModel(mode=c['mode'], d=c['d'], generators=c['generators'], cmap=c['cmap'],
                           sinkhorn_iters=c['sinkhorn_iters'], cmap_eps=c['cmap_eps'], tau=c['tau'], beta=c['beta'],
                           gen_init=c['gen_init'], seed=c['seed'], compose=c.get('compose', 'code'))
    model.load_state_dict(ck['model'])
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(inst['test_in']), batch):
            sl = slice(s, s + batch)
            din = torch.as_tensor(inst['demo_in'][sl, :k].astype(np.int64), device=device)
            dout = torch.as_tensor(inst['demo_out'][sl, :k].astype(np.int64), device=device)
            qin = torch.as_tensor(inst['test_in'][sl].astype(np.int64), device=device)
            mask = torch.ones(din.shape[:2], dtype=torch.bool, device=device)
            preds.append(model(din, dout, mask, qin)['probs'].argmax(-1).cpu().numpy())
    return np.concatenate(preds).reshape(-1, 4, 4)


def check_training_tasks(tasks, inst=None):
    inst = inst or D.load_instances()
    tasks = np.unique(np.asarray(tasks))
    groups = np.bincount(inst['group'][tasks], minlength=3)
    if groups[1] or groups[2]:
        raise AssertionError(f'training used held-out tasks: group counts {groups.tolist()}')
    return groups.tolist()
