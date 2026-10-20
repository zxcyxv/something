"""Part-selection executor, version 3: invented predicates and a learned prior over programs, with Bayesian model
averaging, trained across tasks and evaluated by 5-fold cross-validation over tasks (no held-out test output is used).

Library: version 1 (toy_g/pse.py) plus predicates invented by a fixed grammar over its features (learning predicate
    functions from 149 tasks would memorise; what is learned is which ones are cheap):
        [f >= k], k = 1, 2, 3, for the 16 count features; [dist >= k], k = 2, 3, for the 8 distances; same4 / same8 =
        number of 4- / 8-neighbours of the cell's own colour; enc1 = background cell whose four neighbours are all of
        one non-background colour
Candidates: every consistent program with a complete test prediction (feature sets minimal by inclusion, m <= 2,
    families table / role / role2), the K shortest per task by the version-1 bit length (recomputed for 126 features).
Prior: log P(p | demos) = -w . phi(p) + const over the task's candidates; phi = bit length, family, source kind, number
    of features of each kind, table keys, keys seen on one cell, share of demonstration cells explained by copying, m.
    w starts at (ln 2 on the bit length, 0 elsewhere): the version-1 code.
Prediction: MAP program (all maximal programs must agree), or the posterior mode over test outputs (BMA).
Training: maximise sum_t log P(y_t | demos_t) over the training tasks that have a correct candidate, L2 towards the
    start; Adam on standardised phi.

    python -m toy_g.pse3 export      # candidates of all tasks -> scratch file (GPU, about a minute)
    python -m toy_g.pse3 cv          # 5-fold cross-validation table -> runs/pse3_cv.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import miniarc_grammar as MG
from . import pse
from .miniarc_ceiling import load_tasks

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/tmp/claude-0/-workspace-something/ac319210-15d9-4a7a-9dc0-1f7bf199b6ce/scratchpad/pse3_candidates.pt')
COUNT_FEATS = ['rowc', 'colc', 'diagc', 'antic', 'n4c', 'n8c', 'globc', 'objsize'] + [f'cnt_{d}' for d in pse.DIR_NAMES]
DIST_FEATS = [f'dist_{d}' for d in pse.DIR_NAMES]
KINDS = ['pred', 'count', 'cnt', 'dist', 'pos', 'grav', 'thr', 'same']
SKINDS = ['ident', 'd4', 'shift', 'd4shift', 'anchor', 'object', 'scan', 'pool']
PHI = ['bits'] + [f'fam_{f}' for f in pse.FAMILIES] + [f'src_{k}' for k in SKINDS] + [f'feat_{k}' for k in KINDS] + \
      ['keys', 'singles', 'copy_frac', 'm']


def invented(g, f):
    out = {}
    for k in COUNT_FEATS:
        for t in (1, 2, 3):
            out[f'{k}>={t}'] = (f[k] >= t).astype(int)
    for k in DIST_FEATS:
        for t in (2, 3):
            out[f'{k}>={t}'] = (f[k] >= t).astype(int)
    out['same4'] = np.array([sum(g[j] == g[i] for j in MG.N4[i]) for i in range(25)])
    out['same8'] = np.array([sum(g[j] == g[i] for j in MG.N8[i]) for i in range(25)])
    out['enc1'] = np.array([int(g[i] == 0 and len(MG.N4[i]) == 4 and g[MG.N4[i][0]] != 0
                                and len({int(g[j]) for j in MG.N4[i]}) == 1) for i in range(25)])
    return out


def bank3(task):
    B = pse.bank(task, True)
    grids = np.concatenate([task['x'], task['tx'][None]])
    names = B['feat_names']
    inv = [invented(g, {k: B['feat'][names.index(k), gi] for k in COUNT_FEATS + DIST_FEATS}) for gi, g in enumerate(grids)]
    knames = list(inv[0])
    B['feat'] = np.concatenate([B['feat'], np.stack([np.stack([d[k] for d in inv]) for k in knames])])
    B['feat_names'] = names + knames
    return B


def feature_kind(n):
    if n in MG.PRED_NAMES:
        return 'pred'
    if '>=' in n:
        return 'thr'
    if n in ('same4', 'same8', 'enc1'):
        return 'same'
    if n.startswith('cnt_'):
        return 'cnt'
    if n.startswith('dist_'):
        return 'dist'
    if n in ('row', 'col', 'rpc', 'rmc'):
        return 'pos'
    if n.startswith('grav_'):
        return 'grav'
    return 'count'


def source_kind(i, n):
    if i == pse.IDENT:
        return 'ident'
    if n.startswith('D') and '+(' in n:
        k, shift = int(n[1]), n.split('+')[1] != '(0,0)'
        return 'd4shift' if k and shift else ('d4' if k else 'shift')
    if n.startswith('anchor_'):
        return 'anchor'
    if n.startswith('object_'):
        return 'object'
    if n.startswith('scan_') or n.startswith('nearest_'):
        return 'scan'
    return 'pool'


@torch.no_grad()
def export_task(B, dev, K):
    T = pse.Task(B, dev)
    fk = torch.tensor([KINDS.index(feature_kind(n)) for n in B['feat_names']], device=dev)
    sk = torch.tensor([SKINDS.index(source_kind(i, n)) for i, n in enumerate(B['src_names'])], device=dev)
    F = T.F
    lf = torch.log2(torch.tensor([1.0, F, F * (F - 1) / 2], device=dev, dtype=torch.float64))
    parts = []
    for fi, fam in enumerate(pse.FAMILIES):
        cand = T.candidates(fam)
        if len(cand) == 0:
            continue
        ev = pse.eval_chunks(T, fam, cand)
        keep = ev['valid'] & ev['complete']
        if not keep.any():
            continue
        cand, ev = cand[keep], {k: v[keep] for k, v in ev.items()}
        s, f, g = cand[:, 0], cand[:, 1], cand[:, 2]
        m = (f >= 0).long() + (g >= 0).long()
        bits = T.bits[s] + float(np.log2(3)) + lf[m] + ev['table_bits'] + float(np.log2(3))
        C = len(s)
        phi = torch.zeros(C, len(PHI), device=dev, dtype=torch.float64)
        phi[:, 0] = bits
        phi[:, 1 + fi] = 1
        phi[torch.arange(C, device=dev), 4 + sk[s]] = 1
        for h in (f, g):
            on = h >= 0
            phi[on.nonzero()[:, 0], 12 + fk[h[on]]] += 1
        phi[:, 20], phi[:, 21] = ev['keys'].double(), ev['singles'].double()
        phi[:, 22], phi[:, 23] = ev['copy_cells'].double() / T.N, m.double()
        parts.append((phi, ev['pred'], torch.stack([torch.full_like(s, fi), s, f, g], 1)))
    if not parts:
        return None
    phi, pred, cand = (torch.cat([p[i] for p in parts]) for i in range(3))
    n_all = len(phi)
    if n_all > K:
        top = torch.topk(-phi[:, 0], K).indices
        phi, pred, cand = phi[top], pred[top], cand[top]
    return dict(phi=phi.float().cpu(), pred=pred.to(torch.int8).cpu(), cand=cand.int().cpu(), n_all=n_all,
                correct=(pred == T.ty).all(1).cpu())


def export(K=30000):
    t0 = time.time()
    tasks, exact = load_tasks()
    assert exact
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(32) as ex:
        banks = list(ex.map(bank3, tasks))
    print(f'banks {time.time() - t0:.0f} s; {banks[0]["src"].shape[0]} sources, {banks[0]["feat"].shape[0]} features',
          flush=True)
    dev = torch.device('cuda')
    data = []
    for i, B in enumerate(banks):
        data.append(export_task(B, dev, K))
        if (i + 1) % 25 == 0:
            print(f'  {i + 1} tasks {time.time() - t0:.0f} s', flush=True)
    meta = dict(src_names=banks[0]['src_names'], feat_names=banks[0]['feat_names'],
                ty=[B['ty'].tolist() for B in banks])
    torch.save(dict(data=data, meta=meta), CACHE)
    n = [d['n_all'] if d else 0 for d in data]
    print(f'tasks with a complete consistent candidate {sum(x > 0 for x in n)}; candidates per task: median '
          f'{int(np.median(n))}, max {max(n)}; capped at K = {K} in {sum(x > K for x in n)} tasks; oracle within '
          f'the kept candidates {sum(bool(d["correct"].any()) for d in data if d)}; {time.time() - t0:.0f} s')


def prepare(data, ids, dev):
    """Padded tensors over tasks: phi [T, K, D], mask, correct, group ids of identical predictions."""
    K = max(len(data[i]['phi']) for i in ids if data[i])
    D = len(PHI)
    phi = torch.zeros(len(ids), K, D, device=dev)
    mask = torch.zeros(len(ids), K, dtype=torch.bool, device=dev)
    corr = torch.zeros(len(ids), K, dtype=torch.bool, device=dev)
    grp = torch.full((len(ids), K), -1, dtype=torch.long, device=dev)
    for r, i in enumerate(ids):
        d = data[i]
        if not d:
            continue
        n = len(d['phi'])
        phi[r, :n], mask[r, :n], corr[r, :n] = d['phi'].to(dev), True, d['correct'].to(dev)
        grp[r, :n] = torch.unique(d['pred'].to(dev).long(), dim=0, return_inverse=True)[1]
    return phi, mask, corr, grp


def scores(phi, w, mu, sd):
    return -(((phi - mu) / sd) @ w)


def predict(s, mask, corr, grp):
    """Per task: MAP correct (every maximal candidate correct) and BMA correct (the unique top output is right)."""
    s = s.masked_fill(~mask, -float('inf'))
    top = s.max(1, keepdim=True).values
    at = mask & (s >= top - 1e-9)
    has = mask.any(1)
    map_ok = has & (at & ~corr).sum(1).eq(0) & (at & corr).any(1)
    G = int(grp.max()) + 2
    mass = torch.full((len(s), G), -float('inf'), device=s.device)
    mass = mass.scatter_reduce(1, grp.clamp(min=0), s, 'amax')
    lse = torch.zeros(len(s), G, device=s.device)
    lse = lse.scatter_add(1, grp.clamp(min=0), torch.exp(s - mass.gather(1, grp.clamp(min=0))) * mask)
    lmass = mass + torch.log(lse.clamp(min=1e-30))
    lmass = lmass.masked_fill(~torch.isfinite(mass), -float('inf'))
    best = lmass.max(1, keepdim=True).values
    winners = (lmass >= best - 1e-6) & torch.isfinite(lmass)
    gcorr = torch.zeros(len(s), G, dtype=torch.long, device=s.device).scatter_reduce(
        1, grp.clamp(min=0), (corr & mask).long(), 'amax').bool()
    bma_ok = has & (winners.sum(1) == 1) & (winners & gcorr).any(1)
    return map_ok, bma_ok


def train(phi, mask, corr, steps=500, l2=1e-2, lr=0.05):
    m = mask.float().unsqueeze(-1)
    mu = (phi * m).sum((0, 1)) / m.sum()
    sd = (((phi - mu) ** 2) * m).sum((0, 1)).div(m.sum()).sqrt().clamp(min=1e-6)
    w0 = torch.zeros(len(PHI), device=phi.device)
    w0[0] = float(np.log(2)) * sd[0]
    use = (corr & mask).any(1)
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        s = scores(phi[use], w, mu, sd)
        lp = torch.logsumexp(s.masked_fill(~(corr[use] & mask[use]), -float('inf')), 1) - \
            torch.logsumexp(s.masked_fill(~mask[use], -float('inf')), 1)
        loss = -lp.mean() + l2 * ((w - w0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), w0, mu, sd


def cv(folds=5, seed=0):
    t0 = time.time()
    blob = torch.load(CACHE, weights_only=False)
    data, meta = blob['data'], blob['meta']
    dev = torch.device('cuda')
    n = len(data)
    fold = np.random.default_rng(seed).permutation(n) % folds
    rows = {k: np.zeros(n, bool) for k in ('oracle', 'map_bits', 'bma_bits', 'map_learned', 'bma_learned')}
    weights = []
    for k in range(folds):
        tr, te = [i for i in range(n) if fold[i] != k], [i for i in range(n) if fold[i] == k]
        P = prepare(data, tr, dev)
        w, w0, mu, sd = train(P[0], P[1], P[2])
        weights.append(((w / sd).cpu().numpy(), (w0 / sd).cpu().numpy()))
        Q = prepare(data, te, dev)
        for name, ww in (('bits', w0), ('learned', w)):
            map_ok, bma_ok = predict(scores(Q[0], ww, mu, sd), Q[1], Q[2], Q[3])
            rows[f'map_{name}'][te] = map_ok.cpu().numpy()
            rows[f'bma_{name}'][te] = bma_ok.cpu().numpy()
        rows['oracle'][te] = (Q[2] & Q[1]).any(1).cpu().numpy()
    print(f'5-fold cross-validation over {n} tasks (training uses only training folds), {time.time() - t0:.0f} s')
    for k, v in rows.items():
        print(f'  {k:12s} {int(v.sum()):4d}')
    wl = np.mean([a for a, _ in weights], 0)
    print('learned weights (per unit of phi, nats; mean over folds; start: bits = ln 2 = 0.693):')
    for name, v in zip(PHI, wl):
        print(f'  {name:12s} {v:+.3f}')
    out = ROOT / 'runs' / 'pse3_cv.json'
    out.write_text(json.dumps(dict(counts={k: int(v.sum()) for k, v in rows.items()},
                                   tasks={k: np.flatnonzero(v).tolist() for k, v in rows.items()},
                                   weights=dict(zip(PHI, wl.tolist())), folds=folds, seed=seed), indent=1))
    print('->', out)


def nested(folds=5, inner=4, seed=0, grid=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Outer 5-fold CV; inside each training fold, an inner 4-fold CV picks the L2 strength by BMA accuracy.
    Two prior forms: all 24 components, and 4 (bits, keys, singles, m)."""
    blob = torch.load(CACHE, weights_only=False)
    data = blob['data']
    dev = torch.device('cuda')
    n = len(data)
    fold = np.random.default_rng(seed).permutation(n) % folds
    forms = {'all24': list(range(len(PHI))), 'small4': [PHI.index(k) for k in ('bits', 'keys', 'singles', 'm')]}
    out = {}
    for fname, cols in forms.items():
        ok = np.zeros(n, bool)
        chosen = []
        for k in range(folds):
            tr, te = [i for i in range(n) if fold[i] != k], [i for i in range(n) if fold[i] == k]
            ifold = np.random.default_rng(seed + 1 + k).permutation(len(tr)) % inner
            acc = {}
            for l2 in grid:
                hits = 0
                for j in range(inner):
                    itr = [tr[q] for q in range(len(tr)) if ifold[q] != j]
                    ite = [tr[q] for q in range(len(tr)) if ifold[q] == j]
                    P = prepare(data, itr, dev)
                    w, mu, sd = train_cols(P, cols, l2)
                    Q = prepare(data, ite, dev)
                    hits += int(predict(scores(Q[0][..., cols], w, mu, sd), Q[1], Q[2], Q[3])[1].sum())
                acc[l2] = hits
            best = max(grid, key=lambda l: (acc[l], l))          # ties -> stronger regularisation
            chosen.append(best)
            P = prepare(data, tr, dev)
            w, mu, sd = train_cols(P, cols, best)
            Q = prepare(data, te, dev)
            ok[te] = predict(scores(Q[0][..., cols], w, mu, sd), Q[1], Q[2], Q[3])[1].cpu().numpy()
        out[fname] = dict(bma_correct=int(ok.sum()), l2_chosen=chosen, tasks=np.flatnonzero(ok).tolist())
        print(f'  {fname:7s} nested-CV BMA correct {int(ok.sum()):3d}   L2 chosen per outer fold {chosen}', flush=True)
    return out


def train_cols(P, cols, l2):
    """train() on a subset of the prior components (the bits column must be among them; it is column 0 of cols)."""
    phi, mask, corr = P[0][..., cols], P[1], P[2]
    m = mask.float().unsqueeze(-1)
    mu = (phi * m).sum((0, 1)) / m.sum()
    sd = (((phi - mu) ** 2) * m).sum((0, 1)).div(m.sum()).sqrt().clamp(min=1e-6)
    w0 = torch.zeros(len(cols), device=phi.device)
    w0[0] = float(np.log(2)) * sd[0]
    use = (corr & mask).any(1)
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(500):
        s = scores(phi[use], w, mu, sd)
        lp = torch.logsumexp(s.masked_fill(~(corr[use] & mask[use]), -float('inf')), 1) - \
            torch.logsumexp(s.masked_fill(~mask[use], -float('inf')), 1)
        loss = -lp.mean() + l2 * ((w - w0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), mu, sd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['export', 'cv', 'nested'])
    ap.add_argument('--K', type=int, default=30000)
    a = ap.parse_args()
    if a.cmd == 'export':
        export(a.K)
    elif a.cmd == 'cv':
        cv()
    else:
        res = nested()
        (ROOT / 'runs' / 'pse3_nested.json').write_text(json.dumps(res, indent=1))
