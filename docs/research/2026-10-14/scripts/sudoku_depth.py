"""Logical depth of every cell under synchronous propagation (L0 singles; L1 singles + one-assumption contradiction),
and the model's decision block against it.

Usage (repository root):
  python docs/research/2026-10-14/scripts/sudoku_depth.py --solver-only [N]
  python docs/research/2026-10-14/scripts/sudoku_depth.py RUN_DIR STEP [N=512] [BLOCKS=128] [BATCH=256]

State = the candidate set of every cell (givens: singletons; blanks: digits absent from the 20 neighbours' fixed values).
L0 round: one synchronous pass of naked singles (one candidate) and hidden singles (a digit with one place in a row / column /
box), then the fixed digits are removed from their neighbours.  L1 round: in addition, every candidate (t, d) of the round's
starting state is assumed and singles are propagated to closure; a contradiction (a cell with no candidate, a digit with no
place in a unit, a digit fixed twice in a unit) removes d from t.  All removals of a round are computed from the same starting
state and applied together.  depth(t) = the round at which cell t's candidate set becomes a singleton; givens 0; cells never
resolved = "beyond" (depth -1).  Every fixed digit is checked against the solution.  The candidate sets after each L1 round
are kept for the probe.

Model side (the run's own kernel, no hooks): per block the argmax; at selected blocks the hidden state and logits.
decision(t) = the block from which the argmax stays at its final value.
  1. decision block by L1 depth (solved lanes): mean / median per depth, Spearman(depth, decision) pooled and per lane,
     blocks per round (least-squares slope), |d_t - d_s| for same-depth vs different-depth cell pairs; beyond cells.
  2. candidate dynamics: the model's candidate count (softmax > 0.1 over the nine digits) per block by depth; a ridge probe
     hidden state at block b -> L1 round-r candidate set (9 binary targets) on cells NOT yet fixed after round r, 70/30 lane
     split, against the majority baseline and the same probe from the 11 logits.
Writes docs/research/2026-10-14/results/depth_<run>_step<STEP>_n<N>_b<BLOCKS>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

UNITS = []
for i in range(9):
    UNITS.append([i * 9 + j for j in range(9)])
    UNITS.append([j * 9 + i for j in range(9)])
for bi in range(3):
    for bj in range(3):
        UNITS.append([(bi * 3 + r) * 9 + bj * 3 + c for r in range(3) for c in range(3)])
UNITS = np.array(UNITS)                                                    # [27, 9]
CELL_UNITS = np.array([[u for u in range(27) if t in UNITS[u]] for t in range(81)])   # [81, 3]
POP = np.array([bin(i).count('1') for i in range(512)], dtype=np.int8)
FULL = np.uint16(0x1FF)
SHIFTS = np.arange(9, dtype=np.uint16)


def bits(cand):
    return ((cand[..., None] >> SHIFTS) & 1).astype(bool)                     # [..., 81, 9]


def singles_pass(cand):
    """One synchronous pass: hidden singles, naked singles (fixed = singleton), neighbour elimination.
    Returns new cand and a contradiction flag per lane."""
    N = cand.shape[0]
    bad = np.zeros(N, dtype=bool)
    pc = POP[cand]
    bad |= (pc == 0).any(1)
    b = bits(cand)                                                          # [N,81,9]
    fixed = pc == 1
    # hidden singles: a digit with exactly one place in a unit
    ub = b[:, UNITS, :]                                                     # [N,27,9 cells,9 digits]
    cnt = ub.sum(2)                                                         # [N,27,9]
    bad |= (cnt == 0).any((1, 2))
    fcnt = (ub & fixed[:, UNITS, None]).sum(2)
    bad |= (fcnt > 1).any((1, 2))
    hs = np.zeros((N, 81), dtype=np.uint16)
    pos = ub.argmax(2)                                                      # [N,27,9]: cell index inside the unit
    one = cnt == 1
    lanes, us, ds = np.nonzero(one)
    cells = UNITS[us, pos[lanes, us, ds]]
    np.bitwise_or.at(hs, (lanes, cells), (np.uint16(1) << ds.astype(np.uint16)))
    multi = POP[hs] > 1
    bad |= multi.any(1)
    new = np.where(hs > 0, hs, cand)
    bad |= ((new & cand) == 0).any(1)
    # eliminate fixed digits from neighbours
    pc2 = POP[new]
    fixed2 = pc2 == 1
    fmask = np.where(fixed2, new, 0).astype(np.uint16)
    unit_fixed = np.bitwise_or.reduce(fmask[:, UNITS], axis=2)              # [N,27]
    cell_or = np.bitwise_or.reduce(unit_fixed[:, CELL_UNITS], axis=2)       # [N,81] (own fixed digit included)
    new = np.where(fixed2, new, new & ~cell_or).astype(np.uint16)
    bad |= (POP[new] == 0).any(1)
    return new, bad


def passes(cand, k):
    """k synchronous singles passes (k = 0: closure), contradiction flags accumulated."""
    cand = cand.copy()
    bad = np.zeros(cand.shape[0], dtype=bool)
    for _ in range(k if k > 0 else 100):
        new, b = singles_pass(cand)
        bad |= b
        if (new == cand).all():
            break
        cand = new
    return cand, bad


def initial_state(x):
    cand = np.where(x > 0, (np.uint16(1) << np.maximum(x - 1, 0).astype(np.uint16)), FULL).astype(np.uint16)
    fmask = np.where(x > 0, cand, 0).astype(np.uint16)
    unit_fixed = np.bitwise_or.reduce(fmask[:, UNITS], axis=2)
    cell_or = np.bitwise_or.reduce(unit_fixed[:, CELL_UNITS], axis=2)
    return np.where(x > 0, cand, cand & ~cell_or).astype(np.uint16)


def propagate(x, y, lookahead, inner=1):
    """lookahead: assume each candidate and run `inner` synchronous singles passes (0 = closure); a contradiction removes it.
    Returns depth [N,81] (0 given, r >= 1, -1 beyond) and the list of candidate states after rounds 0..R."""
    N = x.shape[0]
    sol = (np.uint16(1) << (y - 1).astype(np.uint16)).astype(np.uint16)
    cand = initial_state(x)
    depth = np.where(x > 0, 0, -1)
    states = [cand.copy()]
    for r in range(1, 200):
        start = cand
        if lookahead:
            elim = np.zeros_like(cand)
            pc = POP[start]
            for t in range(81):
                for d in range(9):
                    bit = np.uint16(1 << d)
                    lanes = np.nonzero((pc[:, t] > 1) & ((start[:, t] & bit) > 0) & (sol[:, t] != bit))[0]
                    if lanes.size == 0:
                        continue
                    trial = start[lanes].copy()
                    trial[:, t] = bit
                    _, bad = passes(trial, inner)
                    elim[lanes[bad], t] |= bit
            cand = (start & ~elim).astype(np.uint16)
        cand, bad = singles_pass(cand)
        if bad.any():
            raise RuntimeError(f'contradiction in a valid puzzle at round {r}: lanes {np.nonzero(bad)[0][:5]}')
        fixed = POP[cand] == 1
        if ((cand != sol) & fixed).any():
            raise RuntimeError('a fixed digit contradicts the solution')
        newly = fixed & (depth == -1)
        depth[newly] = r
        states.append(cand.copy())
        if (cand == start).all():
            states.pop()
            break
    return depth, states


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    ra = a.argsort().argsort(); rb = b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def load_puzzles(n):
    with np.load(REPO / 'data/sudoku_lt_1k.npz') as z:
        x, y = z['test_inputs'][:n], z['test_labels'][:n]
    return x.reshape(-1, 81).astype(int), y.reshape(-1, 81).astype(int)


def solver_report(depth, name):
    N = depth.shape[0]
    blank = depth != 0
    beyond = depth == -1
    maxd = int(depth.max())
    hist = {int(r): int((depth == r).sum()) for r in range(1, maxd + 1)}
    print(f'[{name}] {N} puzzles: givens/puzzle {(~blank).sum() / N:.1f}; fully resolved {int((~beyond.any(1)).sum())}/{N}; '
          f'beyond cells {int(beyond.sum())} ({beyond.sum() / blank.sum() * 100:.1f}% of blanks); max depth {maxd}')
    print('   cells per depth: ' + ', '.join(f'{r}:{c}' for r, c in hist.items()))
    return dict(n=N, resolved=int((~beyond.any(1)).sum()), beyond_cells=int(beyond.sum()), blanks=int(blank.sum()),
                max_depth=maxd, cells_per_depth=hist)


def ridge_probe(Xtr, Ytr, Xte, Yte, lam=1.0):
    Xtr, Xte = Xtr.double(), Xte.double()
    mu = Xtr.mean(0, keepdim=True)
    Xtr, Xte = Xtr - mu, Xte - mu
    ym = Ytr.double().mean(0)
    f = Xtr.shape[1]
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(f, device=Xtr.device, dtype=Xtr.dtype), Xtr.T @ (Ytr.double() - ym))
    pred = (Xte @ W + ym) > 0.5
    acc = (pred == Yte.bool()).double().mean().item()
    p = Yte.double().mean()
    return acc, torch.maximum(p, 1 - p).mean().item()


def main():
    if sys.argv[1] == '--solver-only':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        x, y = load_puzzles(n)
        t0 = time.monotonic()
        d0, _ = propagate(x, y, lookahead=False)
        solver_report(d0, 'L0 singles')
        for k in (1, 2, 3, 0):
            d1, s1 = propagate(x, y, lookahead=True, inner=k)
            solver_report(d1, f'L1 one assumption + {k if k else "closure"} inner passes')
        print(f'{time.monotonic() - t0:.0f}s')
        return
    from lt import train as t
    from lt import research_phase_clock_stdp as rp
    from lt.kv_stability import ORIGINAL_MODEL_ID
    run, step = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    PROBE_BLOCKS = [b for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if b <= BLOCKS]
    protocol = json.loads((run / 'protocol.json').read_text())
    ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'], global_batch_size=BATCH)
    cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    _, _, x_np, y_np, _, _ = t.load_data(cfg)
    x_np, y_np = x_np[:N], y_np[:N]
    began = time.monotonic()
    x, y = x_np.reshape(-1, 81).astype(int), y_np.reshape(-1, 81).astype(int)
    depth, states = propagate(x, y, lookahead=False)                 # gradation: L0 synchronous singles rounds
    solver0 = solver_report(depth, 'L0 singles')
    depth1, _ = propagate(x, y, lookahead=True, inner=0)              # tier: one assumption + singles closure
    solver1 = solver_report(depth1, 'L1 one assumption + closure')
    R = len(states) - 1
    tier = np.where(depth == 0, 0, np.where(depth > 0, 1, np.where(depth1 > 0, 2, 3)))     # 0 given, 1 L0, 2 L1-only, 3 beyond
    TIER = {1: 'L0', 2: 'L1-only', 3: 'beyond-L1'}
    puzzle_tier = np.where(~(depth == -1).any(1), 1, np.where(~(depth1 == -1).any(1), 2, 3))
    print(f'solver {time.monotonic() - began:.0f}s')
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'])
    inner = base.model.inner
    layer = inner.layers[0]
    T = 81
    preds_all, ncand_all = [], []
    hidden = {b: [] for b in PROBE_BLOCKS}
    logits_at = {b: [] for b in PROBE_BLOCKS}
    with torch.no_grad():
        for batch in t.eval_batches(x_np, y_np, BATCH, 0, 1):
            b = {k: v.to(device) for k, v in batch.items()}
            B = b['inputs'].shape[0]
            inj = inner.injection(b)
            h = inner.init_hidden.float().expand(B, T, inner.d).clone()
            mem = ek = ev = None
            preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
            ncand = torch.zeros(BLOCKS, B, T, dtype=torch.int8, device=device)
            for blk in range(1, BLOCKS + 1):
                h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                logits = inner.w_cls(h).float()
                preds[blk - 1] = logits.argmax(-1).to(torch.int16)
                prob = torch.softmax(logits[..., 2:11], -1)                             # digits 1..9
                ncand[blk - 1] = (prob > 0.1).sum(-1).to(torch.int8)
                if blk in PROBE_BLOCKS:
                    hidden[blk].append(h.half().cpu()); logits_at[blk].append(logits.cpu())
            preds_all.append(preds.cpu()); ncand_all.append(ncand.cpu())
    preds = torch.cat(preds_all, 1).long().numpy()                                       # [BLOCKS, N, 81]
    ncand = torch.cat(ncand_all, 1).numpy().astype(int)
    labels = y + 1
    final = preds[-1]
    same = preds == final[None]
    dec = (np.flip(np.cumprod(np.flip(same, 0), 0), 0).astype(bool)).argmax(0) + 1         # [N, 81]
    right = (preds == labels[None]).all(-1)
    solved = np.flip(np.cumprod(np.flip(right, 0), 0), 0).astype(bool)[-1]                # [N]
    blank = depth != 0
    finite = blank & (depth > 0)
    beyond = depth == -1
    out = dict(run=run.name, step=step, n=N, blocks=BLOCKS, solver_L0=solver0, solver_L1=solver1, solved=int(solved.sum()))
    S = solved
    rows = {}
    for r in range(1, int(depth.max()) + 1):
        m = S[:, None] & (depth == r)
        if m.any():
            rows[r] = dict(cells=int(m.sum()), mean=float(dec[m].mean()), median=float(np.median(dec[m])))
    for k_, name in TIER.items():
        m = S[:, None] & (tier == k_)
        rows[name] = dict(cells=int(m.sum()), mean=float(dec[m].mean()) if m.any() else None, median=float(np.median(dec[m])) if m.any() else None)
    tier_solve = {name: dict(puzzles=int((puzzle_tier == k_).sum()), model_solved=int((S & (puzzle_tier == k_)).sum())) for k_, name in TIER.items()}
    tier_acc = {name: float((final == labels)[tier == k_].mean()) for k_, name in TIER.items() if (tier == k_).any()}
    out.update(tier_solve=tier_solve, tier_cell_accuracy=tier_acc)
    mf = S[:, None] & finite
    pooled = spearman(depth[mf], dec[mf])
    per_lane = [spearman(depth[i][finite[i]], dec[i][finite[i]]) for i in np.nonzero(S)[0]]
    per_lane = [v for v in per_lane if not np.isnan(v)]
    slope = float(np.polyfit(depth[mf].astype(float), dec[mf].astype(float), 1)[0]) if mf.sum() > 2 else None
    same_d, diff_d = [], []
    for i in np.nonzero(S)[0]:
        f = np.nonzero(finite[i])[0]
        if f.size < 3:
            continue
        dd = np.abs(dec[i][f][:, None] - dec[i][f][None]).astype(float)
        eq = depth[i][f][:, None] == depth[i][f][None]
        off = ~np.eye(f.size, dtype=bool)
        if (eq & off).any() and (~eq).any():
            same_d.append(dd[eq & off].mean()); diff_d.append(dd[~eq].mean())
    out.update(decision_by_depth=rows, spearman_pooled=pooled,
               spearman_per_lane_mean=float(np.mean(per_lane)) if per_lane else None,
               spearman_per_lane_median=float(np.median(per_lane)) if per_lane else None,
               blocks_per_round=slope, pair_same_depth=float(np.mean(same_d)) if same_d else None,
               pair_diff_depth=float(np.mean(diff_d)) if diff_d else None,
               beyond_accuracy_all=float((final == labels)[beyond].mean()) if beyond.any() else None,
               beyond_accuracy_unsolved_lanes=float((final == labels)[(~S)[:, None] & beyond].mean()) if ((~S)[:, None] & beyond).any() else None,
               solved_lanes_with_beyond_cells=int((S & beyond.any(1)).sum()),
               unsolved_lanes_fully_resolved_by_L1=int((~S & ~beyond.any(1)).sum()),
               model_solved_among_L0_resolved=int((S & ~beyond.any(1)).sum()), L0_resolved=int((~beyond.any(1)).sum()))
    # 2a. candidate counts
    show = [b for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if b <= BLOCKS]
    cand_model = {}
    for r in list(range(1, int(depth.max()) + 1)) + ['L1-only', 'beyond-L1']:
        m = (depth == r) if isinstance(r, int) else (tier == {'L1-only': 2, 'beyond-L1': 3}[r])
        if m.any():
            cand_model[r] = [float(ncand[b - 1][m].mean()) for b in show]
    cand_solver = {r: [float(POP[states[k]][depth == r].mean()) for k in range(0, min(R, r) + 1)] for r in range(1, int(depth.max()) + 1) if (depth == r).any()}
    out.update(cand_model_blocks=show, cand_model=cand_model, cand_solver=cand_solver)
    # 2b. probe
    gen = np.random.default_rng(0)
    perm = gen.permutation(N)
    tr_l, te_l = np.zeros(N, bool), np.zeros(N, bool)
    tr_l[perm[:int(0.7 * N)]] = True; te_l[perm[int(0.7 * N):]] = True
    probe = {}
    rounds = [r for r in (1, 2, 3, 5, 8) if r <= R]
    for blk in PROBE_BLOCKS:
        H = torch.cat(hidden[blk], 0).float().to(device)
        L = torch.cat(logits_at[blk], 0).float().to(device)
        probe[blk] = {}
        for r in rounds:
            unfixed = POP[states[r]] > 1
            Y = torch.from_numpy(bits(states[r])).to(device)                               # [N,81,9]
            mtr = torch.from_numpy(unfixed & tr_l[:, None]).to(device)
            mte = torch.from_numpy(unfixed & te_l[:, None]).to(device)
            if mte.sum() < 50:
                continue
            acc_h, base_ = ridge_probe(H[mtr], Y[mtr].float(), H[mte], Y[mte].float())
            acc_l, _ = ridge_probe(L[mtr], Y[mtr].float(), L[mte], Y[mte].float())
            probe[blk][r] = dict(cells_test=int(mte.sum()), hidden=acc_h, logits=acc_l, majority=base_)
    out['probe'] = probe
    out['seconds'] = time.monotonic() - began
    path = REPO / 'docs/research/2026-10-14/results' / f'depth_{run.name}_step{step}_n{N}_b{BLOCKS}.json'
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
    print(f"\nstep {step}: model solved {int(solved.sum())}/{N} within {BLOCKS} blocks; L0 resolves {out["L0_resolved"]} puzzles, of which the model solved "
          f"{out['model_solved_among_L0_resolved']}; {out['seconds']:.0f}s")
    print('[0] puzzles by tier (L0-resolved / L1-only / beyond-L1): ' + '; '.join(f"{n}: {v['puzzles']} puzzles, model solved {v['model_solved']}" for n, v in tier_solve.items())
          + '\n    final cell accuracy by cell tier: ' + ', '.join(f'{n} {a:.3f}' for n, a in tier_acc.items()))
    print('[1] decision block by L0 depth / tier (solved lanes): ' + '; '.join(f"{r}: n={v['cells']} mean {v['mean']:.1f} med {v['median']:.0f}" if v['mean'] is not None else f'{r}: n=0'
                                                                      for r, v in rows.items()))
    print(f"    Spearman(depth, decision): pooled {pooled:.3f}, per-lane mean {out['spearman_per_lane_mean']}, median {out['spearman_per_lane_median']}; "
          f"blocks per round {slope}; |d_t - d_s| same depth {out['pair_same_depth']} vs different depth {out['pair_diff_depth']}")
    print(f"    beyond-L1 cells: accuracy all lanes {out['beyond_accuracy_all']}, in unsolved lanes {out['beyond_accuracy_unsolved_lanes']}; "
          f"solved lanes with beyond cells {out['solved_lanes_with_beyond_cells']}; unsolved lanes that L0 resolves {out['unsolved_lanes_fully_resolved_by_L1']}")
    print('[2a] model candidate count (softmax > 0.1) by L0 depth / tier, blocks ' + ' '.join(f'{b:5d}' for b in show))
    for r, v in cand_model.items():
        print(f'    depth {str(r):6s} ' + ' '.join(f'{c:5.2f}' for c in v))
    print('     solver candidate count by depth after rounds 0..r: ' + '; '.join(f'{r}: ' + ' '.join(f'{c:.2f}' for c in v) for r, v in list(cand_solver.items())[:6]))
    print('[2b] ridge probe accuracy, hidden -> L0 round-r candidate set on cells unfixed after round r (hidden / logits / majority)')
    print('    block   ' + ' '.join(f'{"r=" + str(r):>22s}' for r in rounds))
    for blk in PROBE_BLOCKS:
        print(f'    {blk:5d}   ' + ' '.join(f"{v['hidden']:.3f} / {v['logits']:.3f} / {v['majority']:.3f}" if (v := probe[blk].get(r)) else ' ' * 22 for r in rounds))
    print('WROTE', path)


if __name__ == '__main__':
    main()
