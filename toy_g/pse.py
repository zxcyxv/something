"""Part-selection executor, version 1, on Mini-ARC (design of 2026-10-20; docs/claude_2026-10-20_arc_logic_foundations.md
section 10 is the grammar it extends).

Exact parts write per-cell registers on every grid (demonstrations and the test input). A program picks a source
register, at most two feature registers and a colour-map family; the per-task table is counted from the demonstrations;
the program with the shortest description is chosen.

Sources (colour valued):
    MOVE   r0 read at an address of toy_g/miniarc_grammar.py (1,251, same costs)
    SCAN   first non-background cell in each of 8 directions, own cell included or not (16), nearest non-background
           cell along each of 4 lines (4); cost 2
    POOL   majority / minority colour of the grid, majority of the row, of the column, majority / minority of the
           8-connected object, the other colour of a two-colour grid (7); cost 2
Features (small integers, cost 1): the 27 predicates of miniarc_grammar; non-background counts in the row, column, both
    diagonals, 4- and 8-neighbourhood, grid, object; beyond the cell in each direction (8); distance to the first
    non-background cell in each direction (8); row, column, r + c, r - c; gravity comparisons [c >= 5 - row count],
    [c < row count], [r >= 5 - column count], [r < column count]
Families:
    table   key (source colour, features) -> colour
    role    key (source is foreground, features) -> copy source | constant           (copy wins a tie)
    role2   key (r0 and source foreground, features) -> copy r0 | copy source | constant, cost + 1
Scores: 'grammar' = integer costs of miniarc_grammar (address cost + features + 1 for a second source; table + 1 in
    the configurations); 'bits' = two-part description length in bits, no free parameter:
        family log2(#families) + source log2 5 (type) + variant (fixed move: 1 + [k != 0] log2 7 + 1 + [shift != 0]
        log2 80; anchor log2 600; object log2 3; scan log2 20; pool log2 7) + features log2 3 + log2 C(F, m)
        + table: (#keys seen in the demonstrations) log2 A - sum_k log2 |A_k|   (A = 10 / 11 / 12 actions for
        table / role / role2, A_k = actions consistent with key k; refining a key never lowers this, so feature
        sets minimal by inclusion still contain every minimum)
Search: for every source, the demonstration cell pairs the table would have to tell apart; feature sets of size <= 2
    separating every pair come from one matrix product per source. Pairwise separation is exact for table and role
    (miniarc_grammar) and only necessary for role2, whose candidates are then checked bucket by bucket. Feature sets
    are minimal by inclusion; the flags per task are those of miniarc_grammar.

    python -m toy_g.pse --mode repro    # MOVE sources, 27 predicates, table/role: must equal runs/miniarc_grammar.json
    python -m toy_g.pse --mode ext      # all registers, table/role/role2 -> runs/pse_ext.json
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import miniarc_grammar as MG
from .miniarc_ceiling import load_tasks

ROOT = Path(__file__).resolve().parents[1]
RR, CC = MG.RR, MG.CC
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR_NAMES = ['up', 'down', 'left', 'right', 'ul', 'ur', 'dl', 'dr']
LINES = [(0, 1, 'vert'), (2, 3, 'horiz'), (4, 7, 'diag'), (5, 6, 'anti')]
TERMS = MG.address_terms()
IDENT = next(i for i, t in enumerate(TERMS) if t[0] == 'D0+(0,0)')
FAMILIES = ('table', 'role', 'role2')
EXTRA_SOURCE = {'table': 0, 'role': 0, 'role2': 1}
CONFIGS = {'repro': {'table': {'table': 0}, 'table+role': {'role': 0, 'table': 1}},
           'ext': {'table+role': {'role': 0, 'table': 1}, 'table+role+role2': {'role': 0, 'role2': 0, 'table': 1},
                   'bits': {'role': 0, 'role2': 0, 'table': 0}}}
SCORE = {'bits': 'bits'}                                   # configurations scored in bits; the others by grammar costs
FLAGS = ('consistent', 'mdl_correct', 'mdl_correct_iddef', 'oracle', 'oracle_iddef')


def ray(i, d):
    dr, dc = DIRS[d]
    r, c, out = RR[i] + dr, CC[i] + dc, []
    while 0 <= r < 5 and 0 <= c < 5:
        out.append(r * 5 + c)
        r, c = r + dr, c + dc
    return out


RAYS = [[ray(i, d) for i in range(25)] for d in range(8)]


def first_hit(g, d, inclusive):
    """Colour and distance of the first non-background cell from each cell in direction d (0, 0 if none)."""
    col, dist = np.zeros(25, int), np.zeros(25, int)
    for i in range(25):
        if inclusive and g[i]:
            col[i] = g[i]
            continue
        for k, j in enumerate(RAYS[d][i], 1):
            if g[j]:
                col[i], dist[i] = g[j], k
                break
    return col, dist


def nearest_line(g, d1, d2):
    """Own colour if foreground, else the nearer first hit along the line (0 on a tie between two colours)."""
    c1, k1 = first_hit(g, d1, False)
    c2, k2 = first_hit(g, d2, False)
    a, b = np.where(c1 > 0, k1, 99), np.where(c2 > 0, k2, 99)
    out = np.where(a < b, c1, np.where(b < a, c2, np.where((a < 99) & (c1 == c2), c1, 0)))
    return np.where(g != 0, g, out)


def majority(v):
    cnt = np.bincount(v[v > 0], minlength=10)
    return int(np.argmax(cnt)) if cnt.any() else 0


def minority(v):
    cnt = np.bincount(v[v > 0], minlength=10)
    present = np.flatnonzero(cnt)
    return int(present[np.argmin(cnt[present])]) if len(present) else 0


def pool_regs(g):
    G2 = g.reshape(5, 5)
    lab = MG.components(g, MG.N8, False)
    objm, objn = np.zeros(25, int), np.zeros(25, int)
    for o in range(lab.max() + 1):
        m = lab == o
        objm[m], objn[m] = majority(g[m]), minority(g[m])
    present = np.flatnonzero(np.bincount(g[g > 0], minlength=10))
    other = np.zeros(25, int)
    if len(present) == 2:
        a, b = present
        other = np.where(g == a, b, np.where(g == b, a, 0))
    return {'glob_major': np.full(25, majority(g)), 'glob_minor': np.full(25, minority(g)),
            'row_major': np.array([majority(G2[r]) for r in range(5)])[RR],
            'col_major': np.array([majority(G2[:, c]) for c in range(5)])[CC],
            'obj_major': objm, 'obj_minor': objn, 'other2': other}


def feature_regs(g):
    nb = (g != 0).astype(int)
    f = {k: v.astype(int) for k, v in zip(MG.PRED_NAMES, MG.predicates(g))}
    N2 = nb.reshape(5, 5)
    rowc, colc = N2.sum(1)[RR], N2.sum(0)[CC]
    f['rowc'], f['colc'] = rowc, colc
    f['diagc'] = np.array([nb[(RR - CC) == (RR[i] - CC[i])].sum() for i in range(25)])
    f['antic'] = np.array([nb[(RR + CC) == (RR[i] + CC[i])].sum() for i in range(25)])
    f['n4c'] = np.array([nb[MG.N4[i]].sum() for i in range(25)])
    f['n8c'] = np.array([nb[MG.N8[i]].sum() for i in range(25)])
    f['globc'] = np.full(25, min(int(nb.sum()), 9))
    lab = MG.components(g, MG.N8, False)
    sizes = np.bincount(lab[lab >= 0], minlength=1)
    f['objsize'] = np.where(lab >= 0, np.minimum(sizes[np.maximum(lab, 0)], 9), 0)
    for d in range(8):
        f[f'cnt_{DIR_NAMES[d]}'] = np.array([nb[RAYS[d][i]].sum() for i in range(25)])
    for d in range(8):
        f[f'dist_{DIR_NAMES[d]}'] = first_hit(g, d, False)[1]
    f['row'], f['col'], f['rpc'], f['rmc'] = RR.copy(), CC.copy(), RR + CC, RR - CC + 4
    f['grav_right'] = (CC >= 5 - rowc).astype(int)
    f['grav_left'] = (CC < rowc).astype(int)
    f['grav_down'] = (RR >= 5 - colc).astype(int)
    f['grav_up'] = (RR < colc).astype(int)
    return f


def bank(task, ext):
    """Registers of one task on its n demonstration inputs and its test input (row n)."""
    grids = np.concatenate([task['x'], task['tx'][None]])
    names, costs, vals = [], [], []
    for t in TERMS:
        names.append(t[0])
        costs.append(t[1])
        vals.append(np.stack([MG.read(g, MG.src_of(t, g)) for g in grids]))
    if ext:
        for d in range(8):
            for inc in (False, True):
                names.append(f'scan_{DIR_NAMES[d]}' + ('_incl' if inc else ''))
                costs.append(2)
                vals.append(np.stack([first_hit(g, d, inc)[0] for g in grids]))
        for d1, d2, nm in LINES:
            names.append(f'nearest_{nm}')
            costs.append(2)
            vals.append(np.stack([nearest_line(g, d1, d2) for g in grids]))
        pr = [pool_regs(g) for g in grids]
        for k in pr[0]:
            names.append(k)
            costs.append(2)
            vals.append(np.stack([p[k] for p in pr]))
    fr = [feature_regs(g) for g in grids]
    fnames = list(fr[0]) if ext else list(MG.PRED_NAMES)
    lg = np.log2
    bits = [lg(5) + 1 + (lg(7) if k else 0) + 1 + (lg(80) if (a, b) != (0, 0) else 0)
            for k in range(8) for a in range(-4, 5) for b in range(-4, 5)]
    bits += [lg(5) + lg(600)] * 600 + [lg(5) + lg(3)] * 3
    if ext:
        bits += [lg(5) + lg(20)] * 20 + [lg(5) + lg(7)] * 7
    return dict(src_names=names, src_cost=np.array(costs), src_bits=np.array(bits), src=np.stack(vals).astype(np.int64),
                feat_names=fnames,
                feat=np.stack([np.stack([f[k] for f in fr]) for k in fnames]).astype(np.int64), n=len(task['x']),
                y=task['y'].astype(np.int64), ty=task['ty'].astype(np.int64))


class Task:
    """GPU tensors of one task's registers."""

    def __init__(self, B, dev):
        self.B, self.dev, n = B, dev, B['n']
        self.N = N = n * 25
        src = torch.as_tensor(B['src'], device=dev)
        self.S = S = src.shape[0]
        self.sd, self.st = src[:, :n].reshape(S, N), src[:, n]
        feat = torch.as_tensor(B['feat'], device=dev)
        self.F = F = feat.shape[0]
        self.fd, self.ft = feat[:, :n].reshape(F, N), feat[:, n]
        self.card = feat.amax(dim=(1, 2)) + 1
        self.y, self.ty = torch.as_tensor(B['y'].ravel(), device=dev), torch.as_tensor(B['ty'], device=dev)
        self.r0d, self.r0t = self.sd[IDENT], self.st[IDENT]
        self.cost = torch.as_tensor(B['src_cost'], device=dev)
        self.bits = torch.as_tensor(B['src_bits'], device=dev, dtype=torch.float64)

    def candidates(self, fam, chunk=64):
        """[C, 3] (source, feature or -1, feature or -1): feature sets minimal by inclusion that separate every
        demonstration cell pair the family's table would have to tell apart."""
        sd, y, dev = self.sd, self.y, self.dev
        pi, pj = torch.triu_indices(self.N, self.N, 1, device=dev)
        same = (self.fd[:, pi] == self.fd[:, pj]).half()                              # [F, P]
        eqy = y[pi] == y[pj]
        if fam == 'table':
            conf = (sd[:, pi] == sd[:, pj]) & ~eqy
        elif fam == 'role':
            fg, cp = sd != 0, sd == y
            conf = (fg[:, pi] == fg[:, pj]) & ~(eqy | (cp[:, pi] & cp[:, pj]))
        else:
            cls = 2 * (self.r0d != 0).long() + (sd != 0).long()
            cps, cp0 = sd == y, self.r0d == y
            conf = (cls[:, pi] == cls[:, pj]) & ~(eqy | (cp0[pi] & cp0[pj]) | (cps[:, pi] & cps[:, pj]))
        tot = conf.sum(1)
        upper = torch.ones(self.F, self.F, dtype=torch.bool, device=dev).triu(1)
        out = []
        for a in range(0, self.S, chunk):
            c = conf[a:a + chunk]
            miss = torch.matmul(same.unsqueeze(0) * c.half().unsqueeze(1), same.T)     # [ch, F, F], 0 = separated
            none = tot[a:a + chunk] == 0
            single = (miss.diagonal(dim1=1, dim2=2) == 0) & ~none[:, None]
            pair = (miss == 0) & upper & ~single[:, :, None] & ~single[:, None, :] & ~none[:, None, None]
            s0 = torch.nonzero(none)[:, 0]
            s1, f1 = torch.nonzero(single, as_tuple=True)
            s2, f2, g2 = torch.nonzero(pair, as_tuple=True)
            m1 = -torch.ones_like(s0)
            out += [torch.stack([s0 + a, m1, m1], 1), torch.stack([s1 + a, f1, -torch.ones_like(f1)], 1),
                    torch.stack([s2 + a, f2, g2], 1)]
        cand = torch.cat(out)
        if fam == 'role2':
            cand = cand[cand[:, 0] != IDENT]
        return cand

    def evaluate(self, fam, cand):
        """Per candidate: valid (exact consistency), complete test prediction flag, prediction (copy wins a tie),
        prediction with unseen keys -> default, oracle (complete and every cell right under some consistent action),
        oracle with unseen keys -> default."""
        s, f, g = cand[:, 0], cand[:, 1], cand[:, 2]
        C, N, dev, y, ty = len(s), self.N, self.dev, self.y, self.ty
        if fam == 'table':
            cd, ct, ncls = self.sd[s], self.st[s], 10
        elif fam == 'role':
            cd, ct, ncls = (self.sd[s] != 0).long(), (self.st[s] != 0).long(), 2
        else:
            cd = 2 * (self.r0d != 0).long()[None] + (self.sd[s] != 0).long()
            ct = 2 * (self.r0t != 0).long()[None] + (self.st[s] != 0).long()
            ncls = 4
        fv = lambda h, X: torch.where(h[:, None] >= 0, X[h.clamp(min=0)], 0)
        cf = torch.where(f >= 0, self.card[f.clamp(min=0)], 1)
        cg = torch.where(g >= 0, self.card[g.clamp(min=0)], 1)
        Bm = int((ncls * cf * cg).max())
        base = torch.arange(C, device=dev)[:, None] * Bm
        idx = (base + cd + ncls * (fv(f, self.fd) + cf[:, None] * fv(g, self.fd))).ravel()
        tix = base + ct + ncls * (fv(f, self.ft) + cf[:, None] * fv(g, self.ft))
        yv = y.expand(C, N).ravel()
        ymin = torch.full((C * Bm,), 99, device=dev).scatter_reduce(0, idx, yv, 'amin')
        ymax = torch.full((C * Bm,), -1, device=dev).scatter_reduce(0, idx, yv, 'amax')
        seen = ymax >= 0
        tseen, yc = seen[tix], ymin[tix]
        ysame = yc == ymax[tix]
        valid = torch.ones(C, dtype=torch.bool, device=dev)
        if fam == 'table':
            pred, ok, default = yc, yc == ty, self.st[s]
        else:
            allcps = torch.ones(C * Bm, dtype=torch.long, device=dev).scatter_reduce(
                0, idx, (self.sd[s] == y).long().ravel(), 'amin')
            cs, sv = allcps[tix] == 1, self.st[s]
            if fam == 'role':
                pred = torch.where(cs, sv, yc)
                ok = (cs & (sv == ty)) | (ysame & (yc == ty))
                default = sv
            else:
                allcp0 = torch.ones(C * Bm, dtype=torch.long, device=dev).scatter_reduce(
                    0, idx, (self.r0d == y).long().expand(C, N).ravel(), 'amin')
                valid = (~seen | (ymin == ymax) | (allcp0 == 1) | (allcps == 1)).view(C, Bm).all(1)
                c0, r0 = allcp0[tix] == 1, self.r0t.expand(C, 25)
                pred = torch.where(c0, r0, torch.where(cs, sv, yc))
                ok = (c0 & (r0 == ty)) | (cs & (sv == ty)) | (ysame & (yc == ty))
                default = r0
        if fam == 'table':
            nact, A = seen.long(), 10
        elif fam == 'role':
            nact, A = (allcps == 1).long() + (ymin == ymax).long(), 11
        else:
            nact, A = (allcp0 == 1).long() + (allcps == 1).long() + (ymin == ymax).long(), 12
        keys = seen.view(C, Bm).sum(1)
        ties = torch.where(seen, torch.log2(nact.clamp(min=1).double()), 0).view(C, Bm).sum(1)
        cnt = torch.zeros(C * Bm, device=dev).scatter_add_(0, idx, torch.ones(len(idx), device=dev))
        if fam == 'table':
            copyb = torch.zeros_like(seen)
        elif fam == 'role':
            copyb = seen & (allcps == 1)
        else:
            copyb = seen & ((allcp0 == 1) | (allcps == 1))
        complete = tseen.all(1)
        pred_id = torch.where(tseen, pred, default)
        return dict(valid=valid, complete=complete, pred=pred, pred_id=pred_id, oracle=complete & ok.all(1),
                    oracle_id=torch.where(tseen, ok, pred_id == ty).all(1), keys=keys,
                    table_bits=keys.double() * float(np.log2(A)) - ties,
                    singles=(cnt == 1).view(C, Bm).sum(1),                  # keys seen on one demonstration cell
                    copy_cells=torch.where(copyb, cnt, 0).view(C, Bm).sum(1))   # demonstration cells copied


def eval_chunks(T, fam, cand, budget=2 ** 24):
    """Evaluate candidates in chunks bounded by the bucket array size."""
    if len(cand) == 0:
        return None
    card = int(T.card.max())
    width = {'table': 10, 'role': 2, 'role2': 4}[fam] * card * card
    step = max(1, budget // width)
    parts = [T.evaluate(fam, cand[a:a + step]) for a in range(0, len(cand), step)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


@torch.no_grad()
def solve(B, families, configs, dev):
    T = Task(B, dev)
    rows = []                                              # per family: cand, cost_base, m, valid, oracle, oracle_id
    for fam in families:
        cand = T.candidates(fam)
        if len(cand) == 0:
            continue
        ev = eval_chunks(T, fam, cand)
        m = (cand[:, 1] >= 0).long() + (cand[:, 2] >= 0).long()
        fbits = float(np.log2(3)) + torch.log2(torch.tensor([1.0, T.F, T.F * (T.F - 1) / 2], device=dev,
                                                             dtype=torch.float64))[m]
        rows.append(dict(fam=fam, cand=cand, cost=T.cost[cand[:, 0]] + m + EXTRA_SOURCE[fam], m=m, valid=ev['valid'],
                         bits=T.bits[cand[:, 0]] + fbits + ev['table_bits'], keys=ev['keys'],
                         oracle=ev['oracle'] & ev['valid'], oracle_id=ev['oracle_id'] & ev['valid']))
    res = {}
    for cname, extra in configs.items():
        res[cname] = {}
        nfam = len(extra)
        score = (lambda r: r['bits'] + float(np.log2(nfam))) if SCORE.get(cname) == 'bits' else \
            (lambda r: (r['cost'] + extra[r['fam']]).double())
        for level in (0, 1, 2):
            sel = [(r, r['valid'] & (r['m'] <= level)) for r in rows if r['fam'] in extra]
            sel = [(r, mask) for r, mask in sel if mask.any()]
            if not sel:
                res[cname][level] = dict(consistent=False, mdl_correct=False, mdl_correct_iddef=False, oracle=False,
                                         oracle_iddef=False, n=0)
                continue
            best = min(float(score(r)[mask].min()) for r, mask in sel)
            preds, preds_id, examples, n_best = set(), set(), [], 0
            for r, mask in sel:
                top = mask & ((score(r) - best).abs() < 1e-6)
                if not top.any():
                    continue
                ct = r['cand'][top]
                ev = eval_chunks(T, r['fam'], ct)
                n_best += len(ct)
                for p, c in zip(ev['pred'].tolist(), ev['complete'].tolist()):
                    preds.add(tuple(p) if c else None)
                preds_id |= {tuple(p) for p in ev['pred_id'].tolist()}
                for s, f, g in ct[:3].tolist():
                    examples.append((r['fam'], B['src_names'][s], [B['feat_names'][h] for h in (f, g) if h >= 0]))
            ty = tuple(B['ty'].tolist())
            res[cname][level] = dict(
                consistent=True, n=int(sum(int(mask.sum()) for _, mask in sel)),
                best_cost=int(round(best)) if SCORE.get(cname) != 'bits' else round(best, 3), n_best=n_best,
                mdl_correct=len(preds) == 1 and next(iter(preds)) == ty,
                mdl_correct_iddef=len(preds_id) == 1 and next(iter(preds_id)) == ty,
                oracle=any(bool(r['oracle'][mask].any()) for r, mask in sel),
                oracle_iddef=any(bool(r['oracle_id'][mask].any()) for r, mask in sel), example=examples[:3])
    return res


def build_banks(tasks, ext, workers=32):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(bank, tasks, [ext] * len(tasks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['repro', 'ext'], default='repro')
    ap.add_argument('--tasks', type=int, nargs='*')
    a = ap.parse_args()
    t0 = time.time()
    tasks, exact = load_tasks()
    assert exact, 'Mini-ARC reconstruction is not bit-exact'
    ids = a.tasks if a.tasks else list(range(len(tasks)))
    ext = a.mode == 'ext'
    banks = build_banks([tasks[i] for i in ids], ext)
    families = FAMILIES if ext else ('table', 'role')
    print(f'{len(ids)} tasks, {banks[0]["src"].shape[0]} sources, {banks[0]["feat"].shape[0]} features, banks in '
          f'{time.time() - t0:.0f} s', flush=True)
    dev = torch.device('cuda')
    results = {}
    for k, (i, B) in enumerate(zip(ids, banks)):
        results[i] = solve(B, families, CONFIGS[a.mode], dev)
        if (k + 1) % 25 == 0:
            print(f'  {k + 1} tasks, {time.time() - t0:.0f} s', flush=True)
    table = {}
    for cname in CONFIGS[a.mode]:
        print(f'colour maps: {cname}  {CONFIGS[a.mode][cname]}')
        print('  level      ' + ' '.join(f'{f:>18s}' for f in FLAGS))
        for level in (0, 1, 2):
            row = {f: sum(r[cname][level][f] for r in results.values()) for f in FLAGS}
            table[f'{cname}/m<={level}'] = row
            print(f'  m <= {level}     ' + ' '.join(f'{row[f]:18d}' for f in FLAGS))
    out = ROOT / 'runs' / f'pse_{a.mode}.json'
    if not a.tasks:
        out.write_text(json.dumps(dict(table=table, configs=CONFIGS[a.mode], sources=banks[0]['src_names'],
                                       features=banks[0]['feat_names'], seconds=time.time() - t0,
                                       tasks={str(i): {c: {str(l): v for l, v in r[c].items()} for c in r}
                                              for i, r in results.items()}), default=str, indent=1))
        print('->', out)
    print(f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
