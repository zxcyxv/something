"""Anchors + per-cell content predicates on Mini-ARC (docs/claude_2026-10-20_arc_logic_foundations.md section 9).

Program: out(o) = f( x(a(o; x)), pi_1(o; x), ..., pi_m(o; x) ), read width 1, m = 0, 1, 2 predicates, f a table
(read colour, predicate bits) -> output colour fitted on the demonstrations (it must be a function on every
demonstration cell). Output grid = input grid size (all Mini-ARC grids are 5x5; a read outside the grid gives 0).

Address terms a (cost):
    oblivious   D_k then a shift (a, b) in -4..4, background fill (648; cost [k != 0] + [shift != 0])
    anchored    D_k, then the shift that puts the top-left / floor centre / bottom-right of the bounding box of the
                non-background cells of D_k(x) at a target cell (3 x 8 x 25; cost 2 + [k != 0]; FO-definable)
    per-object  inside the bounding box of each 8-connected non-background object: horizontal flip, vertical flip or
                rotation by 180 within the box; identity elsewhere (3; cost 2; needs connectivity)
Predicates (cost 1 each): see PRED_NAMES.
Per task: consistent (some program fits every demonstration), mdl_correct (every consistent program of minimal cost
= address cost + number of predicates predicts the same complete test output and it is right), mdl_correct_iddef (the
same, a test key unseen in the demonstrations takes the read colour), oracle (some consistent program predicts the
test output exactly). Levels: m <= 0, m <= 1, m <= 2.

    python -m toy_g.miniarc_grammar      # CPU, one process per task; writes runs/miniarc_grammar.json
"""
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .miniarc_ceiling import IDX, SHIFT, dihedral, load_tasks

ROOT = Path(__file__).resolve().parents[1]
RR, CC = np.divmod(np.arange(25), 5)
N4 = [[j for j in range(25) if abs(RR[i] - RR[j]) + abs(CC[i] - CC[j]) == 1] for i in range(25)]
N8 = [[j for j in range(25) if j != i and max(abs(RR[i] - RR[j]), abs(CC[i] - CC[j])) == 1] for i in range(25)]
PRED_NAMES = ['obj_largest', 'obj_smallest', 'obj_single', 'obj_border', 'col4_largest', 'col4_single', 'enclosed',
              'row_has', 'col_has', 'rowcol_has', 'between_row', 'between_col', 'n4_ge1', 'n4_ge2', 'n8_ge1',
              'diag_has', 'border', 'centre', 'row_even', 'col_even', 'checker', 'main_diag', 'anti_diag', 'in_bbox',
              'bbox_edge', 'majority_colour', 'minority_colour']


def components(g, nbrs, same_colour):
    """Labels [25] of connected non-background components (-1 on background)."""
    lab, n = -np.ones(25, int), 0
    for s in range(25):
        if g[s] == 0 or lab[s] >= 0:
            continue
        lab[s], q = n, deque([s])
        while q:
            i = q.popleft()
            for j in nbrs[i]:
                if g[j] != 0 and lab[j] < 0 and (not same_colour or g[j] == g[i]):
                    lab[j] = n
                    q.append(j)
        n += 1
    return lab


def predicates(g):
    """[27, 25] booleans, PRED_NAMES order."""
    nb = g != 0
    p = {}
    for name, nbrs, same in (('obj', N8, False), ('col4', N4, True)):
        lab = components(g, nbrs, same)
        sizes = np.bincount(lab[lab >= 0]) if nb.any() else np.zeros(0, int)
        size = np.where(lab >= 0, sizes[np.maximum(lab, 0)] if len(sizes) else 0, 0)
        p[f'{name}_largest'] = nb & (size == (sizes.max() if len(sizes) else -1))
        p[f'{name}_single'] = nb & (size == 1)
        if name == 'obj':
            p['obj_smallest'] = nb & (size == (sizes.min() if len(sizes) else -1))
            edge = (RR == 0) | (RR == 4) | (CC == 0) | (CC == 4)
            touching = set(lab[edge & nb].tolist())
            p['obj_border'] = nb & np.isin(lab, list(touching))
    reach, q = np.zeros(25, bool), deque([i for i in range(25) if (RR[i] in (0, 4) or CC[i] in (0, 4)) and g[i] == 0])
    for i in q:
        reach[i] = True
    while q:
        i = q.popleft()
        for j in N4[i]:
            if g[j] == 0 and not reach[j]:
                reach[j] = True
                q.append(j)
    p['enclosed'] = ~nb & ~reach
    G2 = g.reshape(5, 5)
    rowc, colc = (G2 != 0).sum(1), (G2 != 0).sum(0)
    p['row_has'], p['col_has'] = rowc[RR] > 0, colc[CC] > 0
    p['rowcol_has'] = p['row_has'] | p['col_has']
    btw_r, btw_c = np.zeros(25, bool), np.zeros(25, bool)
    for i in range(25):
        r, c = RR[i], CC[i]
        for a in range(5):
            for b in range(a + 2, 5):
                if G2[r, a] != 0 and G2[r, a] == G2[r, b] and a < c < b:
                    btw_r[i] = True
                if G2[a, c] != 0 and G2[a, c] == G2[b, c] and a < r < b:
                    btw_c[i] = True
    p['between_row'], p['between_col'] = btw_r, btw_c
    n4 = np.array([sum(nb[j] for j in N4[i]) for i in range(25)])
    n8 = np.array([sum(nb[j] for j in N8[i]) for i in range(25)])
    p['n4_ge1'], p['n4_ge2'], p['n8_ge1'] = n4 >= 1, n4 >= 2, n8 >= 1
    p['diag_has'] = np.array([any(nb[j] for j in range(25) if j != i and abs(RR[i] - RR[j]) == abs(CC[i] - CC[j]))
                              for i in range(25)])
    p['border'] = (RR == 0) | (RR == 4) | (CC == 0) | (CC == 4)
    p['centre'] = (RR == 2) & (CC == 2)
    p['row_even'], p['col_even'], p['checker'] = RR % 2 == 0, CC % 2 == 0, (RR + CC) % 2 == 0
    p['main_diag'], p['anti_diag'] = RR == CC, RR + CC == 4
    if nb.any():
        r0, r1, c0, c1 = RR[nb].min(), RR[nb].max(), CC[nb].min(), CC[nb].max()
        inb = (RR >= r0) & (RR <= r1) & (CC >= c0) & (CC <= c1)
        p['in_bbox'] = inb
        p['bbox_edge'] = inb & ((RR == r0) | (RR == r1) | (CC == c0) | (CC == c1))
        counts = np.bincount(g[nb], minlength=10)
        present = np.flatnonzero(counts)
        p['majority_colour'] = nb & (g == present[np.argmax(counts[present])])
        p['minority_colour'] = nb & (g == present[np.argmin(counts[present])])
    else:
        for k in ('in_bbox', 'bbox_edge', 'majority_colour', 'minority_colour'):
            p[k] = np.zeros(25, bool)
    return np.stack([p[k] for k in PRED_NAMES])


def anchored_src(g, k, which, ta, tb):
    """Source index (25 = outside) of every output cell: D_k, then the shift putting the anchor of D_k(g) at (ta, tb)."""
    d, idx = dihedral(g.reshape(5, 5), k), dihedral(IDX, k)
    nz = np.argwhere(d != 0)
    if len(nz) == 0:
        return idx.ravel()
    lo, hi = nz.min(0), nz.max(0)
    anc = {'tl': lo, 'c': (lo + hi) // 2, 'br': hi}[which]
    si, sj = RR.reshape(5, 5) - (ta - anc[0]), CC.reshape(5, 5) - (tb - anc[1])
    inside = (si >= 0) & (si < 5) & (sj >= 0) & (sj < 5)
    return np.where(inside, idx[np.clip(si, 0, 4), np.clip(sj, 0, 4)], 25).ravel()


def object_src(g, mode):
    """Inside each 8-connected object's bounding box (smallest box first), flip h / v or rotate 180; identity elsewhere."""
    lab = components(g, N8, False)
    src = np.arange(25)
    boxes = []
    for o in range(lab.max() + 1):
        m = lab == o
        boxes.append((RR[m].min(), RR[m].max(), CC[m].min(), CC[m].max()))
    boxes.sort(key=lambda b: (b[1] - b[0] + 1) * (b[3] - b[2] + 1), reverse=True)
    for r0, r1, c0, c1 in boxes:                                         # smaller boxes overwrite larger ones
        for i in range(25):
            r, c = RR[i], CC[i]
            if r0 <= r <= r1 and c0 <= c <= c1:
                rr = r0 + r1 - r if mode in ('v', 'r') else r
                cc = c0 + c1 - c if mode in ('h', 'r') else c
                src[i] = rr * 5 + cc
    return src


def address_terms():
    """[(name, cost, kind, params)]."""
    terms = []
    for k in range(8):
        for a in range(-4, 5):
            for b in range(-4, 5):
                terms.append((f'D{k}+({a},{b})', int(k != 0) + int((a, b) != (0, 0)), 'fixed', SHIFT[k, a + 4, b + 4]))
    for which in ('tl', 'c', 'br'):
        for k in range(8):
            for ta in range(5):
                for tb in range(5):
                    terms.append((f'anchor_{which}:D{k}->({ta},{tb})', 2 + int(k != 0), 'anchor', (k, which, ta, tb)))
    for mode in ('h', 'v', 'r'):
        terms.append((f'object_{mode}', 2, 'object', mode))
    return terms


def src_of(term, g):
    _, _, kind, par = term
    if kind == 'fixed':
        return par
    if kind == 'anchor':
        return anchored_src(g, *par)
    return object_src(g, par)


def read(g, src):
    return np.where(src < 25, g[np.minimum(src, 24)], 0)


def keys_of(c, P, S):
    """Table key of every cell: read colour * 4 + the bits of the predicates in S."""
    k = c * 4
    for q, s in enumerate(S):
        k = k + (P[s].astype(int) << (1 - q))
    return k


def predicate_sets(P, pa, pb):
    """Predicate sets of size <= 2, minimal by inclusion, that separate every conflicting cell pair (pa[i], pb[i])."""
    if len(pa) == 0:
        return [()]
    sep = P[:, pa] != P[:, pb]                                           # [27, pairs]
    single = sep.all(1)
    miss = (~sep).astype(np.int32)
    both = (miss @ miss.T) == 0                                          # [27, 27]: together they separate every pair
    return [(i,) for i in np.flatnonzero(single)] + \
        [(i, j) for i, j in zip(*np.nonzero(np.triu(both, 1))) if not single[i] and not single[j]]


def programs(task, terms):
    """Every consistent program, minimal by inclusion in its predicate set, of both colour-map families:
    'table'  f(read colour, bits) -> colour (any function on the keys seen in the demonstrations)
    'role'   f(read is foreground, bits) -> copy the read colour | one constant colour (copy wins a tie)
    -> [(address cost + m, m, family, term index, S, complete test prediction or None, prediction with an unseen key
    -> the read colour, the prediction with every tied key (copy and one constant both fit) set to the constant)]."""
    X, Y, tx = task['x'], task['y'], task['tx']
    P = np.concatenate([predicates(g) for g in X], 1)                    # [27, n*25]
    Pt = predicates(tx)
    y = Y.ravel()
    eq_y = y[:, None] == y[None, :]
    out = []
    for ti, term in enumerate(terms):
        c = np.concatenate([read(g, src_of(term, g)) for g in X])
        ct = read(tx, src_of(term, tx))
        cp = y == c
        conflicts = {'table': (c[:, None] == c[None, :]) & ~eq_y,
                     'role': ((c[:, None] != 0) == (c[None, :] != 0)) & ~(eq_y | (cp[:, None] & cp[None, :]))}
        for fam, conf in conflicts.items():
            for S in predicate_sets(P, *np.nonzero(np.triu(conf))):
                if fam == 'table':
                    tab = np.full(40, -1)
                    tab[keys_of(c, P, S)] = y
                    pred = tab[keys_of(ct, Pt, S)]
                    alt = pred
                else:
                    kd, kt = keys_of((c != 0).astype(int), P, S), keys_of((ct != 0).astype(int), Pt, S)
                    seen, nocopy, mixed, const = np.zeros(8, bool), np.zeros(8, bool), np.zeros(8, bool), np.full(8, -1)
                    seen[kd] = True
                    np.logical_or.at(nocopy, kd, ~cp)
                    np.logical_or.at(mixed, kd, y != y[np.argmax(kd[:, None] == kd[None, :], 1)])
                    const[kd] = y
                    pred = np.where(seen[kt], np.where(nocopy[kt], const[kt], ct), -1)
                    alt = np.where(seen[kt] & ~nocopy[kt] & ~mixed[kt], const[kt], pred)   # the tied constant
                out.append((term[1] + len(S), len(S), fam, ti, S, None if (pred < 0).any() else pred,
                            np.where(pred < 0, ct, pred), alt))
    return out


def summarise(progs, ty, terms, extra):
    """Per level m <= 0, 1, 2: consistent, MDL-correct (every minimal-cost program predicts the true test output; cost =
    address cost + m + extra[family]; a tied role key copies), the same with unseen key -> read colour, and oracle
    (some program is right, a tied role key taking either action)."""
    res = {}
    for level in (0, 1, 2):
        cand = [(cost + extra[fam], m, fam, ti, S, pr, pi, alt) for cost, m, fam, ti, S, pr, pi, alt in progs
                if m <= level and fam in extra]
        if not cand:
            res[level] = dict(consistent=False, mdl_correct=False, mdl_correct_iddef=False, oracle=False,
                              oracle_iddef=False, n=0)
            continue
        best = min(p[0] for p in cand)
        top = [p for p in cand if p[0] == best]
        vals = {None if p[5] is None else tuple(p[5]) for p in top}
        vals_id = {tuple(p[6]) for p in top}
        res[level] = dict(consistent=True, n=len(cand), best_cost=int(best), n_best=len(top),
                          oracle=any(p[5] is not None and ((p[5] == ty) | (p[7] == ty)).all() for p in cand),
                          oracle_iddef=any(((p[6] == ty) | (p[7] == ty)).all() for p in cand),
                          mdl_correct=len(vals) == 1 and None not in vals and np.array_equal(next(iter(vals)), ty),
                          mdl_correct_iddef=len(vals_id) == 1 and np.array_equal(next(iter(vals_id)), ty),
                          example=[(p[2], terms[p[3]][0], [PRED_NAMES[s] for s in p[4]]) for p in top[:3]])
    return res


CONFIGS = {'table': {'table': 0}, 'table+role': {'role': 0, 'table': 1}}
FLAGS = ('consistent', 'mdl_correct', 'mdl_correct_iddef', 'oracle', 'oracle_iddef')


def evaluate_task(task, terms):
    progs = programs(task, terms)
    return {name: summarise(progs, task['ty'], terms, extra) for name, extra in CONFIGS.items()}


def main():
    t0 = time.time()
    tasks, exact = load_tasks()
    assert exact, 'Mini-ARC reconstruction is not bit-exact'
    terms = address_terms()
    with ProcessPoolExecutor(min(32, os.cpu_count())) as ex:
        results = list(ex.map(evaluate_task, tasks, [terms] * len(tasks)))
    print(f'{len(tasks)} tasks, {len(terms)} address terms, {len(PRED_NAMES)} predicates, {time.time() - t0:.0f} s')
    table = {}
    for name in CONFIGS:
        print(f'colour maps: {name}   (cost extras {CONFIGS[name]})')
        print('  level      ' + ' '.join(f'{f:>18s}' for f in FLAGS))
        for level in (0, 1, 2):
            row = {f: sum(r[name][level][f] for r in results) for f in FLAGS}
            table[f'{name}/m<={level}'] = row
            print(f'  m <= {level}     ' + ' '.join(f'{row[f]:18d}' for f in FLAGS))
    out = ROOT / 'runs' / 'miniarc_grammar.json'
    out.write_text(json.dumps(dict(table=table, configs=CONFIGS, predicates=PRED_NAMES, seconds=time.time() - t0,
                                   tasks=[{name: {str(k): v for k, v in r[name].items()} for name in r}
                                          for r in results]), default=str, indent=1))
    print('->', out)


if __name__ == '__main__':
    main()
