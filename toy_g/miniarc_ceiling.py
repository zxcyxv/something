"""Ceiling of "each output cell reads one input cell at a content-independent address, then a colour map" on Mini-ARC.

The 149 original tasks (5x5, colours 0..9, 0 = background) are rebuilt from data/miniarc.npz: identity version task * 32,
demonstrations in their original order, checked by replaying data/prep_miniarc.py's RNG and rebuilding the npz bit-exactly.
A hypothesis (r, kappa) predicts out(o) = kappa(in(r(o))); a read outside the grid reads colour 0 (then kappa applies).

Address families r (grids are not tori; D_k = dihedral(., k) of data/prep_miniarc.py):
    id; D4 (8); D4_cyc: D_k then cyclic shift (a, b) in 0..4 (200); D4_shift: D_k then shift (a, b) in -4..4 (648);
    D4_anchor_tl / D4_anchor_c: D_k then the shift that puts the top-left corner / floor centre of the bounding box of the
    non-background cells of D_k(in) at a target (ta, tb) (200 each; content-dependent, so outside F_any);
    any: every output cell reads one fixed input cell or is a free constant (exact backtracking CSP jointly with kappa).
Colour maps kappa: func (any map 0..9 -> 0..9) and perm (permutation of 1..9, 0 -> 0); constants of 'any' are free in both.
Per task: consistent (some hypothesis reproduces every demonstration), determined (all consistent hypotheses predict one
test output; a test colour on which kappa is left free counts as several predictions), determined_correct, oracle_correct
(some consistent hypothesis predicts the true test output, i.e. the demonstrations plus the test pair are consistent).
'any' is decided exactly without enumerating hypotheses: determined iff no test cell can take a second value, one CSP
query per (cell, value); no cap is needed.

    python -m toy_g.miniarc_ceiling [--quick]  # CPU; writes runs/miniarc_ceiling.json, prints the table and the checks
                                               # ~5 min (main pass ~10 s); --quick skips the 31-version invariance check
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['id', 'D4', 'D4_cyc', 'D4_shift', 'D4_anchor_tl', 'D4_anchor_c', 'any']
CLOSED = ['id', 'D4', 'D4_cyc', 'D4_shift', 'any']          # closed under conjugation by D4 and colour permutations
NESTED = [('id', 'D4'), ('D4', 'D4_cyc'), ('D4', 'D4_shift'), ('D4_cyc', 'any'), ('D4_shift', 'any')]
CLASSES = ['func', 'perm']
FLAGS = ['consistent', 'determined', 'determined_correct', 'oracle_correct']
IDX, (II, JJ) = np.arange(25).reshape(5, 5), np.mgrid[:5, :5]


def dihedral(g, k):
    """data/prep_miniarc.py: rot90 k % 4 times, then transpose if k >= 4."""
    g = np.rot90(g, k % 4)
    return g.T if k >= 4 else g


def shift_src(k, a, b, cyclic):
    """Input cell read by every output cell for D_k followed by the translation (a, b); 25 = outside (reads colour 0)."""
    s, si, sj = dihedral(IDX, k), II - a, JJ - b
    inside = cyclic | ((si >= 0) & (si < 5) & (sj >= 0) & (sj < 5))
    return np.where(inside, s[si % 5, sj % 5], 25).ravel()


SHIFT = np.array([[[shift_src(k, a, b, False) for b in range(-4, 5)] for a in range(-4, 5)] for k in range(8)])
FIXED = {'id': [((0, 0, 0), shift_src(0, 0, 0, True))],
         'D4': [((k, 0, 0), shift_src(k, 0, 0, True)) for k in range(8)],
         'D4_cyc': [((k, a, b), shift_src(k, a, b, True)) for k in range(8) for a in range(5) for b in range(5)],
         'D4_shift': [((k, a, b), SHIFT[k, a + 4, b + 4]) for k in range(8) for a in range(-4, 5) for b in range(-4, 5)]}


def load_tasks(path=ROOT / 'data/miniarc.npz'):
    """Original tasks [dict(x [n, 25], y [n, 25], tx [25], ty [25], perms)] and whether rebuilding the npz from them
    (replaying the prep RNG: 3 colour permutations per task, then the shuffle of the training rows) is bit-exact."""
    d = np.load(path)
    X, Y, T = d['train_inputs'].astype(int) - 1, d['train_labels'].astype(int) - 1, d['train_task']
    rng = np.random.default_rng(0)
    perms = [[np.arange(10)] + [np.concatenate([[0], 1 + rng.permutation(9)]) for _ in range(3)] for _ in range(149)]
    order = rng.permutation(len(T))
    inv, n = np.argsort(order), np.bincount(T // 32) // 32
    start = np.concatenate([[0], np.cumsum(32 * n)])
    tasks, rx, ry, rt = [], [], [], []
    for i in range(len(d['test_task'])):
        assert d['test_task'][i] == 32 * i
        rows = inv[start[i] + np.arange(n[i])]             # the identity version's rows, in the original order
        tasks.append(dict(x=X[rows].reshape(-1, 25), y=Y[rows].reshape(-1, 25), perms=perms[i],
                          tx=d['test_inputs'][i].astype(int).ravel() - 1, ty=d['test_labels'][i].astype(int).ravel() - 1))
        for c, perm in enumerate(perms[i]):
            for k in range(8):
                rx += [dihedral(perm[x], k) for x in X[rows]]
                ry += [dihedral(perm[y], k) for y in Y[rows]]
                rt += [32 * i + 8 * c + k] * n[i]
    exact = all((np.array(r)[order] == a).all() for r, a in ((rx, X), (ry, Y), (rt, T)))
    return tasks, exact


def version(task, c, k):
    """Version 8c + k of a task (colour permutation c, then D_k) with its test pair transformed the same way; the
    demonstrations equal the npz rows of version 32 * task + 8c + k (the bit-exact rebuild in load_tasks)."""
    tf = lambda g: dihedral(task['perms'][c][g.reshape(5, 5)], k).ravel()
    return dict(x=np.array([tf(g) for g in task['x']]), y=np.array([tf(g) for g in task['y']]), tx=tf(task['tx']),
                ty=tf(task['ty']), perms=task['perms'])


# ---- explicit families: enumerate r, read the colours, check kappa ---------------------------------------------------

def anchor(grid, k, centre):
    nz = np.argwhere(dihedral(grid.reshape(5, 5), k) != 0)
    if len(nz) == 0:
        return None
    lo, hi = nz.min(0), nz.max(0)
    return (lo + hi) // 2 if centre else lo


def reads(fam, grids):
    """Hypothesis labels [H] and the colour each output cell reads in each grid [H, G, 25] (grids [G, 25])."""
    if fam in FIXED:
        lab, src = zip(*FIXED[fam])
        src = np.repeat(np.array(src)[:, None], len(grids), 1)
    else:
        lab, src = [], []
        for k in range(8):
            anc = [anchor(g, k, fam == 'D4_anchor_c') for g in grids]
            for ta, tb in itertools.product(range(5), range(5)):
                lab.append((k, ta, tb))
                src.append([SHIFT[k, 4, 4] if c is None else SHIFT[k, ta - c[0] + 4, tb - c[1] + 4] for c in anc])
        src = np.array(src)
    pad = np.concatenate([grids, np.zeros((len(grids), 1), int)], 1)
    return list(lab), pad[np.arange(len(grids))[None, :, None], src]


def complete_perm(kappa):
    """kappa(0) = 0, and the last free colour of 1..9 is forced when the other eight are assigned."""
    kappa = kappa.copy()
    kappa[0] = 0
    free = np.flatnonzero(kappa[1:] < 0) + 1
    if len(free) == 1:
        kappa[free[0]] = min(set(range(1, 10)) - set(kappa[1:].tolist()))
    return kappa


def fit(rd, out, cls):
    """rd [H, N] colours read, out [N] colours wanted -> consistent [H] and kappa [H, 10] (-1 = free)."""
    pos = np.zeros((len(rd), 10, 10), bool)
    pos[np.arange(len(rd))[:, None], rd, out[None]] = True
    ok = pos.sum(2).max(1) <= 1
    if cls == 'perm':
        ok &= (pos.sum(1).max(1) <= 1) & ~pos[:, 0, 1:].any(1) & ~pos[:, 1:, 0].any(1)
    kappa = np.where(pos.any(2), pos.argmax(2), -1)
    if cls == 'perm' and ok.any():
        kappa[ok] = [complete_perm(k) for k in kappa[ok]]
    return ok, kappa


def unique_pred(kappa, rd_test, ok, ty):
    """(determined, determined_correct) of the consistent hypotheses' test predictions; -1 = free colour."""
    pred = kappa[np.arange(len(kappa))[:, None], rd_test][ok]
    det = len(set(map(tuple, pred))) == 1 and bool(pred.min() >= 0)
    return det, det and bool((pred[0] == ty).all())


def eval_explicit(task, fam, cls):
    lab, rd = reads(fam, np.concatenate([task['x'], task['tx'][None]]))
    H = len(lab)
    ok, kappa = fit(rd[:, :-1].reshape(H, -1), task['y'].ravel(), cls)
    orc, kappa_t = fit(rd.reshape(H, -1), np.concatenate([task['y'].ravel(), task['ty']]), cls)
    det, det_ok = unique_pred(kappa, rd[:, -1], ok, task['ty'])
    kd = np.where(kappa >= 0, kappa, np.arange(10))     # diagnostic: free colours default to identity (perm: if c is unused)
    if cls == 'perm':
        kd[(kappa < 0) & (kappa[:, :, None] == np.arange(10)).any(1)] = -1
    res = dict(consistent=bool(ok.any()), determined=det, determined_correct=det_ok, oracle_correct=bool(orc.any()),
               determined_correct_id_default=unique_pred(kd, rd[:, -1], ok, task['ty'])[1], n_consistent=int(ok.sum()),
               labels=[lab[h] for h in np.flatnonzero(ok)], example=None)
    if ok.any():                                         # prefer a hypothesis that also predicts the test output
        h, kap = (np.flatnonzero(orc)[0], kappa_t) if orc.any() else (np.flatnonzero(ok)[0], kappa)
        res['example'] = dict(zip(('k', 'ta', 'tb') if 'anchor' in fam else ('k', 'a', 'b'), map(int, lab[h])),
                              kappa=kap[h].tolist(), predicts_test=bool(orc.any()))
    return res


def apply_explicit(fam, ex, grids):
    lab, rd = reads(fam, grids)
    return np.array(ex['kappa'])[rd[lab.index(tuple(list(ex.values())[:3]))]]       # -1 where kappa is free


# ---- F_any: exact constraint search over (per-cell source or constant, kappa) ----------------------------------------

def cell_candidates(X, col, cls):
    """Ways to explain one output cell with values col [P] (-1 = unconstrained) over inputs X [P, 25]: None if it can be a
    constant, else {requirement: source}, a requirement being the (input colour, output colour) pairs kappa must contain."""
    spec = col >= 0
    if len(set(col[spec].tolist())) <= 1:
        return None
    cand = {}
    for s, xs in enumerate(X[spec].T.tolist()):
        req = {}
        if all(req.setdefault(a, b) == b for a, b in zip(xs, col[spec].tolist())) and (cls == 'func' or (
                len(set(req.values())) == len(req) and all((a == 0) == (b == 0) for a, b in req.items()))):
            cand.setdefault(tuple(sorted(req.items())), s)
    return cand


def solve(cells, cls):
    """kappa (list of 10, -1 = free) under which every non-constant cell has a source whose requirement it contains, or
    None. Depth-first, most constrained cell first; every branch assigns >= 1 colour, so the depth is <= 10; failed partial
    maps are memoised (what remains to be satisfied depends only on kappa)."""
    if any(c is not None and not c for c in cells):
        return None
    cells, failed = [list(c) for c in cells if c is not None], set()
    fits = lambda req, kap: all(kap[a] == b or (kap[a] < 0 and (cls == 'func' or b not in kap)) for a, b in req)

    def rec(kap):
        if tuple(kap) in failed:
            return None
        best = None
        for cand in cells:
            comp = []
            for req in cand:
                if all(kap[a] == b for a, b in req):
                    break                                                           # already satisfied
                if fits(req, kap):
                    comp.append(req)
            else:
                if not comp:
                    failed.add(tuple(kap))
                    return None
                best = comp if best is None or len(comp) < len(best) else best
        if best is None:
            return kap
        for req in best:
            k2 = list(kap)
            for a, b in req:
                k2[a] = b
            if (r := rec(k2)) is not None:
                return r
        failed.add(tuple(kap))
        return None

    return rec([0 if (cls == 'perm' and c == 0) else -1 for c in range(10)])


def realise(cells, consts, kap, cls):
    """A full hypothesis from a solved kappa: per output cell a source (-1 = constant) and the constant values."""
    src = [-1 if c is None else next(s for req, s in c.items() if all(kap[a] == b for a, b in req)) for c in cells]
    kappa = complete_perm(np.array(kap)) if cls == 'perm' else np.array(kap)
    return dict(src=src, const={o: consts[o] for o in range(25) if src[o] < 0}, kappa=kappa.tolist())


def apply_any(ex, grids):
    k = np.array(ex['kappa'])
    return np.array([[ex['const'][o] if s < 0 else k[g[s]] for o, s in enumerate(ex['src'])] for g in grids])


def eval_any(task, cls, count=True):
    """count: number of ambiguous test cells (else stop at the first; the flags are the same)."""
    X, Y, tx, ty = task['x'], task['y'], task['tx'], task['ty']
    X1, consts = np.concatenate([X, tx[None]]), Y[0].tolist()
    cells = [cell_candidates(X, Y[:, o], cls) for o in range(25)]
    if (kap := solve(cells, cls)) is None:
        return dict(consistent=False, determined=False, determined_correct=False, oracle_correct=False, example=None)
    hyp = realise(cells, consts, kap, cls)
    pred = apply_any(hyp, tx[None])[0]                                            # -1 = free colour
    alt = lambda o, v: solve(cells[:o] + [cell_candidates(X1, np.append(Y[:, o], v), cls)] + cells[o + 1:], cls)
    amb = (o for o in range(25) if pred[o] < 0 or any(alt(o, v) is not None for v in range(10) if v != pred[o]))
    ambiguous = list(amb) if count else list(itertools.islice(amb, 1))
    cells_t = [cell_candidates(X1, np.append(Y[:, o], ty[o]), cls) for o in range(25)]
    kap_t = solve(cells_t, cls)
    ex = dict(realise(cells_t, consts, kap_t, cls), predicts_test=True) if kap_t is not None else dict(hyp, predicts_test=False)
    return dict(consistent=True, determined=not ambiguous, determined_correct=not ambiguous and bool((pred == ty).all()),
                oracle_correct=kap_t is not None, ambiguous_test_cells=len(ambiguous) if count else None, example=ex)


def evaluate(task, fam, cls, count=True):
    return eval_any(task, cls, count) if fam == 'any' else eval_explicit(task, fam, cls)


# ---- sanity checks ---------------------------------------------------------------------------------------------------

def brute_any(task, cls, chunk=10000):
    """Independent check of eval_any for tasks whose inputs use few colours: enumerate kappa on those colours (in chunks)
    and, per kappa, the sources each output cell may read; the marginal test values per cell decide determinedness."""
    X, Y, tx, ty = task['x'], task['y'], task['tx'], task['ty']
    cols = sorted(set(X.ravel().tolist()) | set(tx.tolist()))
    nz = [c for c in cols if c]
    K = np.array(list(itertools.product(range(10), repeat=len(cols)))) if cls == 'func' else \
        np.array([[0] * (len(cols) > len(nz)) + list(p) for p in itertools.permutations(range(1, 10), len(nz))])
    lut, const = np.searchsorted(cols, np.arange(10)), (Y == Y[0]).all(0)
    poss, consistent, oracle = const[:, None] & (Y[0][:, None] == np.arange(10)), False, False      # poss [o, value]
    for k in np.array_split(K, -(-len(K) // chunk)):
        rx, rt = k[:, lut[X]], k[:, lut[tx]]                                # [n, P, 25], [n, 25] colours after kappa
        compat = (rx[:, :, None, :] == Y[None, :, :, None]).all(1)          # [n, o, s] reading s explains o
        valid = (compat.any(2) | const).all(1)
        c, t = compat[valid], rt[valid]
        consistent |= bool(valid.any())
        poss |= (c[..., None] & (t[:, None, :, None] == np.arange(10))).any((0, 2))
        oracle |= bool(((c & (t[:, None, :] == ty[None, :, None])).any(2) | (const & (Y[0] == ty))).all(1).any())
    det = consistent and bool((poss.sum(1) == 1).all())
    return dict(consistent=consistent, determined=det, determined_correct=det and bool((poss.argmax(1) == ty).all()),
                oracle_correct=oracle)


def check_examples(task, res):
    """Every reported example reproduces every demonstration (and the test output when it claims to)."""
    bad = []
    for (fam, cls), r in res.items():
        if (ex := r['example']) is None:
            continue
        n = len(task['x']) + ex['predicts_test']
        grids, want = np.concatenate([task['x'], task['tx'][None]])[:n], np.concatenate([task['y'], task['ty'][None]])[:n]
        got = apply_any(ex, grids) if fam == 'any' else apply_explicit(fam, ex, grids)
        bad += [] if (got == want).all() else [(fam, cls)]
    return bad


def check_synthetic(task, rng):
    """Outputs generated from the task's own inputs by a random (r, kappa) of each explicit family must be recognised
    (generating r among the consistent ones, oracle correct), also by 'any' for the content-independent families;
    each D_k with the identity colour map must be recognised by D4."""
    fails, grids = [], np.concatenate([task['x'], task['tx'][None]])
    for fam in FAMILIES[:-1]:
        lab, rd = reads(fam, grids)
        for cls in CLASSES:
            h = rng.integers(len(lab))
            kap = rng.integers(0, 10, 10) if cls == 'func' else np.concatenate([[0], 1 + rng.permutation(9)])
            syn = dict(x=task['x'], y=kap[rd[h, :-1]], tx=task['tx'], ty=kap[rd[h, -1]])
            r = eval_explicit(syn, fam, cls)
            if not (r['consistent'] and r['oracle_correct'] and lab[h] in r['labels']):
                fails.append((fam, cls, lab[h]))
            if fam in CLOSED and not eval_any(syn, cls, count=False)['oracle_correct']:
                fails.append(('any', cls, fam, lab[h]))
    for k in range(8):
        out = np.array([dihedral(g.reshape(5, 5), k).ravel() for g in grids])
        r = eval_explicit(dict(x=task['x'], y=out[:-1], tx=task['tx'], ty=out[-1]), 'D4', 'perm')
        if not (r['oracle_correct'] and (k, 0, 0) in r['labels']):
            fails.append(('D4 identity colours', k))
    return fails


def check_versions(tasks, results):
    """Flags of the closed families must not change on the other 31 versions of every task (colour permutation c, then
    D_k; demonstrations = the npz rows of that version, test pair transformed the same way)."""
    return [(i, v, f, c) for i, t in enumerate(tasks) for v in range(1, 32) for tv in [version(t, v // 8, v % 8)]
            for f in CLOSED for c in CLASSES if any(evaluate(tv, f, c, count=False)[k] != results[i][f, c][k] for k in FLAGS)]


def main():
    t0 = time.time()
    tasks, exact = load_tasks()
    nd = sum(len(t['x']) for t in tasks)
    print(f'{len(tasks)} tasks, {nd} demonstrations (x 32 versions = {32 * nd} rows); npz rebuilt bit-exactly: {exact}')
    results = [{(f, c): evaluate(t, f, c) for f in FAMILIES for c in CLASSES} for t in tasks]
    t_main = time.time() - t0
    count = lambda f, c, key: sum(bool(r[f, c].get(key)) for r in results)
    table = {f: {c: {k: count(f, c, k) for k in FLAGS + ['determined_correct_id_default'] * (f != 'any')} for c in CLASSES}
             for f in FAMILIES}
    print(f'\ncounts out of {len(tasks)}\n{"family":13s}' + ''.join(f'| {c}: consistent det_correct oracle ' for c in CLASSES))
    for f in FAMILIES:
        print(f'{f:13s}' + ''.join(f'| {table[f][c]["consistent"]:>16d} {table[f][c]["determined_correct"]:>11d} '
                                   f'{table[f][c]["oracle_correct"]:>6d} ' for c in CLASSES))
    print(f'\ndiagnostics\n{"family":13s}' + ''.join(f'| {c}: determined det_wrong det_correct(free colour = id) '
                                                    for c in CLASSES))
    for f in FAMILIES:
        print(f'{f:13s}' + ''.join(f'| {t["determined"]:>16d} {t["determined"] - t["determined_correct"]:>9d} '
                                   f'{t.get("determined_correct_id_default", "-"):>29} ' for t in table[f].values()))

    checks = dict(npz_rebuilt_bit_exactly=exact)
    ident = [i for i, t in enumerate(tasks) if (t['tx'] == t['ty']).all()]
    checks['identity_test_tasks'] = {i: {f'id/{c}': {k: results[i]['id', c][k] for k in FLAGS} for c in CLASSES}
                                     for i in ident}
    checks['examples_failing_replay'] = {i: b for i, r in enumerate(results) if (b := check_examples(tasks[i], r))}
    checks['nesting_violations'] = {f'{a}<={b}/{c}': [i for i, r in enumerate(results) if any(
        r[a, c][k] and not r[b, c][k] for k in ('consistent', 'oracle_correct'))] for a, b in NESTED for c in CLASSES}
    checks['anchor_consistent_not_any'] = {f'{f}/{c}': [i for i, r in enumerate(results) if r[f, c]['consistent'] and
                                                        not r['any', c]['consistent']] for f in FAMILIES[4:6] for c in CLASSES}
    small = [i for i, t in enumerate(tasks) if len(set(t['x'].ravel().tolist()) | set(t['tx'].tolist())) <= 5]
    checks['brute_force_any'] = dict(tasks=len(small), mismatches=[(i, c) for i in small for c in CLASSES if any(
        brute_any(tasks[i], c)[k] != results[i]['any', c][k] for k in FLAGS)])
    rng = np.random.default_rng(0)
    checks['synthetic_failures'] = {i: fl for i, t in enumerate(tasks) if (fl := check_synthetic(t, rng))}
    checks['version_invariance_failures'] = 'skipped' if '--quick' in sys.argv else check_versions(tasks, results)
    same = {}
    for i, t in enumerate(tasks):
        same.setdefault((t['tx'].tobytes(), t['ty'].tobytes()), []).append(i)
    checks['tasks_with_identical_test_pairs'] = [g for g in same.values() if len(g) > 1]
    print('\nchecks:', json.dumps(checks, default=str))

    surprising = {f'{f}/{c}': dict(
        consistent_not_oracle=[i for i, r in enumerate(results) if r[f, c]['consistent'] and not r[f, c]['oracle_correct']],
        determined_wrong=[i for i, r in enumerate(results) if r[f, c]['determined'] and not r[f, c]['determined_correct']],
        oracle_not_determined=[i for i, r in enumerate(results) if r[f, c]['oracle_correct'] and not r[f, c]['determined']])
        for f in FAMILIES for c in CLASSES}
    print('\nsurprising:\n' + '\n'.join(f'  {k}: {json.dumps(v)}' for k, v in surprising.items()))
    out = dict(doc=__doc__, table=table, checks=checks, surprising=surprising, seconds_main=round(t_main, 1),
               seconds_total=round(time.time() - t0, 1),
               tasks={i: dict(n_demos=len(t['x']), test_is_identity=bool((t['tx'] == t['ty']).all()),
                              results={f'{f}/{c}': {k: v for k, v in results[i][f, c].items() if k != 'labels'}
                                       for f in FAMILIES for c in CLASSES}) for i, t in enumerate(tasks)})
    (ROOT / 'runs').mkdir(exist_ok=True)
    (ROOT / 'runs/miniarc_ceiling.json').write_text(json.dumps(out, indent=1, default=int) + '\n')
    print(f'\nmain pass {t_main:.1f} s, total {time.time() - t0:.1f} s -> runs/miniarc_ceiling.json')


if __name__ == '__main__':
    main()
