"""Signature audit of ARC-AGI-1/2 for the prior-universality analysis (docs/claude_2026-10-21_prior_universality.md).

Per task (train and test pairs; the public test outputs are used), it records format facts and whether the
input-output relation lies in a few operator classes. Coverage is an ORACLE upper bound: some member of the class maps
every input of the task (train and test) to its output. It is not a solver and says nothing about learnability.
For the enumerable classes it also records DETERMINED: every member consistent with the train pairs gives the test
outputs (the class plus the demonstrations fix the answer without any further prior).

Classes (codes as in the document):
  A1   Y = f(X), f a global colour map
  A2   Y = f(Z(X)), Z a content-independent address map fixed by the shapes: D4 x {identity, upscale, tile,
       subsample, block crop, fixed-size crop (anchored at a corner), shift (bbox-aligned candidates only), cyclic roll,
       per-tile-D4 mosaic}
  ADDR general content-independent address map on fixed-size tasks: Y[p] = f(X[pi(p)]) for one map pi shared by all
       pairs (needs every input to share one shape and every output to share one shape); variants with f = identity,
       with a general f, and with constant output cells allowed (affine term)
  A3   multi-source cellwise maps: overlay of two halves (optional separator line, optional flip of the second half),
       Kronecker self-product (fractal)
  B1   3x3 neighbourhood lookup (one cellular-automaton step), consistent / determined
  B2   fill of enclosed background, rays (added cells reachable in a straight run from an input cell), gravity
  C    per-object recolour, object filter (whole objects removed), object rearrangement (multiset of objects kept)
  D    crop of one input object, output found anywhere inside the input, tiny summary outputs
  E    symmetry completion (grid-aligned D4 element), periodic output with a period shorter than the grid

    python -I docs/evidence/2026-10-21/arc_prior_audit.py ARC_AGI_DIR ARC_AGI_2_DIR --out docs/evidence/2026-10-21/arc_prior_audit.json
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np

D4 = [
    ('id', lambda a: a),
    ('r90', lambda a: np.rot90(a, 1)),
    ('r180', lambda a: np.rot90(a, 2)),
    ('r270', lambda a: np.rot90(a, 3)),
    ('tr', lambda a: a.T),
    ('fx', lambda a: a[:, ::-1]),
    ('fy', lambda a: a[::-1, :]),
    ('atr', lambda a: a[::-1, ::-1].T),
]
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CSP_NODES = 200


# ---------------------------------------------------------------------------- loading
def load_sets(arc1, arc2):
    sets = collections.OrderedDict()
    for name, d in (('arc1_train', f'{arc1}/data/training'), ('arc1_eval', f'{arc1}/data/evaluation'),
                    ('arc2_train', f'{arc2}/data/training'), ('arc2_eval', f'{arc2}/data/evaluation')):
        tasks = []
        for f in sorted(glob.glob(d + '/*.json')):
            j = json.load(open(f))
            conv = lambda ps: [(np.array(p['input'], np.int16), np.array(p['output'], np.int16)) for p in ps]
            tasks.append((os.path.basename(f)[:-5], conv(j['train']), conv(j['test'])))
        sets[name] = tasks
    return sets


# ---------------------------------------------------------------------------- colour maps
def cmap_fit(srcs, dsts):
    """Global map f with f(src) = dst cellwise over all array pairs (src may hold -1 = outside); None if none."""
    s = np.concatenate([a.ravel() for a in srcs]).astype(np.int32)
    d = np.concatenate([b.ravel() for b in dsts]).astype(np.int32)
    u = np.unique((s + 1) * 16 + d)
    keys = u // 16
    if len(np.unique(keys)) != len(keys):
        return None
    return {int(k) - 1: int(v) for k, v in zip(keys, u % 16)}


def cmap_apply(f, a, identity_default=False):
    """f applied to a; None if a holds a colour outside the domain of f (identity_default: such colours map to
    themselves)."""
    lut = np.full(17, -99, np.int32)
    if identity_default:
        lut[1:11] = np.arange(10)
    for k, v in f.items():
        lut[k + 1] = v
    z = lut[a.astype(np.int32) + 1]
    return None if (z == -99).any() else z


# ---------------------------------------------------------------------------- A2: content-independent address maps
def a2_members(train):
    """Members of the A2 family that the first train pair allows (complete for every subfamily except shift, whose
    offsets are only the bounding-box alignments). Each member: (name, function X -> Z or None)."""
    X0, Y0 = train[0]
    bg = background(train)
    out = []
    for tname, T in D4:
        Z0 = T(X0)
        h, w = Z0.shape
        H, W = Y0.shape
        out.append((tname, lambda X, T=T: T(X)))
        if H % h == 0 and W % w == 0 and (H, W) != (h, w):
            a, b = H // h, W // w
            out.append((f'{tname}+up{a}x{b}', lambda X, T=T, a=a, b=b: np.kron(T(X), np.ones((a, b), np.int16))))
            out.append((f'{tname}+tile{a}x{b}', lambda X, T=T, a=a, b=b: np.tile(T(X), (a, b))))
        if h % H == 0 and w % W == 0 and (H, W) != (h, w):
            a, b = h // H, w // W
            for i in range(a):
                for j in range(b):
                    out.append((f'{tname}+sub{a}x{b}@{i},{j}', lambda X, T=T, a=a, b=b, i=i, j=j: T(X)[i::a, j::b]))
                    out.append((f'{tname}+block{a}x{b}@{i},{j}',
                                lambda X, T=T, a=a, b=b, i=i, j=j: blockcrop(T(X), a, b, i, j)))
        if H <= h and W <= w and (H, W) != (h, w):
            for r0 in range(h - H + 1):
                for c0 in range(w - W + 1):
                    if cmap_fit([Z0[r0:r0 + H, c0:c0 + W]], [Y0]) is None:
                        continue
                    for anc in ('tl', 'tr', 'bl', 'br'):
                        dr = r0 if anc[0] == 't' else h - H - r0
                        dc = c0 if anc[1] == 'l' else w - W - c0
                        out.append((f'{tname}+crop{anc}{dr},{dc}',
                                    lambda X, T=T, anc=anc, dr=dr, dc=dc, H=H, W=W: anchored_crop(T(X), anc, dr, dc, H, W)))
        if (H, W) == (h, w):
            for dr, dc in bbox_shifts(Z0, Y0, bg):
                out.append((f'{tname}+shift{dr},{dc}', lambda X, T=T, dr=dr, dc=dc: shift(T(X), dr, dc)))
            for dr in range(h):
                for dc in range(w):
                    if (dr, dc) != (0, 0):
                        out.append((f'{tname}+roll{dr},{dc}', lambda X, T=T, dr=dr, dc=dc: np.roll(T(X), (dr, dc), (0, 1))))
    return out


def blockcrop(Z, a, b, i, j):
    h, w = Z.shape
    if h % a or w % b:
        return None
    H, W = h // a, w // b
    return Z[i * H:(i + 1) * H, j * W:(j + 1) * W]


def anchored_crop(Z, anc, dr, dc, H, W):
    h, w = Z.shape
    r0 = dr if anc[0] == 't' else h - H - dr
    c0 = dc if anc[1] == 'l' else w - W - dc
    if r0 < 0 or c0 < 0 or r0 + H > h or c0 + W > w:
        return None
    return Z[r0:r0 + H, c0:c0 + W]


def shift(Z, dr, dc):
    h, w = Z.shape
    out = np.full((h, w), -1, np.int16)
    rs, re = max(0, dr), min(h, h + dr)
    cs, ce = max(0, dc), min(w, w + dc)
    if rs < re and cs < ce:
        out[rs:re, cs:ce] = Z[rs - dr:re - dr, cs - dc:ce - dc]
    return out


def bbox_shifts(Z, Y, bg):
    mz, my = np.argwhere(Z != bg), np.argwhere(Y != bg)
    if len(mz) == 0 or len(my) == 0:
        return []
    cz = [(mz[:, 0].min(), mz[:, 1].min()), (mz[:, 0].max(), mz[:, 1].max()),
          (mz[:, 0].min(), mz[:, 1].max()), (mz[:, 0].max(), mz[:, 1].min())]
    cy = [(my[:, 0].min(), my[:, 1].min()), (my[:, 0].max(), my[:, 1].max()),
          (my[:, 0].min(), my[:, 1].max()), (my[:, 0].max(), my[:, 1].min())]
    s = {(int(b[0] - a[0]), int(b[1] - a[1])) for a, b in zip(cz, cy)}
    s.discard((0, 0))
    return sorted(s)


def member_fits(fn, pairs):
    Zs = []
    for X, Y in pairs:
        Z = fn(X)
        if Z is None or Z.shape != Y.shape:
            return None
        Zs.append(Z)
    return cmap_fit(Zs, [Y for _, Y in pairs])


def a2_check(train, test):
    """(oracle member names, determined, determined_relaxed) for the A2 family plus the mosaic subfamily.
    determined: every train-consistent member gives the test outputs (strict colour domain).
    determined_relaxed: the same without the cyclic rolls and with unseen colours mapped to themselves."""
    allp = train + test
    oracle, consistent = [], []
    for name, fn in a2_members(train):
        f0 = member_fits(fn, train[:1])
        if f0 is None:
            continue
        ft = member_fits(fn, train)
        if ft is None:
            continue
        consistent.append((name, fn, ft))
        if member_fits(fn, allp) is not None:
            oracle.append(name)
    m = mosaic_check(allp)
    if m:
        oracle.append(m)

    def determined(members, relaxed):
        if not members:
            return False
        for name, fn, ft in members:
            for X, Y in test:
                Z = fn(X)
                P = None if Z is None else cmap_apply(ft, Z, identity_default=relaxed)
                if P is None or P.shape != Y.shape or not (P == Y).all():
                    return False
        return True

    return oracle, determined(consistent, False), determined([m for m in consistent if 'roll' not in m[0]], True)


def mosaic_check(pairs):
    """Output = a x b mosaic of D4 images of the input (tile (i, j) uses its own D4 element), one colour map."""
    shapes = {(X.shape, Y.shape) for X, Y in pairs}
    tiles = None
    for (h, w), (H, W) in shapes:
        cand = {(a, b) for a in range(1, 6) for b in range(1, 6) if (a, b) != (1, 1) and
                ((a * h, b * w) == (H, W) or (a * w, b * h) == (H, W))}
        tiles = cand if tiles is None else tiles & cand
    for a, b in sorted(tiles or ()):
        opts = []
        for i in range(a):
            for j in range(b):
                tile_opts = []
                for tname, T in D4:
                    srcs, dsts, ok = [], [], True
                    for X, Y in pairs:
                        Z = T(X)
                        th, tw = Y.shape[0] // a, Y.shape[1] // b
                        if Z.shape != (th, tw):
                            ok = False
                            break
                        srcs.append(Z)
                        dsts.append(Y[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
                    f = cmap_fit(srcs, dsts) if ok else None
                    if f is not None:
                        tile_opts.append((tname, f))
                if not tile_opts:
                    break
                opts.append(tile_opts)
            else:
                continue
            break
        if len(opts) != a * b:
            continue
        sol = mosaic_search(opts, 0, {})
        if sol is not None:
            return f'mosaic{a}x{b}:' + '/'.join(sol)
    return None


def mosaic_search(opts, k, f):
    if k == len(opts):
        return []
    for tname, g in opts[k]:
        if all(f.get(c, v) == v for c, v in g.items()):
            rest = mosaic_search(opts, k + 1, {**f, **g})
            if rest is not None:
                return [tname] + rest
    return None


# ---------------------------------------------------------------------------- ADDR: general address map, fixed sizes
def addr_check(train, test):
    allp = train + test
    if len({X.shape for X, _ in allp}) != 1 or len({Y.shape for _, Y in allp}) != 1:
        return None
    Xs = np.stack([X.ravel() for X, _ in allp])
    Ys = np.stack([Y.ravel() for _, Y in allp])
    match = (Ys.T[:, None, :] == Xs.T[None, :, :]).all(-1)
    has_src, const = match.any(1), (Ys == Ys[:1]).all(0)
    res = {'id': bool(has_src.all()), 'id_const': bool((has_src | const).all())}
    nt = len(train)
    mt = (Ys[:nt].T[:, None, :] == Xs[:nt].T[None, :, :]).all(-1)
    ct = (Ys[:nt] == Ys[:1]).all(0)
    det = True
    for k in range(nt, len(allp)):
        for p in range(Ys.shape[1]):
            q = np.flatnonzero(mt[p])
            if len(q):
                vals = np.unique(Xs[k, q])
                if len(vals) != 1 or vals[0] != Ys[k, p]:
                    det = False
                    break
            elif not (ct[p] and Ys[0, p] == Ys[k, p]):
                det = False
                break
        if not det:
            break
    res['id_determined'] = det
    res['f'] = True if res['id'] else addr_general_f(Xs, Ys, allow_const=False)
    res['f_const'] = True if (res['id_const'] or res['f']) else addr_general_f(Xs, Ys, allow_const=True)
    return res


def addr_general_f(Xs, Ys, allow_const):
    """Exists one colour map f and one address map pi with Ys[k, p] = f(Xs[k, pi(p)]) for all k, p
    (allow_const: output cells constant over k may instead be a fixed colour). True / False / None (search cut)."""
    ui = np.unique(Xs.T, axis=0)
    uo = np.unique(Ys.T, axis=0)
    if allow_const:
        uo = uo[~(uo == uo[:, :1]).all(1)]
        if len(uo) == 0:
            return True
    n = Xs.shape[0]
    phi = -np.ones((len(uo), len(ui), 10), np.int32)
    valid = np.ones((len(uo), len(ui)), bool)
    for c in range(10):
        m = (ui == c)
        if not m.any():
            continue
        big = 99
        vmin = np.where(m[None], uo[:, None, :], big).min(-1)
        vmax = np.where(m[None], uo[:, None, :], -big).max(-1)
        has = m.any(1)[None, :].repeat(len(uo), 0)
        valid &= ~has | (vmin == vmax)
        phi[..., c] = np.where(has, vmin, -1)
    owners, rows = [], []
    for o in range(len(uo)):
        r = np.flatnonzero(valid[o])
        if len(r) == 0:
            return False
        owners.append(np.full(len(r), o))
        rows.append(phi[o, r])
    C = np.concatenate(rows)
    starts = np.concatenate([[0], np.cumsum([len(r) for r in rows])[:-1]])
    nodes = [0]

    def propagate(f):
        """Unit propagation: an output class whose compatible candidates all force f(c) = v fixes it.
        Returns the extended f or None on a dead end."""
        f = f.copy()
        while True:
            comp = ((C == -1) | (f[None] == -1) | (C == f[None])).all(1)
            if not np.maximum.reduceat(comp.astype(np.int8), starts).all():
                return None
            lo = np.minimum.reduceat(np.where(comp[:, None], C, 99), starts, axis=0)
            hi = np.maximum.reduceat(np.where(comp[:, None], C, -99), starts, axis=0)
            forced = (lo == hi) & (lo >= 0)
            new = False
            for o, c in zip(*np.nonzero(forced)):
                v = lo[o, c]
                if f[c] == -1:
                    f[c] = v
                    new = True
                elif f[c] != v:
                    return None
            if not new:
                return f

    def dfs(f):
        nodes[0] += 1
        if nodes[0] > CSP_NODES:
            raise TimeoutError
        f = propagate(f)
        if f is None:
            return False
        comp = ((C == -1) | (f[None] == -1) | (C == f[None])).all(1)
        open_c = [c for c in range(10) if f[c] == -1 and (C[comp, c] >= 0).any()]
        if not open_c:
            return True
        c = open_c[0]
        for v in sorted({int(v) for v in C[comp, c] if v >= 0}, key=lambda v: (v != c, v)):
            g = f.copy()
            g[c] = v
            if dfs(g):
                return True
        return False

    try:
        return bool(dfs(-np.ones(10, np.int32)))
    except TimeoutError:
        return None


# ---------------------------------------------------------------------------- A3: multi-source cellwise
def a3_overlay(pairs):
    """Output = F(part1, part2) cellwise for two halves of the input (separator line allowed), F one lookup."""
    found = []
    for axis in (0, 1):
        for flip in ('none', 'flip'):
            srcs, dsts, ok = [], [], True
            for X, Y in pairs:
                n_in, n_out = X.shape[axis], Y.shape[axis]
                if X.shape[1 - axis] != Y.shape[1 - axis] or n_in not in (2 * n_out, 2 * n_out + 1):
                    ok = False
                    break
                p1 = X[:n_out] if axis == 0 else X[:, :n_out]
                p2 = X[n_in - n_out:] if axis == 0 else X[:, n_in - n_out:]
                if flip == 'flip':
                    p2 = p2[::-1] if axis == 0 else p2[:, ::-1]
                srcs.append(p1 * 10 + p2)
                dsts.append(Y)
            if ok and cmap_fit(srcs, dsts) is not None:
                found.append(f'overlay_axis{axis}_{flip}')
    return found


def a3_kron(pairs, bg):
    found = []
    for name, fn in (('kron_mask_x', lambda X: np.kron((X != bg).astype(np.int16), X + 1) - 1),
                     ('kron_invmask_x', lambda X: np.kron((X == bg).astype(np.int16), X + 1) - 1),
                     ('kron_x_mask', lambda X: np.kron(X + 1, (X != bg).astype(np.int16)) - 1)):
        if member_fits(fn, pairs) is not None:
            found.append(name)
    return found


# ---------------------------------------------------------------------------- B: local rules
def neigh_codes(X):
    h, w = X.shape
    P = np.full((h + 2, w + 2), -1, np.int64)
    P[1:-1, 1:-1] = X
    code = np.zeros((h, w), np.int64)
    for dy in range(3):
        for dx in range(3):
            code = code * 11 + (P[dy:dy + h, dx:dx + w] + 1)
    return code


def b1_lookup(train, test):
    if any(X.shape != Y.shape for X, Y in train + test):
        return None
    codes = lambda ps: (np.concatenate([neigh_codes(X).ravel() for X, _ in ps]), np.concatenate([Y.ravel() for _, Y in ps]))
    c, y = codes(train + test)
    u = np.unique(c * 16 + y)
    consistent = len(np.unique(u // 16)) == len(u)
    ct, yt = codes(train)
    ut = np.unique(ct * 16 + yt)
    table = dict(zip((ut // 16).tolist(), (ut % 16).tolist())) if len(np.unique(ut // 16)) == len(ut) else None
    det = False
    if table is not None:
        cs, ys = codes(test)
        det = all(table.get(k, -1) == v for k, v in zip(cs.tolist(), ys.tolist()))
    return {'consistent': bool(consistent), 'determined': bool(det)}


def border_bg(X, bg):
    h, w = X.shape
    seen = np.zeros((h, w), bool)
    st = [(r, c) for r in range(h) for c in (0, w - 1)] + [(r, c) for c in range(w) for r in (0, h - 1)]
    st = [(r, c) for r, c in st if X[r, c] == bg]
    for r, c in st:
        seen[r, c] = True
    while st:
        r, c = st.pop()
        for dr, dc in NB4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and X[rr, cc] == bg:
                seen[rr, cc] = True
                st.append((rr, cc))
    return seen


def b2_checks(pairs, bg, change):
    out = {}
    if change != ('add',):
        return out
    sub, full, rays, thin, longest = True, True, True, True, 0
    for X, Y in pairs:
        added = (X == bg) & (Y != bg)
        enclosed = (X == bg) & ~border_bg(X, bg)
        sub &= bool((added <= enclosed).all())
        full &= bool((added == enclosed).all())
        thin &= not (added[:-1, :-1] & added[1:, :-1] & added[:-1, 1:] & added[1:, 1:]).any()
        longest = max(longest, longest_run(added))
        if rays:
            h, w = X.shape
            for r, c in np.argwhere(added):
                hit = False
                for dr, dc in NB8:
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < h and 0 <= cc < w and added[rr, cc]:
                        rr, cc = rr - dr, cc - dc
                    if 0 <= rr < h and 0 <= cc < w and X[rr, cc] != bg:
                        hit = True
                        break
                if not hit:
                    rays = False
                    break
    out['fill_enclosed_subset'] = sub
    out['fill_enclosed_all'] = full
    out['rays'] = rays and thin and longest >= 3 and not sub
    return out


def longest_run(mask):
    """Longest straight run (row, column or diagonal) of True cells."""
    h, w = mask.shape
    best = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r, c in np.argwhere(mask):
            pr, pc = r - dr, c - dc
            if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                continue
            n, rr, cc = 0, r, c
            while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                n, rr, cc = n + 1, rr + dr, cc + dc
            best = max(best, n)
    return best


def gravity(X, bg, d):
    Z = np.rot90(X, d).copy()
    for j in range(Z.shape[1]):
        col = Z[:, j]
        nz = col[col != bg]
        Z[:, j] = bg
        if len(nz):
            Z[-len(nz):, j] = nz
    return np.rot90(Z, -d)


# ---------------------------------------------------------------------------- C, D: objects
def components(X, bg, same_colour=True):
    h, w = X.shape
    a = X.tolist()
    lab = [[-1] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if a[r][c] == bg or lab[r][c] >= 0:
                continue
            col, k, st, cells = a[r][c], len(comps), [(r, c)], []
            lab[r][c] = k
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in NB8:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and lab[yy][xx] < 0 and a[yy][xx] != bg and \
                            (not same_colour or a[yy][xx] == col):
                        lab[yy][xx] = k
                        st.append((yy, xx))
            comps.append(cells)
    return comps


def shape_key(cells, colour, d4=False):
    pts = np.array(cells)
    keys = []
    for k in (range(8) if d4 else range(1)):
        p = pts.copy()
        if k & 1:
            p = p[:, ::-1]
        if k & 2:
            p[:, 0] = -p[:, 0]
        if k & 4:
            p[:, 1] = -p[:, 1]
        p = p - p.min(0)
        keys.append(tuple(sorted(map(tuple, p.tolist()))))
    return (colour, min(keys))


def c_checks(pairs, bg, change, same):
    out = {}
    if same and change == ('recolor',):
        ok, moved = True, False
        for X, Y in pairs:
            for cells in components(X, bg):
                vals = {int(Y[r, c]) for r, c in cells}
                if len(vals) != 1:
                    ok = False
                    break
            if not ok:
                break
        out['obj_recolor'] = ok
    if same and change == ('remove',):
        ok = True
        for X, Y in pairs:
            for cells in components(X, bg):
                kept = [Y[r, c] != bg for r, c in cells]
                if any(kept) and not all(kept):
                    ok = False
                    break
            if not ok:
                break
        out['obj_filter'] = ok
    if same and change and change != ('add',):
        for d4 in (False, True):
            ok, moved = True, False
            for X, Y in pairs:
                kx = collections.Counter(shape_key(cs, int(X[cs[0]]), d4) for cs in components(X, bg))
                ky = collections.Counter(shape_key(cs, int(Y[cs[0]]), d4) for cs in components(Y, bg))
                if kx != ky:
                    ok = False
                    break
                moved |= bool((X != Y).any())
            out['obj_rearrange' + ('_d4' if d4 else '')] = ok and moved
    return out


def d_checks(pairs, bg):
    out = {'select_object': True, 'crop_anywhere': True}
    for X, Y in pairs:
        H, W = Y.shape
        if H > X.shape[0] or W > X.shape[1]:
            out['select_object'] = out['crop_anywhere'] = False
            break
        if out['select_object']:
            hit = False
            for same_colour in (True, False):
                for cells in components(X, bg, same_colour):
                    pts = np.array(cells)
                    r0, c0 = pts.min(0)
                    r1, c1 = pts.max(0)
                    if (r1 - r0 + 1, c1 - c0 + 1) == (H, W) and (X[r0:r1 + 1, c0:c1 + 1] == Y).all():
                        hit = True
                        break
                if hit:
                    break
            out['select_object'] = hit
        if out['crop_anywhere']:
            win = np.lib.stride_tricks.sliding_window_view(X, (H, W))
            out['crop_anywhere'] = bool((win == Y).all((-1, -2)).any())
    out['tiny_output'] = all(Y.size < X.size and (Y.size <= 9 or min(Y.shape) == 1) for X, Y in pairs)
    return out


# ---------------------------------------------------------------------------- E: regularity
def e_checks(pairs, same):
    """Completion classes: the output has a regularity the input breaks, and most cells are kept
    (mean agreement >= 0.5, output not a single colour)."""
    out = {}
    if not same:
        return out
    agree = np.mean([(X == Y).mean() for X, Y in pairs])
    varied = all(len(np.unique(Y)) > 1 for _, Y in pairs)
    if agree < 0.5 or not varied:
        out['symmetry_completion'], out['periodic_output'] = [], False
        return out
    sym = []
    for name, T in D4[1:]:
        ok_y, broken_x = True, False
        for X, Y in pairs:
            TY = T(Y)
            if TY.shape != Y.shape or not (TY == Y).all():
                ok_y = False
                break
            broken_x |= not (T(X) == X).all()
        if ok_y and broken_x:
            sym.append(name)
    out['symmetry_completion'] = sym
    per, broken = True, False
    for X, Y in pairs:
        h, w = Y.shape
        pr = next(p for p in range(1, h + 1) if p == h or (Y[p:] == Y[:-p]).all())
        pc = next(p for p in range(1, w + 1) if p == w or (Y[:, p:] == Y[:, :-p]).all())
        if not ((pr < h and pr <= h // 2) or (pc < w and pc <= w // 2)):
            per = False
            break
        broken |= (pr < h and not (X[pr:] == X[:-pr]).all()) or (pc < w and not (X[:, pc:] == X[:, :-pc]).all())
    out['periodic_output'] = per and broken
    return out


# ---------------------------------------------------------------------------- per task
def background(pairs):
    v = np.concatenate([X.ravel() for X, _ in pairs])
    return int(np.bincount(v, minlength=10).argmax())


def size_relation(pairs):
    rel = set()
    for X, Y in pairs:
        (h, w), (H, W) = X.shape, Y.shape
        if (H, W) == (h, w):
            rel.add('same')
        elif (H, W) == (w, h):
            rel.add('transpose')
        elif H % h == 0 and W % w == 0:
            rel.add(f'up{H // h}x{W // w}')
        elif h % H == 0 and w % W == 0:
            rel.add(f'down{h // H}x{w // W}')
        elif H <= h and W <= w:
            rel.add('smaller')
        elif H >= h and W >= w:
            rel.add('larger')
        else:
            rel.add('mixed')
    if len({Y.shape for _, Y in pairs}) == 1 and rel != {'same'}:
        return 'const_out' if len(rel) > 1 else rel.pop()
    return rel.pop() if len(rel) == 1 else 'varies'


def change_type(pairs, bg):
    kinds = set()
    for X, Y in pairs:
        if X.shape != Y.shape:
            return None
        if ((X == bg) & (Y != bg)).any():
            kinds.add('add')
        if ((X != bg) & (Y == bg)).any():
            kinds.add('remove')
        if ((X != bg) & (Y != bg) & (X != Y)).any():
            kinds.add('recolor')
    return tuple(sorted(kinds))


def audit_task(tid, train, test):
    allp = train + test
    bg = background(allp)
    same = all(X.shape == Y.shape for X, Y in allp)
    change = change_type(allp, bg) if same else None
    rec = {
        'id': tid, 'K': len(train), 'n_test': len(test),
        'max_side': int(max(max(a.shape) for p in allp for a in p)),
        'n_colours': int(len(np.unique(np.concatenate([a.ravel() for p in allp for a in p])))),
        'bg': bg, 'bg0_every_grid': all(int(np.bincount(a.ravel(), minlength=10).argmax()) == 0 for p in allp for a in p),
        'same_shape': same,
        'inputs_one_shape': len({X.shape for X, _ in allp}) == 1,
        'outputs_one_shape': len({Y.shape for _, Y in allp}) == 1,
        'size_rel': size_relation(allp),
        'change': '+'.join(change) if change is not None else None,
        'occupancy_kept': all(int((X != bg).sum()) == int((Y != bg).sum()) for X, Y in allp),
    }
    a1 = same and cmap_fit([X for X, _ in allp], [Y for _, Y in allp]) is not None
    rec['A1'] = bool(a1)
    if same:
        ft = cmap_fit([X for X, _ in train], [Y for _, Y in train])
        rec['A1_determined'] = bool(ft is not None and all(
            (lambda P: P is not None and (P == Y).all())(cmap_apply(ft, X)) for X, Y in test))
    else:
        rec['A1_determined'] = False
    a2, a2_det, a2_det_relaxed = a2_check(train, test)
    rec['A2'] = sorted(set(a2))
    rec['A2_determined'] = bool(a2_det)
    rec['A2_determined_relaxed'] = bool(a2_det_relaxed)
    rec['changed_frac'] = float(np.mean([(X != Y).mean() for X, Y in allp])) if same else None
    rec['ADDR'] = addr_check(train, test)
    rec['A3'] = a3_overlay(allp) + a3_kron(allp, bg)
    rec['B1'] = b1_lookup(train, test)
    rec.update(b2_checks(allp, bg, change))
    rec['gravity'] = same and any(all((gravity(X, bg, d) == Y).all() for X, Y in allp) for d in range(4)) \
        and any((X != Y).any() for X, Y in allp)
    rec.update(c_checks(allp, bg, change, same))
    rec.update(d_checks(allp, bg))
    rec.update(e_checks(allp, same))
    return rec


# ---------------------------------------------------------------------------- summary
def classes_of(r):
    """Which document classes the detectors put the task in (oracle sense)."""
    c = set()
    if r['A1']:
        c.add('A1')
    if r['A2']:
        c.add('A2')
        if any('roll' in n for n in r['A2']) and not any('roll' not in n for n in r['A2']) and not r['A1']:
            c.add('A2_roll_only')
    ad = r['ADDR'] or {}
    if ad.get('f') or ad.get('id'):
        c.add('ADDR')
    if ad.get('f_const') or ad.get('id_const'):
        c.add('ADDR_const')
    if r['A3']:
        c.add('A3')
    if r['B1'] and r['B1']['determined']:
        c.add('B1_det')
    if r['B1'] and r['B1']['consistent']:
        c.add('B1_cons')
    for k in ('fill_enclosed_subset', 'rays', 'gravity'):
        if r.get(k):
            c.add('B2')
    for k in ('obj_recolor', 'obj_filter', 'obj_rearrange', 'obj_rearrange_d4'):
        if r.get(k):
            c.add('C')
    if r.get('select_object') or r.get('crop_anywhere'):
        c.add('D_crop')
    if r.get('tiny_output'):
        c.add('D_tiny')
    if r.get('symmetry_completion') or r.get('periodic_output'):
        c.add('E')
    return c


def summarise(recs):
    n = len(recs)
    pct = lambda k: round(100.0 * k / n, 1)
    cls = [classes_of(r) for r in recs]
    cnt = collections.Counter(x for c in cls for x in c)
    rank1 = sum(1 for r, c in zip(recs, cls) if c & {'A1', 'A2', 'ADDR'})
    rank1_const = sum(1 for r, c in zip(recs, cls) if c & {'A1', 'A2', 'ADDR', 'ADDR_const'})
    level_a = sum(1 for c in cls if c & {'A1', 'A2', 'ADDR', 'ADDR_const', 'A3'})
    any_det = sum(1 for c in cls if c - {'B1_cons', 'D_tiny'})
    s = {
        'tasks': n,
        'K_hist': dict(sorted(collections.Counter(r['K'] for r in recs).items())),
        'n_test_hist': dict(sorted(collections.Counter(r['n_test'] for r in recs).items())),
        'max_side_le4': pct(sum(r['max_side'] <= 4 for r in recs)),
        'max_side_ge20': pct(sum(r['max_side'] >= 20 for r in recs)),
        'same_shape': pct(sum(r['same_shape'] for r in recs)),
        'inputs_one_shape': pct(sum(r['inputs_one_shape'] for r in recs)),
        'fixed_size_task': pct(sum(r['inputs_one_shape'] and r['outputs_one_shape'] for r in recs)),
        'bg0_every_grid': pct(sum(r['bg0_every_grid'] for r in recs)),
        'occupancy_kept': pct(sum(r['occupancy_kept'] for r in recs)),
        'size_rel': {k: pct(v) for k, v in collections.Counter(
            'same' if r['size_rel'] == 'same' else 'up' if r['size_rel'].startswith('up') else
            'down' if r['size_rel'].startswith('down') else r['size_rel'] for r in recs).most_common()},
        'change_same_shape': {k: pct(v) for k, v in collections.Counter(r['change'] for r in recs if r['same_shape']).most_common()},
        'class_pct': {k: pct(v) for k, v in sorted(cnt.items())},
        'A1_determined': pct(sum(r['A1_determined'] for r in recs)),
        'A2_determined': pct(sum(r['A2_determined'] for r in recs)),
        'A2_determined_relaxed': pct(sum(r['A2_determined_relaxed'] for r in recs)),
        'A2_oracle_noroll': pct(sum(any('roll' not in n for n in r['A2']) for r in recs)),
        'changed_frac_median_same_shape': round(float(np.median([r['changed_frac'] for r in recs if r['same_shape']] or [0])), 3),
        'n_colours_median': float(np.median([r['n_colours'] for r in recs])),
        'ADDR_id_determined': pct(sum(bool(r['ADDR'] and r['ADDR']['id_determined']) for r in recs)),
        'ADDR_csp_cut': sum(1 for r in recs if r['ADDR'] and (r['ADDR']['f'] is None or r['ADDR']['f_const'] is None)),
        'rank1_CXW': pct(rank1),
        'rank1_CXW_affine': pct(rank1_const),
        'rank1_needs_size_frames': pct(sum(1 for r, c in zip(recs, cls) if (c & {'A1', 'A2', 'ADDR'}) and not r['inputs_one_shape'])),
        'level_A': pct(level_a),
        'any_detector': pct(any_det),
    }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arc1')
    ap.add_argument('arc2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default=None, help='comma-separated task ids (debug)')
    args = ap.parse_args()
    sets = load_sets(args.arc1, args.arc2)
    arc1_ids = {tid for name in ('arc1_train', 'arc1_eval') for tid, _, _ in sets[name]}
    only = set(args.only.split(',')) if args.only else None
    result = {'sources': {'arc1': args.arc1, 'arc2': args.arc2}, 'sets': {}, 'tasks': []}
    for name, tasks in sets.items():
        t0 = time.time()
        recs = []
        for tid, train, test in tasks:
            if only and tid not in only:
                continue
            r = audit_task(tid, train, test)
            r['set'] = name
            r['in_arc1'] = tid in arc1_ids
            r['classes'] = sorted(classes_of(r))
            recs.append(r)
        if not recs:
            continue
        result['tasks'] += recs
        result['sets'][name] = summarise(recs)
        new = [r for r in recs if not r['in_arc1']]
        if name == 'arc2_train' and new:
            result['sets']['arc2_train_new'] = summarise(new)
        print(f'{name}: {len(recs)} tasks, {time.time() - t0:.0f}s', file=sys.stderr)
    json.dump(result, open(args.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    for name, s in result['sets'].items():
        print(name, json.dumps({k: v for k, v in s.items() if k not in ('K_hist', 'n_test_hist')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
