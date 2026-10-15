"""Logical depth of every Sudoku cell under synchronous constraint propagation (used by interventions.py).

State = the candidate set of every cell (givens: singletons; blanks: digits absent from the 20 neighbours' givens).
L0 round: one synchronous pass of naked singles and hidden singles, then fixed digits are removed from neighbours.
L1 round: in addition every candidate of the round's starting state is assumed and singles are propagated to closure;
a contradiction removes it. depth(t) = the round at which cell t becomes a singleton; givens 0; never resolved -1.
Run A: the model's decision block is 1.06 blocks per L0 round (Spearman 0.64 on solved puzzles).
"""
import numpy as np

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

