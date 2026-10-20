"""FP64 checks of the identities in docs/claude_2026-10-20_rule_synapse_theory.md (no training, CPU, seconds).

    python -m toy_g.theory_checks
"""
import numpy as np

from . import group as G
from .data import random_inputs

C = G.COLORS


def perm_op(t):
    """Cell-basis operator of the address map: P e_o = e_{SRC[t, o]}."""
    P = np.zeros((G.CELLS, G.CELLS))
    P[G.SRC[t], np.arange(G.CELLS)] = 1
    return P


def onehot(grid):
    """[16] colours 0..4 -> [16, 4] indicators of colours 1..4 (background row = 0)."""
    return (grid.reshape(-1)[:, None] == np.arange(1, C + 1)[None, :]).astype(float)


def colour_pair_cells(I, O):
    """Cell-basis colour-pair synapses R[a, b][x, o] = 1[I(x) = a, O(o) = b] and grams A[a, c][x, y] = 1[I(x) = a, I(y) = c]."""
    X, Y = onehot(I), onehot(O)
    return np.einsum('xa,ob->abxo', X, Y), np.einsum('xa,yc->acxy', X, X)


def to_codes(M, Q):
    """Cell-basis operator -> code basis, codes g(x) = Q^T e_x (Q: 16 x d with orthonormal columns)."""
    return Q.T @ M @ Q


def main():
    rng = np.random.default_rng(0)
    worst = {}
    Q_orth = np.linalg.qr(rng.standard_normal((G.CELLS, G.CELLS)))[0]           # generic orthonormal codes, d = 16
    ones = np.ones((G.CELLS, 1)) / 4
    Q_zero = np.linalg.qr(np.hstack([ones, rng.standard_normal((G.CELLS, G.CELLS - 1))]))[0][:, 1:]  # zero-mean, d = 15
    for trial in range(200):
        I = random_inputs(rng, 1)[0]
        t, c = int(rng.integers(G.NUM_T)), int(rng.integers(G.NUM_C))
        O = G.apply_task(I, t, c)
        kappa = G.CMAPS[c]                                                    # kappa[a] = b
        Rab, Aac = colour_pair_cells(I, O)
        P = perm_op(t)
        R = sum(Rab[a - 1, kappa[a] - 1] for a in range(1, C + 1))             # matched-colour synapse (kappa known)
        A = sum(Aac[a - 1, a - 1] for a in range(1, C + 1))                    # same-colour gram of the input
        for name, Q in (('orth16', Q_orth), ('zero15', Q_zero)):
            # Theorem 1: R = A P (code basis), and per colour pair R[a, b] = A[a, kappa^-1(b)] P
            e1 = np.abs(to_codes(R, Q) - to_codes(A, Q) @ to_codes(P, Q)).max()
            inv = np.argsort(kappa)
            e2 = max(np.abs(to_codes(Rab[a, b], Q) - to_codes(Aac[a, inv[b + 1] - 1], Q) @ to_codes(P, Q)).max()
                     for a in range(C) for b in range(C))
            # Corollary: <R, P(t')> = match count (orth16), up to a t'-independent constant (zero15)
            m = np.array([sum(1 for o in range(G.CELLS) if O.flat[o] and I.flat[G.SRC[tp, o]]
                              and O.flat[o] == kappa[I.flat[G.SRC[tp, o]]]) for tp in range(G.NUM_T)])
            s = np.array([np.sum(to_codes(R, Q) * to_codes(perm_op(tp), Q)) for tp in range(G.NUM_T)])
            e3 = np.ptp(s - m)
            for k, v in (('factor', e1), ('factor_pairs', e2), ('score_minus_count_spread', e3)):
                worst[(name, k)] = max(worst.get((name, k), 0.0), v)
    for k, v in sorted(worst.items()):
        print(f'{k[0]:7s} {k[1]:26s} max |err| {v:.2e}')

    # Theorem 1 for arbitrary address maps: non-injective r on the same grid, and a 2x3 output reading a 4x4 input
    err = 0.0
    for trial in range(500):
        I = random_inputs(rng, 1)[0].reshape(-1)
        n_out = G.CELLS if trial % 2 == 0 else 6
        r = rng.integers(G.CELLS, size=n_out)                                  # any function, repeats allowed
        kappa = G.CMAPS[int(rng.integers(G.NUM_C))]
        O = kappa[I[r]]
        X = onehot(I)
        R = np.einsum('xa,oa->xo', X, onehot(O) @ np.eye(C)[kappa[1:] - 1].T)     # 1[I(x) = a, O(o) = kappa(a)]
        A = X @ X.T
        P = np.zeros((G.CELLS, n_out))
        P[r, np.arange(n_out)] = 1                                             # P = sum_o e_{r(o)} e_o^T
        err = max(err, np.abs(R - A @ P).max())
    print(f'arbitrary address maps (non-injective, 16 -> 6 cells): max |R - A P| {err:.2e}')

    # Proposition 2: E[A] = (alpha - beta) I + beta J for the toy's input distribution
    s = np.arange(3, 9)
    alpha, beta = s.mean() / G.CELLS, (s * (s - 1)).mean() / (G.CELLS * (G.CELLS - 1)) / C
    expect = (alpha - beta) * np.eye(G.CELLS) + beta * np.ones((G.CELLS, G.CELLS))
    for n in (10_000, 100_000, 1_000_000):
        X = (random_inputs(rng, n).reshape(n, -1)[:, :, None] == np.arange(1, C + 1)).astype(float)
        EA = np.einsum('nxa,nya->xy', X, X) / n
        print(f'E[A] Monte Carlo n={n:>9,}: max |mean - ((a-b) I + b J)| = {np.abs(EA - expect).max():.2e} '
              f'(alpha {alpha:.5f}, beta {beta:.5f})')

    # Proposition 3: monomial structure in the Fourier basis, translations diagonal, D4 orbits on frequencies
    xy = np.array([(r, cc) for r in range(G.N) for cc in range(G.N)])
    F = np.exp(2j * np.pi * (xy @ xy.T) / G.N) / G.N                          # F[x, k], unitary
    maps, diag_ok, mono_err = {}, True, 0.0
    for t in range(G.NUM_T):
        PF = F.conj().T @ perm_op(t) @ F
        mag = np.abs(PF)
        mono_err = max(mono_err, np.abs(np.sort(mag, axis=0)[-1] - 1).max(), np.sort(mag, axis=0)[-2].max())
        k_of = mag.argmax(axis=0)                                            # column k' -> row k
        p = G.decode_t(t)[0]
        maps.setdefault(p, set()).add(tuple(k_of))
        if p == 0:
            diag_ok &= bool((k_of == np.arange(G.CELLS)).all())
    print(f'Fourier: max deviation from a monomial matrix {mono_err:.2e}; translations diagonal: {diag_ok}; '
          f'frequency map depends only on the point operation: {all(len(v) == 1 for v in maps.values())}')
    perms = [np.array(next(iter(v))) for v in maps.values()]
    seen, orbits = set(), []
    for k in range(G.CELLS):
        if k in seen:
            continue
        orb = {int(pm[k]) for pm in perms}
        seen |= orb
        orbits.append(sorted(orb))
    print('D4 orbits on frequencies (k = 4 k1 + k2):', [[tuple(xy[i]) for i in o] for o in orbits])

    # Involutions and word lengths
    invol = [t for t in range(G.NUM_T) if (G.SRC[t][G.SRC[t]] == np.arange(G.CELLS)).all()]
    lengths = np.array([len(G.word(G.inverse(t))) for t in range(G.NUM_T)])
    print(f'involutions (r o r = id): {len(invol)} of {G.NUM_T} (identity included)')
    fix = np.array([(G.SRC[t] == np.arange(G.CELLS)).sum() for t in range(G.NUM_T)])
    print(f'fixed cells of the non-identity elements: max {fix[np.arange(G.NUM_T) != G.IDENTITY].max()}, '
          f'histogram {np.bincount(fix).tolist()}')
    print(f'word lengths of word(inverse(t)): max {lengths.max()}, mean {lengths.mean():.2f}, '
          f'histogram {np.bincount(lengths).tolist()}')


if __name__ == '__main__':
    main()
