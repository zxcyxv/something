"""Hand-set (untrained) check of the rule-synapse architecture of docs/claude_2026-10-20_rule_synapse_architecture.md
on the fixed toy instances (3 demonstrations, in context, no task identifier).

    p(x)   = e(colour) (x) g(x)                 conjunctive code, e(background) = 0, g = cell one-hot (orthonormal)
    V_k    = sum_x p(I_k(x), x),  U_k = sum_o p(O_k(o), o)      grid summaries (lag-0 read of each grid)
    W      = token-level bands window over two blocks: block 0 state p(o), block 1 state V_k (demo output tokens)
    score  = <W, T_(r,kappa)^T> = sum_k <U_k, T V_k>,  T = K_kappa (x) P(r)^T     (Hopfield over 128 x 24 operators)
    answer = argmax-score operator applied to the query input

Window variants (two real poles z = +-0.5): causal (l(0) = 0, l(1) = 1, l(-1) = 0), odd (l(-1) = -1), even (l(-1) = +1),
causal + l(0) = 1.

    python -m toy_g.design_check
"""
import numpy as np

from . import group as G

C = G.COLORS


def summaries(grids):
    """[..., 4, 4] colours 0..4 -> [..., 4 (colour 1..4), 16 (cell)] conjunctive summaries sum_x e(c) (x) e_x."""
    flat = grids.reshape(*grids.shape[:-2], G.CELLS)
    return (flat[..., None, :] == np.arange(1, C + 1)[:, None]).astype(float)


def bands_window(z, a, b, k_states, v_states):
    """Literal bands recursion over the blocks (lt/model.py): returns sum_blocks S, with S = sum_r Re(a_r v E_K,r^T +
    b_r D_V,r k^T). k_states, v_states: [blocks, tokens, dim] (real); z, a, b: [R] complex; per-token traces."""
    gamma = np.sqrt(1 - np.abs(z) ** 2)
    n_tok, dim = k_states.shape[1:]
    EK = np.zeros((len(z), n_tok, dim), complex)
    EV = np.zeros((len(z), n_tok, dim), complex)
    W = np.zeros((dim, dim))
    for k, v in zip(k_states, v_states):
        EK = z[:, None, None] * EK + gamma[:, None, None] * k
        DV = z[:, None, None] * EV
        EV = DV + gamma[:, None, None] * v
        W += np.einsum('r,tv,rtk->vk', a, v, EK).real + np.einsum('r,rtv,tk->vk', b, DV, k).real
    return W


def main():
    inst = np.load(G.__file__.replace('toy_g/group.py', 'data/gtorus_instances.npz'))
    din, dout, tin, tout, group = inst['demo_in'], inst['demo_out'], inst['test_in'], inst['test_out'], inst['group']
    n = len(group)
    V, U = summaries(din), summaries(dout)                                        # [n, 3, 4, 16]
    kap = G.CMAPS[:, 1:] - 1                                                      # [24, 4]: kappa_c(a) - 1
    invol_r = np.array([(G.SRC[t][G.SRC[t]] == np.arange(G.CELLS)).all() for t in range(G.NUM_T)])
    invol_c = np.array([(G.CMAPS[c][G.CMAPS[c]] == np.arange(C + 1)).all() for c in range(G.NUM_C)])
    t_true, c_true = G.split_task(inst['task'])
    invol_task = invol_r[t_true] & invol_c[c_true]
    preds = G.CMAPS[np.arange(G.NUM_C)[None, :, None], tin.reshape(n, G.CELLS)[:, G.SRC][:, :, None, :]]  # [n, 128, 24, 16]
    correct_grid = (preds == tout.reshape(n, 1, 1, G.CELLS)).all(-1)              # [n, 128, 24] hypothesis -> exact

    def evaluate(Wcp, name):
        """Wcp [n, 4, 16, 4, 16] = colour-pair synapses R_ab[x, o]; scores over all (r, kappa)."""
        S = Wcp[:, :, :, :, :][:, :, G.SRC, :, :]                                # gather rows x = SRC[r, o]
        # S[n, a, r, o, b, o'] -> take o = o' diagonal
        S = np.einsum('narobo->nabr', S)                                         # [n, 4, 4, 128] = <R_ab, P(r)>
        score = sum(S[:, a, kap[:, a], :] for a in range(C))                     # [n, 24, 128]
        score = score.transpose(0, 2, 1)                                         # [n, 128, 24]
        best = score.reshape(n, -1).max(1)
        top = np.isclose(score, best[:, None, None])
        exact = (correct_grid & top).sum((1, 2)) == top.sum((1, 2))              # every top-scoring operator is right
        line = ' | '.join(f'g{g} {100 * exact[group == g].mean():5.1f}%' for g in range(3))
        print(f'{name:34s} {line} | involutive tasks {100 * exact[invol_task].mean():5.1f}% '
              f'({invol_task.sum()}) | others {100 * exact[~invol_task].mean():5.1f}%')

    # Direct synapse W = sum_k V_k U_k^T (grid level)
    evaluate(np.einsum('nkax,nkbo->naxbo', V, U), 'grid-level W = sum V U^T')

    # Token-level bands window: demo output tokens, block 0 state p(o), block 1 state V_k
    z = np.array([0.5, -0.5], complex)
    gam = np.sqrt(1 - np.abs(z) ** 2)
    a_causal = np.array([1, -1]) / gam                                            # l(0) = 0, l(1) = 1
    variants = {'bands causal (l0=0, l-1=0)': (a_causal, np.zeros(2)),
                'bands odd STDP (l-1=-1)': (a_causal, -a_causal),
                'bands even (l-1=+1)': (a_causal, a_causal),
                'bands causal + l(0)=1': (np.array([1.5, -0.5]) / gam, np.zeros(2))}
    for name, (a, b) in variants.items():
        l = lambda D: (gam * a * z ** D).real.sum() if D >= 0 else (gam * b * z ** (-D)).real.sum()
        Wcp = np.zeros((n, C, G.CELLS, C, G.CELLS))
        for i in range(n):
            # token o of demo k: block-0 state = e(O_k(o)) (x) e_o, block-1 state = V_k (flattened 4 x 16)
            blk0, blk1 = [], []
            for k in range(3):
                col = dout[i, k].reshape(-1)
                own = np.zeros((G.CELLS, C, G.CELLS))
                m = col > 0
                own[np.arange(G.CELLS)[m], col[m] - 1, np.arange(G.CELLS)[m]] = 1
                blk0.append(own.reshape(G.CELLS, -1))
                blk1.append(np.repeat(V[i, k].reshape(1, -1), G.CELLS, 0))
            states = np.stack([np.concatenate(blk0), np.concatenate(blk1)])          # [2 blocks, 48 tokens, 64]
            W = bands_window(z, a.astype(complex), b.astype(complex), states, states)  # value rows, key columns
            Wcp[i] = W.reshape(C, G.CELLS, C, G.CELLS)
        evaluate(Wcp, f'{name} [l(-1),l(0),l(1)]=[{l(-1):+.0f},{l(0):+.0f},{l(1):+.0f}]')


if __name__ == '__main__':
    main()
