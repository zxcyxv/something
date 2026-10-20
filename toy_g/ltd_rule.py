"""Rule synapse written in context by error-driven LTP/LTD with a sparsity LTD, in a learned structure-code basis
(docs/claude_2026-10-20_rule_synapse_theory.md: theorem 1 R = A P(r), Yule-Walker P(r) = A^-1 R, section 2.6 completion
of the unobserved cells; the LTD discussion of 2026-10-20). 4x4 torus toy, same data and split as version C.

Per task (fast, in context; nothing per task is a trained parameter):
    codes        g(x) = row x of Q (16 x 16 unitary)
    images       u_a = sum_{x: colour a} g(x) for every demonstration input (a = 1..4), v_b likewise for outputs
    synapses     W (16 x 16 complex, positions) from 0; C (4 x 4, colours, columns sum to 1, >= 0) from uniform
    T steps      prediction  v^_b = sum_a C[b, a] W u_a;  error e_b = v_b - v^_b  (the third factor)
                 W += eta_w * sum_{k,a} (sum_b C[b, a] e_b) u_a^H, then complex soft threshold theta_w (sparsity LTD)
                 C += eta_c * sum_k Re<e_b, W u_a>, then >= 0, soft threshold theta_c, columns normalised
                 eta_w = s_w / lambda_max(sum u u^H), eta_c = s_c / sum ||W u||^2   (per task, from the data)
    read         query images through C W; colour score of cell o = Re<g(o), v^_b>; background if all scores are low
Slow (trained across tasks): Q = Q0 matrix_exp(A - A^H) (Q0 Haar random, A from 0) and six scalars (s_w, theta_w, s_c,
theta_c, read scale, background level). Q fixed to the Fourier basis, the identity or a random unitary gives the
reference conditions (hand-set, labelled as such).

    python -m toy_g.ltd_rule check                     # no learning: fixed Q, joint recovery of W and C
    python -m toy_g.ltd_rule train --steps 3000 --out runs/ltd/learned_Q
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from . import group as G

CD = torch.complex64


def fourier(device):
    z = np.array([[r, c] for r in range(G.N) for c in range(G.N)])
    return torch.as_tensor(np.exp(2j * np.pi * (z @ z.T) / G.N) / G.N, dtype=CD, device=device)


def haar(n, seed, device):
    g = torch.Generator().manual_seed(seed)
    a = torch.complex(torch.randn(n, n, generator=g), torch.randn(n, n, generator=g))
    q, r = torch.linalg.qr(a)
    return (q * (torch.diagonal(r) / torch.diagonal(r).abs())[None]).to(CD).to(device)


def images(grids, Q):
    """grids [..., 16] long -> per-colour images [..., 4, 16] (complex)."""
    return torch.stack([(grids == a).to(Q.dtype) @ Q for a in range(1, G.COLORS + 1)], -2)


def soft(x, t):
    """Soft threshold of magnitudes (complex or real), t >= 0 broadcastable."""
    mag = x.abs()
    return x * torch.clamp(1 - t / mag.clamp(min=1e-12), min=0)


def infer(U, V, mask, s_w, theta_w, s_c, theta_c, T=100, T_grad=None):
    """U, V [B, K, 4, 16] demonstration images, mask [B, K] -> W [B, 16, 16], C [B, 4, 4]. With T_grad, only the
    last T_grad steps are differentiated (the earlier ones run without a graph and are treated as a constant start)."""
    B = U.shape[0]
    m = mask.to(U.real.dtype)[:, :, None, None]
    U, V = U * m, V * m
    gram = torch.einsum('bkad,bkae->bde', U, U.conj())                      # sum of u u^H (pre-synaptic correlation)
    lam = torch.linalg.eigvalsh(gram)[:, -1].clamp(min=1e-6)                 # largest eigenvalue per task
    W = torch.zeros(B, 16, 16, dtype=U.dtype, device=U.device)
    Y, t = W, 1.0                                                            # FISTA extrapolation point of W
    C = torch.full((B, 4, 4), 0.25, device=U.device)
    for it in range(T):
        if T_grad is not None and it == T - T_grad:
            W, Y, C = W.detach(), Y.detach(), C.detach()
            torch.set_grad_enabled(grad_on)
        elif T_grad is not None and it == 0:
            grad_on = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        YU = U @ Y.transpose(-1, -2)[:, None]                                # Y u_a as rows [B, K, 4, 16]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), YU)) * m
        back = torch.einsum('bca,bkcd->bkad', C.to(U.dtype), E)              # error sent back through the colour map
        cn = C.sum(2).amax(1).clamp(min=1e-3)                                # >= ||C||_2^2 (columns sum <= 1)
        eta_w = (s_w / (lam * cn))[:, None, None]
        Wn = soft(Y + eta_w * torch.einsum('bkad,bkae->bde', back, U.conj()), eta_w * theta_w)
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        Y, W, t = Wn + ((t - 1) / tn) * (Wn - W), Wn, tn
        WU = U @ W.transpose(-1, -2)[:, None]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), WU)) * m
        eta_c = (s_c / (WU.abs() ** 2).sum((1, 2, 3)).clamp(min=1e-6))[:, None, None]
        C = soft(torch.clamp(C + eta_c * torch.einsum('bkcd,bkad->bca', E, WU.conj()).real, min=0), eta_c * theta_c)
        C = C / C.sum(1, keepdim=True).clamp(min=1e-6).maximum(torch.ones_like(C[:, :1]))
    return W, C


def read(Uq, W, C, Q):
    """Uq [B, 4, 16] query images -> colour scores [B, 4, 16] (cells on the last axis)."""
    Vq = torch.einsum('bca,bad->bcd', C.to(Uq.dtype), Uq @ W.transpose(-1, -2))
    return (Vq @ Q.conj().T).real


class LTDRuleModel(nn.Module):
    def __init__(self, basis='learned', seed=0, T=300, T_grad=50):
        super().__init__()
        self.basis, self.T, self.T_grad = basis, T, T_grad
        self.register_buffer('Q0', haar(16, seed, 'cpu') if basis in ('learned', 'random') else
                             (fourier('cpu') if basis == 'fourier' else torch.eye(16, dtype=CD)))
        self.A = nn.Parameter(torch.zeros(16, 16, 2)) if basis == 'learned' else None
        self.log_s_w = nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_theta_w = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_s_c = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_theta_c = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bg = nn.Parameter(torch.tensor(0.5))

    def codes(self):
        if self.A is None:
            return self.Q0
        A = torch.complex(self.A[..., 0], self.A[..., 1])
        return self.Q0 @ torch.matrix_exp(A - A.conj().T)

    def forward(self, demo_in, demo_out, demo_mask, query_in):
        B, K = demo_in.shape[:2]
        Q = self.codes()
        U = images(demo_in.reshape(B, K, 16).long(), Q)
        V = images(demo_out.reshape(B, K, 16).long(), Q)
        W, C = infer(U, V, demo_mask, self.log_s_w.exp().clamp(max=1.0), self.log_theta_w.exp(),
                     self.log_s_c.exp().clamp(max=1.0), self.log_theta_c.exp(), self.T,
                     self.T_grad if self.training else None)
        s = read(images(query_in.reshape(B, 16).long(), Q), W, C, Q)          # [B, 4, 16]
        logits = self.log_scale.exp() * torch.cat([self.bg.expand(B, 1, 16), s], 1)
        return dict(probs=logits.softmax(1).transpose(1, 2), W=W, C=C)      # probs [B, 16, 5]


@torch.no_grad()
def lawful_sparsity(Q, thresh=0.05):
    """Entries above thresh of the 128 toy transformations and of 1,000 random permutations, in the basis Q."""
    def nnz(src):
        P = torch.zeros(len(src), 16, 16, dtype=Q.dtype, device=Q.device)
        P[torch.arange(len(src))[:, None], torch.arange(16)[None], src] = 1
        M = Q.T @ P @ Q.conj()                                               # the W with W u = v (column convention)
        return (M.abs() > thresh).sum((1, 2)).float()
    lawful = nnz(torch.as_tensor(G.SRC, device=Q.device))
    rng = np.random.default_rng(0)
    rand = nnz(torch.as_tensor(np.stack([rng.permutation(16) for _ in range(1000)]), device=Q.device))
    return dict(lawful_median=float(lawful.median()), lawful_max=float(lawful.max()), random_median=float(rand.median()))


@torch.no_grad()
def check(T=100):
    """No learning: joint recovery with Q fixed (Fourier / identity / random), all 3,072 instances, 3 demonstrations."""
    dev = torch.device('cuda')
    inst = D.load_instances()
    group = inst['group'][inst['task']]
    din = torch.as_tensor(inst['demo_in'].reshape(-1, 3, 16).astype(np.int64), device=dev)
    dout = torch.as_tensor(inst['demo_out'].reshape(-1, 3, 16).astype(np.int64), device=dev)
    qin = torch.as_tensor(inst['test_in'].reshape(-1, 16).astype(np.int64), device=dev)
    target = torch.as_tensor(inst['test_out'].reshape(-1, 16).astype(np.int64), device=dev)
    mask = torch.ones(len(din), 3, dtype=torch.bool, device=dev)
    for name, Q in (('fourier', fourier(dev)), ('identity', torch.eye(16, dtype=CD, device=dev)), ('random', haar(16, 0, dev))):
      for s_w, theta_w in ((1.0, 1e-3), (1.0, 1e-2)):
        t0 = time.time()
        U, V = images(din, Q), images(dout, Q)
        W, C = infer(U, V, mask, s_w, theta_w, 0.5, 0.01, T)
        s = read(images(qin, Q), W, C, Q)
        pred = torch.cat([torch.full_like(s[:, :1], 0.5), s], 1).argmax(1)
        ok = (pred == target).all(1).cpu().numpy()
        cmap = torch.as_tensor(G.CMAPS[inst['task'] % G.NUM_C][:, 1:] - 1, device=dev)
        c_ok = (C.argmax(1) == cmap).all(1).float().mean().item()
        print(f'{name:8s} T={T} s_w={s_w} theta_w={theta_w}: exact g0 {ok[group == 0].mean():.3f} g1 {ok[group == 1].mean():.3f} g2 {ok[group == 2].mean():.3f}'
              f' | colour map argmax right {c_ok:.3f} | W entries > 0.05 median {(W.abs() > 0.05).sum((1, 2)).float().median().item():.0f}'
              f' | basis {lawful_sparsity(Q)} | {time.time() - t0:.1f} s', flush=True)


def train(args):
    from .model import nll
    from .train_g import TorchSampler, evaluate, format_eval
    torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=1))
    split = D.make_split(seed=0)
    inst = D.load_instances()
    model = LTDRuleModel(args.basis, args.seed, args.T, args.T_grad).to(dev)
    params = [p for p in model.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    sampler = TorchSampler(np.sort(split['train']), dev, seed=10_000 + args.seed)

    def run_eval(step):
        model.eval()
        res = evaluate(model, inst, dev, ks=(3,))
        sp = lawful_sparsity(model.codes().detach())
        model.train()
        sc = dict(s_w=model.log_s_w.exp().item(), theta_w=model.log_theta_w.exp().item(), s_c=model.log_s_c.exp().item(),
                  theta_c=model.log_theta_c.exp().item(), scale=model.log_scale.exp().item(), bg=model.bg.item())
        with open(out / 'evals.jsonl', 'a') as f:
            f.write(json.dumps(dict(step=step, eval=res, sparsity=sp, scalars=sc)) + '\n')
        print(f'--- eval at step {step}\n{format_eval(res)}\n  basis {sp} | {sc}', flush=True)

    run_eval(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        b = sampler.batch(args.batch, (1, 2, 3), (3, 8))
        res = model(b['demo_in'], b['demo_out'], b['demo_mask'], b['query_in'])
        loss = nll(res['probs'], b['query_out'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if step % args.log_every == 0:
            exact = (res['probs'].argmax(-1) == b['query_out']).all(1).float().mean().item()
            print(f'step {step:5d} loss {loss.item():.4f} exact {exact:.3f} gnorm {gn:.3f} {time.time() - t0:.0f}s', flush=True)
        if step % args.eval_every == 0 and step != args.steps:
            run_eval(step)
    run_eval(args.steps)
    torch.save(model.state_dict(), out / 'final.pt')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'train'])
    ap.add_argument('--basis', default='learned', choices=['learned', 'fourier', 'identity', 'random'])
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--T-grad', type=int, default=50, help='steps differentiated in training (the last ones)')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--out', default='runs/ltd/learned_Q')
    a = ap.parse_args()
    check(a.T) if a.cmd == 'check' else train(a)
