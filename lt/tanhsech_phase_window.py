"""Exact tanh*sech STDP window with free per-token phases, as a fused kernel.

    G[i,j] = mean_n V[n,i] K[n,j] L(phiV[n,i] - phiK[n,j]),   L(d) = tanh(d) sech(d)

L is odd, has slope 1 at the origin, peak 1/2 at d=asinh(1) and the tail
2 exp(-|d|). No finite separable rank represents it exactly, so the pair sum
is evaluated directly, without materializing the [T,D,D] pair tensor.

Per-pair exponentials are removed algebraically. With r = e^d and p = r^2,
    L(d) = 2 r (p - 1) / (p + 1)^2,
and r = e^{phiV} e^{-phiK} factorizes. Precomputing per token/channel
    A = V e^{phiV},  P = e^{2 phiV},  B = K e^{-phiK},  Q = e^{-2 phiK}
gives V K L = A B h(P Q) with h(p) = 2 (p - 1) / (p + 1)^2, so every pair needs
one reciprocal and a few FMAs. h'(p) = 2 (3 - p) / (p + 1)^3 reuses it.
The kernel differentiates A, P, B, Q; autograd chains through the O(T D)
precompute. Phases must stay moderate (|phi| <= 10) so P*Q cannot overflow;
the models bound them by (pi/2) tanh.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


PHASE_BOUND = 10.0


def window(delta):
    return delta.tanh() / delta.cosh()


def derivative(delta):
    sech = 1 / delta.cosh()
    return sech ** 3 - delta.tanh().square() * sech


def direct_write(k, v, pk, pv):
    """Reference [...,T,D,D] pair evaluation, any device and dtype."""
    delta = pv[..., :, None] - pk[..., None, :]
    return (v[..., :, None] * k[..., None, :] * window(delta)).mean(-3)


@triton.jit
def _rcp(x):
    # Approximate reciprocal (rcp.approx, ~1 ulp) instead of IEEE division.
    return tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;', '=r,r', [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _write_fwd_kernel(A, P, B, Q, G, T, D, scale,
               BI: tl.constexpr, BJ: tl.constexpr):
    # Outer-product accumulation over tokens n into a register tile of G.
    bh, ib, jb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i = ib * BI + tl.arange(0, BI)
    j = jb * BJ + tl.arange(0, BJ)
    mi, mj = i < D, j < D
    base = bh * T * D
    acc = tl.zeros((BI, BJ), dtype=tl.float32)
    for n in range(T):
        row = base + n * D
        a = tl.load(A + row + i, mask=mi, other=0.)
        pi = tl.load(P + row + i, mask=mi, other=0.)
        b = tl.load(B + row + j, mask=mj, other=0.)
        qj = tl.load(Q + row + j, mask=mj, other=0.)
        p = pi[:, None] * qj[None, :]
        inv = _rcp(p + 1.)
        acc += (a[:, None] * b[None, :]) * ((p - 1.) * inv * inv)
    out = G + bh * D * D + i[:, None] * D + j[None, :]
    tl.store(out, acc * scale, mask=mi[:, None] & mj[None, :])


@triton.jit
def _write_bwd_kernel(XT, XPT, OT, OPT, OQT, dGT, dXT, dXPT, T, D, scale,
                      BI: tl.constexpr, BN: tl.constexpr):
    """One side of the backward, without atomics or cross-thread reductions.

    Everything is in the transposed [D, T] layout so tokens are contiguous.
    Own side X, XP; other side O, OP and OQ = O * OP. For the rows,
    own = (A, P) and other = (B, Q):
        dA[n,i] = sum_j dG[i,j] B[n,j] h(P[n,i] Q[n,j])
        dP[n,i] = A[n,i] sum_j dG[i,j] B[n,j] Q[n,j] h'(P[n,i] Q[n,j])
    The columns are the same with the roles swapped and dG transposed.
    dGT holds dG with the other side's channel as the row: element (j, i)
    pairs other channel j with own channel i. The tile is (own channels i,
    tokens n); the sum over the other side's channels is a sequential
    outer-product accumulation, like the forward. Each thread keeps several
    channels for one token, so per step it loads three token values and a
    broadcast cotangent row.
    """
    bh, ib, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i = ib * BI + tl.arange(0, BI)
    n = nb * BN + tl.arange(0, BN)
    mi, mn = i < D, n < T
    m = mi[:, None] & mn[None, :]
    tile = bh * D * T + i[:, None] * T + n[None, :]
    x = tl.load(XT + tile, mask=m, other=0.)
    xp = tl.load(XPT + tile, mask=m, other=0.)
    acc_h = tl.zeros((BI, BN), dtype=tl.float32)
    acc_d = tl.zeros((BI, BN), dtype=tl.float32)
    g_row = dGT + bh * D * D + i
    column = bh * D * T + n
    for j in range(D):
        g = tl.load(g_row + j * D, mask=mi, other=0.)
        o = tl.load(OT + column + j * T, mask=mn, other=0.)
        op = tl.load(OPT + column + j * T, mask=mn, other=0.)
        oq = tl.load(OQT + column + j * T, mask=mn, other=0.)
        p = xp * op[None, :]
        inv = _rcp(p + 1.)
        inv2 = inv * inv
        acc_h += (g[:, None] * o[None, :]) * ((p - 1.) * inv2)
        acc_d += (g[:, None] * oq[None, :]) * ((3. - p) * inv2 * inv)
    tl.store(dXT + tile, acc_h * scale, mask=m)
    tl.store(dXPT + tile, x * acc_d * scale, mask=m)


def set_blocks(forward=(16, 128, 1), backward=(32, 32, 1)):
    """Tile shapes (rows, columns, num_warps); defaults from the RTX 3090 sweep.

    Each kernel is wrapped in a single-config autotuner because torch.compile
    does not forward a num_warps launch argument to user Triton kernels; a
    Config is honoured in eager mode and under compile alike.
    """
    global BLOCK_FWD, BLOCK_BWD, _write_fwd, _write_bwd_side
    BLOCK_FWD, BLOCK_BWD = tuple(forward), tuple(backward)
    _write_fwd = triton.autotune([triton.Config(dict(BI=forward[0], BJ=forward[1]), num_warps=forward[2])],
                                 key=[])(_write_fwd_kernel)
    _write_bwd_side = triton.autotune([triton.Config(dict(BI=backward[0], BN=backward[1]), num_warps=backward[2])],
                                      key=[])(_write_bwd_kernel)


set_blocks()


def _grid(a, block):
    d = a.shape[-1]
    return (a.numel() // (a.shape[-2] * d), triton.cdiv(d, block[0]), triton.cdiv(d, block[1]))


@triton_op('lt::tanhsech_write', mutates_args={})
def _pair_write(a: torch.Tensor, p: torch.Tensor, b: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    *lead, t, d = a.shape
    a, p, b, q = (x.contiguous() for x in (a, p, b, q))
    g = torch.empty(*lead, d, d, device=a.device, dtype=torch.float32)
    wrap_triton(_write_fwd)[_grid(a, BLOCK_FWD)](a, p, b, q, g, t, d, 2. / t)
    return g


@triton_op('lt::tanhsech_write_backward', mutates_args={})
def _pair_write_backward(a: torch.Tensor, p: torch.Tensor, b: torch.Tensor, q: torch.Tensor,
                         dg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t, d = a.shape[-2:]
    tr = lambda x: x.transpose(-1, -2).contiguous()
    at, pt, bt, qt = (tr(x) for x in (a, p, b, q))
    dat, dpt, dbt, dqt = (torch.empty_like(x) for x in (at, pt, bt, qt))
    bi, bn, _ = BLOCK_BWD
    grid = (a.numel() // (t * d), triton.cdiv(d, bi), triton.cdiv(t, bn))
    kernel = wrap_triton(_write_bwd_side)
    # Rows: own (A, P), other (B, Q); other channel j indexes rows of dG.T.
    kernel[grid](at, pt, bt, qt, bt * qt, tr(dg), dat, dpt, t, d, 2. / t)
    # Columns: own (B, Q), other (A, P); other channel i indexes rows of dG.
    kernel[grid](bt, qt, at, pt, at * pt, dg.contiguous(), dbt, dqt, t, d, 2. / t)
    da, dp, db, dq = (tr(x) for x in (dat, dpt, dbt, dqt))
    return da, dp, db, dq


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dg):
    return _pair_write_backward(*ctx.saved_tensors, dg)


_pair_write.register_autograd(_backward, setup_context=_setup)


def fused_write(k, v, pk, pv):
    """FP32 CUDA kernel path. Inputs [..., T, D]; returns [..., D, D] FP32."""
    k, v, pk, pv = (x.float() for x in (k, v, pk, pv))
    ev, ek = pv.exp(), (-pk).exp()
    return _pair_write(*(x.contiguous() for x in (v * ev, ev.square(), k * ek, ek.square())))


def write(k, v, pk, pv):
    """Model entry: fused kernel for CUDA FP32/BF16, direct pairs otherwise (tests, FP64)."""
    if k.is_cuda and k.dtype != torch.float64:
        return fused_write(k, v, pk, pv)
    return direct_write(k, v, pk, pv)
