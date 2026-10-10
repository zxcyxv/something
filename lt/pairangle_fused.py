"""Fused pairangle attention: L2 norm, RoPE, phase features, write and read in Triton.

For one layer and block, per (batch, head), with x_hat = x / (||x||_2 + eps) per token:

    qr = RoPE(q_hat),  kr = RoPE(k_hat)             (learned per-pair angles [H, T, P])
    G_ab = sum_n v_na kr_nb (hebb + alpha_h W(phiV_n,pair(a) - phiK_n,pair(b)))
    W(d) = sum_r c_r sin(r d),   read = qr G^T

phiV is the angle of the value pair, phiK the angle of the key pair before RoPE
('unrotated', equal to the angle of the raw key) or after it ('rotated').
Unit phasors come from the pair itself, cos phi = x/rho, sin phi = y/rho (cos 1,
sin 0 at rho = 0, as atan2(0, 0) = 0), and the harmonics from the angle-addition
recurrence, so the features need no transcendental function. With A_m, B_m the
value-side and key-side features of the separable window,
    G = sum_m A_m^T B_m,  m in {Hebbian, (r, sin), (r, cos)},
accumulated by channel-parity blocks with tensor-core dots; no feature tensor is
written to memory. G is returned without gradient: the models ignore it in the
next block and detach it at the segment boundary.

Backward: dG = dR^T qr, then dq (through RoPE and the L2 norm), dv and dk
(through the phases, d phi = (x dy - y dx) / rho^2), per-(batch, head) partial
sums for alpha and per-(batch, head) angle gradients summed over the batch.

bf16=True rounds dot operands to BF16 with FP32 accumulation; bf16=False uses
IEEE FP32 dots (reference precision, slow). Elementwise work, norms and phases
are FP32 in both. Registered as opaque custom ops: under torch.compile the kernels are
launched by Triton's own JIT (inductor-recompiled user kernels ran 3-5x slower). q, k, v may be strided [B, H, T, D] views (unit stride in D).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import custom_op


@triton.jit
def _dot(a, b, BF16: tl.constexpr):
    if BF16:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision="ieee")


@triton.jit
def _ld(ptr, base, st, t, p, T, P, PAR):
    """[BT, BP] FP32 tile of channel parity PAR (component 2p + PAR) for tokens t."""
    m = (t[:, None] < T) & (p[None, :] < P)
    return tl.load(ptr + base + t[:, None] * st + 2 * p[None, :] + PAR, mask=m, other=0.).to(tl.float32)


@triton.jit
def _ldT(ptr, base, st, t, p, T, P, PAR):
    """Transposed [BP, BT] FP32 tile of channel parity PAR."""
    m = (p[:, None] < P) & (t[None, :] < T)
    return tl.load(ptr + base + t[None, :] * st + 2 * p[:, None] + PAR, mask=m, other=0.).to(tl.float32)


@triton.jit
def _st(ptr, base, st, t, p, T, P, PAR, x):
    m = (t[:, None] < T) & (p[None, :] < P)
    tl.store(ptr + base + t[:, None] * st + 2 * p[None, :] + PAR, x, mask=m)


@triton.jit
def _angles(ANG, h, t, p, T, P, TRANS: tl.constexpr):
    """cos/sin of the RoPE angles [H, T, P] as [BT, BP] (or [BP, BT] when TRANS)."""
    if TRANS:
        m = (p[:, None] < P) & (t[None, :] < T)
        a = tl.load(ANG + h * T * P + t[None, :] * P + p[:, None], mask=m, other=0.)
    else:
        m = (t[:, None] < T) & (p[None, :] < P)
        a = tl.load(ANG + h * T * P + t[:, None] * P + p[None, :], mask=m, other=0.)
    return tl.cos(a), tl.sin(a)


@triton.jit
def _unit(e, o):
    r2 = e * e + o * o
    pos = r2 > 0.
    inv = tl.where(pos, 1. / tl.sqrt(tl.where(pos, r2, 1.)), 0.)
    return tl.where(pos, e * inv, 1.), o * inv, tl.where(pos, 1. / tl.where(pos, r2, 1.), 0.)


@triton.jit
def _norm_rope(e, o, ca, sa, eps, AXIS: tl.constexpr):
    """x_hat = x / (||x|| + eps) over the channel axis, then the pair rotation."""
    inv = 1. / (tl.sqrt(tl.sum(e * e + o * o, axis=AXIS)) + eps)
    if AXIS == 1:
        inv = inv[:, None]
    else:
        inv = inv[None, :]
    e, o = e * inv, o * inv
    return e * ca - o * sa, e * sa + o * ca


@triton.jit
def _gld(G, gbase, p, P, D, ROW, COL, TRANS: tl.constexpr):
    """Block G[2i + ROW, 2j + COL] as [i, j], or as [j, i] when TRANS."""
    m = (p[:, None] < P) & (p[None, :] < P)
    if TRANS:
        off = (2 * p[None, :] + ROW) * D + 2 * p[:, None] + COL
    else:
        off = (2 * p[:, None] + ROW) * D + 2 * p[None, :] + COL
    return tl.load(G + gbase + off, mask=m, other=0.)


@triton.jit
def _gstT(G, gbase, p, P, D, ROW, COL, x):
    """Store x[j, i] into G[2i + ROW, 2j + COL]."""
    m = (p[:, None] < P) & (p[None, :] < P)
    tl.store(G + gbase + (2 * p[None, :] + ROW) * D + 2 * p[:, None] + COL, x, mask=m)


@triton.jit
def _fwd_kernel(Q, K, V, ANG, ALPHA, COEF, READ, G,
                sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, srb, srh, srt,
                H, T, D, P, eps,
                R: tl.constexpr, HEBB: tl.constexpr, BF16: tl.constexpr, ROTATED: tl.constexpr,
                BT: tl.constexpr, BP: tl.constexpr):
    """Program (batch*head, row parity x): G[a_x, :] and read[:, a_x].

    Accumulates U_y[b, a] = G[a_x, b_y] = sum_n B[n, b_y] A[n, a_x] with the key side
    loaded transposed, so read_x = qr_e U_e + qr_o U_o needs no transpose.
    """
    bh = tl.program_id(0)
    X = tl.program_id(1)
    b, h = bh // H, bh % H
    alpha = tl.load(ALPHA + h)
    qb, kb, vb, rb = b * sqb + h * sqh, b * skb + h * skh, b * svb + h * svh, b * srb + h * srh
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    ue = tl.zeros((BP, BP), dtype=tl.float32)
    uo = tl.zeros((BP, BP), dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tt
        ve = _ld(V, vb, svt, t, p, T, P, 0)
        vo = _ld(V, vb, svt, t, p, T, P, 1)
        vx = ve + (vo - ve) * X
        cv, sv, _ = _unit(ve, vo)
        k_e = _ldT(K, kb, skt, t, p, T, P, 0)
        k_o = _ldT(K, kb, skt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, True)
        ke, ko = _norm_rope(k_e, k_o, ca, sa, eps, 0)
        if ROTATED:
            ck, sk, _ = _unit(ke, ko)
        else:
            ck, sk, _ = _unit(k_e, k_o)
        if HEBB:
            ue += _dot(ke, vx, BF16)
            uo += _dot(ko, vx, BF16)
        crv, srv, crk, srk = cv, sv, ck, sk
        for r in tl.static_range(R):
            c = tl.load(COEF + r) * alpha
            a_s = c * vx * srv
            a_c = -c * vx * crv
            ue += _dot(ke * crk, a_s, BF16) + _dot(ke * srk, a_c, BF16)
            uo += _dot(ko * crk, a_s, BF16) + _dot(ko * srk, a_c, BF16)
            crv, srv = crv * cv - srv * sv, srv * cv + crv * sv
            crk, srk = crk * ck - srk * sk, srk * ck + crk * sk
    gbase = bh * D * D
    _gstT(G, gbase, p, P, D, X, 0, ue)
    _gstT(G, gbase, p, P, D, X, 1, uo)
    for t0 in range(0, T, BT):
        t = t0 + tt
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        qe, qo = _norm_rope(_ld(Q, qb, sqt, t, p, T, P, 0), _ld(Q, qb, sqt, t, p, T, P, 1), ca, sa, eps, 1)
        _st(READ, rb, srt, t, p, T, P, X, _dot(qe, ue, BF16) + _dot(qo, uo, BF16))


@triton.jit
def _bwd_dg_kernel(DR, Q, ANG, DG, sdb, sdh, sdt, sqb, sqh, sqt, H, T, D, P, eps,
                   BF16: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr):
    """Program (batch*head, row parity s): dG[a_s, b_y] = sum_t dR[t, a_s] qr[t, b_y]."""
    bh = tl.program_id(0)
    S = tl.program_id(1)
    b, h = bh // H, bh % H
    db, qb = b * sdb + h * sdh, b * sqb + h * sqh
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    we = tl.zeros((BP, BP), dtype=tl.float32)
    wo = tl.zeros((BP, BP), dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tt
        drs = _ld(DR, db, sdt, t, p, T, P, S)
        ca, sa = _angles(ANG, h, t, p, T, P, True)
        qe, qo = _norm_rope(_ldT(Q, qb, sqt, t, p, T, P, 0), _ldT(Q, qb, sqt, t, p, T, P, 1), ca, sa, eps, 0)
        we += _dot(qe, drs, BF16)
        wo += _dot(qo, drs, BF16)
    gbase = bh * D * D
    _gstT(DG, gbase, p, P, D, S, 0, we)
    _gstT(DG, gbase, p, P, D, S, 1, wo)


@triton.jit
def _bwd_q_kernel(DR, Q, ANG, G, DQ, DANG, sdb, sdh, sdt, sqb, sqh, sqt, sob, soh, sot, H, T, D, P, eps,
                  BF16: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr):
    """dqr = dR G, then through RoPE (angle gradient) and the L2 norm."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    db, qb, ob = b * sdb + h * sdh, b * sqb + h * sqh, b * sob + h * soh
    gbase = bh * D * D
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    gee = _gld(G, gbase, p, P, D, 0, 0, False)
    geo = _gld(G, gbase, p, P, D, 0, 1, False)
    goe = _gld(G, gbase, p, P, D, 1, 0, False)
    goo = _gld(G, gbase, p, P, D, 1, 1, False)
    for t0 in range(0, T, BT):
        t = t0 + tt
        dre = _ld(DR, db, sdt, t, p, T, P, 0)
        dro = _ld(DR, db, sdt, t, p, T, P, 1)
        dqe = _dot(dre, gee, BF16) + _dot(dro, goe, BF16)
        dqo = _dot(dre, geo, BF16) + _dot(dro, goo, BF16)
        q_e = _ld(Q, qb, sqt, t, p, T, P, 0)
        q_o = _ld(Q, qb, sqt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        n = tl.sqrt(tl.sum(q_e * q_e + q_o * q_o, axis=1))
        inv = 1. / (n + eps)
        he, ho = q_e * inv[:, None], q_o * inv[:, None]
        # qr = (he c - ho s, he s + ho c);  d angle = -dqr_e qr_o + dqr_o qr_e
        dang = -dqe * (he * sa + ho * ca) + dqo * (he * ca - ho * sa)
        m = (t[:, None] < T) & (p[None, :] < P)
        tl.store(DANG + bh * T * P + t[:, None] * P + p[None, :], dang, mask=m)
        ge = dqe * ca + dqo * sa
        go = -dqe * sa + dqo * ca
        # x_hat = x / (n + eps):  dx = g / (n + eps) - x (x . g) / (n (n + eps)^2)
        dot_xg = tl.sum(q_e * ge + q_o * go, axis=1)
        cn = tl.where(n > 0., dot_xg * inv * inv / tl.where(n > 0., n, 1.), 0.)
        _st(DQ, ob, sot, t, p, T, P, 0, ge * inv[:, None] - q_e * cn[:, None])
        _st(DQ, ob, sot, t, p, T, P, 1, go * inv[:, None] - q_o * cn[:, None])


@triton.jit
def _bwd_v_kernel(DG, K, V, ANG, ALPHA, COEF, DV, DALPHA,
                  skb, skh, skt, svb, svh, svt, sob, soh, sot, H, T, D, P, eps,
                  R: tl.constexpr, HEBB: tl.constexpr, BF16: tl.constexpr, ROTATED: tl.constexpr,
                  BT: tl.constexpr, BP: tl.constexpr):
    """Value side: dA_x = B_e dG_xe^T + B_o dG_xo^T; A_s = c v sin(r phiV), A_c = -c v cos(r phiV)."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    alpha = tl.load(ALPHA + h)
    kb, vb = b * skb + h * skh, b * svb + h * svh
    gbase = bh * D * D
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    tee = _gld(DG, gbase, p, P, D, 0, 0, True)
    teo = _gld(DG, gbase, p, P, D, 0, 1, True)
    toe = _gld(DG, gbase, p, P, D, 1, 0, True)
    too = _gld(DG, gbase, p, P, D, 1, 1, True)
    dal = 0.
    for t0 in range(0, T, BT):
        t = t0 + tt
        ve = _ld(V, vb, svt, t, p, T, P, 0)
        vo = _ld(V, vb, svt, t, p, T, P, 1)
        k_e = _ld(K, kb, skt, t, p, T, P, 0)
        k_o = _ld(K, kb, skt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        ke, ko = _norm_rope(k_e, k_o, ca, sa, eps, 1)
        cv, sv, iv = _unit(ve, vo)
        if ROTATED:
            ck, sk, _ = _unit(ke, ko)
        else:
            ck, sk, _ = _unit(k_e, k_o)
        if HEBB:
            dve = _dot(ke, tee, BF16) + _dot(ko, teo, BF16)
            dvo = _dot(ke, toe, BF16) + _dot(ko, too, BF16)
        else:
            dve = tl.zeros((BT, BP), dtype=tl.float32)
            dvo = tl.zeros((BT, BP), dtype=tl.float32)
        gpv = tl.zeros((BT, BP), dtype=tl.float32)
        crv, srv, crk, srk = cv, sv, ck, sk
        for r in tl.static_range(R):
            cf = tl.load(COEF + r)
            c = cf * alpha
            rr = r + 1.
            das_e = _dot(ke * crk, tee, BF16) + _dot(ko * crk, teo, BF16)
            das_o = _dot(ke * crk, toe, BF16) + _dot(ko * crk, too, BF16)
            dac_e = _dot(ke * srk, tee, BF16) + _dot(ko * srk, teo, BF16)
            dac_o = _dot(ke * srk, toe, BF16) + _dot(ko * srk, too, BF16)
            dve += c * (das_e * srv - dac_e * crv)
            dvo += c * (das_o * srv - dac_o * crv)
            gpv += (c * rr) * ((das_e * ve + das_o * vo) * crv + (dac_e * ve + dac_o * vo) * srv)
            dal += cf * tl.sum(ve * (das_e * srv - dac_e * crv) + vo * (das_o * srv - dac_o * crv))
            crv, srv = crv * cv - srv * sv, srv * cv + crv * sv
            crk, srk = crk * ck - srk * sk, srk * ck + crk * sk
        # phi = atan2(o, e):  d phi/d e = -o / rho^2,  d phi/d o = e / rho^2
        _st(DV, b * sob + h * soh, sot, t, p, T, P, 0, dve - vo * iv * gpv)
        _st(DV, b * sob + h * soh, sot, t, p, T, P, 1, dvo + ve * iv * gpv)
    tl.store(DALPHA + bh, dal)


@triton.jit
def _bwd_k_kernel(DG, K, V, ANG, ALPHA, COEF, DK, DANG,
                  skb, skh, skt, svb, svh, svt, sob, soh, sot, H, T, D, P, eps,
                  R: tl.constexpr, HEBB: tl.constexpr, BF16: tl.constexpr, ROTATED: tl.constexpr,
                  BT: tl.constexpr, BP: tl.constexpr):
    """Key side: dB_y = A_e dG_ey + A_o dG_oy; B_c = kr cos(r phiK), B_s = kr sin(r phiK);
    then the phase, RoPE (angle gradient) and L2-norm chain to the raw key."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    alpha = tl.load(ALPHA + h)
    kb, vb = b * skb + h * skh, b * svb + h * svh
    gbase = bh * D * D
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    dee = _gld(DG, gbase, p, P, D, 0, 0, False)
    deo = _gld(DG, gbase, p, P, D, 0, 1, False)
    doe = _gld(DG, gbase, p, P, D, 1, 0, False)
    doo = _gld(DG, gbase, p, P, D, 1, 1, False)
    for t0 in range(0, T, BT):
        t = t0 + tt
        ve = _ld(V, vb, svt, t, p, T, P, 0)
        vo = _ld(V, vb, svt, t, p, T, P, 1)
        k_e = _ld(K, kb, skt, t, p, T, P, 0)
        k_o = _ld(K, kb, skt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        n = tl.sqrt(tl.sum(k_e * k_e + k_o * k_o, axis=1))
        inv = 1. / (n + eps)
        he, ho = k_e * inv[:, None], k_o * inv[:, None]
        ke, ko = he * ca - ho * sa, he * sa + ho * ca
        cv, sv, _ = _unit(ve, vo)
        if ROTATED:
            ck, sk, ik = _unit(ke, ko)
            pe, po = ke, ko
        else:
            ck, sk, ik = _unit(k_e, k_o)
            pe, po = k_e, k_o
        if HEBB:
            dke = _dot(ve, dee, BF16) + _dot(vo, doe, BF16)
            dko = _dot(ve, deo, BF16) + _dot(vo, doo, BF16)
        else:
            dke = tl.zeros((BT, BP), dtype=tl.float32)
            dko = tl.zeros((BT, BP), dtype=tl.float32)
        gpk = tl.zeros((BT, BP), dtype=tl.float32)
        crv, srv, crk, srk = cv, sv, ck, sk
        for r in tl.static_range(R):
            c = tl.load(COEF + r) * alpha
            rr = r + 1.
            ase, aso = c * ve * srv, c * vo * srv
            ace, aco = -c * ve * crv, -c * vo * crv
            dbc_e = _dot(ase, dee, BF16) + _dot(aso, doe, BF16)
            dbc_o = _dot(ase, deo, BF16) + _dot(aso, doo, BF16)
            dbs_e = _dot(ace, dee, BF16) + _dot(aco, doe, BF16)
            dbs_o = _dot(ace, deo, BF16) + _dot(aco, doo, BF16)
            dke += dbc_e * crk + dbs_e * srk
            dko += dbc_o * crk + dbs_o * srk
            gpk += rr * ((dbs_e * ke + dbs_o * ko) * crk - (dbc_e * ke + dbc_o * ko) * srk)
            crv, srv = crv * cv - srv * sv, srv * cv + crv * sv
            crk, srk = crk * ck - srk * sk, srk * ck + crk * sk
        # Phase gradient on its source pair (kr when rotated, the raw key otherwise).
        dpe, dpo = -po * ik * gpk, pe * ik * gpk
        if ROTATED:
            dke += dpe
            dko += dpo
        m = (t[:, None] < T) & (p[None, :] < P)
        tl.store(DANG + bh * T * P + t[:, None] * P + p[None, :], -dke * ko + dko * ke, mask=m)
        ge = dke * ca + dko * sa
        go = -dke * sa + dko * ca
        dot_xg = tl.sum(k_e * ge + k_o * go, axis=1)
        cn = tl.where(n > 0., dot_xg * inv * inv / tl.where(n > 0., n, 1.), 0.)
        dxe = ge * inv[:, None] - k_e * cn[:, None]
        dxo = go * inv[:, None] - k_o * cn[:, None]
        if not ROTATED:
            dxe += dpe
            dxo += dpo
        _st(DK, b * sob + h * soh, sot, t, p, T, P, 0, dxe)
        _st(DK, b * sob + h * soh, sot, t, p, T, P, 1, dxo)


FWD_CONFIG = dict(BT=16, num_warps=4, num_stages=1)
BWD_CONFIG = dict(BT=16, num_warps=4, num_stages=1)


def set_config(forward=None, backward=None):
    """Tile/warp choice. A one-config autotuner carries num_warps under torch.compile."""
    global _fwd, _bwd_dg, _bwd_q, _bwd_v, _bwd_k
    if forward: FWD_CONFIG.update(forward)
    if backward: BWD_CONFIG.update(backward)
    cfg = lambda c: triton.Config(dict(BT=c['BT']), num_warps=c['num_warps'], num_stages=c['num_stages'])
    _fwd = triton.autotune([cfg(FWD_CONFIG)], key=[])(_fwd_kernel)
    _bwd_dg, _bwd_q, _bwd_v, _bwd_k = (triton.autotune([cfg(BWD_CONFIG)], key=[])(k)
                                       for k in (_bwd_dg_kernel, _bwd_q_kernel, _bwd_v_kernel, _bwd_k_kernel))


set_config()


def _strides(x):
    if x.dim() != 4 or x.stride(-1) != 1:
        raise ValueError('expected [B, H, T, D] with unit stride in D')
    return x.stride(0), x.stride(1), x.stride(2)


def _bp(d):
    if d % 2 or d // 2 > 128:
        raise ValueError(f'head dim {d} must be even and at most 256')
    return max(16, triton.next_power_of_2(d // 2))


@custom_op('lt::pairangle_attention', mutates_args=())
def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, angles: torch.Tensor,
               alpha: torch.Tensor, coef: torch.Tensor, eps: float, hebbian: bool, bf16: bool,
               rotated: bool) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, h, t, d = q.shape
    # read is written as [B, T, H, D] (heads merge for free); outputs are never views
    read = torch.empty(bsz, t, h, d, device=q.device, dtype=torch.float32)
    g = torch.empty(bsz, h, d, d, device=q.device, dtype=torch.float32)
    _fwd[(bsz * h, 2)](q, k, v, angles, alpha, coef, read, g,
                                    *_strides(q), *_strides(k), *_strides(v), t * h * d, d, h * d,
                                    h, t, d, d // 2, eps, R=coef.numel(), HEBB=hebbian, BF16=bf16,
                                    ROTATED=rotated, BP=_bp(d))
    return read, g


@custom_op('lt::pairangle_attention_backward', mutates_args=())
def _attention_backward(dr: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        angles: torch.Tensor, alpha: torch.Tensor, coef: torch.Tensor, g: torch.Tensor,
                        eps: float, hebbian: bool, bf16: bool, rotated: bool
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, h, t, d = q.shape
    p, bp, n = d // 2, _bp(d), bsz * h
    # Gradients get their own (dense) layout; q, k, v may be views with gaps (e.g. qkv chunks).
    dq, dk, dv = (torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))
    dg = torch.empty_like(g)
    dalpha = torch.empty(n, device=q.device, dtype=torch.float32)
    dang_q = torch.empty(bsz, h, t, p, device=q.device, dtype=torch.float32)
    dang_k = torch.empty(bsz, h, t, p, device=q.device, dtype=torch.float32)
    flags = dict(R=coef.numel(), HEBB=hebbian, BF16=bf16, ROTATED=rotated, BP=bp)
    _bwd_dg[(n, 2)](dr, q, angles, dg, *_strides(dr), *_strides(q), h, t, d, p, eps,
                                 BF16=bf16, BP=bp)
    _bwd_q[(n,)](dr, q, angles, g, dq, dang_q, *_strides(dr), *_strides(q), *_strides(dq),
                              h, t, d, p, eps,
                              BF16=bf16, BP=bp)
    _bwd_v[(n,)](dg, k, v, angles, alpha, coef, dv, dalpha, *_strides(k), *_strides(v),
                              *_strides(dv), h, t, d, p, eps, **flags)
    _bwd_k[(n,)](dg, k, v, angles, alpha, coef, dk, dang_k, *_strides(k), *_strides(v),
                              *_strides(dk), h, t, d, p, eps, **flags)
    return dq, dk, dv, dalpha.view(bsz, h).sum(0), (dang_q + dang_k).sum(0)


@_attention.register_fake
def _attention_fake(q, k, v, angles, alpha, coef, eps, hebbian, bf16, rotated):
    bsz, h, t, d = q.shape
    return q.new_empty(bsz, t, h, d, dtype=torch.float32), q.new_empty(bsz, h, d, d, dtype=torch.float32)


@_attention_backward.register_fake
def _attention_backward_fake(dr, q, k, v, angles, alpha, coef, g, eps, hebbian, bf16, rotated):
    bsz, h, t, d = q.shape
    return (*(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v)),
            alpha.new_empty(h), angles.new_empty(angles.shape))


def _setup(ctx, inputs, output):
    q, k, v, angles, alpha, coef, eps, hebbian, bf16, rotated = inputs
    ctx.save_for_backward(q, k, v, angles, alpha, coef, output[1])
    ctx.flags = (eps, hebbian, bf16, rotated)
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, dr, dg):
    q, k, v, angles, alpha, coef, g = ctx.saved_tensors
    dq, dk, dv, dalpha, dang = _attention_backward(dr.transpose(1, 2), q, k, v, angles, alpha, coef, g,
                                                   *ctx.flags)
    return dq, dk, dv, dang, dalpha, None, None, None, None, None


_attention.register_autograd(_backward, setup_context=_setup)


def pairangle_attention(q, k, v, angles, alpha, coef, eps, hebbian=True, bf16=False, rotated=False):
    """q, k, v [B, H, T, D] (CUDA, FP32 or BF16, unit stride in D); angles [H, T, D/2] RoPE angles;
    alpha [H]; coef [R] (window c_r for frequencies 1..R). Token-sum write.

    Returns (read [B, H, T, D] FP32, a view of [B, T, H, D] storage; G [B, H, D, D] FP32, no grad).
    """
    read, g = _attention(q, k, v, angles.float().contiguous(), alpha.float().contiguous(),
                         coef.float().contiguous(), float(eps), bool(hebbian), bool(bf16), bool(rotated))
    return read.transpose(1, 2), g


def reference_attention(q, k, v, angles, alpha, coef, eps, hebbian=True, rotated=False):
    """Direct torch evaluation with atan2 phases and the model's norm/RoPE (tests)."""
    q, k = (x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + eps) for x in (q, k))
    c, s = angles.cos()[None], angles.sin()[None]
    def rope(x):
        e, o = x[..., 0::2], x[..., 1::2]
        return torch.stack((e * c - o * s, e * s + o * c), -1).reshape_as(x)
    qr, kr = rope(q), rope(k)
    def angle(x):
        pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        return torch.atan2(pair[..., 1], pair[..., 0]).repeat_interleave(2, -1)
    pk, pv = angle(kr if rotated else k), angle(v)
    delta = pv[..., :, None] - pk[..., None, :]
    r = torch.arange(1, coef.numel() + 1, dtype=q.dtype, device=q.device)
    window = (coef.to(q.dtype) * torch.sin(delta[..., None] * r)).sum(-1)
    gain = alpha.to(q.dtype)[:, None, None, None] * window
    if hebbian:
        gain = gain + 1
    g = (v[..., :, None] * kr[..., None, :] * gain).sum(-3)
    return qr @ g.transpose(-1, -2), g
