"""Fused Triton block of the phase-clock STDP memory (forward and backward).

For one block, per (batch, head), with x_hat = x / (||x||_2 + eps) per token, RoPE the learned per-pair
rotation (angles [H, T, P]) and the per-pair event tags / retentions tau_K, tau_V, lam_K, lam_V [B, H, T, P]
(both components of a pair share them):

    qr = RoPE(q_hat),  kr = RoPE(k_hat)
    E_K' = lam_K E_K + (1 - lam_K) k_hat e^{-i tau_K}        (stored before RoPE)
    D_V  = lam_V E_V,   E_V' = D_V + (1 - lam_V) v e^{+i tau_V}
    G    = sum_n v kr^T + alpha_h Im[ (v e^{i tau_V})^T RoPE(E_K') + D_V^T (kr e^{-i tau_K}) ]
    read = qr G^T

RoPE commutes with the per-pair scalars, so RoPE(E_K') = lam_K RoPE(E_K) + (1 - lam_K) kr e^{-i tau_K} is formed
in registers from the unrotated trace. The Hebbian carrier and the four timing features are accumulated by
channel-parity blocks with tensor-core dots in one program per (batch, head), so every operand is read once and no
feature tensor touches memory; the new traces are written by the same program. The kernels are bandwidth bound
(FP32 traces in and out).

Backward (four kernels): dG = dR^T qr and dq (through RoPE and the L2 norm), then the value side (dv, dE_V, d tau_V,
d lam_V, d alpha) and the key side (dk through RoPE and the L2 norm, dE_K, d tau_K, d lam_K, d angles). The upstream
gradients of the new traces (from the next block) enter both. G is returned without gradient.

bf16=True rounds dot operands to BF16 with FP32 accumulation; bf16=False uses IEEE FP32 dots (reference precision).
Elementwise work, norms, tags and traces are FP32 in both modes. Registered as opaque custom ops: under torch.compile
the kernels are launched by Triton's own JIT (Inductor-recompiled user kernels ran 3-5x slower). q, k, v may be
strided [B, H, T, D] views (unit stride in D); traces are contiguous [B, H, T, 2D] (real half, imaginary half); tags
and retentions contiguous [B, H, T, P].
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
def _ldp(ptr, base, st, t, p, T, P):
    """[BT, BP] FP32 tile of a [.., T, P] tensor (one value per pair) for tokens t."""
    m = (t[:, None] < T) & (p[None, :] < P)
    return tl.load(ptr + base + t[:, None] * st + p[None, :], mask=m, other=0.).to(tl.float32)


@triton.jit
def _stp(ptr, base, st, t, p, T, P, x):
    m = (t[:, None] < T) & (p[None, :] < P)
    tl.store(ptr + base + t[:, None] * st + p[None, :], x, mask=m)


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
def _ldpT(ptr, base, st, t, p, T, P):
    """Transposed [BP, BT] FP32 tile of a [.., T, P] tensor."""
    m = (p[:, None] < P) & (t[None, :] < T)
    return tl.load(ptr + base + t[None, :] * st + p[:, None], mask=m, other=0.).to(tl.float32)


@triton.jit
def _stT(ptr, base, st, t, p, T, P, PAR, x):
    """Store the transposed tile x[p, t] into component 2p + PAR of tokens t."""
    m = (p[:, None] < P) & (t[None, :] < T)
    tl.store(ptr + base + t[None, :] * st + 2 * p[:, None] + PAR, x, mask=m)


@triton.jit
def _fwd_kernel(Q, K, V, EK, EV, PHK, PHV, LK, LV, ANG, ALPHA, READ, G, EKO, EVO,
                sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, srb, srh, srt,
                H, T, D, P, eps, BF16: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr):
    """Program (batch*head): G, read, E_K' and E_V' for both channel parities.

    U_xy[b, a] = G[a_x, b_y] accumulates with the key side transposed ([BP, BT]); one program per
    (batch, head) loads every operand once (the kernels are bandwidth bound).
    """
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    alpha = tl.load(ALPHA + h)
    qb, kb, vb, rb = b * sqb + h * sqh, b * skb + h * skh, b * svb + h * svh, b * srb + h * srh
    pb, eb, se = bh * T * P, bh * T * 2 * D, 2 * D
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    uee = tl.zeros((BP, BP), dtype=tl.float32)
    ueo = tl.zeros((BP, BP), dtype=tl.float32)
    uoe = tl.zeros((BP, BP), dtype=tl.float32)
    uoo = tl.zeros((BP, BP), dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tt
        # key side [BP, BT]: unit key, RoPE, phase e^{-i phi_K}, new trace (unrotated) and its rotation
        k_e = _ldT(K, kb, skt, t, p, T, P, 0)
        k_o = _ldT(K, kb, skt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, True)
        inv = 1. / (tl.sqrt(tl.sum(k_e * k_e + k_o * k_o, axis=0)) + eps)
        he, ho = k_e * inv[None, :], k_o * inv[None, :]
        ke, ko = he * ca - ho * sa, he * sa + ho * ca
        pk = _ldpT(PHK, pb, P, t, p, T, P)
        ck, sk = tl.cos(pk), tl.sin(pk)
        lk = _ldpT(LK, pb, P, t, p, T, P)
        wk = 1. - lk
        nkr_e = lk * _ldT(EK, eb, se, t, p, T, P, 0) + wk * he * ck
        nkr_o = lk * _ldT(EK, eb, se, t, p, T, P, 1) + wk * ho * ck
        nki_e = lk * _ldT(EK, eb + D, se, t, p, T, P, 0) - wk * he * sk
        nki_o = lk * _ldT(EK, eb + D, se, t, p, T, P, 1) - wk * ho * sk
        _stT(EKO, eb, se, t, p, T, P, 0, nkr_e)
        _stT(EKO, eb, se, t, p, T, P, 1, nkr_o)
        _stT(EKO, eb + D, se, t, p, T, P, 0, nki_e)
        _stT(EKO, eb + D, se, t, p, T, P, 1, nki_o)
        rr_e, rr_o = nkr_e * ca - nkr_o * sa, nkr_e * sa + nkr_o * ca
        ri_e, ri_o = nki_e * ca - nki_o * sa, nki_e * sa + nki_o * ca
        kc_e, kc_o, ks_e, ks_o = ke * ck, ko * ck, ke * sk, ko * sk
        # value side [BT, BP]: carrier v, phase e^{i phi_V}, decayed trace D_V and the new trace
        pv = _ldp(PHV, pb, P, t, p, T, P)
        cv, sv = tl.cos(pv), tl.sin(pv)
        lv = _ldp(LV, pb, P, t, p, T, P)
        wv = 1. - lv
        for X in tl.static_range(2):
            vx = _ld(V, vb, svt, t, p, T, P, X)
            dvr = lv * _ld(EV, eb, se, t, p, T, P, X)
            dvi = lv * _ld(EV, eb + D, se, t, p, T, P, X)
            _st(EVO, eb, se, t, p, T, P, X, dvr + wv * vx * cv)
            _st(EVO, eb + D, se, t, p, T, P, X, dvi + wv * vx * sv)
            # G[a, b] = sum_n v kr + alpha [ (v sv) rr + (v cv) ri + dvi (kr ck) - dvr (kr sk) ]
            a1, a2, a3, a4 = alpha * vx * sv, alpha * vx * cv, alpha * dvi, -alpha * dvr
            ue = (_dot(ke, vx, BF16) + _dot(rr_e, a1, BF16) + _dot(ri_e, a2, BF16)
                  + _dot(kc_e, a3, BF16) + _dot(ks_e, a4, BF16))
            uo = (_dot(ko, vx, BF16) + _dot(rr_o, a1, BF16) + _dot(ri_o, a2, BF16)
                  + _dot(kc_o, a3, BF16) + _dot(ks_o, a4, BF16))
            if X == 0:
                uee += ue
                ueo += uo
            else:
                uoe += ue
                uoo += uo
    gbase = bh * D * D
    _gstT(G, gbase, p, P, D, 0, 0, uee)
    _gstT(G, gbase, p, P, D, 0, 1, ueo)
    _gstT(G, gbase, p, P, D, 1, 0, uoe)
    _gstT(G, gbase, p, P, D, 1, 1, uoo)
    for t0 in range(0, T, BT):
        t = t0 + tt
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        qe, qo = _norm_rope(_ld(Q, qb, sqt, t, p, T, P, 0), _ld(Q, qb, sqt, t, p, T, P, 1), ca, sa, eps, 1)
        _st(READ, rb, srt, t, p, T, P, 0, _dot(qe, uee, BF16) + _dot(qo, ueo, BF16))
        _st(READ, rb, srt, t, p, T, P, 1, _dot(qe, uoe, BF16) + _dot(qo, uoo, BF16))


@triton.jit
def _bwd_v_kernel(DG, K, V, EK, EV, PHK, PHV, LK, LV, ANG, ALPHA, DEVO, DV, DEV, DPHV, DLV, DALPHA,
                  skb, skh, skt, svb, svh, svt, sob, soh, sot, H, T, D, P, eps,
                  BF16: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr):
    """Value side: for each key feature B_m, dA_m = B_m dG^T; then dv, d phi_V, dE_V, d lam_V, d alpha."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    alpha = tl.load(ALPHA + h)
    kb, vb, ob = b * skb + h * skh, b * svb + h * svh, b * sob + h * soh
    pb, eb, se = bh * T * P, bh * T * 2 * D, 2 * D
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
        # key features [BT, BP]
        k_e = _ld(K, kb, skt, t, p, T, P, 0)
        k_o = _ld(K, kb, skt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        inv = 1. / (tl.sqrt(tl.sum(k_e * k_e + k_o * k_o, axis=1)) + eps)
        he, ho = k_e * inv[:, None], k_o * inv[:, None]
        ke, ko = he * ca - ho * sa, he * sa + ho * ca
        pk = _ldp(PHK, pb, P, t, p, T, P)
        ck, sk = tl.cos(pk), tl.sin(pk)
        lk = _ldp(LK, pb, P, t, p, T, P)
        wk = 1. - lk
        nkr_e = lk * _ld(EK, eb, se, t, p, T, P, 0) + wk * he * ck
        nkr_o = lk * _ld(EK, eb, se, t, p, T, P, 1) + wk * ho * ck
        nki_e = lk * _ld(EK, eb + D, se, t, p, T, P, 0) - wk * he * sk
        nki_o = lk * _ld(EK, eb + D, se, t, p, T, P, 1) - wk * ho * sk
        rr_e, rr_o = nkr_e * ca - nkr_o * sa, nkr_e * sa + nkr_o * ca
        ri_e, ri_o = nki_e * ca - nki_o * sa, nki_e * sa + nki_o * ca
        dh_e = _dot(ke, tee, BF16) + _dot(ko, teo, BF16)
        dh_o = _dot(ke, toe, BF16) + _dot(ko, too, BF16)
        d1_e = _dot(rr_e, tee, BF16) + _dot(rr_o, teo, BF16)
        d1_o = _dot(rr_e, toe, BF16) + _dot(rr_o, too, BF16)
        d2_e = _dot(ri_e, tee, BF16) + _dot(ri_o, teo, BF16)
        d2_o = _dot(ri_e, toe, BF16) + _dot(ri_o, too, BF16)
        d3_e = _dot(ke * ck, tee, BF16) + _dot(ko * ck, teo, BF16)
        d3_o = _dot(ke * ck, toe, BF16) + _dot(ko * ck, too, BF16)
        d4_e = _dot(ke * sk, tee, BF16) + _dot(ko * sk, teo, BF16)
        d4_o = _dot(ke * sk, toe, BF16) + _dot(ko * sk, too, BF16)
        # value side
        ve = _ld(V, vb, svt, t, p, T, P, 0)
        vo = _ld(V, vb, svt, t, p, T, P, 1)
        pv = _ldp(PHV, pb, P, t, p, T, P)
        cv, sv = tl.cos(pv), tl.sin(pv)
        lv = _ldp(LV, pb, P, t, p, T, P)
        wv = 1. - lv
        evr_e = _ld(EV, eb, se, t, p, T, P, 0)
        evr_o = _ld(EV, eb, se, t, p, T, P, 1)
        evi_e = _ld(EV, eb + D, se, t, p, T, P, 0)
        evi_o = _ld(EV, eb + D, se, t, p, T, P, 1)
        gvr_e = _ld(DEVO, eb, se, t, p, T, P, 0)
        gvr_o = _ld(DEVO, eb, se, t, p, T, P, 1)
        gvi_e = _ld(DEVO, eb + D, se, t, p, T, P, 0)
        gvi_o = _ld(DEVO, eb + D, se, t, p, T, P, 1)
        # A_hebb = v, A1 = v sv, A2 = v cv, A3 = D_V_im, A4 = -D_V_re;  E_V' = D_V + (1 - lv) v (cv, sv)
        _st(DV, ob, sot, t, p, T, P, 0, dh_e + alpha * (d1_e * sv + d2_e * cv) + wv * (gvr_e * cv + gvi_e * sv))
        _st(DV, ob, sot, t, p, T, P, 1, dh_o + alpha * (d1_o * sv + d2_o * cv) + wv * (gvr_o * cv + gvi_o * sv))
        v1, v2 = d1_e * ve + d1_o * vo, d2_e * ve + d2_o * vo
        gr, gi = gvr_e * ve + gvr_o * vo, gvi_e * ve + gvi_o * vo
        _stp(DPHV, pb, P, t, p, T, P, alpha * (v1 * cv - v2 * sv) + wv * (gi * cv - gr * sv))
        ddr_e, ddr_o = gvr_e - alpha * d4_e, gvr_o - alpha * d4_o
        ddi_e, ddi_o = gvi_e + alpha * d3_e, gvi_o + alpha * d3_o
        _st(DEV, eb, se, t, p, T, P, 0, lv * ddr_e)
        _st(DEV, eb, se, t, p, T, P, 1, lv * ddr_o)
        _st(DEV, eb + D, se, t, p, T, P, 0, lv * ddi_e)
        _st(DEV, eb + D, se, t, p, T, P, 1, lv * ddi_o)
        _stp(DLV, pb, P, t, p, T, P, ddr_e * evr_e + ddr_o * evr_o + ddi_e * evi_e + ddi_o * evi_o - gr * cv - gi * sv)
        dal += tl.sum(v1 * sv + v2 * cv + lv * (evi_e * d3_e + evi_o * d3_o - evr_e * d4_e - evr_o * d4_o))
    tl.store(DALPHA + bh, dal)


@triton.jit
def _bwd_k_kernel(DG, K, V, EK, EV, PHK, PHV, LK, LV, ANG, ALPHA, DEKO, DK, DEK, DPHK, DLK, DANG,
                  skb, skh, skt, svb, svh, svt, sob, soh, sot, H, T, D, P, eps,
                  BF16: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr):
    """Key side: for each value feature A_m, dB_m = A_m dG; then the trace, phase, RoPE (angle) and
    L2-norm chains to the raw key, dE_K, d phi_K and d lam_K."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    alpha = tl.load(ALPHA + h)
    kb, vb, ob = b * skb + h * skh, b * svb + h * svh, b * sob + h * soh
    pb, eb, se = bh * T * P, bh * T * 2 * D, 2 * D
    gbase = bh * D * D
    p = tl.arange(0, BP)
    tt = tl.arange(0, BT)
    dee = _gld(DG, gbase, p, P, D, 0, 0, False)
    deo = _gld(DG, gbase, p, P, D, 0, 1, False)
    doe = _gld(DG, gbase, p, P, D, 1, 0, False)
    doo = _gld(DG, gbase, p, P, D, 1, 1, False)
    for t0 in range(0, T, BT):
        t = t0 + tt
        # value features [BT, BP]
        ve = _ld(V, vb, svt, t, p, T, P, 0)
        vo = _ld(V, vb, svt, t, p, T, P, 1)
        pv = _ldp(PHV, pb, P, t, p, T, P)
        cv, sv = tl.cos(pv), tl.sin(pv)
        lv = _ldp(LV, pb, P, t, p, T, P)
        dvr_e = lv * _ld(EV, eb, se, t, p, T, P, 0)
        dvr_o = lv * _ld(EV, eb, se, t, p, T, P, 1)
        dvi_e = lv * _ld(EV, eb + D, se, t, p, T, P, 0)
        dvi_o = lv * _ld(EV, eb + D, se, t, p, T, P, 1)
        bh_e = _dot(ve, dee, BF16) + _dot(vo, doe, BF16)
        bh_o = _dot(ve, deo, BF16) + _dot(vo, doo, BF16)
        b1_e = _dot(ve * sv, dee, BF16) + _dot(vo * sv, doe, BF16)
        b1_o = _dot(ve * sv, deo, BF16) + _dot(vo * sv, doo, BF16)
        b2_e = _dot(ve * cv, dee, BF16) + _dot(vo * cv, doe, BF16)
        b2_o = _dot(ve * cv, deo, BF16) + _dot(vo * cv, doo, BF16)
        b3_e = _dot(dvi_e, dee, BF16) + _dot(dvi_o, doe, BF16)
        b3_o = _dot(dvi_e, deo, BF16) + _dot(dvi_o, doo, BF16)
        b4_e = -(_dot(dvr_e, dee, BF16) + _dot(dvr_o, doe, BF16))
        b4_o = -(_dot(dvr_e, deo, BF16) + _dot(dvr_o, doo, BF16))
        # key side
        k_e = _ld(K, kb, skt, t, p, T, P, 0)
        k_o = _ld(K, kb, skt, t, p, T, P, 1)
        ca, sa = _angles(ANG, h, t, p, T, P, False)
        n = tl.sqrt(tl.sum(k_e * k_e + k_o * k_o, axis=1))
        inv = 1. / (n + eps)
        he, ho = k_e * inv[:, None], k_o * inv[:, None]
        ke, ko = he * ca - ho * sa, he * sa + ho * ca
        pk = _ldp(PHK, pb, P, t, p, T, P)
        ck, sk = tl.cos(pk), tl.sin(pk)
        lk = _ldp(LK, pb, P, t, p, T, P)
        wk = 1. - lk
        ekr_e = _ld(EK, eb, se, t, p, T, P, 0)
        ekr_o = _ld(EK, eb, se, t, p, T, P, 1)
        eki_e = _ld(EK, eb + D, se, t, p, T, P, 0)
        eki_o = _ld(EK, eb + D, se, t, p, T, P, 1)
        nkr_e, nkr_o = lk * ekr_e + wk * he * ck, lk * ekr_o + wk * ho * ck
        nki_e, nki_o = lk * eki_e - wk * he * sk, lk * eki_o - wk * ho * sk
        rr_e, rr_o = nkr_e * ca - nkr_o * sa, nkr_e * sa + nkr_o * ca
        ri_e, ri_o = nki_e * ca - nki_o * sa, nki_e * sa + nki_o * ca
        # gradients on the rotated traces rr = RoPE(E_K'_re), ri = RoPE(E_K'_im)
        dyr_e, dyr_o = alpha * b1_e, alpha * b1_o
        dyi_e, dyi_o = alpha * b2_e, alpha * b2_o
        # total gradient on the unrotated new trace: upstream + RoPE^T
        gkr_e = _ld(DEKO, eb, se, t, p, T, P, 0) + dyr_e * ca + dyr_o * sa
        gkr_o = _ld(DEKO, eb, se, t, p, T, P, 1) - dyr_e * sa + dyr_o * ca
        gki_e = _ld(DEKO, eb + D, se, t, p, T, P, 0) + dyi_e * ca + dyi_o * sa
        gki_o = _ld(DEKO, eb + D, se, t, p, T, P, 1) - dyi_e * sa + dyi_o * ca
        _st(DEK, eb, se, t, p, T, P, 0, lk * gkr_e)
        _st(DEK, eb, se, t, p, T, P, 1, lk * gkr_o)
        _st(DEK, eb + D, se, t, p, T, P, 0, lk * gki_e)
        _st(DEK, eb + D, se, t, p, T, P, 1, lk * gki_o)
        # kr receives the Hebbian carrier and the phase-tagged current-key features B3 = kr ck, B4 = kr sk
        dkr_e = bh_e + alpha * (b3_e * ck + b4_e * sk)
        dkr_o = bh_o + alpha * (b3_o * ck + b4_o * sk)
        dang = (-dkr_e * ko + dkr_o * ke - dyr_e * rr_o + dyr_o * rr_e - dyi_e * ri_o + dyi_o * ri_e)
        m = (t[:, None] < T) & (p[None, :] < P)
        tl.store(DANG + bh * T * P + t[:, None] * P + p[None, :], dang, mask=m)
        hr, hi = gkr_e * he + gkr_o * ho, gki_e * he + gki_o * ho
        k3, k4 = b3_e * ke + b3_o * ko, b4_e * ke + b4_o * ko
        _stp(DPHK, pb, P, t, p, T, P, -wk * (hr * sk + hi * ck) + alpha * (k4 * ck - k3 * sk))
        _stp(DLK, pb, P, t, p, T, P, gkr_e * (ekr_e - he * ck) + gkr_o * (ekr_o - ho * ck)
             + gki_e * (eki_e + he * sk) + gki_o * (eki_o + ho * sk))
        # unit key: RoPE^T of dkr plus the unrotated trace path; then the L2-norm chain
        ge = dkr_e * ca + dkr_o * sa + wk * (gkr_e * ck - gki_e * sk)
        go = -dkr_e * sa + dkr_o * ca + wk * (gkr_o * ck - gki_o * sk)
        dot_xg = tl.sum(k_e * ge + k_o * go, axis=1)
        cn = tl.where(n > 0., dot_xg * inv * inv / tl.where(n > 0., n, 1.), 0.)
        _st(DK, ob, sot, t, p, T, P, 0, ge * inv[:, None] - k_e * cn[:, None])
        _st(DK, ob, sot, t, p, T, P, 1, go * inv[:, None] - k_o * cn[:, None])



FWD_CONFIG = dict(BT=16, num_warps=8, num_stages=1)
BWD_CONFIG = dict(BT=16, num_warps=8, num_stages=1)
QG_CONFIG = dict(BT=16, num_warps=4, num_stages=1)      # dG and dq kernels


def set_config(forward=None, backward=None):
    """Tile/warp choice. A one-config autotuner carries num_warps under torch.compile."""
    global _fwd, _bwd_dg, _bwd_q, _bwd_v, _bwd_k
    if forward: FWD_CONFIG.update(forward)
    if backward: BWD_CONFIG.update(backward)
    cfg = lambda c: triton.Config(dict(BT=c['BT']), num_warps=c['num_warps'], num_stages=c['num_stages'])
    _fwd = triton.autotune([cfg(FWD_CONFIG)], key=[])(_fwd_kernel)
    _bwd_dg, _bwd_q = (triton.autotune([cfg(QG_CONFIG)], key=[])(k) for k in (_bwd_dg_kernel, _bwd_q_kernel))
    _bwd_v, _bwd_k = (triton.autotune([cfg(BWD_CONFIG)], key=[])(k) for k in (_bwd_v_kernel, _bwd_k_kernel))


set_config()


def _strides(x):
    if x.dim() != 4 or x.stride(-1) != 1:
        raise ValueError('expected [B, H, T, D] with unit stride in D')
    return x.stride(0), x.stride(1), x.stride(2)


def _bp(d):
    if d % 2 or d // 2 > 128:
        raise ValueError(f'head dim {d} must be even and at most 256')
    return max(16, triton.next_power_of_2(d // 2))

def _launch_forward(q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, eps, bf16):
    bsz, h, t, d = q.shape
    read = torch.empty(bsz, t, h, d, device=q.device, dtype=torch.float32)
    g = torch.empty(bsz, h, d, d, device=q.device, dtype=torch.float32)
    eko, evo = torch.empty_like(ek), torch.empty_like(ev)
    _fwd[(bsz * h,)](q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, read, g, eko, evo,
                       *_strides(q), *_strides(k), *_strides(v), t * h * d, d, h * d,
                       h, t, d, d // 2, eps, BF16=bf16, BP=_bp(d))
    return read, g, eko, evo


def _launch_backward(dr, deko, devo, q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, g, eps, bf16):
    bsz, h, t, d = q.shape
    p, bp, n = d // 2, _bp(d), bsz * h
    dq, dk, dv = (torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v))
    dg = torch.empty_like(g)
    dek, dev = torch.empty_like(ek), torch.empty_like(ev)
    dphk, dphv, dlk, dlv = (torch.empty_like(x) for x in (phk, phv, lk, lv))
    dalpha = torch.empty(n, device=q.device, dtype=torch.float32)
    dang_q = torch.empty(bsz, h, t, p, device=q.device, dtype=torch.float32)
    dang_k = torch.empty(bsz, h, t, p, device=q.device, dtype=torch.float32)
    _bwd_dg[(n, 2)](dr, q, angles, dg, *_strides(dr), *_strides(q), h, t, d, p, eps, BF16=bf16, BP=bp)
    _bwd_q[(n,)](dr, q, angles, g, dq, dang_q, *_strides(dr), *_strides(q), *_strides(dq),
                    h, t, d, p, eps, BF16=bf16, BP=bp)
    _bwd_v[(n,)](dg, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, devo, dv, dev, dphv, dlv, dalpha,
                 *_strides(k), *_strides(v), *_strides(dv), h, t, d, p, eps, BF16=bf16, BP=bp)
    _bwd_k[(n,)](dg, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, deko, dk, dek, dphk, dlk, dang_k,
                 *_strides(k), *_strides(v), *_strides(dk), h, t, d, p, eps, BF16=bf16, BP=bp)
    return dq, dk, dv, dek, dev, dphk, dphv, dlk, dlv, dalpha.view(bsz, h).sum(0), (dang_q + dang_k).sum(0)


@custom_op('lt::phase_clock_attention', mutates_args=())
def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor,
               phk: torch.Tensor, phv: torch.Tensor, lk: torch.Tensor, lv: torch.Tensor, angles: torch.Tensor,
               alpha: torch.Tensor, eps: float, bf16: bool
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_forward(q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, eps, bf16)


@custom_op('lt::phase_clock_attention_backward', mutates_args=())
def _attention_backward(dr: torch.Tensor, deko: torch.Tensor, devo: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                        v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor, phk: torch.Tensor, phv: torch.Tensor,
                        lk: torch.Tensor, lv: torch.Tensor, angles: torch.Tensor, alpha: torch.Tensor,
                        g: torch.Tensor, eps: float, bf16: bool
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor]:
    return _launch_backward(dr, deko, devo, q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, g, eps, bf16)


@_attention.register_fake
def _attention_fake(q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, eps, bf16):
    bsz, h, t, d = q.shape
    return (q.new_empty(bsz, t, h, d, dtype=torch.float32), q.new_empty(bsz, h, d, d, dtype=torch.float32),
            torch.empty_like(ek), torch.empty_like(ev))


@_attention_backward.register_fake
def _attention_backward_fake(dr, deko, devo, q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, g, eps, bf16):
    return (*(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v)),
            torch.empty_like(ek), torch.empty_like(ev), *(torch.empty_like(x) for x in (phk, phv, lk, lv)),
            alpha.new_empty(q.shape[1]), angles.new_empty(angles.shape))


def _setup(ctx, inputs, output):
    q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, eps, bf16 = inputs
    ctx.save_for_backward(q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, output[1])
    ctx.flags = (eps, bf16)
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, dr, dg, deko, devo):
    q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, g = ctx.saved_tensors
    deko = torch.zeros_like(ek) if deko is None else deko.contiguous()
    devo = torch.zeros_like(ev) if devo is None else devo.contiguous()
    grads = _attention_backward(dr.transpose(1, 2), deko, devo, q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha,
                                g, *ctx.flags)
    dq, dk, dv, dek, dev, dphk, dphv, dlk, dlv, dalpha, dang = grads
    return dq, dk, dv, dek, dev, dphk, dphv, dlk, dlv, dang, dalpha, None, None


_attention.register_autograd(_backward, setup_context=_setup)


def phase_clock_attention(q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, eps, bf16=False):
    """q, k, v [B, H, T, D] (CUDA, FP32 or BF16, unit stride in D); ek, ev [B, H, T, 2D] traces
    (real half, imaginary half; the K trace before RoPE); phk, phv, lk, lv [B, H, T, D/2] phases
    (already advanced for this block) and retentions; angles [H, T, D/2]; alpha [H].

    Returns (read [B, H, T, D] FP32 (a view of [B, T, H, D] storage), G [B, H, D, D] FP32 without
    gradient, new K trace, new V trace). Gradients flow to every other input.
    """
    read, g, eko, evo = _attention(q, k, v, ek.float().contiguous(), ev.float().contiguous(),
                                   phk.float().contiguous(), phv.float().contiguous(),
                                   lk.float().contiguous(), lv.float().contiguous(),
                                   angles.float().contiguous(), alpha.float().contiguous(), float(eps), bool(bf16))
    return read.transpose(1, 2), g, eko, evo


def reference_phase_clock(q, k, v, ek, ev, phk, phv, lk, lv, angles, alpha, eps):
    """Direct torch evaluation of the same block (tests; any dtype, e.g. FP64)."""
    d = q.shape[-1]
    q, k = (x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + eps) for x in (q, k))
    c, s = angles.cos()[None], angles.sin()[None]

    def rope(x):
        e, o = x[..., 0::2], x[..., 1::2]
        return torch.stack((e * c - o * s, e * s + o * c), -1).reshape_as(x)
    rep = lambda x: x.repeat_interleave(2, -1)
    qr, kr = rope(q), rope(k)
    ck, sk, cv, sv = rep(phk.cos()), rep(phk.sin()), rep(phv.cos()), rep(phv.sin())
    lk, lv = rep(lk), rep(lv)
    ekr, eki = ek[..., :d], ek[..., d:]
    evr, evi = ev[..., :d], ev[..., d:]
    nkr = lk * ekr + (1 - lk) * k * ck
    nki = lk * eki - (1 - lk) * k * sk
    dvr, dvi = lv * evr, lv * evi
    im_s = ((v * sv).transpose(-1, -2) @ rope(nkr) + (v * cv).transpose(-1, -2) @ rope(nki)
            + dvi.transpose(-1, -2) @ (kr * ck) - dvr.transpose(-1, -2) @ (kr * sk))
    g = v.transpose(-1, -2) @ kr + alpha.to(q.dtype)[:, None, None] * im_s
    eko = torch.cat((nkr, nki), -1)
    evo = torch.cat((dvr + (1 - lv) * v * cv, dvi + (1 - lv) * v * sv), -1)
    return qr @ g.transpose(-1, -2), g, eko, evo
