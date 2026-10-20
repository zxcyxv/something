"""Fused Triton band step of the band_product memory (forward and backward).

For one block, per (batch, head), with k~ (unit, RoPE) and v [T, D] FP32, R bands with per-head coefficients
z_r = zr + i zi, gamma_r and gamma_r c_r = gr + i gi, and the carried traces E_K, E_V [T, 2, R, D] (Re, Im):

    E_K,r' = z_r E_K,r + gamma_r k~;   D_V,r = z_r E_V,r;   E_V,r' = D_V,r + gamma_r v        (this block included)
    Y_r    = gamma_r c_r E_V,r'
    M      = sum_r sum_t Re(Y_r)^T Re(E_K,r') + Im(Y_r)^T Im(E_K,r')         [D x D, value rows, key cols]

Forward: one program per (batch, head) streams the T tiles and bands once, writes the new traces and accumulates M in
registers, so Y and the stacked [2RT, D] operands never touch memory. Backward, given dM (two kernels): the value
side forms dY = E_K' dM^T and runs the E_V chain (dv, dE_V, d(gamma c), and dz, dgamma from the value side); the key
side forms Y dM and runs the E_K chain (dk~, dE_K, and dz, dgamma from the key side). The upstream gradients of the
new traces (from the next block) enter both. Per-(head, band) parameter gradients are per-program partial sums,
reduced over the batch in torch. Traffic per block: 4 trace volumes forward, 8 backward (bandwidth bound). The
backward contracts D in chunks of BJ (an IEEE dot with a [D, D] operand does not fit in registers), and the band and
chunk loops are dynamic: unrolled with tl.static_range, every load of a step was hoisted and the kernels spilled.

All arithmetic is FP32 with IEEE FP32 dots (no TF32 or BF16): the results equal lt.model's torch path up to FP32
rounding (summation order). Registered as opaque custom ops, as lt/kernel.py. k~ and v may be strided [B, H, T, D]
views (unit stride in D); traces are contiguous [B, H, T, 2RD] ((Re, Im) x R bands of D); coef [5, H, R] =
(zr, zi, gamma, gr, gi); M [B, H, D, D]; D <= 128.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import custom_op


@triton.jit
def _coef(COEF, h, r, HR, R: tl.constexpr):
    """(zr, zi, gamma, gr, gi) of head h, band r."""
    o = h * R + r
    return (tl.load(COEF + o), tl.load(COEF + HR + o), tl.load(COEF + 2 * HR + o), tl.load(COEF + 3 * HR + o),
            tl.load(COEF + 4 * HR + o))


@triton.jit
def _fwd_kernel(KR, V, EK, EV, COEF, M, EKO, EVO, skb, skh, skt, svb, svh, svt, H, T, D,
                R: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """Program (batch*head): new traces and M = sum Y^T E_K' (accumulated over T tiles and bands)."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    d = tl.arange(0, BD)
    tt = tl.arange(0, BT)
    dm = d < D
    acc = tl.zeros((BD, BD), dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tt
        m2 = (t[:, None] < T) & dm[None, :]
        kr = tl.load(KR + b * skb + h * skh + t[:, None] * skt + d[None, :], mask=m2, other=0.)
        v = tl.load(V + b * svb + h * svh + t[:, None] * svt + d[None, :], mask=m2, other=0.)
        row = (bh * T + t[:, None]) * (2 * R * D) + d[None, :]
        for r in range(R):
            zr, zi, ga, gr, gi = _coef(COEF, h, r, H * R, R)
            ore = row + r * D
            oim = ore + R * D
            ekr = tl.load(EK + ore, mask=m2, other=0.)
            eki = tl.load(EK + oim, mask=m2, other=0.)
            evr = tl.load(EV + ore, mask=m2, other=0.)
            evi = tl.load(EV + oim, mask=m2, other=0.)
            nkr = zr * ekr - zi * eki + ga * kr
            nki = zr * eki + zi * ekr
            dvi = zr * evi + zi * evr
            nvr = zr * evr - zi * evi + ga * v
            tl.store(EKO + ore, nkr, mask=m2)
            tl.store(EKO + oim, nki, mask=m2)
            tl.store(EVO + ore, nvr, mask=m2)
            tl.store(EVO + oim, dvi, mask=m2)
            yr = gr * nvr - gi * dvi
            yi = gr * dvi + gi * nvr
            acc += tl.dot(tl.trans(yr), nkr, input_precision="ieee")
            acc += tl.dot(tl.trans(yi), nki, input_precision="ieee")
    tl.store(M + bh * D * D + d[:, None] * D + d[None, :], acc, mask=dm[:, None] & dm[None, :])


@triton.jit
def _bwd_v_kernel(DMT, KR, V, EK, EV, COEF, GEV, DEV, DV, DCOEF, skb, skh, skt, svb, svh, svt, sdb, sdh, sdt,
                  H, T, D, R: tl.constexpr, RP: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BJ: tl.constexpr,
                  HAS_G: tl.constexpr, WRITE_DE: tl.constexpr):
    """Value side: dY = E_K' dM^T (DMT = dM^T contiguous, read by rows; the key dim contracted in chunks of BJ, so no
    [D, D] operand is held), then dv, dE_V and the (zr, zi, gamma, gr, gi) partial sums of this program."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    d = tl.arange(0, BD)
    jj = tl.arange(0, BJ)
    tt = tl.arange(0, BT)
    dm = d < D
    ri = tl.arange(0, RP)
    s_zr = tl.zeros((RP,), dtype=tl.float32)
    s_zi = tl.zeros((RP,), dtype=tl.float32)
    s_ga = tl.zeros((RP,), dtype=tl.float32)
    s_gr = tl.zeros((RP,), dtype=tl.float32)
    s_gi = tl.zeros((RP,), dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tt
        tm = t < T
        m2 = tm[:, None] & dm[None, :]
        v = tl.load(V + b * svb + h * svh + t[:, None] * svt + d[None, :], mask=m2, other=0.)
        dv = tl.zeros((BT, BD), dtype=tl.float32)
        trow = (bh * T + t[:, None]) * (2 * R * D)
        row = trow + d[None, :]
        for r in range(R):
            zr, zi, ga, gr, gi = _coef(COEF, h, r, H * R, R)
            dyr = tl.zeros((BT, BD), dtype=tl.float32)
            dyi = tl.zeros((BT, BD), dtype=tl.float32)
            for jc in range(BD // BJ):
                j = jc * BJ + jj
                jm = j < D
                mj = tm[:, None] & jm[None, :]
                dmt = tl.load(DMT + bh * D * D + j[:, None] * D + d[None, :], mask=jm[:, None] & dm[None, :], other=0.)
                krj = tl.load(KR + b * skb + h * skh + t[:, None] * skt + j[None, :], mask=mj, other=0.)
                ekr = tl.load(EK + trow + r * D + j[None, :], mask=mj, other=0.)
                eki = tl.load(EK + trow + (R + r) * D + j[None, :], mask=mj, other=0.)
                nkr = zr * ekr - zi * eki + ga * krj
                nki = zr * eki + zi * ekr
                dyr += tl.dot(nkr, dmt, input_precision="ieee")
                dyi += tl.dot(nki, dmt, input_precision="ieee")
            ore = row + r * D
            oim = ore + R * D
            evr = tl.load(EV + ore, mask=m2, other=0.)
            evi = tl.load(EV + oim, mask=m2, other=0.)
            dvi = zr * evi + zi * evr
            nvr = zr * evr - zi * evi + ga * v
            sel = ri == r
            s_gr += tl.where(sel, tl.sum(dyr * nvr + dyi * dvi), 0.)
            s_gi += tl.where(sel, tl.sum(dyi * nvr - dyr * dvi), 0.)
            gvr = gr * dyr + gi * dyi                       # total gradient on the new trace E_V,r'
            gvi = gr * dyi - gi * dyr
            if HAS_G:
                gvr += tl.load(GEV + ore, mask=m2, other=0.)
                gvi += tl.load(GEV + oim, mask=m2, other=0.)
            dv += ga * gvr
            s_ga += tl.where(sel, tl.sum(gvr * v), 0.)
            s_zr += tl.where(sel, tl.sum(gvr * evr + gvi * evi), 0.)
            s_zi += tl.where(sel, tl.sum(gvi * evr - gvr * evi), 0.)
            if WRITE_DE:
                tl.store(DEV + ore, zr * gvr + zi * gvi, mask=m2)
                tl.store(DEV + oim, zr * gvi - zi * gvr, mask=m2)
        tl.store(DV + b * sdb + h * sdh + t[:, None] * sdt + d[None, :], dv, mask=m2)
    rm = ri < R
    base = bh * 5 * R + ri
    tl.store(DCOEF + base, s_zr, mask=rm)
    tl.store(DCOEF + base + R, s_zi, mask=rm)
    tl.store(DCOEF + base + 2 * R, s_ga, mask=rm)
    tl.store(DCOEF + base + 3 * R, s_gr, mask=rm)
    tl.store(DCOEF + base + 4 * R, s_gi, mask=rm)


@triton.jit
def _bwd_k_kernel(DM, KR, V, EK, EV, COEF, GEK, DEK, DKR, DCOEF, skb, skh, skt, svb, svh, svt, sdb, sdh, sdt,
                  H, T, D, R: tl.constexpr, RP: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BJ: tl.constexpr,
                  HAS_G: tl.constexpr, WRITE_DE: tl.constexpr):
    """Key side: dE_K' = Y dM (the value dim contracted in chunks of BJ), then dk~, dE_K and the (zr, zi, gamma)
    partial sums of this program."""
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    d = tl.arange(0, BD)
    jj = tl.arange(0, BJ)
    tt = tl.arange(0, BT)
    dm = d < D
    ri = tl.arange(0, RP)
    s_zr = tl.zeros((RP,), dtype=tl.float32)
    s_zi = tl.zeros((RP,), dtype=tl.float32)
    s_ga = tl.zeros((RP,), dtype=tl.float32)
    for t0 in range(0, T, BT):
        t = t0 + tt
        tm = t < T
        m2 = tm[:, None] & dm[None, :]
        kr = tl.load(KR + b * skb + h * skh + t[:, None] * skt + d[None, :], mask=m2, other=0.)
        dk = tl.zeros((BT, BD), dtype=tl.float32)
        trow = (bh * T + t[:, None]) * (2 * R * D)
        row = trow + d[None, :]
        for r in range(R):
            zr, zi, ga, gr, gi = _coef(COEF, h, r, H * R, R)
            gkr = tl.zeros((BT, BD), dtype=tl.float32)          # total gradient on the new trace E_K,r'
            gki = tl.zeros((BT, BD), dtype=tl.float32)
            for ic in range(BD // BJ):
                i = ic * BJ + jj
                im = i < D
                mi = tm[:, None] & im[None, :]
                dmi = tl.load(DM + bh * D * D + i[:, None] * D + d[None, :], mask=im[:, None] & dm[None, :], other=0.)
                vi = tl.load(V + b * svb + h * svh + t[:, None] * svt + i[None, :], mask=mi, other=0.)
                evr = tl.load(EV + trow + r * D + i[None, :], mask=mi, other=0.)
                evi = tl.load(EV + trow + (R + r) * D + i[None, :], mask=mi, other=0.)
                dvi = zr * evi + zi * evr
                nvr = zr * evr - zi * evi + ga * vi
                yr = gr * nvr - gi * dvi
                yi = gr * dvi + gi * nvr
                gkr += tl.dot(yr, dmi, input_precision="ieee")
                gki += tl.dot(yi, dmi, input_precision="ieee")
            ore = row + r * D
            oim = ore + R * D
            ekr = tl.load(EK + ore, mask=m2, other=0.)
            eki = tl.load(EK + oim, mask=m2, other=0.)
            if HAS_G:
                gkr += tl.load(GEK + ore, mask=m2, other=0.)
                gki += tl.load(GEK + oim, mask=m2, other=0.)
            dk += ga * gkr
            sel = ri == r
            s_ga += tl.where(sel, tl.sum(gkr * kr), 0.)
            s_zr += tl.where(sel, tl.sum(gkr * ekr + gki * eki), 0.)
            s_zi += tl.where(sel, tl.sum(gki * ekr - gkr * eki), 0.)
            if WRITE_DE:
                tl.store(DEK + ore, zr * gkr + zi * gki, mask=m2)
                tl.store(DEK + oim, zr * gki - zi * gkr, mask=m2)
        tl.store(DKR + b * sdb + h * sdh + t[:, None] * sdt + d[None, :], dk, mask=m2)
    rm = ri < R
    base = bh * 5 * R + ri
    tl.store(DCOEF + base, s_zr, mask=rm)
    tl.store(DCOEF + base + R, s_zi, mask=rm)
    tl.store(DCOEF + base + 2 * R, s_ga, mask=rm)


FWD_CONFIG = dict(BT=16, num_warps=8, num_stages=1)
BWD_CONFIG = dict(BT=16, BJ=64, num_warps=8, num_stages=1)


def set_config(forward=None, backward=None):
    """Tile/warp choice. A one-config autotuner carries num_warps under torch.compile."""
    global _fwd, _bwd_v, _bwd_k
    if forward:
        FWD_CONFIG.update(forward)
    if backward:
        BWD_CONFIG.update(backward)
    cfg = lambda c: triton.Config(dict(BT=c['BT']), num_warps=c['num_warps'], num_stages=c['num_stages'])
    _fwd = triton.autotune([cfg(FWD_CONFIG)], key=[])(_fwd_kernel)
    _bwd_v, _bwd_k = (triton.autotune([cfg(BWD_CONFIG)], key=[])(k) for k in (_bwd_v_kernel, _bwd_k_kernel))


set_config()


def _strides(x):
    if x.dim() != 4 or x.stride(-1) != 1:
        raise ValueError('expected [B, H, T, D] with unit stride in D')
    return x.stride(0), x.stride(1), x.stride(2)


def _bd(d):
    if d > 128:
        raise ValueError(f'head dim {d} must be at most 128 (the [D, D] accumulator of one program)')
    return max(16, triton.next_power_of_2(d))


def _launch_forward(kr, v, ek, ev, coef):
    bsz, h, t, d = kr.shape
    r = coef.shape[-1]
    m = torch.empty(bsz, h, d, d, device=kr.device, dtype=torch.float32)
    eko, evo = torch.empty_like(ek), torch.empty_like(ev)
    _fwd[(bsz * h,)](kr, v, ek, ev, coef, m, eko, evo, *_strides(kr), *_strides(v), h, t, d, R=r, BD=_bd(d))
    return m, eko, evo


def _launch_backward(dm, deko, devo, kr, v, ek, ev, coef, has_gk, has_gv, write_de):
    bsz, h, t, d = kr.shape
    r = coef.shape[-1]
    dkr = torch.empty(bsz, h, t, d, device=kr.device, dtype=torch.float32)
    dv = torch.empty_like(dkr)
    dek = torch.empty_like(ek) if write_de else ek.new_empty(0)
    dev = torch.empty_like(ev) if write_de else ev.new_empty(0)
    dcoef = torch.zeros(2, bsz * h, 5, r, device=kr.device, dtype=torch.float32)
    rp, bd = max(2, triton.next_power_of_2(r)), _bd(d)
    bj = min(BWD_CONFIG['BJ'], bd)                      # contraction chunk (a launch argument: clamped to BD)
    dmt = dm.transpose(-1, -2).contiguous()             # the value side reads dM^T by rows
    _bwd_v[(bsz * h,)](dmt, kr, v, ek, ev, coef, devo, dev, dv, dcoef[0], *_strides(kr), *_strides(v),
                       *_strides(dv), h, t, d, R=r, RP=rp, BD=bd, BJ=bj, HAS_G=has_gv, WRITE_DE=write_de)
    _bwd_k[(bsz * h,)](dm, kr, v, ek, ev, coef, deko, dek, dkr, dcoef[1], *_strides(kr), *_strides(v),
                       *_strides(dkr), h, t, d, R=r, RP=rp, BD=bd, BJ=bj, HAS_G=has_gk, WRITE_DE=write_de)
    return dkr, dv, dek, dev, dcoef.view(2, bsz, h, 5, r).sum((0, 1)).transpose(0, 1).contiguous()


@custom_op('lt::band_product', mutates_args=())
def _band_product(kr: torch.Tensor, v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor, coef: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_forward(kr, v, ek, ev, coef)


@custom_op('lt::band_product_backward', mutates_args=())
def _band_product_backward(dm: torch.Tensor, deko: torch.Tensor, devo: torch.Tensor, kr: torch.Tensor,
                           v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor, coef: torch.Tensor, has_gk: bool,
                           has_gv: bool, write_de: bool
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch_backward(dm, deko, devo, kr, v, ek, ev, coef, has_gk, has_gv, write_de)


@_band_product.register_fake
def _band_product_fake(kr, v, ek, ev, coef):
    bsz, h, t, d = kr.shape
    return kr.new_empty(bsz, h, d, d, dtype=torch.float32), torch.empty_like(ek), torch.empty_like(ev)


@_band_product_backward.register_fake
def _band_product_backward_fake(dm, deko, devo, kr, v, ek, ev, coef, has_gk, has_gv, write_de):
    bsz, h, t, d = kr.shape
    return (kr.new_empty(bsz, h, t, d, dtype=torch.float32), kr.new_empty(bsz, h, t, d, dtype=torch.float32),
            torch.empty_like(ek) if write_de else ek.new_empty(0), torch.empty_like(ev) if write_de else ev.new_empty(0),
            coef.new_empty(coef.shape))


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, dm, deko, devo):
    kr, v, ek, ev, coef = ctx.saved_tensors
    write_de = bool(ctx.needs_input_grad[2] or ctx.needs_input_grad[3])
    dm = torch.zeros(kr.shape[:2] + (kr.shape[-1],) * 2, device=kr.device) if dm is None else dm.contiguous()
    has_gk, has_gv = deko is not None, devo is not None
    dkr, dv, dek, dev, dcoef = _band_product_backward(dm, deko.contiguous() if has_gk else ek,
                                                      devo.contiguous() if has_gv else ev, kr, v, ek, ev, coef,
                                                      has_gk, has_gv, write_de)
    return dkr, dv, (dek if write_de else None), (dev if write_de else None), dcoef


_band_product.register_autograd(_backward, setup_context=_setup)


def band_product(kr, v, ek, ev, coef):
    """kr = k~, v [B, H, T, D] FP32 (unit stride in D); ek, ev [B, H, T, 2RD] traces; coef [5, H, R] =
    (zr, zi, gamma, gamma Re c, gamma Im c). Returns (M [B, H, D, D], new E_K, new E_V); gradients flow to every
    input."""
    return _band_product(kr.float(), v.float(), ek.float().contiguous(), ev.float().contiguous(),
                         coef.float().contiguous())


def reference_band_product(kr, v, ek, ev, coef):
    """Direct torch evaluation of the same block (tests; any dtype, e.g. FP64)."""
    bsz, h, t, d = kr.shape
    r = coef.shape[-1]
    zr, zi, ga, gr, gi = (c[None, :, None, :, None] for c in coef)
    ekr, eki = ek.reshape(bsz, h, t, 2, r, d).unbind(3)
    evr, evi = ev.reshape(bsz, h, t, 2, r, d).unbind(3)
    nkr = zr * ekr - zi * eki + ga * kr.unsqueeze(3)
    nki = zr * eki + zi * ekr
    dvi = zr * evi + zi * evr
    nvr = zr * evr - zi * evi + ga * v.unsqueeze(3)
    yr, yi = gr * nvr - gi * dvi, gr * dvi + gi * nvr
    m = torch.einsum('bhtrd,bhtre->bhde', yr, nkr) + torch.einsum('bhtrd,bhtre->bhde', yi, nki)
    return m, torch.stack((nkr, nki), 3).reshape(bsz, h, t, -1), torch.stack((nvr, dvi), 3).reshape(bsz, h, t, -1)
