"""Current-G STDP with unit phase direction and a Gaussian envelope.

uV=exp(i phiV), uK=exp(i phiK), Im(uV conj(uK))=sin(phiV-phiK).
A is the scalar unit direction of that imaginary inner product, with A=0
at an exact tie. L=A exp(-(phiV-phiK)^2/tau). tau is a squared phase width.
For unwrapped differences in (-pi,pi), A follows the sign of the difference.
Outside that interval unit phases alias; this module does not unwrap them.

The CUDA implementation evaluates the equivalent sin(difference) directly to
avoid subtracting two rounded unit-vector products near a tie. It fuses the
channel-pair write and has an ordinary first-order backward through the Gaussian
envelope. It does not save a B,H,T,D,D tensor, sort phases, or truncate gradients.
Signed K,V activities remain separate from the signed learning-rate coefficient.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def unit_inner_gaussian_reference(q, k, v, pk, pv, tau=1., gain=1.):
    """Literal unit-phase inner products; float64 mathematical reference."""
    ck, sk, cv, sv = pk.cos(), pk.sin(), pv.cos(), pv.sin()
    imaginary = sv[..., :, None] * ck[..., None, :] - cv[..., :, None] * sk[..., None, :]
    delta = pv[..., :, None] - pk[..., None, :]
    window = gain * imaginary.sign() * torch.exp(-delta.square() / tau)
    g = (v[..., :, None] * k[..., None, :] * window).mean(-3)
    return q @ g.transpose(-1, -2)


def unit_phase_gaussian_reference(q, k, v, pk, pv, tau=1., gain=1.):
    """Equivalent relative-angle form, avoiding unit-vector cancellation."""
    delta = pv[..., :, None] - pk[..., None, :]
    window = gain * delta.sin().sign() * torch.exp(-delta.square() / tau)
    g = (v[..., :, None] * k[..., None, :] * window).mean(-3)
    return q @ g.transpose(-1, -2)


def gaussian_lag_reference(q, k, v, pk, pv, tau=1., gain=1.):
    """Same Gaussian, direct lag sign; equivalent within (-pi,pi)."""
    delta = pv[..., :, None] - pk[..., None, :]
    window = gain * delta.sign() * torch.exp(-delta.square() / tau)
    g = (v[..., :, None] * k[..., None, :] * window).mean(-3)
    return q @ g.transpose(-1, -2)


@triton.jit
def _window(delta, sk, ck, sv, cv, valid,
            TAU: tl.constexpr, GAIN: tl.constexpr, MODE: tl.constexpr):
    if MODE == 2:
        imaginary = sv[:, :, None] * ck[:, None, :] - cv[:, :, None] * sk[:, None, :]
        # This is an evaluation switch, not a softened sign or phase dead zone.
        # Near cancellation, the equivalent sin(delta) preserves the direction.
        uncertain = valid & (tl.abs(imaginary) < 1e-5) & (delta != 0.)
        count = tl.sum(tl.sum(tl.sum(uncertain.to(tl.int32), axis=2), axis=1), axis=0)
        if count > 0:
            imaginary = tl.where(uncertain, tl.sin(delta), imaginary)
        imaginary = tl.where(delta == 0., 0., imaginary)
    elif MODE == 1:
        imaginary = tl.sin(delta)
    else:
        imaginary = delta
    direction = tl.where(imaginary > 0., 1., tl.where(imaginary < 0., -1., 0.))
    return GAIN * direction * tl.exp(-delta * delta / TAU)


@triton.jit
def _write(K, V, PK, PV, SK, CK, SV, CV, G, T: tl.constexpr, D: tl.constexpr,
           TAU: tl.constexpr, GAIN: tl.constexpr, MODE: tl.constexpr,
           BT: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr):
    iv = tl.program_id(0) * BV + tl.arange(0, BV)
    ik = tl.program_id(1) * BK + tl.arange(0, BK)
    bh = tl.program_id(2)
    ot = tl.arange(0, BT)
    accum = tl.full((BV, BK), 0., tl.float32)
    for start in range(0, T, BT):
        it = start + ot
        ki = (bh * T + it[:, None]) * D + ik[None, :]
        vi = (bh * T + it[:, None]) * D + iv[None, :]
        kvalid = (it[:, None] < T) & (ik[None, :] < D)
        vvalid = (it[:, None] < T) & (iv[None, :] < D)
        k = tl.load(K + ki, kvalid, 0.)
        v = tl.load(V + vi, vvalid, 0.)
        pk = tl.load(PK + ki, kvalid, 0.)
        pv = tl.load(PV + vi, vvalid, 0.)
        delta = pv[:, :, None] - pk[:, None, :]
        if MODE == 2:
            sk, ck = tl.load(SK + ki, kvalid, 0.), tl.load(CK + ki, kvalid, 1.)
            sv, cv = tl.load(SV + vi, vvalid, 0.), tl.load(CV + vi, vvalid, 1.)
        else:
            sk, ck, sv, cv = 0., 1., 0., 1.
        valid = vvalid[:, :, None] & kvalid[:, None, :]
        window = _window(delta, sk, ck, sv, cv, valid, TAU, GAIN, MODE)
        accum += tl.sum(v[:, :, None] * k[:, None, :] * window, axis=0)
    gi = bh * D * D + iv[:, None] * D + ik[None, :]
    tl.store(G + gi, accum / T, (iv[:, None] < D) & (ik[None, :] < D))


@triton.jit
def _backward_v(DG, K, V, PK, PV, SK, CK, SV, CV, DV, DPV, T: tl.constexpr, D: tl.constexpr,
                TAU: tl.constexpr, GAIN: tl.constexpr, MODE: tl.constexpr,
                BT: tl.constexpr, BV: tl.constexpr, RD: tl.constexpr):
    it = tl.program_id(0) * BT + tl.arange(0, BT)
    iv = tl.program_id(1) * BV + tl.arange(0, BV)
    ik = tl.arange(0, RD)
    bh = tl.program_id(2)
    vi = (bh * T + it[:, None]) * D + iv[None, :]
    ki = (bh * T + it[:, None]) * D + ik[None, :]
    vvalid = (it[:, None] < T) & (iv[None, :] < D)
    kvalid = (it[:, None] < T) & (ik[None, :] < D)
    v, pv = tl.load(V + vi, vvalid, 0.), tl.load(PV + vi, vvalid, 0.)
    k, pk = tl.load(K + ki, kvalid, 0.), tl.load(PK + ki, kvalid, 0.)
    dg = tl.load(DG + bh * D * D + iv[:, None] * D + ik[None, :],
                 (iv[:, None] < D) & (ik[None, :] < D), 0.)
    delta = pv[:, :, None] - pk[:, None, :]
    if MODE == 2:
        sk, ck = tl.load(SK + ki, kvalid, 0.), tl.load(CK + ki, kvalid, 1.)
        sv, cv = tl.load(SV + vi, vvalid, 0.), tl.load(CV + vi, vvalid, 1.)
    else:
        sk, ck, sv, cv = 0., 1., 0., 1.
    valid = vvalid[:, :, None] & kvalid[:, None, :]
    mass = dg[None, :, :] * k[:, None, :] * _window(delta, sk, ck, sv, cv, valid, TAU, GAIN, MODE)
    dv = tl.sum(mass, axis=2) / T
    dpv = v * tl.sum(mass * (-2. * delta / TAU), axis=2) / T
    tl.store(DV + vi, dv, vvalid)
    tl.store(DPV + vi, dpv, vvalid)


@triton.jit
def _backward_k(DG, K, V, PK, PV, SK, CK, SV, CV, DK, DPK, T: tl.constexpr, D: tl.constexpr,
                TAU: tl.constexpr, GAIN: tl.constexpr, MODE: tl.constexpr,
                BT: tl.constexpr, BK: tl.constexpr, RD: tl.constexpr):
    it = tl.program_id(0) * BT + tl.arange(0, BT)
    ik = tl.program_id(1) * BK + tl.arange(0, BK)
    iv = tl.arange(0, RD)
    bh = tl.program_id(2)
    ki = (bh * T + it[:, None]) * D + ik[None, :]
    vi = (bh * T + it[:, None]) * D + iv[None, :]
    kvalid = (it[:, None] < T) & (ik[None, :] < D)
    vvalid = (it[:, None] < T) & (iv[None, :] < D)
    k, pk = tl.load(K + ki, kvalid, 0.), tl.load(PK + ki, kvalid, 0.)
    v, pv = tl.load(V + vi, vvalid, 0.), tl.load(PV + vi, vvalid, 0.)
    dg = tl.load(DG + bh * D * D + iv[:, None] * D + ik[None, :],
                 (iv[:, None] < D) & (ik[None, :] < D), 0.)
    delta = pv[:, :, None] - pk[:, None, :]
    if MODE == 2:
        sk, ck = tl.load(SK + ki, kvalid, 0.), tl.load(CK + ki, kvalid, 1.)
        sv, cv = tl.load(SV + vi, vvalid, 0.), tl.load(CV + vi, vvalid, 1.)
    else:
        sk, ck, sv, cv = 0., 1., 0., 1.
    valid = vvalid[:, :, None] & kvalid[:, None, :]
    mass = dg[None, :, :] * v[:, :, None] * _window(delta, sk, ck, sv, cv, valid, TAU, GAIN, MODE)
    dk = tl.sum(mass, axis=1) / T
    dpk = k * tl.sum(mass * (2. * delta / TAU), axis=1) / T
    tl.store(DK + ki, dk, kvalid)
    tl.store(DPK + ki, dpk, kvalid)


class _GaussianWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, pk, pv, sk, ck, sv, cv, tau, gain, mode):
        if not k.is_cuda or any(x.dtype != torch.float32 or x.device != k.device
                               for x in (k, v, pk, pv)):
            raise ValueError('The fused unit-phase operator requires CUDA FP32.')
        if not (k.shape == v.shape == pk.shape == pv.shape):
            raise ValueError('K,V,phiK,phiV must have the same B,H,T,D shape.')
        b, h, t, d = k.shape
        g = torch.empty((b, h, d, d), device=k.device, dtype=k.dtype)
        _write[(triton.cdiv(d, 8), triton.cdiv(d, 16), b * h)](
            k, v, pk, pv, sk, ck, sv, cv, g, t, d, tau, gain, mode, 64, 8, 16, num_warps=4)
        ctx.save_for_backward(k, v, pk, pv, sk, ck, sv, cv)
        ctx.tau, ctx.gain, ctx.mode = tau, gain, mode
        return g

    @staticmethod
    def backward(ctx, dg):
        k, v, pk, pv, sk, ck, sv, cv = ctx.saved_tensors
        b, h, t, d = k.shape
        dg = dg.contiguous()
        dk, dv, dpk, dpv = (torch.empty_like(x) for x in (k, v, pk, pv))
        rd = triton.next_power_of_2(d)
        _backward_v[(triton.cdiv(t, 8), triton.cdiv(d, 4), b * h)](
            dg, k, v, pk, pv, sk, ck, sv, cv, dv, dpv, t, d, ctx.tau, ctx.gain,
            ctx.mode, 8, 4, rd, num_warps=4)
        _backward_k[(triton.cdiv(t, 8), triton.cdiv(d, 4), b * h)](
            dg, k, v, pk, pv, sk, ck, sv, cv, dk, dpk, t, d, ctx.tau, ctx.gain,
            ctx.mode, 8, 4, rd, num_warps=4)
        # The direction has the ordinary sign gradient (zero). All phase
        # envelope gradients take the direct pk/pv route above.
        return dk, dv, dpk, dpv, None, None, None, None, None, None, None


def _current_g(k, v, pk, pv, tau, gain, mode):
    if tau <= 0. or gain <= 0.:
        raise ValueError('Gaussian squared width and learning-rate magnitude must be positive.')
    if mode == 2:
        sk, ck = pk.sin().contiguous(), pk.cos().contiguous()
        sv, cv = pv.sin().contiguous(), pv.cos().contiguous()
    else:
        unused = torch.empty(0, device=k.device, dtype=k.dtype)
        sk, ck, sv, cv = unused, unused, unused, unused
    return _GaussianWrite.apply(k.contiguous(), v.contiguous(), pk.contiguous(), pv.contiguous(),
                                sk, ck, sv, cv, tau, gain, mode)


def unit_phase_gaussian_write(k, v, pk, pv, tau=1., gain=1.):
    """Return current G[B,H,D,D] for the unit-direction Gaussian rule.

    This exposes the same fused write used by unit_phase_gaussian, allowing
    recurrent models to record G without computing a second write or storing
    a channel-pair tensor for every token. No activity/phasor norm is applied.
    """
    return _current_g(k, v, pk, pv, tau, gain, 1)


def _read(q, k, v, pk, pv, tau, gain, mode):
    return q @ _current_g(k, v, pk, pv, tau, gain, mode).transpose(-1, -2)


def unit_phase_gaussian(q, k, v, pk, pv, tau=1., gain=1.):
    """Fused current G and all-Q read: unit phase direction, Gaussian decay.

    Input phases are unwrapped and their pairwise differences must lie in
    (-pi,pi) for the direction to match temporal order. tau/gain are constants.
    CUDA FP32; first-order gradients for Q,K,V,phiK,phiV are all preserved.
    """
    return _read(q, k, v, pk, pv, tau, gain, 1)


def unit_phase_precomputed(q, k, v, pk, pv, tau=1., gain=1.):
    """Same unit-phase Gaussian with sin/cos prepared once per neuron.

    Ill-conditioned inner products are reevaluated using the equivalent
    sin(delta); this never smooths A or discards small phase differences.
    """
    return _read(q, k, v, pk, pv, tau, gain, 2)


def gaussian_lag_triton(q, k, v, pk, pv, tau=1., gain=1.):
    """Matched fused control using direct lag signs instead of unit phases."""
    return _read(q, k, v, pk, pv, tau, gain, 0)
