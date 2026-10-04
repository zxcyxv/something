"""Exact phase-STDP write using one shared ordering per token.

The ordering depends on the whole current K phase vector, rather than fixed
independent per-neuron features. K is sorted once and every V phase is located
in that ordering. A shared prefix lookup then encodes strict LTP/LTD membership;
the write kernels never compare a K phase against a V phase. All Q rows read
the same G after the write. This is not an STDP-window approximation.

The custom backward differentiates the exponential envelopes while treating
ordering as locally constant, exactly as sign/abs in the direct implementation.
Exact phase ties contribute zero, including their ordinary autograd gradient.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def order_metadata(pk, pv):
    """K rank and strict prefix/suffix boundaries, once per observation."""
    phase, order = pk.detach().sort(-1)
    ranks = torch.arange(pk.shape[-1], device=pk.device, dtype=torch.int32)
    rank = torch.empty_like(order, dtype=torch.int32).scatter(
        -1, order, ranks.expand_as(order))
    left = torch.searchsorted(phase.contiguous(), pv.detach().contiguous(),
                              right=False, out_int32=True)
    right = torch.searchsorted(phase.contiguous(), pv.detach().contiguous(),
                               right=True, out_int32=True)
    # This small table depends only on D, not on any activity or phase value.
    prefix = (torch.arange(pk.shape[-1] + 1, device=pk.device)[:, None]
              > ranks[None, :]).to(torch.uint8).contiguous()
    return rank, left, right, prefix


def shared_order_reference(q, k, v, pk, pv):
    """Torch reference. Materializes channel-pair lookups; use small cases."""
    rank, left, right, prefix = order_metadata(pk, pv)
    positive = prefix[left[..., :, None], rank[..., None, :]].bool()
    negative = ~prefix[right[..., :, None], rank[..., None, :]].bool()
    vm, vp = v * torch.exp(-pv), v * torch.exp(pv)
    kp, km = k * torch.exp(pk), k * torch.exp(-pk)
    g = (torch.where(positive, vm[..., :, None] * kp[..., None, :], 0.)
         - torch.where(negative, vp[..., :, None] * km[..., None, :], 0.)).mean(-3)
    return q @ g.transpose(-1, -2)


@triton.jit
def _memberships(rank, left, right, PREFIX, valid,
                 D: tl.constexpr, ORDERED: tl.constexpr):
    if ORDERED:
        positive = tl.load(PREFIX + left[:, :, None] * D + rank[:, None, :],
                           valid, 0).to(tl.float32)
        before_or_equal = tl.load(PREFIX + right[:, :, None] * D + rank[:, None, :],
                                  valid, 1).to(tl.float32)
    else:
        # Matched control: identical kernels, with direct phase comparisons.
        positive = (left[:, :, None] > rank[:, None, :]).to(tl.float32)
        before_or_equal = (right[:, :, None] >= rank[:, None, :]).to(tl.float32)
    return positive, before_or_equal


@triton.jit
def _write_g(VM, VP, KP, KM, RANK, LEFT, RIGHT, PREFIX, G,
             T: tl.constexpr, D: tl.constexpr,
             BT: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
             ORDERED: tl.constexpr):
    iv = tl.program_id(0) * BV + tl.arange(0, BV)
    ik = tl.program_id(1) * BK + tl.arange(0, BK)
    bh = tl.program_id(2)
    ot = tl.arange(0, BT)
    accum = tl.full((BV, BK), 0., tl.float32)
    for start in range(0, T, BT):
        it = start + ot
        vi = (bh * T + it[:, None]) * D + iv[None, :]
        ki = (bh * T + it[:, None]) * D + ik[None, :]
        vm = tl.load(VM + vi, (it[:, None] < T) & (iv[None, :] < D), 0.)
        vp = tl.load(VP + vi, (it[:, None] < T) & (iv[None, :] < D), 0.)
        kp = tl.load(KP + ki, (it[:, None] < T) & (ik[None, :] < D), 0.)
        km = tl.load(KM + ki, (it[:, None] < T) & (ik[None, :] < D), 0.)
        rank = tl.load(RANK + ki, (it[:, None] < T) & (ik[None, :] < D), 0)
        left = tl.load(LEFT + vi, (it[:, None] < T) & (iv[None, :] < D), 0)
        right = tl.load(RIGHT + vi, (it[:, None] < T) & (iv[None, :] < D), 0)
        valid = ((it[:, None, None] < T) & (iv[None, :, None] < D)
                 & (ik[None, None, :] < D))
        positive, before_or_equal = _memberships(rank, left, right, PREFIX, valid, D, ORDERED)
        term = (vm[:, :, None] * kp[:, None, :] * positive
                - vp[:, :, None] * km[:, None, :] * (1. - before_or_equal))
        accum += tl.sum(term, axis=0)
    gi = bh * D * D + iv[:, None] * D + ik[None, :]
    tl.store(G + gi, accum / T, (iv[:, None] < D) & (ik[None, :] < D))


@triton.jit
def _backward_v(DG, KP, KM, RANK, LEFT, RIGHT, PREFIX, DVM, DVP,
                T: tl.constexpr, D: tl.constexpr,
                BT: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
                ORDERED: tl.constexpr):
    it = tl.program_id(0) * BT + tl.arange(0, BT)
    iv = tl.program_id(1) * BV + tl.arange(0, BV)
    bh = tl.program_id(2)
    vi = (bh * T + it[:, None]) * D + iv[None, :]
    vvalid = (it[:, None] < T) & (iv[None, :] < D)
    left = tl.load(LEFT + vi, vvalid, 0)
    right = tl.load(RIGHT + vi, vvalid, 0)
    plus = tl.full((BT, BV), 0., tl.float32)
    minus = tl.full((BT, BV), 0., tl.float32)
    for start in range(0, D, BK):
        ik = start + tl.arange(0, BK)
        ki = (bh * T + it[:, None]) * D + ik[None, :]
        kvalid = (it[:, None] < T) & (ik[None, :] < D)
        kp = tl.load(KP + ki, kvalid, 0.)
        km = tl.load(KM + ki, kvalid, 0.)
        rank = tl.load(RANK + ki, kvalid, 0)
        dg = tl.load(DG + bh * D * D + iv[:, None] * D + ik[None, :],
                     (iv[:, None] < D) & (ik[None, :] < D), 0.)
        valid = vvalid[:, :, None] & (ik[None, None, :] < D)
        positive, before_or_equal = _memberships(rank, left, right, PREFIX, valid, D, ORDERED)
        plus += tl.sum(dg[None, :, :] * kp[:, None, :] * positive, axis=2)
        minus -= tl.sum(dg[None, :, :] * km[:, None, :] * (1. - before_or_equal), axis=2)
    tl.store(DVM + vi, plus / T, vvalid)
    tl.store(DVP + vi, minus / T, vvalid)


@triton.jit
def _backward_k(DG, VM, VP, RANK, LEFT, RIGHT, PREFIX, DKP, DKM,
                T: tl.constexpr, D: tl.constexpr,
                BT: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
                ORDERED: tl.constexpr):
    it = tl.program_id(0) * BT + tl.arange(0, BT)
    ik = tl.program_id(1) * BK + tl.arange(0, BK)
    bh = tl.program_id(2)
    ki = (bh * T + it[:, None]) * D + ik[None, :]
    kvalid = (it[:, None] < T) & (ik[None, :] < D)
    rank = tl.load(RANK + ki, kvalid, 0)
    plus = tl.full((BT, BK), 0., tl.float32)
    minus = tl.full((BT, BK), 0., tl.float32)
    for start in range(0, D, BV):
        iv = start + tl.arange(0, BV)
        vi = (bh * T + it[:, None]) * D + iv[None, :]
        vvalid = (it[:, None] < T) & (iv[None, :] < D)
        vm = tl.load(VM + vi, vvalid, 0.)
        vp = tl.load(VP + vi, vvalid, 0.)
        left = tl.load(LEFT + vi, vvalid, 0)
        right = tl.load(RIGHT + vi, vvalid, 0)
        dg = tl.load(DG + bh * D * D + iv[:, None] * D + ik[None, :],
                     (iv[:, None] < D) & (ik[None, :] < D), 0.)
        valid = (it[:, None, None] < T) & (iv[None, :, None] < D) & (ik[None, None, :] < D)
        positive, before_or_equal = _memberships(rank, left, right, PREFIX, valid, D, ORDERED)
        plus += tl.sum(dg[None, :, :] * vm[:, :, None] * positive, axis=1)
        minus -= tl.sum(dg[None, :, :] * vp[:, :, None] * (1. - before_or_equal), axis=1)
    tl.store(DKP + ki, plus / T, kvalid)
    tl.store(DKM + ki, minus / T, kvalid)


class _ExponentialWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vm, vp, kp, km, rank, left, right, prefix, ordered):
        b, h, t, d = kp.shape
        g = torch.empty((b, h, d, d), device=kp.device, dtype=kp.dtype)
        _write_g[(triton.cdiv(d, 8), triton.cdiv(d, 16), b * h)](
            vm, vp, kp, km, rank, left, right, prefix, g,
            t, d, 32, 8, 16, ordered, num_warps=4)
        ctx.save_for_backward(vm, vp, kp, km, rank, left, right, prefix)
        ctx.ordered = ordered
        return g

    @staticmethod
    def backward(ctx, dg):
        vm, vp, kp, km, rank, left, right, prefix = ctx.saved_tensors
        b, h, t, d = kp.shape
        dg = dg.contiguous()
        dvm, dvp = torch.empty_like(vm), torch.empty_like(vp)
        dkp, dkm = torch.empty_like(kp), torch.empty_like(km)
        _backward_v[(triton.cdiv(t, 16), triton.cdiv(d, 8), b * h)](
            dg, kp, km, rank, left, right, prefix, dvm, dvp,
            t, d, 16, 8, 32, ctx.ordered, num_warps=4)
        _backward_k[(triton.cdiv(t, 16), triton.cdiv(d, 16), b * h)](
            dg, vm, vp, rank, left, right, prefix, dkp, dkm,
            t, d, 16, 16, 16, ctx.ordered, num_warps=4)
        return dvm, dvp, dkp, dkm, None, None, None, None, None


def shared_order_triton(q, k, v, pk, pv):
    """Exact current G and all-Q read, with a shared order and custom backward.

    CUDA FP32, contiguous B,H,T,D operands. Equal key/value dimensions are the
    current project setting. No model parameters or state are added.
    """
    rank, left, right, prefix = order_metadata(pk, pv)
    vm, vp = v * torch.exp(-pv), v * torch.exp(pv)
    kp, km = k * torch.exp(pk), k * torch.exp(-pk)
    g = _ExponentialWrite.apply(vm, vp, kp, km, rank, left, right, prefix, True)
    return q @ g.transpose(-1, -2)


def direct_triton(q, k, v, pk, pv):
    """Matched control: same fused write/backward, direct phase comparisons.

    This distinguishes gains from shared ordering from gains due to custom
    kernels and the custom backward. It is not the proposed order-based method.
    """
    vm, vp = v * torch.exp(-pv), v * torch.exp(pv)
    kp, km = k * torch.exp(pk), k * torch.exp(-pk)
    unused = torch.empty(0, dtype=torch.uint8, device=k.device)
    g = _ExponentialWrite.apply(vm, vp, kp, km, pk.contiguous(),
                               pv.contiguous(), pv.contiguous(), unused, False)
    return q @ g.transpose(-1, -2)
