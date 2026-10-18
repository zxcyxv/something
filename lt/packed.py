"""Packed (padding-free) segments for ARC with the bands memory (prototype; lt.train does not use it yet).

The dense path runs every lane on the full 30x30 canvas plus the prefix (901 positions), although a training pair
needs only the crop that holds its grids and the EOS row/column (22% of the canvas on average). Here each lane keeps
only that crop: the tokens of all lanes are concatenated into one flat [N] sequence, and

    tokenwise ops (embedding, projections, RMSNorm, FFN, classifier, the band traces E_K, E_V)   on [N, ...]
    per-lane token sums  S_b = sum_{t in b} (v_t k_mix,t^T + v_mix,t k~_t^T),  Hebb_b = sum_{t in b} v_t k~_t^T
    per-lane read        read_t = q~_t (Hebb_b + W_b)^T                                      W [B, H, D, D]

The sums use lane-aligned chunks: every lane's tokens are padded to a multiple of `chunk` with dummy tokens, so the
tokens reshape to [Nc, chunk] with one lane per chunk; a chunk's outer products are one batched matmul [D,C] @ [C,D]
and a one-hot [B+1, Nc] matmul adds the chunks of a lane (and copies the lane's matrix back to its chunks for the
read). Dummy tokens have v = k~ = 0 in the sums, so they never touch a lane (their own hidden states are garbage and
never read). Whole dummy chunks (padding the total to a multiple of `bucket` tokens, which bounds the number of
distinct shapes torch.compile sees) belong to an extra lane B that is dropped.

RoPE takes each token's canvas coordinates (row, col), so the random translation of a training pair costs nothing
(the same angles as on the dense canvas); prefix tokens have coordinates (0, 0), i.e. angle 0 as in the dense path.

The model is the dense LT (same parameters, same block: u = h + sqrt(d) inj; RMSNorm(u + W_O read); RMSNorm(h' +
FFN(h'))). Difference from the dense path on real ARC data: the dense lanes also sum the pad cells outside the crop
(input token 1) into S and Hebb; packed lanes do not. With a crop that covers the whole dense grid the two are equal
(tests/test_packed.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .model import LT, rms_norm
from .train import log_stablemax


# ---------------------------------------------------------------------------- layout
@dataclass
class Pack:
    """Flat token layout of one batch of lanes (fixed for the `loops` segments of the lanes' puzzles)."""
    inputs: torch.Tensor          # [N] tokens (0 for dummies)
    labels: torch.Tensor          # [N] label tokens (1 = ignored: prefix, dummies and pad cells)
    lane: torch.Tensor            # [N] lane index, B for whole dummy chunks
    row: torch.Tensor             # [N] canvas row (0 for prefix and dummies), float
    col: torch.Tensor             # [N] canvas column
    prefix: torch.Tensor          # [N] bool: prefix token
    prefix_slot: torch.Tensor     # [N] long: index of the prefix token within the lane's prefix (0 elsewhere)
    valid: torch.Tensor           # [N] float: 1 real token (prefix or cell), 0 dummy
    chunk_lane: torch.Tensor      # [Nc] lane of each chunk (B for dummy chunks)
    onehot: torch.Tensor          # [B + 1, Nc] float: chunk membership
    src: torch.Tensor             # [N] long: lane * T + canvas position of a cell token, -1 otherwise
    puzzle_identifiers: torch.Tensor   # [B]
    chunk: int
    num_lanes: int
    canvas_len: int               # T of the dense layout (for unpack)

    @property
    def num_tokens(self):
        return self.inputs.shape[0]

    def to(self, device, float_dtype=None):
        out = {}
        for name, value in self.__dict__.items():
            if torch.is_tensor(value):
                value = value.to(device, non_blocking=True)
                if float_dtype is not None and value.is_floating_point():
                    value = value.to(float_dtype)
            out[name] = value
        return Pack(**out)


def crop_boxes(inputs, labels, width):
    """Per lane (r0, c0, h, w): bounding box of the non-pad cells (token > 1) of input and label, i.e. the grids and
    their EOS row/column. inputs, labels [B, width * width] tokens."""
    keep = ((inputs > 1) | (labels > 1)).reshape(len(inputs), width, width)
    rows, cols = keep.any(2), keep.any(1)
    boxes = []
    for b in range(len(inputs)):
        r = torch.nonzero(rows[b]).flatten()
        c = torch.nonzero(cols[b]).flatten()
        if len(r) == 0:
            boxes.append((0, 0, 1, 1))
        else:
            boxes.append((int(r[0]), int(c[0]), int(r[-1]) - int(r[0]) + 1, int(c[-1]) - int(c[0]) + 1))
    return boxes


def build_pack(batch, width, prefix_len=1, chunk=32, bucket=256, boxes=None):
    """Packed layout of a dense batch (lt.data: inputs, labels [B, width^2] tokens, puzzle_identifiers [B]).

    Lane b = prefix_len prefix tokens, then the cells of its crop box (row-major, canvas coordinates), padded with
    dummy tokens to a multiple of `chunk`; the total is padded with dummy chunks to a multiple of `bucket` tokens
    (bucket a multiple of chunk; 0 = no bucketing). boxes: per lane (r0, c0, h, w), default crop_boxes."""
    inputs, labels = batch['inputs'].cpu(), batch['labels'].cpu()
    B, T = inputs.shape
    if T != width * width:
        raise ValueError('inputs must be [B, width^2]')
    if bucket and bucket % chunk:
        raise ValueError('bucket must be a multiple of chunk')
    boxes = crop_boxes(inputs, labels, width) if boxes is None else boxes
    pos_l, lane_l, pre_l, slot_l, valid_l = [], [], [], [], []
    for b, (r0, c0, h, w) in enumerate(boxes):
        rr, cc = np.meshgrid(np.arange(r0, r0 + h), np.arange(c0, c0 + w), indexing='ij')
        pos = (rr * width + cc).reshape(-1)
        n = prefix_len + len(pos)
        n_pad = -(-n // chunk) * chunk
        pos_l.append(np.concatenate([np.full(prefix_len, -1), pos, np.full(n_pad - n, -1)]))
        lane_l.append(np.full(n_pad, b))
        pre_l.append(np.concatenate([np.ones(prefix_len, bool), np.zeros(n_pad - prefix_len, bool)]))
        slot_l.append(np.concatenate([np.arange(prefix_len), np.zeros(n_pad - prefix_len, np.int64)]))
        valid_l.append(np.concatenate([np.ones(n), np.zeros(n_pad - n)]))
    N = sum(len(x) for x in pos_l)
    total = -(-N // bucket) * bucket if bucket else N
    if total > N:
        pos_l.append(np.full(total - N, -1)); lane_l.append(np.full(total - N, B))
        pre_l.append(np.zeros(total - N, bool)); slot_l.append(np.zeros(total - N, np.int64))
        valid_l.append(np.zeros(total - N))
    pos = torch.from_numpy(np.concatenate(pos_l)).long()
    lane = torch.from_numpy(np.concatenate(lane_l)).long()
    cell = pos >= 0
    safe = pos.clamp_min(0)
    lane_c = lane.clamp_max(B - 1)
    tok_in = torch.where(cell, inputs[lane_c, safe], 0)
    tok_lab = torch.where(cell, labels[lane_c, safe], 1)
    chunk_lane = lane[::chunk].clone()
    onehot = F.one_hot(chunk_lane, B + 1).T.contiguous().double()
    return Pack(inputs=tok_in.int(), labels=tok_lab.int(), lane=lane,
                row=torch.where(cell, safe // width, 0).double(), col=torch.where(cell, safe % width, 0).double(),
                prefix=torch.from_numpy(np.concatenate(pre_l)), prefix_slot=torch.from_numpy(np.concatenate(slot_l)),
                valid=torch.from_numpy(np.concatenate(valid_l)), chunk_lane=chunk_lane, onehot=onehot,
                src=torch.where(cell, lane_c * T + safe, -1), puzzle_identifiers=batch['puzzle_identifiers'].cpu(),
                chunk=chunk, num_lanes=B, canvas_len=T)


def unpack(x, pack, fill=0.0):
    """Token values x [N, ...] -> dense [B, T, ...] on the canvas (cells outside the crop get `fill`; the prefix is
    dropped)."""
    out = x.new_full((pack.num_lanes * pack.canvas_len,) + x.shape[1:], fill)
    cell = pack.src >= 0
    out[pack.src[cell]] = x[cell]
    return out.view(pack.num_lanes, pack.canvas_len, *x.shape[1:])


# ---------------------------------------------------------------------------- model
@dataclass
class PackedCarry:
    hidden: torch.Tensor          # [N, d]
    key_trace: torch.Tensor       # [N, H, 2 R D]  (Re, Im) x R bands of E_K, the dense layout per token
    value_trace: torch.Tensor     # [N, H, 2 R D]
    stdp_w: torch.Tensor          # [B, H, D, D]
    steps: torch.Tensor           # [B]
    halted: torch.Tensor          # [B]


class PackedLT(nn.Module):
    """Packed forward of an LT in bands mode; holds the LT (shared parameters), adds none of its own."""
    def __init__(self, model: LT):
        super().__init__()
        cfg = model.config
        if not cfg.bands:
            raise ValueError('PackedLT implements the bands memory only')
        if cfg.task_film or cfg.task_film_up or cfg.task_lowrank or cfg.puzzle_emb_mode == 'film':
            raise ValueError('PackedLT: task_film / task_film_up / task_lowrank / film embedding are not implemented')
        self.model = model
        self.block_fn = None            # e.g. torch.compile(packed.block_step, dynamic=False)

    # -------------------------------------------------------------- parts
    def injection(self, pack):
        m = self.model
        inj = m.embed(pack.inputs.long())
        if m.puzzle_emb is None:
            return inj
        pe = m.puzzle_emb(pack.puzzle_identifiers)                                # [B, ndim]
        lane = pack.lane.clamp_max(pack.num_lanes - 1)
        if m.prefix_len:
            pe = F.pad(pe, (0, m.prefix_len * m.d - pe.shape[-1])).view(-1, m.prefix_len, m.d).to(inj.dtype)
            return torch.where(pack.prefix[:, None], pe[lane, pack.prefix_slot], inj)
        pe = F.pad(pe, (0, m.d - pe.shape[-1])).to(inj.dtype)
        return inj + pe[lane]

    def memory_step(self, q, k, v, state, aux):
        """bands_step on packed tokens: q, k, v [N, H, D]; state (E_K, E_V, W); aux (row, col, valid, onehot)."""
        m, cfg = self.model, self.model.config
        ek, ev, w = state
        row, col, valid, onehot = aux
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            th = m.block.rope_theta.to(dtype)                                      # [H, P, 2]
            angles = row.to(dtype)[:, None, None] * th[..., 0] + col.to(dtype)[:, None, None] * th[..., 1]   # [N,H,P]
            tables = angles.cos(), angles.sin()
            qr, kr = m.rope(unit(q.to(dtype)), tables), m.rope(unit(k.to(dtype)), tables)
            msk = valid.to(dtype)[:, None, None]
            v, kr = v.to(dtype) * msk, kr * msk                                    # dummies out of every sum
            N, H, D = v.shape
            R = cfg.bands
            zr, zi, gamma, ar, ai, br, bi = (x.to(dtype)[None, :, :, None] for x in m.block.bands())   # [1,H,R,1]
            ek_re, ek_im = ek.to(dtype).reshape(N, H, 2, R, D).unbind(2)
            ev_re, ev_im = ev.to(dtype).reshape(N, H, 2, R, D).unbind(2)
            ek_re, ek_im = zr * ek_re - zi * ek_im + gamma * kr.unsqueeze(2), zr * ek_im + zi * ek_re
            dv_re, dv_im = zr * ev_re - zi * ev_im, zr * ev_im + zi * ev_re
            k_mix = (ar * ek_re - ai * ek_im).sum(2)
            v_mix = (br * dv_re - bi * dv_im).sum(2)
            # per chunk [Nc, H, C, D]; X1 = v^T [k_mix | k~], X2 = v_mix^T k~
            nc = onehot.shape[1]
            ch = lambda x: x.reshape(nc, N // nc, H, D).transpose(1, 2)
            vt = ch(v).transpose(-1, -2)
            x1 = vt @ torch.cat((ch(k_mix), ch(kr)), -1)                           # [Nc, H, D, 2D]
            x2 = ch(v_mix).transpose(-1, -2) @ ch(kr)                              # [Nc, H, D, D]
            sums = (onehot.to(dtype) @ torch.cat((x1, x2), -1).reshape(nc, -1))[:-1].view(-1, H, D, 3 * D)
            hebb, s = sums[..., D:2 * D], sums[..., :D] + sums[..., 2 * D:]       # [B, H, D, D]
            w, mem = m.accumulate(w, s, dtype)
            mat = (hebb if cfg.hebb else s) + mem                                  # [B, H, D, D]
            mat = F.pad(mat, (0, 0, 0, 0, 0, 0, 0, 1))                             # lane B (dummy chunks): zeros
            mat_c = (onehot.to(dtype).T @ mat.reshape(mat.shape[0], -1)).view(nc, H, D, D)
            read = (ch(qr) @ mat_c.transpose(-1, -2)).transpose(1, 2).reshape(N, H, D)
            ev_re = dv_re + gamma * v.unsqueeze(2)
            ek = torch.stack((ek_re, ek_im), 2).reshape(N, H, -1)
            ev = torch.stack((ev_re, dv_im), 2).reshape(N, H, -1)
        return read, (ek, ev, w)

    def block_step(self, h, inj, state, aux):
        """LT.block_step on [N, d] tokens (same residual-then-RMSNorm structure)."""
        m = self.model
        L = m.block
        u = h + m.embed_scale * inj
        N = u.shape[0]
        heads = lambda x: x.reshape(N, m.H, m.dh)
        q, k, v = heads(L.q_proj(u)), heads(L.k_proj(u)), heads(L.v_proj(u))
        read, state = self.memory_step(q, k, v, state, aux)
        read = read.reshape(N, m.d).to(u.dtype)
        h = rms_norm(u + L.out_proj(read))
        g, a = L.ffn_gate_up(h).chunk(2, dim=-1)
        return rms_norm(h + L.ffn_down(F.silu(g) * a)), state

    def initial_carry(self, pack):
        m = self.model
        N, B, device = pack.num_tokens, pack.num_lanes, m.init_hidden.device
        dtype = torch.float64 if m.forward_dtype == torch.float64 else torch.float32
        trace = 2 * m.dh * m.config.bands
        return PackedCarry(hidden=m.init_hidden.expand(N, m.d).clone(),
                           key_trace=torch.zeros(N, m.H, trace, device=device, dtype=dtype),
                           value_trace=torch.zeros(N, m.H, trace, device=device, dtype=dtype),
                           stdp_w=torch.zeros(B, m.H, m.dh, m.dh, device=device, dtype=dtype),
                           steps=torch.zeros(B, dtype=torch.int32, device=device),
                           halted=torch.zeros(B, dtype=torch.bool, device=device))

    def forward(self, carry: Optional[PackedCarry], pack: Pack):
        """One segment on a packed batch. carry None = every lane starts its puzzle (the layout belongs to these
        puzzles; a new pack needs a new carry). Returns the new carry (detached) and logits [N, vocab]."""
        m, cfg = self.model, self.model.config
        if carry is None:
            carry = self.initial_carry(pack)
        h, state = carry.hidden, (carry.key_trace, carry.value_trace, carry.stdp_w)
        aux = (pack.row, pack.col, pack.valid, pack.onehot)
        amp = m._amp(h.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            inj = self.injection(pack)
            ckpt = cfg.act_ckpt and torch.is_grad_enabled()
            step = self.block_fn or self.block_step
            n_ckpt = cfg.blocks_per_seg if cfg.act_ckpt_blocks < 0 else cfg.act_ckpt_blocks
            for i in range(cfg.blocks_per_seg):
                if ckpt and i < n_ckpt:
                    h, state = torch.utils.checkpoint.checkpoint(step, h, inj, state, aux, use_reentrant=False)
                else:
                    h, state = step(h, inj, state, aux)
            logits = m.classifier(h)
        if amp:
            h, logits = h.float(), logits.float()
        steps = carry.steps + 1
        ek, ev, w = (x.detach() for x in state)
        return PackedCarry(hidden=h.detach(), key_trace=ek, value_trace=ev, stdp_w=w, steps=steps,
                           halted=steps >= cfg.loops), logits


# ---------------------------------------------------------------------------- loss
def packed_loss(logits, pack, halted):
    """lt.train.segment_loss with ignore0 on packed tokens: summed per-lane mean stablemax CE over labelled cells
    (label token != 1) and the metric sums (accuracy / exact over finished lanes)."""
    B = pack.num_lanes
    labels = pack.labels.long()
    lp = log_stablemax(logits.double())
    cell = -torch.gather(lp, -1, labels.unsqueeze(-1)).squeeze(-1)
    keep = labels != 1
    lane_sum = lambda x: torch.zeros(B + 1, dtype=x.dtype, device=x.device).index_add(0, pack.lane, x)[:B]
    count = lane_sum(keep.double()).clamp_min(1)
    loss = (lane_sum(cell * keep) / count).sum()
    with torch.no_grad():
        right = (logits.argmax(-1) == labels) & keep
        acc = lane_sum(right.double()) / count
        wrong = lane_sum((keep & ~right).double())
        metrics = dict(count=halted.sum(), accuracy=torch.where(halted, acc, 0).sum(),
                       exact_accuracy=(halted & (wrong == 0)).sum(), lm_loss=loss.detach())
    return loss, metrics
