"""Shared pieces of the interpretability tools: checkpoint loading, block-by-block inference, and the
order/latency split of the STDP term (docs/architecture.md, section 4).

Order/latency split. A pair k blocks apart has weight (1 - lam) lam^k sin(k omegaT + Delta a), Delta a = latency
difference c sin(phi_V) - c sin(phi_K). Reversing time, omegaT -> -omegaT, along the SAME trajectory flips the sign
of the order part only, so with shadow traces fed the tags tau - 2 n omegaT (same k^, v, phi):
    O = (Im S(+) - Im S(-)) / 2   order part: odd in time, sign = which block came first
    L = (Im S(+) + Im S(-)) / 2   latency part: sign = whose own clock leads
At a fixed point of the recurrence O = 0 and L = kappa SB, SB = sum_t sin(a_V - a_K) v k~^T,
kappa = (1 - lam)(1 - lam^2) / (1 - 2 lam cos omegaT + lam^2).
"""
from __future__ import annotations

import math
from pathlib import Path
import json

import torch

from ..model import LT, ModelConfig, StepRecord, convert_legacy_state_dict, stdp_step
from ..train import FORMAT

REPO = Path(__file__).resolve().parents[2]


def kappa(lam, omega_t):
    return (1 - lam) * (1 - lam ** 2) / (1 - 2 * lam * math.cos(omega_t) + lam ** 2)


def load_model(path, device, weights='ema', **overrides):
    """A trained model in eval mode (torch path, FP32 without autocast by default, as the published analyses). Reads this repository's checkpoints and the
    pre-refactor ones (run A of lt/research_phase_clock_stdp.py with its protocol.json, or the Kaggle model-B cell).
    Returns (model, cfg)."""
    path = Path(path)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    if ck.get('format') == FORMAT:
        sd = ck['model'] if weights == 'ema' else ck['raw_model']
    else:
        sd = ck['model_state_dict'] if weights == 'ema' else ck['raw_model_state_dict']
        if 'b_tag_bound' in cfg:                                            # Kaggle model-B cell
            cfg.update(tag_bound=cfg['b_tag_bound'], block_omega=cfg['b_block_omega'],
                       trace_lambda=cfg['b_trace_lambda'], outer_rho=cfg['b_outer_rho'])
        else:                                                               # research runner (run A)
            protocol = json.loads((path.parent / 'protocol.json').read_text())
            if protocol.get('clock') != 'bounded' or not protocol.get('freeze_lambda'):
                raise ValueError(f'{path}: not a bounded-clock, frozen-lambda run')
            cfg.update(tag_bound=protocol['bound'], block_omega=protocol['omega_t'],
                       trace_lambda=protocol['lambda_init'], outer_rho=0.0)
    cfg.update({'kernel': 'torch', 'amp': False, **overrides})
    if cfg.get('data_npz') and not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = None
    with torch.device(device):
        model = LT(ModelConfig.from_dict(dict(cfg, batch_size=1)))
    model.load_state_dict(convert_legacy_state_dict(sd, model.config))
    return model.eval(), cfg


@torch.no_grad()
def run_blocks(model, batch, blocks, each=None):
    """Inference one block at a time from a fresh carry (all lanes start at block 1, n = b - 1).
    Calls each(block, carry, logits) after every block; returns the argmax per block [blocks, B, T]."""
    cfg = model.config
    saved = cfg.blocks_per_seg, cfg.loops
    cfg.blocks_per_seg, cfg.loops = 1, blocks + 1
    try:
        carry, preds = model.initial_carry(batch), []
        for b in range(1, blocks + 1):
            carry, logits = model(carry, batch)
            if each is not None:
                each(b, carry, logits)
            preds.append(logits.argmax(-1))
    finally:
        cfg.blocks_per_seg, cfg.loops = saved
    return torch.stack(preds)


class OrderLatencyProbe:
    """Model.probe that measures O and L every block with time-reversed shadow traces.

    mode None keeps the read; 'order' reads Hebb + alpha O, 'latency' Hebb + alpha L, 'reversed' Hebb + alpha (L - O)
    (the time-reversed STDP term), 'none' Hebb only -- applied from block `start` on (1-based; lanes count blocks
    from their own puzzle start). self.last holds O, L and the record of the latest block."""
    def __init__(self, omega_t, lam, mode=None, start=1):
        self.omega_t, self.lam, self.mode, self.start = omega_t, lam, mode, start
        self.shadow, self.last = None, None

    def __call__(self, rec: StepRecord):
        fresh = (rec.count[:, :1, :1, :1] == 0)
        shadow = rec.traces if self.shadow is None else tuple(
            torch.where(fresh, torch.zeros_like(x), x) for x in self.shadow)
        shift = 2 * rec.count * self.omega_t
        _, im_s2, self.shadow = stdp_step(rec.k_unit, rec.kr, rec.v, rec.rope, rec.tag_k - shift, rec.tag_v - shift,
                                          self.lam, shadow)
        O, L = (rec.im_s - im_s2) / 2, (rec.im_s + im_s2) / 2
        self.last = dict(O=O, L=L, rec=rec)
        if self.mode is None:
            return None
        swap = {'order': O, 'latency': L, 'reversed': L - O, 'none': torch.zeros_like(O)}[self.mode]
        active = (rec.count[:, :1, :1, :1] + 1 >= self.start)
        return torch.where(active, swap, rec.im_s)


def latency_term(rec: StepRecord, omega_t):
    """SB = sum_t sin(a_V - a_K) v k~^T with a = tau - n omegaT (the same-block latency term without (1 - lam))."""
    from ..model import rep2
    a_k, a_v = rep2(rec.tag_k - rec.count * omega_t), rep2(rec.tag_v - rec.count * omega_t)
    return ((rec.v * a_v.sin()).transpose(-1, -2) @ (rec.kr * a_k.cos())
            - (rec.v * a_v.cos()).transpose(-1, -2) @ (rec.kr * a_k.sin()))


def fro(m):
    return m.flatten(-2).norm(dim=-1)


def suffix_block(flags):
    """flags [blocks, N] -> first block (1-based) from which flags stays true to the end; blocks + 1 if never."""
    blocks = flags.shape[0]
    suffix = flags.flip(0).to(torch.int8).cumprod(0).flip(0).bool()
    return torch.where(suffix[-1], suffix.float().argmax(0) + 1, torch.full_like(suffix[-1], blocks + 1, dtype=torch.float)).long()


def spearman(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    if a.numel() < 3:
        return float('nan')
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))
