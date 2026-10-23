"""Training harness (URM protocol) for the bounded phase-clock STDP model.

One optimizer step = one segment (8 blocks, full BPTT); hidden state, clock, traces (and W) carry across the 16
segments of a puzzle and detach at segment boundaries. A lane whose puzzle has run 16 segments takes the next
puzzle of the batch stream (URM lane replacement). Loss: stablemax cross entropy, mean over cells, every segment.
AdamATan2 (lr 1e-4 after a 2,000-step linear warmup, weight decay 1.0 on matrices, betas 0.9/0.95), SignSGD for the
puzzle embedding, EMA 0.999. Held-out evaluation (EMA weights, fresh carry, 16 segments) every eval_interval epochs.

    python -m lt.train --config configs/a_bounded.json --out runs/a              # model A (rho = 0)
    python -m lt.train --config configs/b_leaky.json --out runs/b --set max_steps=20000
    torchrun --nproc_per_node=2 -m lt.train --config ...                        # batch split across ranks
Rerunning with the same --out resumes from its latest checkpoint (exact: optimizer, EMA, carry and RNG included).
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import glob
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .data import (ArcContextDataset, ArcTrainDataset, SudokuTrainDataset, ctx_packed_length, eval_batches, load_arc,
                   load_arc_ctx, load_data, load_task_ids)
from .model import LT, Carry, ModelConfig

FORMAT = 'lt-bounded-phase-clock-stdp-v1'
DEFAULTS = dict(
    # model (lt.model.ModelConfig)
    hidden_size=832, num_heads=8, blocks_per_seg=8, loops=16, mlp_expansion=4.0, puzzle_emb_ndim=832, eps=1e-4,
    grid=9, num_grids=1, num_symbols=0, ignore_label0=False, mlp_inter=0, linear_attn=False, num_puzzle_identifiers=1, puzzle_emb_mode='add',
    task_film=False, task_film_up=False, task_lowrank=0, task_ids_as_identifier=True,      # multi-task data: False gives the model identifier 0 everywhere (per-task eval kept)
    tag_bound=0.14, block_omega=0.34, trace_lambda=0.5, outer_rho=0.0, outer_eta=0.0, outer_eta_learn=False,
    alpha_init=1.0, hebb=True, event_stdp=False, event_theta=0.0, event_even=False, kv_ema=False, kv_ema_current=False, exp_stdp=False, exp_hebb=False,
    bands=0, band_periods=(2.0, 6.0, 20.0, 64.0), band_coef_std=0.1, band_product=False, outer_eta_cap=False, band_bf16=False,
    rope='learned2d', rope_base=10000.0,
    kernel='auto', kernel_bf16=True, amp=True, act_ckpt=False, act_ckpt_blocks=-1, act=False, act_explore=0.1,
    fast_ffn=0, fast_ffn_eta=0.5, fast_ffn_decay=0.02,
    rule_syn=False, rule_inner=20, rule_theta=0.01, rule_ridge=0.1, rule_prior='mean',
    local_conv=False, frame_rope=False, ctx_packed=False, grad_blocks=0, rule_path=False,
    rule_prior_emb=0, rule_prior_dropout=0.5, rule_write_bias=-2.0, rule_theta_init=1e-3, rule_prior_pe=False, ctx_versions=False,
    # data
    data_npz=None, test_size=2048, num_aug=1000,
    ce='stablemax',                   # 'softmax': plain cross entropy (URM's is stablemax)
    decay_all=False,                  # weight decay on every parameter (default: matrices only, as URM)
    # optimization
    global_batch_size=128, epochs=50000, eval_interval=250, lr=1e-4, lr_warmup_steps=2000, weight_decay=1.0,
    lr_warmup_start=0.0,              # warmup from lr_warmup_start * lr (every optimizer's lr scaled alike)
    beta1=0.9, beta2=0.95, puzzle_emb_lr=1e-4, puzzle_emb_weight_decay=1.0, ema_rate=0.999, ema_puzzle_emb=False, seed=0,
    grad_accum=1,                     # micro-batches per optimizer step, each with its own lanes (like extra ranks)
    carry_offload=False,              # keep the micro-batches' carries in pinned CPU memory between their segments
    matmul_precision='highest', compile='auto', compile_scope='segment',
    # run control
    init_from=None, init_weights='ema',   # new run warm-started from a checkpoint's weights (new parameters keep their init)
    out_dir=None, resume_from=None, max_steps=None, max_hours=None, save_every_steps=10000, keep_last=25,
    save_on_eval=True, save_carry=True,   # save_carry False: checkpoints hold no carry; a resume starts fresh lanes
    save_final=True,                      # False: no checkpoint when the run ends (speed benchmarks)
    eval_batch_size=0,                    # puzzles per evaluation batch per process (0: global_batch_size / ranks)
    milestone_every=0, milestone_extrap_segs=128, milestone_extrap_n=512, log_every=250, stop_check_every=25,
    dataloader_workers=1)
# A checkpoint resumes only under the same values of these keys (plus the data fingerprint).
RECIPE_KEYS = ('hidden_size', 'num_heads', 'blocks_per_seg', 'loops', 'mlp_expansion', 'puzzle_emb_ndim', 'eps',
               'grid', 'num_grids', 'num_symbols', 'ignore_label0', 'mlp_inter', 'linear_attn', 'num_puzzle_identifiers', 'puzzle_emb_mode', 'task_ids_as_identifier', 'task_film', 'task_film_up', 'task_lowrank',
               'tag_bound', 'block_omega', 'trace_lambda', 'outer_rho', 'outer_eta', 'outer_eta_learn', 'alpha_init', 'hebb',
               'event_stdp', 'event_theta', 'event_even', 'kv_ema', 'kv_ema_current', 'exp_stdp', 'exp_hebb',
               'bands', 'band_periods', 'band_coef_std', 'band_product', 'outer_eta_cap', 'band_bf16', 'act', 'act_explore', 'grad_accum',
               'rope', 'rope_base',
               'fast_ffn', 'fast_ffn_eta', 'fast_ffn_decay', 'local_conv', 'frame_rope', 'ctx_packed', 'grad_blocks', 'rule_path', 'rule_prior_emb', 'rule_prior_dropout', 'rule_write_bias', 'rule_theta_init', 'rule_prior_pe', 'ctx_versions', 'init_from', 'init_weights',
               'num_aug', 'test_size', 'ce', 'decay_all',
               'global_batch_size', 'epochs', 'eval_interval', 'lr', 'lr_warmup_steps', 'lr_warmup_start', 'weight_decay', 'beta1',
               'beta2', 'puzzle_emb_lr', 'puzzle_emb_weight_decay', 'ema_rate', 'ema_puzzle_emb', 'seed',
               'data_fingerprint')
METRICS = ('count', 'accuracy', 'exact_accuracy', 'lm_loss', 'q_halt_loss', 'q_halt_accuracy', 'steps')
_STOP = False


# ---------------------------------------------------------------------------- loss
def log_stablemax(x, dim=-1):
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def segment_loss(logits, labels, halted, ignore0=False, softmax=False):
    """Summed per-puzzle mean stablemax CE (FP64) and the metric sums; accuracy counts finished puzzles only.
    ignore0: cells whose raw label is 0 (token 1 after lt.data's +1; in-context demonstrations) count in neither
    the loss nor the metrics."""
    lp = F.log_softmax(logits.double(), -1) if softmax else log_stablemax(logits.double())   # ce: 'softmax' | 'stablemax'
    cell = -torch.gather(lp, -1, labels.long().unsqueeze(-1)).squeeze(-1)          # [B, T]
    if ignore0:
        keep = labels != 1                               # raw label 0 -> token 1
        loss = ((cell * keep).sum(-1) / keep.sum(-1).clamp_min(1)).sum()
        with torch.no_grad():
            right = (logits.argmax(-1) == labels) & keep
            acc = right.sum(-1).double() / keep.sum(-1).clamp_min(1)
            metrics = dict(count=halted.sum(), accuracy=torch.where(halted, acc, 0).sum(),
                           exact_accuracy=(halted & (right | ~keep).all(-1)).sum(), lm_loss=loss.detach())
        return loss, metrics
    loss = cell.mean(-1).sum()
    with torch.no_grad():
        ok = logits.argmax(-1) == labels
        metrics = dict(count=halted.sum(), accuracy=torch.where(halted, ok.float().mean(-1), 0).sum(),
                       exact_accuracy=(halted & ok.all(-1)).sum(), lm_loss=loss.detach())
    return loss, metrics


class TrainStep(torch.nn.Module):
    """Model + loss in one module, so torch.compile sees the whole segment. With ACT (URM ACTLossHead) the loss adds
    0.5 * BCE(q_halt, puzzle solved) summed over lanes."""
    def __init__(self, model, ignore0=False, softmax=False):
        super().__init__()
        self.model = model
        self.ignore0 = ignore0
        self.softmax = softmax

    def forward(self, carry, batch):
        carry, logits = self.model(carry, batch)
        labels, halted = carry.data['labels'], carry.halted
        loss, metrics = segment_loss(logits, labels, halted, self.ignore0, self.softmax)
        zero = loss.detach() * 0
        metrics.update(q_halt_loss=zero, q_halt_accuracy=zero, steps=torch.where(halted, carry.steps, 0).sum())
        if self.model.config.act:
            q = self.model.q_halt_logits
            with torch.no_grad():
                ok = logits.argmax(-1) == labels
                solved = ((ok | (labels == 1)) if self.ignore0 else ok).all(-1)
            q_loss = F.binary_cross_entropy_with_logits(q, solved.to(q.dtype), reduction='sum')
            loss = loss + 0.5 * q_loss
            metrics.update(q_halt_loss=q_loss.detach(), q_halt_accuracy=(halted & ((q >= 0) == solved)).sum())
        return carry, loss, metrics


# ---------------------------------------------------------------------------- optimizers
class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g['betas']
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st['step'] += 1
                t, m, v = st['step'], st['m'], st['v']
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.mul_(1 - g['lr'] * g['weight_decay'])
                p.add_(torch.atan2(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt()), alpha=-g['lr'])


class SignSGDEmbedding(Optimizer):
    """URM sparse update of the puzzle embedding rows used in the batch (gathered across ranks). With gradient
    accumulation, stash() keeps each micro-batch's rows and gradients; step() then updates from all of them."""
    def __init__(self, embedding, world_size, lr=0.0, weight_decay=0.0):
        self.embedding = embedding
        self.pending = []
        super().__init__([embedding.weights, embedding.local_weights, embedding.local_ids],
                         dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def stash(self):
        e = self.embedding
        grad = e.local_weights.grad
        self.pending.append((torch.zeros_like(e.local_weights) if grad is None else grad.clone(), e.local_ids.clone()))
        e.local_weights.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        group, e = self.param_groups[0], self.embedding
        if self.pending:
            grad, ids = (torch.cat(x) for x in zip(*self.pending))
            self.pending = []
        else:
            grad, ids = e.local_weights.grad, e.local_ids
        if grad is None:
            grad = torch.zeros_like(e.local_weights)
        if group['world_size'] > 1:
            all_g = torch.empty((len(grad) * group['world_size'], grad.shape[1]), device=grad.device)
            all_i = torch.empty(len(ids) * group['world_size'], dtype=ids.dtype, device=ids.device)
            dist.all_gather_into_tensor(all_g, grad.contiguous())
            dist.all_gather_into_tensor(all_i, ids.contiguous())
            grad, ids = all_g, all_i
        unique, inverse = ids.unique(return_inverse=True)
        summed = torch.zeros((len(unique), grad.shape[1]), dtype=grad.dtype, device=grad.device)
        summed.scatter_add_(0, inverse[:, None].expand_as(grad), grad)
        p = e.weights[unique]
        p.mul_(1 - group['lr'] * group['weight_decay']).add_(summed.sign(), alpha=-group['lr'])
        e.weights[unique] = p


def create_optimizers(model, cfg, world_size):
    """Weight decay on the projection/FFN/embedding/classifier matrices; none on biases, alpha, RoPE angles and the
    event thresholds."""
    opts, lrs = [], []
    if model.puzzle_emb is not None:
        opts.append(SignSGDEmbedding(model.puzzle_emb, world_size, weight_decay=cfg['puzzle_emb_weight_decay']))
        lrs.append(cfg['puzzle_emb_lr'])
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = lambda n, p: cfg['decay_all'] or (p.ndim >= 2 and 'rope_theta' not in n and 'event_theta_raw' not in n and 'band_' not in n
                          and 'task_film' not in n and 'task_u' not in n and 'task_v' not in n)
    opts.append(AdamATan2([{'params': [p for n, p in named if not decay(n, p)], 'weight_decay': 0.0},
                           {'params': [p for n, p in named if decay(n, p)], 'weight_decay': cfg['weight_decay']}],
                          lr=0, betas=(cfg['beta1'], cfg['beta2'])))
    lrs.append(cfg['lr'])
    return opts, lrs


def lr_at(step, base_lr, warmup, start=0.0):
    return base_lr * (start + (1 - start) * step / max(1, warmup)) if step < warmup else base_lr


class EMA:
    """EMA of the trainable parameters; with_puzzle_emb also averages the SignSGD puzzle-embedding table (a buffer),
    so the evaluated body and the embedding it reads come from the same average."""
    def __init__(self, model, rate, with_puzzle_emb=False):
        self.rate = rate
        self.shadow = {n: p.detach().clone() for n, p in self._tensors(model, with_puzzle_emb)}
        self.with_puzzle_emb = with_puzzle_emb

    @staticmethod
    def _tensors(model, with_puzzle_emb):
        out = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if with_puzzle_emb and model.puzzle_emb is not None:
            out.append(('puzzle_emb.weights', model.puzzle_emb.weights))
        return out

    @torch.no_grad()
    def update(self, model):
        for n, p in self._tensors(model, self.with_puzzle_emb):
            self.shadow[n].mul_(self.rate).add_(p, alpha=1 - self.rate)

    def swap(self, model):
        """Context manager: EMA weights inside, raw weights restored after."""
        ema = self

        class _Swap:
            def __enter__(self):
                self.backup = {}
                with torch.no_grad():
                    for n, p in ema._tensors(model, ema.with_puzzle_emb):
                        if n in ema.shadow:
                            self.backup[n] = p.detach().clone()
                            p.copy_(ema.shadow[n])
                return model

            def __exit__(self, *exc):
                with torch.no_grad():
                    for n, p in ema._tensors(model, ema.with_puzzle_emb):
                        if n in self.backup:
                            p.copy_(self.backup[n])
                return False
        return _Swap()


# ---------------------------------------------------------------------------- checkpoints
_CKPT_RE = re.compile(r'step_(\d+)\.pt$')


def _cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def _map_carry(carry, fn):
    """Apply fn to every tensor of a Carry (data dict included; None fields stay None)."""
    def one(x):
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: fn(v) for k, v in x.items()}
        return fn(x)
    return Carry(**{f.name: one(getattr(carry, f.name)) for f in fields(Carry)})


def carry_cat(carries):
    """Micro-batch carries -> one carry of all their lanes (in order)."""
    def cat(xs):
        if xs[0] is None:
            return None
        if isinstance(xs[0], dict):
            return {k: torch.cat([x[k] for x in xs]) for k in xs[0]}
        return torch.cat(xs)
    return Carry(**{f.name: cat([getattr(c, f.name) for c in carries]) for f in fields(Carry)})


def carry_split(carry, k):
    n = len(carry.steps) // k
    return [_map_carry(carry, lambda x, i=i: x[i * n:(i + 1) * n].clone()) for i in range(k)]


class CarryBank:
    """The carries of the grad_accum micro-batches. offload: kept in pinned CPU memory between their segments; the
    next micro-batch's carry is uploaded and the finished one downloaded on side streams while the GPU computes."""
    def __init__(self, n, offload, device):
        self.n, self.device = n, device
        self.offload = offload and device.type == 'cuda'
        self.slots = [None] * n
        self.ready, self.saved = {}, {}
        if self.offload:
            self.up, self.down = torch.cuda.Stream(device), torch.cuda.Stream(device)

    def prefetch(self, i):
        if not self.offload or self.slots[i] is None or i in self.ready:
            return
        with torch.cuda.stream(self.up):
            if i in self.saved:
                self.up.wait_event(self.saved.pop(i))
            dev = _map_carry(self.slots[i], lambda t: t.to(self.device, non_blocking=True))
            event = torch.cuda.Event()
            event.record(self.up)
        self.ready[i] = (dev, event)

    def take(self, i):
        if not self.offload or self.slots[i] is None:
            return self.slots[i]
        self.prefetch(i)
        dev, event = self.ready.pop(i)
        main = torch.cuda.current_stream(self.device)
        main.wait_event(event)
        _map_carry(dev, lambda t: t.record_stream(main))
        return dev

    def put(self, i, carry):
        if not self.offload:
            self.slots[i] = carry
            return
        if self.slots[i] is None:
            self.slots[i] = _map_carry(carry, lambda t: torch.empty(t.shape, dtype=t.dtype, pin_memory=True))
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        self.down.wait_event(done)
        with torch.cuda.stream(self.down):
            def copy(pair):
                host, t = pair
                host.copy_(t, non_blocking=True)
                t.record_stream(self.down)
            for f in fields(Carry):
                host, t = getattr(self.slots[i], f.name), getattr(carry, f.name)
                if isinstance(t, dict):
                    for k in t:
                        copy((host[k], t[k]))
                elif t is not None:
                    copy((host, t))
            event = torch.cuda.Event()
            event.record(self.down)
        self.saved[i] = event

    def all(self):
        """Every lane's carry (one Carry, CPU) or None before the first step."""
        if self.slots[0] is None:
            return None
        if self.offload:
            torch.cuda.synchronize(self.device)
        return carry_cat([_map_carry(c, lambda t: t.detach().cpu()) for c in self.slots])

    def load(self, carry):
        """Restore from one Carry of all local lanes (device tensors)."""
        for i, c in enumerate(carry_split(carry, self.n)):
            self.put(i, c)


def _rng_state(device):
    return dict(torch=torch.random.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None)


def _restore_rng(state, device):
    torch.random.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if device.type == 'cuda' and state.get('cuda') is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def find_latest_checkpoint(path):
    if not path:
        return None
    if os.path.isfile(path):
        return str(path)
    found = [(int(m.group(1)), p) for p in glob.glob(os.path.join(str(path), 'step_*.pt'))
             if (m := _CKPT_RE.search(os.path.basename(p)))]
    return max(found)[1] if found else None


def save_checkpoint(out_dir, state, model, optimizers, ema, cfg, rank, ws, device, keep_last):
    """Every rank must call. Weights (EMA and raw), optimizers, EMA, per-rank carry and RNG; atomic replace."""
    carry = state['bank'].all() if cfg['save_carry'] else None
    local = dict(carry=None if carry is None else {f.name: _cpu(getattr(carry, f.name)) for f in fields(Carry)},
                 rng=_rng_state(device))
    rank_states = [local]
    if ws > 1:
        rank_states = [None] * ws if rank == 0 else None
        dist.gather_object(local, rank_states, dst=0)
    path = os.path.join(out_dir, f"step_{state['step']}.pt")
    error = [None]
    if rank == 0:
        try:
            os.makedirs(out_dir, exist_ok=True)
            raw = _cpu(model.state_dict())
            with ema.swap(model):
                ema_sd = _cpu(model.state_dict())
            ck = dict(format=FORMAT, step=state['step'], iter_id=state['iter_id'], batch_in_iter=state['batch_in_iter'],
                      world_size=ws, model=ema_sd, raw_model=raw, ema=_cpu(ema.shadow),
                      optimizers=[_cpu(o.state_dict()) for o in optimizers], rank_states=rank_states, cfg=dict(cfg))
            torch.save(ck, path + '.tmp')
            os.replace(path + '.tmp', path)
            if keep_last > 0:
                old = sorted((int(m.group(1)), p) for p in glob.glob(os.path.join(out_dir, 'step_*.pt'))
                             if (m := _CKPT_RE.search(os.path.basename(p))))
                for _, p in old[:-keep_last]:
                    os.remove(p)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if ws > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f'checkpoint save failed: {error[0]}')
    return path


def load_checkpoint(path, model, optimizers, ema, cfg, rank, ws, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)        # trusted files only (RNG/optimizer objects)
    if ck.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} training checkpoint')
    saved = lambda k: ck['cfg'].get(k, DEFAULTS.get(k))           # keys added later default to their old behaviour
    changed = {k: (saved(k), cfg.get(k)) for k in RECIPE_KEYS if saved(k) != cfg.get(k)}
    if changed:
        raise ValueError(f'resume config/data mismatch (use a new out_dir for a new experiment): {changed}')
    model.load_state_dict(ck['raw_model'])
    for opt, sd in zip(optimizers, ck['optimizers'], strict=True):
        opt.load_state_dict(sd)
        if isinstance(opt, SignSGDEmbedding):
            opt.param_groups[0]['world_size'] = ws
    ema.shadow = {n: v.to(device) for n, v in ck['ema'].items()}
    saved = [r['carry'] for r in ck['rank_states']]
    carry = None
    if saved[0] is not None:
        lbs = cfg['global_batch_size'] // ws

        def merge(parts):
            if parts[0] is None:
                return None
            if isinstance(parts[0], dict):
                return {k: merge([p[k] for p in parts]) for k in parts[0]}
            return torch.cat(parts)[rank * lbs:(rank + 1) * lbs].to(device).clone()
        carry = Carry(**{k: merge([c[k] for c in saved]) for k in saved[0]})
    if ws == ck['world_size']:
        _restore_rng(ck['rank_states'][rank]['rng'], device)
    elif rank == 0:
        print(f"[LT] carry resharded {ck['world_size']} -> {ws} ranks (not bitwise reproducible)", flush=True)
    return dict(step=int(ck['step']), iter_id=int(ck['iter_id']), batch_in_iter=int(ck['batch_in_iter']), carry=carry)


# ---------------------------------------------------------------------------- evaluation
def stop_requested(device, deadline):
    flag = torch.tensor(int(_STOP or time.monotonic() >= deadline), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


@torch.no_grad()
def run_segments(model, x, y, segs, gbs, rank, ws, device, deadline=float('inf'), ids=None, ignore0=False):
    """Fresh carry, `segs` segments per puzzle. Returns per-segment sums [3, segs] (cell accuracy, exact, churn),
    the count of puzzles and whether time ran out."""
    sums = torch.zeros(3, segs, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    loops, training = model.config.loops, model.training
    model.config.loops = segs + 1                              # no lane is replaced during the run
    model.eval()
    try:
        pack = (model.config.grid, model.config.num_grids) if model.config.ctx_packed else None
        for batch in eval_batches(x, y, gbs, rank, ws, ids, pack):
            if stop_requested(device, deadline):
                return sums, count, True
            if batch is None:
                continue
            b = {k: v.to(device) for k, v in batch.items()}
            carry, prev = model.initial_carry(b), None
            for s in range(segs):
                carry, logits = model(carry, b)
                pred = logits.argmax(-1)
                ok = pred == b['labels']
                if ignore0:
                    keep = b['labels'] != 1                  # raw label 0 -> token 1
                    sums[0, s] += ((ok & keep).sum(-1).double() / keep.sum(-1).clamp_min(1)).sum()
                    sums[1, s] += (ok | ~keep).all(-1).sum()
                else:
                    sums[0, s] += ok.double().mean(-1).sum()
                    sums[1, s] += ok.all(-1).sum()
                if prev is not None:
                    sums[2, s] += (pred != prev).double().mean(-1).sum()
                prev = pred
            count += len(b['labels'])
    finally:
        model.config.loops = loops
        model.train(training)
    return sums, count, False


def eval_bs(cfg, ws):
    """Global evaluation batch (run_segments splits it across ranks)."""
    return cfg['eval_batch_size'] * ws if cfg['eval_batch_size'] else cfg['global_batch_size']


def evaluate(model, ema, x, y, cfg, rank, ws, device, step, deadline=float('inf'), ids=None, group_ids=None, tag=''):
    with ema.swap(model):
        sums, count, partial = run_segments(model, x, y, cfg['loops'], eval_bs(cfg, ws), rank, ws, device,
                                            deadline, ids, cfg['ignore_label0'])
        groups = ids if group_ids is None else group_ids
        if groups is not None and not rank and len(np.unique(groups)) <= 16:   # per-task results (few tasks; one process)
            parts = []
            for t in np.unique(groups).tolist():
                sel = groups == t
                s_t, c_t, _ = run_segments(model, x[sel], y[sel], cfg['loops'], eval_bs(cfg, 1), 0, 1, device,
                                           deadline, ids[sel], cfg['ignore_label0'])
                parts.append(f'task {t} acc {float(s_t[0, -1]) / max(int(c_t), 1):.4f} exact {int(s_t[1, -1])}/{int(c_t)}')
            print(f'[TASK{tag}] step {step} ' + ' | '.join(parts), flush=True)
    if ws > 1:
        dist.reduce(sums, dst=0)
        dist.reduce(count, dst=0)
    if rank:
        return None
    n = max(int(count), 1)
    acc, exact = float(sums[0, -1]) / n, int(sums[1, -1])
    print(f'[EVAL{tag}] step {step} acc {acc:.4f} exact {exact}/{int(count)}' + (' [partial]' if partial else ''), flush=True)
    if tag:
        return dict(step=step, accuracy=acc, exact=exact, n=int(count), partial=partial)
    if model.config.outer_eta_learn:
        with ema.swap(model):
            eta_ema = torch.sigmoid(model.block.eta_raw.detach()).tolist()
        eta_raw = torch.sigmoid(model.block.eta_raw.detach()).tolist()
        fmt = lambda v: '[' + ' '.join(f'{e:.4f}' for e in v) + ']'
        print(f'[ETA] step {step} mean {sum(eta_ema) / len(eta_ema):.4f} ema {fmt(eta_ema)} raw {fmt(eta_raw)}', flush=True)
    if model.config.exp_stdp:
        with ema.swap(model):
            lp, lm, ap, am = (x.detach().tolist() for x in model.block.window())
        fmt = lambda v: '[' + ' '.join(f'{e:.3f}' for e in v) + ']'
        print(f'[WIN] step {step} (ema) lam+ {fmt(lp)} lam- {fmt(lm)} A+ {fmt(ap)} A- {fmt(am)}', flush=True)
    if model.config.bands:
        with ema.swap(model):
            l0, area, period, efold = (x.tolist() for x in model.block.band_summary())
        fmt = lambda v: '[' + ' '.join(f'{e:.3f}' for e in v) + ']'
        print(f'[BAND] step {step} (ema) l0 {fmt(l0)} area {fmt(area)} period {fmt(period)} efold {fmt(efold)}',
              flush=True)
    return dict(step=step, accuracy=acc, exact=exact, n=int(count), partial=partial)


def extrapolate(model, ema, x, y, cfg, rank, ws, device, step, segs, out_json, deadline=float('inf'), ids=None):
    """Segments beyond the training horizon (milestones): per-segment exact count on the first milestone_extrap_n
    held-out puzzles and the error-reduction rate (best - seg16) / (n - seg16)."""
    x, y = x[:cfg['milestone_extrap_n']], y[:cfg['milestone_extrap_n']]
    ids = None if ids is None else ids[:cfg['milestone_extrap_n']]
    with ema.swap(model):
        sums, count, partial = run_segments(model, x, y, segs, eval_bs(cfg, ws), rank, ws, device, deadline, ids,
                                            cfg['ignore_label0'])
    if ws > 1:
        dist.reduce(sums, dst=0)
        dist.reduce(count, dst=0)
    if rank:
        return None
    n = int(count)
    exact = [int(e) for e in sums[1].tolist()]
    train_seg = exact[cfg['loops'] - 1]
    best = max(exact)
    result = dict(step=step, n=n, segs=segs, partial=partial, exact=exact,
                  accuracy=(sums[0] / max(n, 1)).tolist(), churn=(sums[2] / max(n, 1)).tolist(),
                  train_segment=train_seg, best=best, best_segment=exact.index(best) + 1,
                  error_reduction=(best - train_seg) / max(n - train_seg, 1))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=1) + '\n')
    print(f"[EXTRAP] step {step}: seg{cfg['loops']} {train_seg}/{n} -> best {best} (seg {result['best_segment']}), "
          f"error reduction {100 * result['error_reduction']:.1f}%", flush=True)
    return result


# ---------------------------------------------------------------------------- training
def _allreduce_grads(model, ws):
    if ws == 1:
        return
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat)                                          # SUM: the loss is divided by the GLOBAL batch
    off = 0
    for p in params:
        p.grad.copy_(flat[off:off + p.numel()].view_as(p))
        off += p.numel()


def _check_finite(model, loss, ws, pending=()):
    flags = [torch.isfinite(loss.detach()).all()]
    flags += [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    pe = model.puzzle_emb
    if pe is not None and pe.local_weights.grad is not None:
        flags.append(torch.isfinite(pe.local_weights.grad).all())
    flags += [torch.isfinite(g).all() for g, _ in pending]
    bad = (~torch.stack(flags).all()).to(torch.int32)
    if ws > 1:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError('nonfinite loss/gradient before the optimizer step; last checkpoint left intact')


def train_step(step_fn, model, state, batch, cfg, optimizers, lrs, rank, ws, device):
    """One optimizer step: grad_accum micro-batches (row slices of the local batch), each one segment on its own
    lanes; the loss is divided by the global batch, so the gradient is that of one batch of all lanes."""
    bank, k = state['bank'], cfg['grad_accum']
    pe_opt = next((o for o in optimizers if isinstance(o, SignSGDEmbedding)), None)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    mb = len(batch['inputs']) // k
    total, vals = 0.0, None
    for i in range(k):
        micro = {key: v[i * mb:(i + 1) * mb].to(device, non_blocking=True) for key, v in batch.items()}
        carry = bank.take(i)
        if carry is None:
            carry = model.initial_carry(micro)
        if i + 1 < k:
            bank.prefetch(i + 1)
        carry, loss, metrics = step_fn(carry, micro)
        (loss / cfg['global_batch_size']).backward()
        if pe_opt is not None and k > 1:
            pe_opt.stash()
        bank.put(i, carry)
        total = total + loss.detach()
        v = torch.stack([metrics[key].double() for key in METRICS])
        vals = v if vals is None else vals + v
        if i == 0:
            state['segment'] = carry.steps.max()
    bank.prefetch(0)
    _allreduce_grads(model, ws)
    _check_finite(model, total, ws, pe_opt.pending if pe_opt is not None else ())
    lr = lr_at(state['step'], cfg['lr'], cfg['lr_warmup_steps'], cfg['lr_warmup_start'])
    for opt, base_lr in zip(optimizers, lrs):
        for group in opt.param_groups:
            group['lr'] = lr_at(state['step'], base_lr, cfg['lr_warmup_steps'], cfg['lr_warmup_start'])
        opt.step()
        opt.zero_grad(set_to_none=True)
    model.block.project_eta_(model.config)
    state['step'] += 1
    if ws > 1:
        dist.reduce(vals, dst=0)
    if rank:
        return None
    d = dict(zip(METRICS, vals.tolist()))
    n = max(d['count'], 1)
    out = dict(lm_loss=d['lm_loss'] / cfg['global_batch_size'], accuracy=d['accuracy'] / n,
               exact_accuracy=d['exact_accuracy'] / n, count=d['count'], lr=lr)
    if cfg['act']:
        out.update(q_halt_loss=d['q_halt_loss'] / cfg['global_batch_size'], q_halt_accuracy=d['q_halt_accuracy'] / n,
                   steps=d['steps'] / n)
    return out


def _signal_stop(signum, frame):
    global _STOP
    _STOP = True


def init_distributed():
    ws, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    if torch.cuda.is_available():
        local = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
    else:
        device = torch.device('cpu')
    if ws > 1:
        from datetime import timedelta
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=60))
    return rank, ws, device


def validate(cfg):
    unknown = set(cfg) - set(DEFAULTS) - {'data_fingerprint'}
    if unknown:
        raise ValueError(f'unknown config keys: {sorted(unknown)}')
    if cfg['epochs'] <= 0 or cfg['eval_interval'] <= 0 or cfg['epochs'] % cfg['eval_interval']:
        raise ValueError('epochs must be a positive multiple of eval_interval (both in epochs)')
    if cfg['compile_scope'] not in ('segment', 'block'):
        raise ValueError('compile_scope must be segment or block')
    if cfg['dataloader_workers'] not in (0, 1):
        raise ValueError('dataloader_workers must be 0 or 1')
    if not cfg['out_dir']:
        raise ValueError('out_dir is required')
    if cfg['init_weights'] not in ('ema', 'raw'):
        raise ValueError('init_weights must be ema or raw')
    if cfg['grad_accum'] < 1:
        raise ValueError('grad_accum >= 1')
    if cfg['act'] and cfg['compile'] and cfg['compile_scope'] != 'block':
        raise ValueError('act reads q_halt outside the compiled block: use compile_scope block')


def main(cfg):
    global _STOP
    _STOP = False
    cfg = dict(DEFAULTS, **cfg)
    validate(cfg)
    deadline = float('inf') if cfg['max_hours'] is None else time.monotonic() + cfg['max_hours'] * 3600
    rank, ws, device = init_distributed()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal_stop)
    torch.set_num_threads(1 if cfg['ctx_packed'] else 2)   # packed: the pin-memory thread sets 1; a flip would recompile
    torch.set_float32_matmul_precision(cfg['matmul_precision'])
    torch.manual_seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    random.seed(cfg['seed'] + rank)
    gbs = cfg['global_batch_size']
    if gbs % (ws * cfg['grad_accum']):
        raise ValueError('global_batch_size must be divisible by ranks * grad_accum')
    tr_x, tr_y, te_x, te_y, path, fingerprint = load_data(cfg['data_npz'], cfg['test_size'])
    tr_ids, te_ids = load_task_ids(path, cfg['test_size'])
    arc = load_arc(path)
    ctx = load_arc_ctx(path)                             # in-context ARC: episodes drawn on the fly, no identifiers
    if arc is not None:                                  # ARC: identifiers are task * (num_aug + 1) + version; test v = 0
        te_ids = arc[2][:cfg['test_size']].astype(np.int64) * (cfg['num_aug'] + 1)
        if cfg['num_puzzle_identifiers'] < arc[3] * (cfg['num_aug'] + 1):
            raise ValueError(f"ARC needs num_puzzle_identifiers >= {arc[3] * (cfg['num_aug'] + 1)}")
    te_noid = None
    if ctx is not None and cfg['ctx_versions']:          # in-context ARC, URM identifiers task * (num_aug + 1) + v
        with np.load(path, allow_pickle=False) as z:
            te_ids = z['ctx_test_task'][:len(te_x)].astype(np.int64) * (cfg['num_aug'] + 1)
        need = (int(ctx['task'].max()) + 1) * (cfg['num_aug'] + 1)
        if cfg['num_puzzle_identifiers'] < need:
            raise ValueError(f'ctx_versions needs num_puzzle_identifiers >= {need}')
    prior_ids = ctx is not None and cfg['rule_prior_emb'] and not cfg['ctx_versions']
    if prior_ids:                                        # identifier task + 1 where the task was trained, else 0
        with np.load(path, allow_pickle=False) as z:
            et = z['ctx_test_task'][:len(te_x)].astype(np.int64)
        trained = np.zeros(int(ctx['task'].max()) + 1, bool)
        trained[np.unique(ctx['task'][ctx['trainable']])] = True
        te_ids_prior = np.where(trained[et], et + 1, 0)
    if ctx is not None and cfg['ctx_packed']:            # evaluation batches of similar packed lengths
        o = np.argsort(ctx_packed_length(te_x, cfg['grid'], cfg['num_grids']), kind='stable')
        te_x, te_y = te_x[o], te_y[o]
        te_ids = None if te_ids is None else te_ids[o]
        if prior_ids:
            te_ids_prior = te_ids_prior[o]
    task_te = None if cfg['ctx_versions'] else te_ids    # per-task grouping of the evaluation
    if tr_ids is not None and not cfg['task_ids_as_identifier']:
        tr_ids, te_ids = np.zeros_like(tr_ids), np.zeros_like(te_ids)
    if prior_ids:                                        # [EVAL] with the task priors, [EVAL-NOID] without
        te_noid = np.zeros(len(te_x), np.int64)
        te_ids = te_ids_prior
    if tr_ids is not None and int(max(tr_ids.max(), te_ids.max())) >= cfg['num_puzzle_identifiers']:
        raise ValueError('num_puzzle_identifiers must exceed every task identifier in the data')
    cfg.update(data_npz=path, data_fingerprint=fingerprint)
    steps_per_iter = cfg['eval_interval'] * len(tr_x) // gbs
    total_iters = cfg['epochs'] // cfg['eval_interval']
    stop_at = steps_per_iter * total_iters if cfg['max_steps'] is None else min(cfg['max_steps'], steps_per_iter * total_iters)
    if cfg['compile'] == 'auto':
        cfg['compile'] = device.type == 'cuda' and torch.cuda.get_device_capability(device) >= (7, 0)
    if tr_x.shape[-1] != cfg['grid'] or tr_x[0].size != cfg['num_grids'] * cfg['grid'] ** 2:
        raise ValueError(f"data boards are {tr_x.shape[-1]}x{tr_x.shape[-1]} but grid = {cfg['grid']}")
    with torch.device(device):
        model = LT(ModelConfig.from_dict(dict(cfg, batch_size=gbs // ws // cfg['grad_accum'],
                                              seq_len=cfg['num_grids'] * cfg['grid'] ** 2,
                                              vocab_size=(cfg['num_symbols'] or cfg['grid']) + 2)))
    model.train()
    if ws > 1:
        with torch.no_grad():
            for p in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(p, src=0)
    if cfg['init_from'] and not find_latest_checkpoint(cfg['resume_from'] or cfg['out_dir']):
        ck = torch.load(cfg['init_from'], map_location='cpu', weights_only=False)
        sd = ck['model' if cfg['init_weights'] == 'ema' else 'raw_model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f'init_from: unexpected keys {unexpected}')
        if rank == 0:
            print(f"[LT] init_from {cfg['init_from']} ({cfg['init_weights']} weights, step {ck['step']}); "
                  f"new parameters: {missing}", flush=True)
        del ck, sd
    optimizers, lrs = create_optimizers(model, cfg, ws)
    ema = EMA(model, cfg['ema_rate'], cfg['ema_puzzle_emb'])
    out_dir = os.path.abspath(os.path.expanduser(str(cfg['out_dir'])))
    cfg['out_dir'] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt = find_latest_checkpoint(cfg['resume_from'] or out_dir)
    if cfg['resume_from'] and not ckpt:
        raise FileNotFoundError(f"resume_from has no checkpoint: {cfg['resume_from']}")
    state = (load_checkpoint(ckpt, model, optimizers, ema, cfg, rank, ws, device) if ckpt
             else dict(step=0, iter_id=0, batch_in_iter=0, carry=None))
    carry = state.pop('carry')
    state['bank'] = CarryBank(cfg['grad_accum'], cfg['carry_offload'], device)
    if carry is not None:
        state['bank'].load(carry)
    elif ckpt and rank == 0:
        print('[LT] the checkpoint holds no carry: every lane starts a fresh puzzle', flush=True)
    if state['step'] != state['iter_id'] * steps_per_iter + state['batch_in_iter']:
        raise ValueError('checkpoint cursor/step inconsistency')
    if rank == 0:
        Path(out_dir, 'config.json').write_text(json.dumps(dict(cfg, format=FORMAT), indent=2) + '\n')
    if cfg['compile']:
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
        if cfg['ctx_packed']:                          # one compiled block per packed length (multiples of 128)
            import torch._dynamo.config as dc
            dc.recompile_limit = max(dc.recompile_limit, 64)
            dc.accumulated_recompile_limit = max(dc.accumulated_recompile_limit, 256)
    step_fn = TrainStep(model, cfg['ignore_label0'], cfg['ce'] == 'softmax')
    if cfg['compile'] and cfg['compile_scope'] == 'block':
        # One compiled block reused by every block of the segment; act_ckpt then checkpoints around it, so the
        # activation memory is that of eager checkpointing while the block runs as fused kernels.
        model.block_fn = torch.compile(model.block_step, dynamic=False)
    elif cfg['compile']:
        step_fn = torch.compile(step_fn, dynamic=False)
    if rank == 0:
        mc = model.config
        print(f"[LT] {FORMAT} torch={torch.__version__} device={device} ranks={ws} "
              f"params={sum(p.numel() for p in model.parameters()):,} amp={model._amp(device)} "
              f"kernel={'triton' if model.use_kernel(device, torch.float32) or model.use_band_kernel(device, torch.float32) else 'torch'} compile={cfg['compile']} scope={cfg['compile_scope']}", flush=True)
        print(f"[LT] tau = {mc.tag_bound:g} sin(phi) + n {mc.block_omega:g}; lambda {mc.trace_lambda:g}; "
              f"rho {mc.outer_rho:g}; eta {mc.outer_eta:g}; alpha_init {mc.alpha_init:g}; "
              + (f"EVENT-STDP theta {mc.event_theta:g} even {mc.event_even}; " if mc.event_stdp else '')
              + (('KV-EMA + current Hebb (gain 1); ' if mc.kv_ema_current else 'KV-EMA (plain linear KV, no clock/STDP); ') if mc.kv_ema else '')
              + (('exp-STDP + Hebb (B-EMA form); ' if mc.exp_hebb else 'exp-STDP (scalar channels, exponential pair window, L(0) = 0); ') if mc.exp_stdp else '')
              + (f"BANDS{' PRODUCT' if mc.band_product else ''}{'' if mc.hebb else ' single window, no tonic Hebb'} R={mc.bands} periods {list(mc.band_periods)} coef_std {mc.band_coef_std:g} "
                 + ('M = sum_r gamma_r Re(c_r sum_t E_V,r conj(E_K,r)^T): per-band EMA of S at rate 1 - |z_r|^2, no W/eta; '
                    if mc.band_product else f"eta_learn {mc.outer_eta_learn} cap {mc.outer_eta_cap}; ") if mc.bands else '')
              + f"stop_at {stop_at}; "
              + (f"RESUME {ckpt} step {state['step']}" if ckpt else 'NEW RUN') + f'; out {out_dir}', flush=True)
    if ctx is not None:
        ds = ArcContextDataset(ctx, seed=cfg['seed'], global_batch_size=gbs, rank=rank, world_size=ws,
                               epochs_per_iter=cfg['eval_interval'], start_iter=state['iter_id'],
                               total_iters=total_iters, skip_batches=state['batch_in_iter'],
                               packed=cfg['ctx_packed'], task_ids=bool(prior_ids),
                               id_dropout=cfg['rule_prior_dropout'],
                               num_aug=cfg['num_aug'] if cfg['ctx_versions'] else None)
    elif arc is not None:
        ds = ArcTrainDataset(tr_x, tr_y, arc[0], arc[1], seed=cfg['seed'], num_aug=cfg['num_aug'], global_batch_size=gbs,
                             rank=rank, world_size=ws, epochs_per_iter=cfg['eval_interval'], start_iter=state['iter_id'],
                             total_iters=total_iters, skip_batches=state['batch_in_iter'])
    else:
        ds = SudokuTrainDataset(tr_x, tr_y, seed=cfg['seed'], num_aug=cfg['num_aug'], global_batch_size=gbs, rank=rank,
                                world_size=ws, epochs_per_iter=cfg['eval_interval'], start_iter=state['iter_id'],
                                total_iters=total_iters, skip_batches=state['batch_in_iter'], ids=tr_ids)
    kwargs = dict(batch_size=None, num_workers=cfg['dataloader_workers'], pin_memory=device.type == 'cuda',
                  generator=torch.Generator().manual_seed(cfg['seed'] + rank))
    if cfg['dataloader_workers']:
        kwargs.update(prefetch_factor=4, multiprocessing_context='spawn')
    loader = DataLoader(ds, **kwargs)
    save = lambda d=out_dir, keep=cfg['keep_last']: save_checkpoint(d, state, model, optimizers, ema, cfg, rank, ws,
                                                                    device, keep)
    began, last_eval, last_full = time.monotonic(), -1, None
    stopped = stop_requested(device, deadline)
    log = Path(out_dir, 'train.jsonl')
    cuda = device.type == 'cuda'

    def run_eval(step, ids=te_ids, tag=''):                    # eval.jsonl: one record per evaluation (rank 0)
        rec = evaluate(model, ema, te_x, te_y, cfg, rank, ws, device, step, deadline, ids, task_te, tag=tag)
        if rec is not None:
            with Path(out_dir, 'eval.jsonl').open('a') as f:
                f.write(json.dumps(dict(rec, tag=tag, elapsed=time.monotonic() - began)) + '\n')
    try:
        if not stopped and state['step'] < stop_at:
            for it, batch in loader:
                if int(it) != state['iter_id']:
                    raise RuntimeError(f"data cursor mismatch: {it} != {state['iter_id']}")
                started = time.monotonic()
                metrics = train_step(step_fn, model, state, batch, cfg, optimizers, lrs, rank, ws, device)
                ema.update(model)
                state['batch_in_iter'] += 1
                boundary = state['batch_in_iter'] == steps_per_iter
                if boundary:
                    state['iter_id'] += 1
                    state['batch_in_iter'] = 0
                step = state['step']
                if rank == 0:
                    with log.open('a') as f:
                        f.write(json.dumps(dict(step=step, seconds=time.monotonic() - started,
                                                elapsed=time.monotonic() - began,
                                                segment=int(state['segment']), **metrics,
                                                mem_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else 0.0))
                            + '\n')
                    if metrics['count']:
                        last_full = (step, metrics)
                    if step % cfg['log_every'] == 0:
                        msg = f"[LT] step {step} lm_loss {metrics['lm_loss']:.5f} lr {metrics['lr']:.3g}"
                        if last_full:
                            msg += (f" acc {last_full[1]['accuracy']:.4f} exact {last_full[1]['exact_accuracy']:.4f}"
                                    f" [finished at step {last_full[0]}]")
                        print(msg, flush=True)
                if step % cfg['stop_check_every'] == 0 or boundary or step >= stop_at:
                    stopped = stop_requested(device, deadline)
                if stopped:
                    break
                if (boundary and cfg['save_on_eval']) or (cfg['save_every_steps'] and step % cfg['save_every_steps'] == 0):
                    save()
                if boundary:
                    run_eval(step)
                    if te_noid is not None:
                        run_eval(step, te_noid, '-NOID')
                    last_eval = step
                if cfg['milestone_every'] and step % cfg['milestone_every'] == 0:
                    mdir = os.path.join(out_dir, 'milestones')
                    save(mdir, 0)
                    extrapolate(model, ema, te_x, te_y, cfg, rank, ws, device, step, cfg['milestone_extrap_segs'],
                                os.path.join(mdir, f'extrap_step_{step}.json'), deadline, te_ids)
                if step >= stop_at:
                    break
        path = save() if cfg['save_final'] else None   # before the final evaluation: an interruption loses nothing
        stopped = stopped or stop_requested(device, deadline)
        if not stopped and last_eval != state['step']:
            run_eval(state['step'])
        if rank == 0:
            print(f"[LT] saved step {state['step']}: {path}" + (' [stopped]' if stopped else ''), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return state['step']


def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, help='JSON file of DEFAULTS overrides')
    ap.add_argument('--out', help='output directory (out_dir)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='further overrides (JSON values)')
    opt = ap.parse_args(argv)
    cfg = json.loads(opt.config.read_text()) if opt.config else {}
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}       # '_comment' keys
    if opt.out:
        cfg['out_dir'] = opt.out
    for item in opt.set:
        key, _, value = item.partition('=')
        cfg[key] = _parse_value(value)
    return main(cfg)


if __name__ == '__main__':
    sys.exit(0 if cli() is not None else 1)
