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

from .data import SudokuTrainDataset, eval_batches, load_data
from .model import LT, Carry, ModelConfig

FORMAT = 'lt-bounded-phase-clock-stdp-v1'
DEFAULTS = dict(
    # model (lt.model.ModelConfig)
    hidden_size=832, num_heads=8, blocks_per_seg=8, loops=16, mlp_expansion=4.0, puzzle_emb_ndim=832, eps=1e-4,
    tag_bound=0.14, block_omega=0.34, trace_lambda=0.5, outer_rho=0.0, alpha_init=1.0, hebb=True,
    kernel='auto', kernel_bf16=True, amp=True,
    # data
    data_npz=None, test_size=2048, num_aug=1000,
    # optimization
    global_batch_size=128, epochs=50000, eval_interval=250, lr=1e-4, lr_warmup_steps=2000, weight_decay=1.0,
    beta1=0.9, beta2=0.95, puzzle_emb_lr=1e-4, puzzle_emb_weight_decay=1.0, ema_rate=0.999, seed=0,
    matmul_precision='highest', compile='auto',
    # run control
    out_dir=None, resume_from=None, max_steps=None, max_hours=None, save_every_steps=10000, keep_last=25,
    milestone_every=0, milestone_extrap_segs=128, milestone_extrap_n=512, log_every=250, stop_check_every=25,
    dataloader_workers=1)
# A checkpoint resumes only under the same values of these keys (plus the data fingerprint).
RECIPE_KEYS = ('hidden_size', 'num_heads', 'blocks_per_seg', 'loops', 'mlp_expansion', 'puzzle_emb_ndim', 'eps',
               'tag_bound', 'block_omega', 'trace_lambda', 'outer_rho', 'alpha_init', 'hebb', 'num_aug', 'test_size',
               'global_batch_size', 'epochs', 'eval_interval', 'lr', 'lr_warmup_steps', 'weight_decay', 'beta1',
               'beta2', 'puzzle_emb_lr', 'puzzle_emb_weight_decay', 'ema_rate', 'seed', 'data_fingerprint')
METRICS = ('count', 'accuracy', 'exact_accuracy', 'lm_loss')
_STOP = False


# ---------------------------------------------------------------------------- loss
def log_stablemax(x, dim=-1):
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def segment_loss(logits, labels, halted):
    """Summed per-puzzle mean stablemax CE (FP64) and the metric sums; accuracy counts finished puzzles only."""
    lp = log_stablemax(logits.double())
    cell = -torch.gather(lp, -1, labels.long().unsqueeze(-1)).squeeze(-1)          # [B, T]
    loss = cell.mean(-1).sum()
    with torch.no_grad():
        ok = logits.argmax(-1) == labels
        metrics = dict(count=halted.sum(), accuracy=torch.where(halted, ok.float().mean(-1), 0).sum(),
                       exact_accuracy=(halted & ok.all(-1)).sum(), lm_loss=loss.detach())
    return loss, metrics


class TrainStep(torch.nn.Module):
    """Model + loss in one module, so torch.compile sees the whole segment."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, carry, batch):
        carry, logits = self.model(carry, batch)
        loss, metrics = segment_loss(logits, carry.data['labels'], carry.halted)
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
    """URM sparse update of the puzzle embedding rows used in the batch (gathered across ranks)."""
    def __init__(self, embedding, world_size, lr=0.0, weight_decay=0.0):
        self.embedding = embedding
        super().__init__([embedding.weights, embedding.local_weights, embedding.local_ids],
                         dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad()
    def step(self, closure=None):
        group, e = self.param_groups[0], self.embedding
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
    """Weight decay on the projection/FFN/embedding/classifier matrices; none on biases, alpha and RoPE angles."""
    opts, lrs = [], []
    if model.puzzle_emb is not None:
        opts.append(SignSGDEmbedding(model.puzzle_emb, world_size, weight_decay=cfg['puzzle_emb_weight_decay']))
        lrs.append(cfg['puzzle_emb_lr'])
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = lambda n, p: p.ndim >= 2 and 'rope_theta' not in n
    opts.append(AdamATan2([{'params': [p for n, p in named if not decay(n, p)], 'weight_decay': 0.0},
                           {'params': [p for n, p in named if decay(n, p)], 'weight_decay': cfg['weight_decay']}],
                          lr=0, betas=(cfg['beta1'], cfg['beta2'])))
    lrs.append(cfg['lr'])
    return opts, lrs


def lr_at(step, base_lr, warmup):
    return base_lr * step / max(1, warmup) if step < warmup else base_lr


class EMA:
    def __init__(self, model, rate):
        self.rate = rate
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.rate).add_(p, alpha=1 - self.rate)

    def swap(self, model):
        """Context manager: EMA weights inside, raw weights restored after."""
        ema = self

        class _Swap:
            def __enter__(self):
                self.backup = {}
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema.shadow:
                            self.backup[n] = p.detach().clone()
                            p.copy_(ema.shadow[n])
                return model

            def __exit__(self, *exc):
                with torch.no_grad():
                    for n, p in model.named_parameters():
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
    local = dict(carry=None if state['carry'] is None else {f.name: _cpu(getattr(state['carry'], f.name))
                                                            for f in fields(Carry)}, rng=_rng_state(device))
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
def run_segments(model, x, y, segs, gbs, rank, ws, device, deadline=float('inf')):
    """Fresh carry, `segs` segments per puzzle. Returns per-segment sums [3, segs] (cell accuracy, exact, churn),
    the count of puzzles and whether time ran out."""
    sums = torch.zeros(3, segs, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    loops, training = model.config.loops, model.training
    model.config.loops = segs + 1                              # no lane is replaced during the run
    model.eval()
    try:
        for batch in eval_batches(x, y, gbs, rank, ws):
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


def evaluate(model, ema, x, y, cfg, rank, ws, device, step, deadline=float('inf')):
    with ema.swap(model):
        sums, count, partial = run_segments(model, x, y, cfg['loops'], cfg['global_batch_size'], rank, ws, device,
                                            deadline)
    if ws > 1:
        dist.reduce(sums, dst=0)
        dist.reduce(count, dst=0)
    if rank:
        return None
    n = max(int(count), 1)
    acc, exact = float(sums[0, -1]) / n, int(sums[1, -1])
    print(f'[EVAL] step {step} acc {acc:.4f} exact {exact}/{int(count)}' + (' [partial]' if partial else ''), flush=True)
    return dict(step=step, accuracy=acc, exact=exact, n=int(count), partial=partial)


def extrapolate(model, ema, x, y, cfg, rank, ws, device, step, segs, out_json, deadline=float('inf')):
    """Segments beyond the training horizon (milestones): per-segment exact count on the first milestone_extrap_n
    held-out puzzles and the error-reduction rate (best - seg16) / (n - seg16)."""
    x, y = x[:cfg['milestone_extrap_n']], y[:cfg['milestone_extrap_n']]
    with ema.swap(model):
        sums, count, partial = run_segments(model, x, y, segs, cfg['global_batch_size'], rank, ws, device, deadline)
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


def _check_finite(model, loss, ws):
    flags = [torch.isfinite(loss.detach()).all()]
    flags += [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    pe = model.puzzle_emb
    if pe is not None and pe.local_weights.grad is not None:
        flags.append(torch.isfinite(pe.local_weights.grad).all())
    bad = (~torch.stack(flags).all()).to(torch.int32)
    if ws > 1:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError('nonfinite loss/gradient before the optimizer step; last checkpoint left intact')


def train_step(step_fn, model, state, batch, cfg, optimizers, lrs, rank, ws, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    if state['carry'] is None:
        state['carry'] = model.initial_carry(batch)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    carry, loss, metrics = step_fn(state['carry'], batch)
    (loss / cfg['global_batch_size']).backward()
    _allreduce_grads(model, ws)
    _check_finite(model, loss, ws)
    lr = lr_at(state['step'], cfg['lr'], cfg['lr_warmup_steps'])
    for opt, base_lr in zip(optimizers, lrs):
        for group in opt.param_groups:
            group['lr'] = lr_at(state['step'], base_lr, cfg['lr_warmup_steps'])
        opt.step()
        opt.zero_grad(set_to_none=True)
    state['carry'], state['step'] = carry, state['step'] + 1
    vals = torch.stack([metrics[k].double() for k in METRICS])
    if ws > 1:
        dist.reduce(vals, dst=0)
    if rank:
        return None
    d = dict(zip(METRICS, vals.tolist()))
    n = max(d['count'], 1)
    return dict(lm_loss=d['lm_loss'] / cfg['global_batch_size'], accuracy=d['accuracy'] / n,
                exact_accuracy=d['exact_accuracy'] / n, count=d['count'], lr=lr)


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
    if cfg['dataloader_workers'] not in (0, 1):
        raise ValueError('dataloader_workers must be 0 or 1')
    if not cfg['out_dir']:
        raise ValueError('out_dir is required')


def main(cfg):
    global _STOP
    _STOP = False
    cfg = dict(DEFAULTS, **cfg)
    validate(cfg)
    deadline = float('inf') if cfg['max_hours'] is None else time.monotonic() + cfg['max_hours'] * 3600
    rank, ws, device = init_distributed()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal_stop)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(cfg['matmul_precision'])
    torch.manual_seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    random.seed(cfg['seed'] + rank)
    gbs = cfg['global_batch_size']
    if gbs % ws:
        raise ValueError('global_batch_size must be divisible by the number of processes')
    tr_x, tr_y, te_x, te_y, path, fingerprint = load_data(cfg['data_npz'], cfg['test_size'])
    cfg.update(data_npz=path, data_fingerprint=fingerprint)
    steps_per_iter = cfg['eval_interval'] * len(tr_x) // gbs
    total_iters = cfg['epochs'] // cfg['eval_interval']
    stop_at = steps_per_iter * total_iters if cfg['max_steps'] is None else min(cfg['max_steps'], steps_per_iter * total_iters)
    if cfg['compile'] == 'auto':
        cfg['compile'] = device.type == 'cuda' and torch.cuda.get_device_capability(device) >= (7, 0)
    with torch.device(device):
        model = LT(ModelConfig.from_dict(dict(cfg, batch_size=gbs // ws)))
    model.train()
    if ws > 1:
        with torch.no_grad():
            for p in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(p, src=0)
    optimizers, lrs = create_optimizers(model, cfg, ws)
    ema = EMA(model, cfg['ema_rate'])
    out_dir = os.path.abspath(os.path.expanduser(str(cfg['out_dir'])))
    cfg['out_dir'] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    ckpt = find_latest_checkpoint(cfg['resume_from'] or out_dir)
    if cfg['resume_from'] and not ckpt:
        raise FileNotFoundError(f"resume_from has no checkpoint: {cfg['resume_from']}")
    state = (load_checkpoint(ckpt, model, optimizers, ema, cfg, rank, ws, device) if ckpt
             else dict(step=0, iter_id=0, batch_in_iter=0, carry=None))
    if state['step'] != state['iter_id'] * steps_per_iter + state['batch_in_iter']:
        raise ValueError('checkpoint cursor/step inconsistency')
    if rank == 0:
        Path(out_dir, 'config.json').write_text(json.dumps(dict(cfg, format=FORMAT), indent=2) + '\n')
    if cfg['compile']:
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
    step_fn = TrainStep(model)
    step_fn = torch.compile(step_fn, dynamic=False) if cfg['compile'] else step_fn
    if rank == 0:
        mc = model.config
        print(f"[LT] {FORMAT} torch={torch.__version__} device={device} ranks={ws} "
              f"params={sum(p.numel() for p in model.parameters()):,} amp={model._amp(device)} "
              f"kernel={'triton' if model.use_kernel(device, torch.float32) else 'torch'} compile={cfg['compile']}", flush=True)
        print(f"[LT] tau = {mc.tag_bound:g} sin(phi) + n {mc.block_omega:g}; lambda {mc.trace_lambda:g}; "
              f"rho {mc.outer_rho:g}; alpha_init {mc.alpha_init:g}; stop_at {stop_at}; "
              + (f"RESUME {ckpt} step {state['step']}" if ckpt else 'NEW RUN') + f'; out {out_dir}', flush=True)
    ds = SudokuTrainDataset(tr_x, tr_y, seed=cfg['seed'], num_aug=cfg['num_aug'], global_batch_size=gbs, rank=rank,
                            world_size=ws, epochs_per_iter=cfg['eval_interval'], start_iter=state['iter_id'],
                            total_iters=total_iters, skip_batches=state['batch_in_iter'])
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
                                                segment=int(state['carry'].steps.max()), **metrics)) + '\n')
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
                if boundary or (cfg['save_every_steps'] and step % cfg['save_every_steps'] == 0):
                    save()
                if boundary:
                    evaluate(model, ema, te_x, te_y, cfg, rank, ws, device, step, deadline)
                    last_eval = step
                if cfg['milestone_every'] and step % cfg['milestone_every'] == 0:
                    mdir = os.path.join(out_dir, 'milestones')
                    save(mdir, 0)
                    extrapolate(model, ema, te_x, te_y, cfg, rank, ws, device, step, cfg['milestone_extrap_segs'],
                                os.path.join(mdir, f'extrap_step_{step}.json'), deadline)
                if step >= stop_at:
                    break
        path = save()                                  # before the final evaluation: an interruption loses nothing
        stopped = stopped or stop_requested(device, deadline)
        if not stopped and last_eval != state['step']:
            evaluate(model, ema, te_x, te_y, cfg, rank, ws, device, state['step'], deadline)
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
