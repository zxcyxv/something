"""Train the cognitive-map rule synapse (lt.cmap) on an in-context pair store (data/miniarc_pairs.npz, transductive).

    python -m lt.cmap_train --data data/miniarc_pairs.npz --out runs/cmap_mini [--steps 20000 --batch 32 ...]

Loss: cross entropy of the query output's colours inside its true extent + 0.1 smooth-L1 on the predicted extent.
Evaluation: every official test pair, version 0, the task's puzzle embedding; exact = extent and every cell right.
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cmap import Cmap, CmapConfig
from .cmap_data import CmapEpisodes
from .train import AdamATan2


def to(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def scores(logits, size, batch):
    label, true = batch['label'], batch['size'][:, -1]
    inside = label >= 0
    hit = (logits.argmax(-1) == label) | ~inside
    size_ok = (size.round() == true).all(-1)
    exact = hit.flatten(1).all(1) & size_ok
    cell = ((logits.argmax(-1) == label) & inside).sum().item() / max(int(inside.sum()), 1)
    return exact, cell, size_ok


def run(model, b, train):
    logits, size, _, _ = model(b, train=train)
    return logits, size


def evaluate(model, data, dev, batch_size):
    model.eval()
    ex, cells, n_cell, n, size_ok = 0, 0.0, 0, 0, 0
    with torch.no_grad():
        for b in data.eval_batches(batch_size):
            b = to(b, dev)
            logits, size = run(model, b, False)
            e, c, s = scores(logits, size, b)
            k = int((b['label'] >= 0).sum())
            ex, n, size_ok = ex + int(e.sum()), n + len(e), size_ok + int(s.sum())
            cells, n_cell = cells + c * k, n_cell + k
    model.train()
    return ex, n, cells / max(n_cell, 1), size_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/miniarc_pairs.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true', help='continue from OUT/last.pt (model, EMA, optimizer, step)')
    ap.add_argument('--compile', action='store_true', help='torch.compile perception and frames (not the read: the write '
                    'differentiates it twice, which compiled autograd does not support)')
    ap.add_argument('--opt', choices=['adamw', 'adamatan2'], default='adamw', help='adamatan2: URM optimizer, betas 0.9/0.95')
    ap.add_argument('--wd', type=float, default=0.01, help='weight decay on every parameter (URM ARC: 0.1)')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA rate of the weights used for evaluation (0 = off; URM 0.999)')
    ap.add_argument('--max_demos', type=int, default=5, help='demonstration slots (Mini-ARC: 5 cuts 5 of 149 tasks)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='CmapConfig overrides (JSON values)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = CmapEpisodes(args.data, seed=args.seed, max_demos=args.max_demos)
    over = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    cfg = CmapConfig(**dict(dict(N=data.N, D=data.D, n_tasks=data.n_tasks), **over))
    model = Cmap(cfg).to(dev)
    if args.opt == 'adamatan2':
        opt = AdamATan2(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema > 0 else None
    if args.compile:
        model.perceive = torch.compile(model.perceive, dynamic=False)
        model.frames = torch.compile(model.frames, dynamic=False)
    n_par = sum(p.numel() for p in model.parameters())
    meta = dict(args=vars(args), cfg=asdict(cfg), params=n_par)
    (out / 'config.json').write_text(json.dumps(meta, indent=1))
    print(f'[CMAP] params {n_par:,} cfg {asdict(cfg)}', flush=True)
    start = 1
    if args.resume and (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if ema is not None and ck.get('ema') is not None:
            ema = {k: v.to(dev) for k, v in ck['ema'].items()}
        start = ck['step'] + 1
        print(f'[CMAP] resumed from step {ck["step"]} (data stream reseeded)', flush=True)
    rng = np.random.default_rng(args.seed + start)
    log = open(out / 'train.jsonl', 'a')
    t0 = time.time()
    for step in range(start, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup)
        for g in opt.param_groups:
            g['lr'] = lr
        b = to(data.train_batch(args.batch, rng), dev)
        logits, size = run(model, b, True)
        label = b['label']
        inside = label >= 0
        ce = F.cross_entropy(logits[inside], label[inside])
        sl = F.smooth_l1_loss(size, b['size'][:, -1].to(size.dtype))
        loss = ce + 0.1 * sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            print(f'[CMAP] step {step} nonfinite grad, skipped', flush=True)
            continue
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
        if step % args.log_every == 0:
            e, c, s = scores(logits.detach(), size.detach(), b)
            rec = dict(step=step, loss=loss.item(), ce=ce.item(), size_l1=sl.item(), cell=c, exact=float(e.float().mean()),
                       size_ok=float(s.float().mean()), lr=lr, sec=time.time() - t0)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(f"[CMAP] step {step} loss {rec['loss']:.4f} ce {rec['ce']:.4f} size {rec['size_l1']:.3f} "
                  f"cell {c:.4f} exact {rec['exact']:.3f} size_ok {rec['size_ok']:.3f} {rec['sec']:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if ema is not None:                                                   # evaluate the EMA weights
                live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            ex, n, cell, sok = evaluate(model, data, dev, args.batch)
            with torch.no_grad():                                                 # the same weights without the
                saved = model.task_emb.detach().clone()                           # puzzle embedding (shared prior)
                model.task_emb.zero_()
                ex0, _, cell0, _ = evaluate(model, data, dev, args.batch)
                model.task_emb.copy_(saved)
            print(f'[EVAL-NOEMB] step {step} cell {cell0:.4f} exact {ex0}/{n}', flush=True)
            if ema is not None:
                model.load_state_dict(live)
            print(f'[EVAL] step {step} cell {cell:.4f} exact {ex}/{n} size_ok {sok}/{n}', flush=True)
            log.write(json.dumps(dict(step=step, eval_cell=cell, eval_exact=ex, eval_n=n, eval_size_ok=sok,
                                      eval_noemb_cell=cell0, eval_noemb_exact=ex0)) + '\n')
            log.flush()
            torch.save(dict(model=model.state_dict(), ema=ema, opt=opt.state_dict(), step=step, meta=meta), out / 'last.pt')


if __name__ == '__main__':
    main()
