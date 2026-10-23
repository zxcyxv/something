"""Train the coupled model (lt.coupled) on an in-context pair store (data/arc1_ctx10_trans.npz: ARC-1 grids <= 10,
transductive: the train pairs of every task are trained, the 278 test pairs are evaluated).

    python -m lt.coupled_train --data data/arc1_ctx10_trans.npz --out runs/coupled_x --compile [--set write_every=1]

Loss: stablemax cross entropy (URM) on the query lane's output canvas, pad cells ignored (grid colours and EOS count),
mean over cells per episode, mean over the batch. AdamATan2 (betas 0.9/0.95), linear warmup, weight decay on every
parameter, EMA of the weights for evaluation. Evaluation every eval_every steps: every test pair, version 0, exact =
every labelled cell right (EOS included, so the size too), cell = accuracy inside the true grid (cmap's criterion),
cell_eos = with the EOS cells. OUT/last.pt (model, EMA, optimizer, step; tens of MB) is overwritten at every evaluation.
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .coupled import PAD, Coupled, CoupledConfig, CoupledEpisodes, scores
from .train import AdamATan2, log_stablemax


def to(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def loss_fn(logits, labels, query):
    y = labels[query, -logits.shape[1]:]
    lab = (y != PAD).float()
    lp = log_stablemax(logits[query].double())
    cell = -torch.gather(lp, -1, y[..., None]).squeeze(-1)
    return ((cell * lab).sum(-1) / lab.sum(-1).clamp_min(1)).mean().float()


def evaluate(model, data, dev, batch_size):
    model.eval()
    tot = dict(exact=0, n=0, cell=0, n_cell=0, cell_eos=0, n_eos=0)
    with torch.no_grad():
        for b in data.eval_batches(batch_size):
            b = to(b, dev)
            logits, _ = model(b)
            for k, v in scores(logits, b['labels'], b['query']).items():
                tot[k] += v
    model.train()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/arc1_ctx10_trans.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--batch', type=int, default=64, help='lanes = queries per step (leave-one-out)')
    ap.add_argument('--eval_groups', type=int, default=16, help='test pairs (groups of up to D + 1 lanes) per eval batch')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--num_aug', type=int, default=1000, help='URM: 1000 versions per task')
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--compile', action='store_true', help='torch.compile the bands block (lt.model.LT.block_step)')
    ap.add_argument('--resume', action='store_true', help='continue from OUT/last.pt')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='CoupledConfig overrides (JSON values)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    over = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
    probe = CoupledEpisodes(args.data, seed=args.seed, num_aug=args.num_aug)
    cfg = CoupledConfig(**dict(dict(N=probe.N), **over))
    data = CoupledEpisodes(args.data, seed=args.seed, num_aug=args.num_aug, D=cfg.D, N=cfg.N)
    model = Coupled(cfg).to(dev)
    opt = AdamATan2(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema > 0 else None
    if args.compile:
        model.block_fn = torch.compile(model.lt.block_step, dynamic=False)
    n_par = sum(p.numel() for p in model.parameters())
    meta = dict(args=vars(args), cfg=asdict(cfg), params=n_par, tasks=len(data.tasks),
                test_pairs=sum(len(v) for v in data.test_pairs.values()))
    (out / 'config.json').write_text(json.dumps(meta, indent=1))
    print(f'[COUPLED] params {n_par:,} cfg {asdict(cfg)} tasks {meta["tasks"]} test pairs {meta["test_pairs"]}', flush=True)
    start = 1
    if args.resume and (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        if ema is not None and ck.get('ema') is not None:
            ema = {k: v.to(dev) for k, v in ck['ema'].items()}
        start = ck['step'] + 1
        print(f'[COUPLED] resumed from step {ck["step"]} (data stream reseeded)', flush=True)
    rng = np.random.default_rng(args.seed + start)
    log, elog = open(out / 'train.jsonl', 'a'), open(out / 'eval.jsonl', 'a')
    t0 = time.time()
    for step in range(start, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup)
        for g in opt.param_groups:
            g['lr'] = lr
        b = to(data.train_batch(args.batch, rng), dev)
        logits, _ = model(b)
        loss = loss_fn(logits, b['labels'], b['query'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not torch.isfinite(gn):
            print(f'[COUPLED] step {step} nonfinite grad, skipped', flush=True)
            continue
        opt.step()
        model.project_()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
        if step % args.log_every == 0:
            s = scores(logits.detach(), b['labels'], b['query'])
            rec = dict(step=step, loss=loss.item(), exact=s['exact'] / s['n'], cell=s['cell'] / max(s['n_cell'], 1),
                       cell_eos=s['cell_eos'] / max(s['n_eos'], 1), lr=lr, sec=time.time() - t0,
                       tau=float(model.tau), lam=float(torch.nn.functional.softplus(model.lam_raw)),
                       mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else 0.0)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(f"[COUPLED] step {step} loss {rec['loss']:.4f} cell {rec['cell']:.4f} exact {rec['exact']:.3f} "
                  f"tau {rec['tau']:.3f} lam {rec['lam']:.2e} {rec['sec']:.0f}s {rec['mem_gb']:.1f}GiB", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if ema is not None:
                live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            e = evaluate(model, data, dev, args.eval_groups)
            if ema is not None:
                model.load_state_dict(live)
            rec = dict(step=step, exact=e['exact'], n=e['n'], cell=e['cell'] / max(e['n_cell'], 1),
                       cell_eos=e['cell_eos'] / max(e['n_eos'], 1), sec=time.time() - t0)
            elog.write(json.dumps(rec) + '\n')
            elog.flush()
            print(f"[EVAL] step {step} exact {e['exact']}/{e['n']} cell {rec['cell']:.4f} cell_eos {rec['cell_eos']:.4f}",
                  flush=True)
            torch.save(dict(model=model.state_dict(), ema=ema, opt=opt.state_dict(), step=step, meta=meta),
                       out / 'last.pt')


if __name__ == '__main__':
    main()
