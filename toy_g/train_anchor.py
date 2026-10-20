"""Train the anchored version C (toy_g/anchor_model.py) on the anchor toy (toy_g/anchor_toy.py), in context, no task
identifier. Training: fresh determined samples of the training tasks, k in {1, 2, 3}; evaluation: the fixed instances
(data/anchor_instances.npz) at k = 2 and 3, per kind (fixed / anchored) and group (0 train task, 1 held-out pair,
2 held-out rule), also on the instances whose query colours all occur in the demonstrations (the colour map is any
function, as in version C, so a colour never seen cannot be mapped; the designed parameters are exact there).

    python -m toy_g.train_anchor --model general --steps 3000 --out runs/anchor/C_anchor_d16     # anchor not learned
    python -m toy_g.train_anchor --model phase --steps 3000 --out runs/anchor/phase_d16
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import anchor_toy as A
from . import group as G
from .anchor_model import AnchorStructureModel, PhaseAnchorModel
from .model import nll


@torch.no_grad()
def evaluate(model, inst, device, ks=(2, 3), batch=256):
    n = len(inst['task'])
    kind = inst['task'] // G.NUM_TASKS
    group = inst['group'][inst['task']]
    din_all = inst['demo_in'].reshape(n, -1)
    qin_all = inst['test_in'].reshape(n, -1)
    target = inst['test_out'].reshape(n, -1)
    res = {}
    for k in ks:
        seen = np.array([not (set(qin_all[i][qin_all[i] > 0].tolist()) -
                              set(inst['demo_in'][i, :k].reshape(-1)[inst['demo_in'][i, :k].reshape(-1) > 0].tolist()))
                         for i in range(n)])
        preds, amass = [], []
        for s in range(0, n, batch):
            sl = slice(s, s + batch)
            din = torch.as_tensor(inst['demo_in'][sl, :k].astype(np.int64), device=device)
            dout = torch.as_tensor(inst['demo_out'][sl, :k].astype(np.int64), device=device)
            qin = torch.as_tensor(inst['test_in'][sl].astype(np.int64), device=device)
            r = model(din, dout, torch.ones(din.shape[:2], dtype=torch.bool, device=device), qin)
            preds.append(r['probs'].argmax(-1).cpu())
            amass.append(r['anchored_mass'].cpu())
        exact = (torch.cat(preds).numpy() == target).all(1)
        amass = torch.cat(amass).numpy()
        det = inst['determined_k'][:, k - 1]
        res[f'k{k}'] = {}
        for kname, ksel in (('fixed', kind == 0), ('anchored', kind == 1)):
            for g in range(3):
                sel = ksel & (group == g) & det
                res[f'k{k}'][f'{kname}/g{g}'] = dict(n=int(sel.sum()), exact=float(exact[sel].mean()),
                                                     exact_seen=float(exact[sel & seen].mean()),
                                                     anchored_mass=float(amass[sel].mean()))
    return res


def format_eval(res):
    lines = []
    for k, r in res.items():
        lines.append(f'  {k}: ' + ' | '.join(f'{name} {100 * v["exact"]:5.1f}% (seen {100 * v["exact_seen"]:5.1f}%, '
                                             f'anchored mass {v["anchored_mass"]:.2f})' for name, v in r.items()))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['general', 'phase'], default='phase',
                   help='general: words over K unitary primitives (translations too); phase: words for the '
                        'non-commutative part, translations as bindings with position codes')
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--K', type=int, default=None, help='default 4 (general) / 2 (phase)')
    p.add_argument('--N', type=int, default=None, help='default 5 (general) / 4 (phase)')
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=1))
    split = A.make_split()
    inst = A.load_instances()
    cls = AnchorStructureModel if args.model == 'general' else PhaseAnchorModel
    K = args.K or (4 if args.model == 'general' else 2)
    N = args.N or (5 if args.model == 'general' else 4)
    model = cls(d=args.d, K=K, N=N, seed=args.seed, lam=args.lam).to(device)
    params = list(model.parameters())
    print(f'{args.model}: {sum(p.numel() for p in params)} parameters, {len(model.lens2)} programs; out {out}', flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    sampler = A.AnchorSampler(split['train'], device, seed=10_000 + args.seed)

    def run_eval(step):
        model.eval()
        res = evaluate(model, inst, device)
        diag = model.diagnostics(split['heldout_fixed'].tolist())
        model.train()
        with open(out / 'evals.jsonl', 'a') as f:
            f.write(json.dumps(dict(step=step, eval=res, diag=diag)) + '\n')
        print(f'--- eval at step {step}\n{format_eval(res)}\n  diag {diag}', flush=True)

    run_eval(0)
    acc, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        if step > args.warmup:
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(args.steps - args.warmup, 1))))
        for gr in opt.param_groups:
            gr['lr'] = lr
        b = sampler.batch(args.batch)
        res = model(b['demo_in'], b['demo_out'], b['demo_mask'], b['query_in'])
        loss = nll(res['probs'], b['query_out'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        with torch.no_grad():
            cell = (res['probs'].argmax(-1) == b['query_out']).float()
            anc = (b['task'] >= G.NUM_TASKS)
            acc.append(torch.stack([loss.detach(), cell.min(1).values.mean(), cell.min(1).values[anc].mean(),
                                    cell.min(1).values[~anc].mean(), res['anchored_mass'][anc].mean(),
                                    res['anchored_mass'][~anc].mean()]))
        if step % args.log_every == 0 or step == args.steps:
            m = torch.stack(acc).mean(0).tolist()
            acc = []
            print(f'step {step:5d} loss {m[0]:.4f} exact {m[1]:.3f} (anchored {m[2]:.3f}, fixed {m[3]:.3f}) anchored mass '
                  f'on anchored {m[4]:.2f} / fixed {m[5]:.2f} gnorm {gnorm:.2f} {time.time() - t0:.0f}s', flush=True)
        if args.eval_every and step % args.eval_every == 0 and step != args.steps:
            run_eval(step)
            torch.save(model.state_dict(), out / f'step_{step}.pt')
    run_eval(args.steps)
    torch.save(model.state_dict(), out / 'final.pt')


if __name__ == '__main__':
    main()
