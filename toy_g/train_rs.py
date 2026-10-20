"""Train and evaluate the rule-synapse models (toy_g/rule_synapse.py) on the 4x4 torus toy, in context, no identifier.

Same data and evaluation as toy_g/train_g.py (fresh determined group-0 samples with k in {1, 2, 3}; the fixed
instances by k and group) plus the k = 3 accuracy on involutive tasks (t and c both their own inverse, 440 of 3,072)
against the rest, which separates the window shapes of version B.

    python -m toy_g.train_rs --arch A --mode designed --steps 2000 --out runs/rs/A_designed           # step 1
    python -m toy_g.train_rs --arch A --mode learned --d 32 --compose cell --seed 1 --out runs/rs/A_cell  # step 2
    python -m toy_g.train_rs --arch B --schedule 2block --init odd --steps 3000 --out runs/rs/B_2block_odd  # step 3
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import group as G
from .model import diagnostics, nll
from .learned_structure import FreeStructureModel
from .rule_synapse import BandsRuleModel, RuleSynapseModel
from .train_g import TorchSampler, evaluate, format_diag, format_eval


def involutive_tasks():
    invol_t = np.array([(G.SRC[t][G.SRC[t]] == np.arange(G.CELLS)).all() for t in range(G.NUM_T)])
    invol_c = np.array([(G.CMAPS[c][G.CMAPS[c]] == np.arange(G.COLORS + 1)).all() for c in range(G.NUM_C)])
    task = np.arange(G.NUM_TASKS)
    return invol_t[task // G.NUM_C] & invol_c[task % G.NUM_C]


@torch.no_grad()
def eval_involutive(model, inst, device, batch=1024):
    """k = 3 exact accuracy on involutive tasks and on the rest, per group."""
    preds = []
    for s in range(0, len(inst['task']), batch):
        sl = slice(s, s + batch)
        din = torch.as_tensor(inst['demo_in'][sl].astype(np.int64), device=device)
        dout = torch.as_tensor(inst['demo_out'][sl].astype(np.int64), device=device)
        qin = torch.as_tensor(inst['test_in'][sl].astype(np.int64), device=device)
        mask = torch.ones(din.shape[:2], dtype=torch.bool, device=device)
        preds.append(model(din, dout, mask, qin)['probs'].argmax(-1).cpu())
    exact = (torch.cat(preds).numpy() == inst['test_out'].reshape(-1, G.CELLS)).all(1)
    inv = involutive_tasks()[inst['task']]
    return dict(involutive=float(exact[inv].mean()), others=float(exact[~inv].mean()),
                **{f'g{g}': float(exact[inst['group'] == g].mean()) for g in range(3)})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arch', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--K', type=int, default=4, help='C: number of primitives')
    p.add_argument('--N', type=int, default=6, help='C: maximum word length')
    p.add_argument('--lam', type=float, default=0.0, help='C: description cost per word symbol, in explained cells')
    p.add_argument('--len-aux', type=float, default=0.0, help='C: weight of the expected word length in the loss')
    p.add_argument('--read', default='cos', choices=['cos', 'ls'], help="C: cell read, Hebbian cosine or delta-rule least squares")
    p.add_argument('--code-sep', type=float, default=0.0, help='C: weight of the distinct-address penalty (cos > 0.9)')
    p.add_argument('--mode', default='designed', choices=['designed', 'learned', 'learned_free'])
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--compose', choices=['code', 'cell'], default='code')
    p.add_argument('--tau', type=float, default=3.0)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--schedule', default='2block', choices=['2block', 'persist', 'residual'])
    p.add_argument('--bands', type=int, default=4)
    p.add_argument('--init', default='odd', choices=['odd', 'causal', 'even', 'random'])
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=1))
    split = D.make_split(seed=0)
    inst = D.load_instances()
    train_tasks = np.sort(split['train'])
    heldout_t = split['heldout_t_elements'].tolist()
    if args.arch == 'A':
        model = RuleSynapseModel(mode=args.mode, d=args.d, cmap='perm', tau=args.tau, beta=args.beta, seed=args.seed,
                                 compose=args.compose)
    elif args.arch == 'B':
        model = BandsRuleModel(schedule=args.schedule, bands=args.bands, init=args.init, beta=args.beta, seed=args.seed)
    else:
        model = FreeStructureModel(d=args.d, K=args.K, N=args.N, tau=args.tau, beta=args.beta, seed=args.seed, lam=args.lam,
                                   read=args.read)
    model = model.to(device)
    params = list(model.parameters())
    print(f'arch {args.arch}: {sum(p.numel() for p in params)} trainable parameters; out {out}', flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    sampler = TorchSampler(train_tasks, device, seed=10_000 + args.seed)

    def describe():
        if args.arch == 'A':
            return f'tau {model.tau.item():.2f} beta {model.beta.item():.3f}'
        if args.arch == 'C':
            return 'tau {:.2f} beta {:.3f} beta_c {:.2f}'.format(model.log_tau.exp().item(), model.log_beta.exp().item(),
                                                                 model.log_beta_c.exp().item())
        w = model.window()
        return 'beta {:.3f} window '.format(model.log_beta.exp().item()) + ' '.join(f'l({D:+d})={v:+.3f}' for D, v in w.items())

    def run_eval(step):
        model.eval()
        res = evaluate(model, inst, device)
        inv = eval_involutive(model, inst, device)
        rec = dict(step=step, eval=res, involutive=inv)
        msg = f'--- eval at step {step}\n{format_eval(res)}\n  k=3: involutive {100 * inv["involutive"]:.1f}% ' \
              f'others {100 * inv["others"]:.1f}% | {describe()}'
        if args.arch == 'A' and args.mode != 'designed':
            diag = diagnostics(model, heldout_t)
            rec['diag'] = diag
            msg += f'\n  {format_diag(diag)}'
        if args.arch == 'B':
            rec['window'] = model.window()
        if args.arch == 'C':
            rec['realised'] = r = model.realised(heldout_t)
            msg += (f"\n  words {r['words']}: valid {r['words_valid']}, distinct transformations {r['distinct']}/128, "
                    f"held-out realised {r['heldout_realised']}/{r['heldout_total']}, "
                    f"mean shortest word {r['mean_shortest_len']:.2f}")
        model.train()
        with open(out / 'evals.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(msg, flush=True)

    run_eval(0)
    acc, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        if step > args.warmup:
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(args.steps - args.warmup, 1))))
        for gr in opt.param_groups:
            gr['lr'] = lr
        b = sampler.batch(args.batch, (1, 2, 3), (3, 8))
        res = model(b['demo_in'], b['demo_out'], b['demo_mask'], b['query_in'])
        loss = nll(res['probs'], b['query_out'])
        if args.arch == 'C' and args.len_aux:
            loss = loss + args.len_aux * res['exp_len'].mean()
        if args.arch == 'C' and args.code_sep:
            loss = loss + args.code_sep * model.separation_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        opt.step()
        with torch.no_grad():
            cell = (res['probs'].argmax(-1) == b['query_out']).float()
            acc.append(torch.stack([loss.detach(), cell.min(1).values.mean(), cell.mean()]))
        if step % args.log_every == 0 or step == args.steps:
            m = torch.stack(acc).mean(0).tolist()
            acc = []
            print(f'step {step:6d} loss {m[0]:.4f} exact {m[1]:.3f} cell {m[2]:.4f} gnorm {gnorm:.3f} | {describe()} '
                  f'{time.time() - t0:.0f}s', flush=True)
        if args.eval_every and step % args.eval_every == 0 and step != args.steps:
            run_eval(step)
            torch.save(model.state_dict(), out / f'step_{step}.pt')
    run_eval(args.steps)
    torch.save(model.state_dict(), out / 'final.pt')


if __name__ == '__main__':
    main()
