"""Train and evaluate the structure-code model (toy_g/model.py) on the 4x4 torus toy.

Training data: fresh samples from group-0 (train) tasks whose query output is determined by the k demonstrations,
drawn by TorchSampler (a torch port of toy_g.data.sample_batch with the same distribution and exact checks; runs on the
GPU). Evaluation: the fixed instances data/gtorus_instances.npz for k = 1, 2, 3 (first k demonstrations), by group
(0 train tasks, 1 held-out (t, c) combinations, 2 held-out transformations), over all and over determined_k instances.

    python -m toy_g.train_g --mode designed --steps 0 --tau 50 --beta 20 --out runs/gtorus_g/designed_large
    python -m toy_g.train_g --mode learned --d 16 --steps 20000 --out runs/gtorus_g/learned_d16
    python -m toy_g.train_g --mode learned_free --d 16 --steps 20000 --out runs/gtorus_g/free_d16

Every run writes config.json, train_tasks.json (the training-task set and the per-task draw counts; the script asserts
that no group-1/2 task is ever drawn), train.jsonl, evals.jsonl, eval.json, diag.json and final.pt under --out.
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
from .model import NCOL, StructureModel, diagnostics, nll

DEFAULT_PHASES = [dict(k=[1, 2, 3], sparsity=[3, 8])]


class TorchSampler:
    """Fresh determined samples, as toy_g.data.sample_batch, on any torch device.

    Consistency is checked through pair sets: (t, c) maps a demonstration to its output iff every (moved colour,
    output colour) pair under t is a pair of the colour permutation c. This is equivalent to data.consistent_tasks and
    data.determined (tested against them)."""

    def __init__(self, tasks, device='cpu', seed=0):
        self.device = torch.device(device)
        self.tasks = torch.as_tensor(np.asarray(tasks), device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.src = torch.as_tensor(G.SRC, device=self.device)                         # [128, 16]
        self.cmaps = torch.as_tensor(G.CMAPS, device=self.device)                     # [24, 5]
        pairs = torch.nn.functional.one_hot(self.cmaps, NCOL).reshape(G.NUM_C, NCOL * NCOL)
        self.not_pairs = (1 - pairs).float().T.contiguous()                           # [25, 24]
        self.counts = torch.zeros(G.NUM_TASKS, dtype=torch.long, device=self.device)  # tasks drawn (kept samples)

    def _randint(self, lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=self.gen, device=self.device)

    def random_inputs(self, n, sparsity=(3, 8)):
        """[n, 16] grids: a uniform number of coloured cells in [lo, hi] (distinct cells), colours uniform 1..4."""
        lo, hi = sparsity
        counts = self._randint(lo, hi + 1, (n,))
        order = torch.rand(n, G.CELLS, generator=self.gen, device=self.device).argsort(1)
        colours = self._randint(1, G.COLORS + 1, (n, G.CELLS))
        values = torch.where(torch.arange(G.CELLS, device=self.device)[None] < counts[:, None], colours, 0)
        return torch.zeros(n, G.CELLS, dtype=torch.long, device=self.device).scatter_(1, order, values)

    def apply(self, grids, task):
        """grids [n, m, 16], task [n] -> outputs [n, m, 16]."""
        t, c = task // G.NUM_C, task % G.NUM_C
        moved = grids.gather(-1, self.src[t][:, None, :].expand_as(grids))
        return self.cmaps[c][:, None, :].expand(-1, grids.shape[1], -1).gather(-1, moved)

    def _violations(self, moved, out):
        """moved [n, m, T, 16], out [n, m, 16] -> [n, T, 24] count of pair types that colour map c violates."""
        n, m, T, _ = moved.shape
        idx = (moved * NCOL + out[:, :, None, :]).permute(0, 2, 1, 3).reshape(n, T, m * G.CELLS)
        present = torch.zeros(n, T, NCOL * NCOL, device=self.device).scatter_(2, idx, 1.0)
        return present @ self.not_pairs

    def consistent(self, demo_in, demo_out):
        """demo_in, demo_out [n, k, 16] -> bool [n, 128, 24]: tasks that map every demonstration to its output."""
        return self._violations(demo_in[:, :, self.src], demo_out) == 0

    def determined(self, demo_in, demo_out, query_in, query_out, chunk=2048):
        res = []
        for s in range(0, len(query_in), chunk):
            e = s + chunk
            cons = self.consistent(demo_in[s:e], demo_out[s:e])
            bad = self._violations(query_in[s:e, None][:, :, self.src], query_out[s:e, None]) > 0
            res.append(~(cons & bad).flatten(1).any(1))
        return torch.cat(res)

    def sample(self, n, k, sparsity=(3, 8), max_rounds=100):
        """n determined samples with k demonstrations each (flat [.., 16] grids)."""
        keep, have = [], 0
        for _ in range(max_rounds):
            m = max(2 * (n - have), 256)
            task = self.tasks[self._randint(0, len(self.tasks), (m,))]
            din = self.random_inputs(m * k, sparsity).reshape(m, k, G.CELLS)
            qin = self.random_inputs(m, sparsity)
            dout, qout = self.apply(din, task), self.apply(qin[:, None], task)[:, 0]
            ok = self.determined(din, dout, qin, qout)
            keep.append((din[ok], dout[ok], qin[ok], qout[ok], task[ok]))
            have += int(ok.sum())
            if have >= n:
                break
        else:
            raise RuntimeError('could not draw enough determined samples')
        return [torch.cat(x)[:n] for x in zip(*keep)]

    def batch(self, n, ks=(1, 2, 3), sparsity=(3, 8)):
        """n samples with k drawn uniformly from ks per sample; demonstrations padded to max(ks) with a mask."""
        ks = list(ks)
        kmax = max(ks)
        draw = torch.as_tensor(ks, device=self.device)[self._randint(0, len(ks), (n,))]
        parts = dict(demo_in=[], demo_out=[], demo_mask=[], query_in=[], query_out=[], task=[], k=[])
        for k in sorted(set(ks)):
            nk = int((draw == k).sum())
            if nk == 0:
                continue
            din, dout, qin, qout, task = self.sample(nk, k, sparsity)
            pad = torch.zeros(nk, kmax - k, G.CELLS, dtype=torch.long, device=self.device)
            mask = (torch.arange(kmax, device=self.device) < k)[None].expand(nk, -1)
            for key, val in zip(parts, (torch.cat([din, pad], 1), torch.cat([dout, pad], 1), mask, qin, qout, task,
                                        torch.full((nk,), k, device=self.device))):
                parts[key].append(val)
        out = {key: torch.cat(val) for key, val in parts.items()}
        self.counts += torch.bincount(out['task'], minlength=G.NUM_TASKS)
        return out


@torch.no_grad()
def evaluate(model, inst, device='cpu', ks=(1, 2, 3), idx=None, batch=1024):
    """Accuracy on the fixed instances: res[f'k{k}'][f'g{g}' or 'all'][subset] = dict(n, exact, cell), subset 'all' or
    'det' (instances determined by the first k demonstrations)."""
    device = torch.device(device)
    idx = np.arange(len(inst['task'])) if idx is None else np.asarray(idx)
    group = inst['group'][inst['task'][idx]]
    target = torch.as_tensor(inst['test_out'][idx].astype(np.int64)).reshape(len(idx), G.CELLS)
    res = {}
    for k in ks:
        preds = []
        for s in range(0, len(idx), batch):
            sl = idx[s:s + batch]
            din = torch.as_tensor(inst['demo_in'][sl, :k].astype(np.int64), device=device)
            dout = torch.as_tensor(inst['demo_out'][sl, :k].astype(np.int64), device=device)
            qin = torch.as_tensor(inst['test_in'][sl].astype(np.int64), device=device)
            mask = torch.ones(len(sl), k, dtype=torch.bool, device=device)
            preds.append(model(din, dout, mask, qin)['probs'].argmax(-1).cpu())
        cell = (torch.cat(preds) == target).float().numpy()
        exact = cell.min(1)
        det = inst['determined_k'][idx, k - 1]
        res[f'k{k}'] = {}
        for gname, gsel in [(f'g{g}', group == g) for g in range(3)] + [('all', np.ones(len(idx), bool))]:
            res[f'k{k}'][gname] = {}
            for sname, sel in (('all', gsel), ('det', gsel & det)):
                n = int(sel.sum())
                res[f'k{k}'][gname][sname] = dict(n=n, exact=float(exact[sel].mean()) if n else float('nan'),
                                                  cell=float(cell[sel].mean()) if n else float('nan'))
    return res


def format_eval(res):
    ks = sorted(res)
    lines = ['group  subset ' + ''.join(f'{k:>22}' for k in ks), ' ' * 15 + '     exact / cell  (n)' * len(ks)]
    for g in ('g0', 'g1', 'g2', 'all'):
        for s in ('all', 'det'):
            row = f'{g:<6} {s:<6} '
            for k in ks:
                r = res[k][g][s]
                row += f'  {100 * r["exact"]:6.2f} / {100 * r["cell"]:6.2f} ({r["n"]:4d})'
            lines.append(row)
    return '\n'.join(lines)


def format_diag(diag):
    keys = ['tau', 'beta', 'read_exact_train_t', 'read_exact_heldout_t', 'op_exact_train_t', 'op_exact_heldout_t',
            'margin_min_train_t', 'margin_min_heldout_t', 'code_cos_abs_mean', 'code_cos_max']
    s = ' '.join(f'{k}={diag[k]:.4g}' for k in keys if k in diag)
    rel = ' '.join(f'{k}={v:.3g}' for k, v in diag['relation_orbit_err'].items())
    relabel = ' '.join(f'{k}={diag[k]:.4g}' for k in ('realised_valid', 'realised_distinct', 'realised_label_exact',
                                                       'heldout_elements_realised', 'relabel_margin_min') if k in diag)
    gens = ' '.join(f'{k}->{v["element"]}{tuple(v["decoded"]) if v["decoded"] else ""}(true {v["true"]}, '
                    f'cos {v["cos_min"]:.2f})' for k, v in diag['generator_action'].items())
    return (f'{s}\n  relation errors on the code orbit: {rel}\n  up to relabelling: {relabel}\n'
            f'  generator actions on the codes: {gens}')


def schedule(spec, step):
    """'a,b,n' -> geometric ramp from a to b over n steps (constant after)."""
    a, b, n = (float(x) for x in spec.split(','))
    return a * (b / a) ** min(step / max(n, 1), 1.0)


def phase_at(phases, step):
    for ph in phases:
        if 'until' not in ph or step <= ph['until']:
            return ph
    return phases[-1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', default='learned', choices=['designed', 'learned', 'learned_free'])
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--generators', default='orthogonal', choices=['orthogonal', 'free'])
    p.add_argument('--cmap', default='perm', choices=['perm', 'sinkhorn', 'row'])
    p.add_argument('--sinkhorn-iters', type=int, default=20)
    p.add_argument('--cmap-eps', type=float, default=1e-3)
    p.add_argument('--tau', type=float, default=3.0, help='initial tau')
    p.add_argument('--beta', type=float, default=0.1, help='initial beta')
    p.add_argument('--tau-sched', default=None, help="'a,b,n': tau fixed on a geometric ramp a -> b over n steps")
    p.add_argument('--beta-sched', default=None, help="'a,b,n': beta fixed on a geometric ramp a -> b over n steps")
    p.add_argument('--gen-init', type=float, default=1.0, help='std of the raw generator entries, times 1/sqrt(d)')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--lr-min-frac', type=float, default=0.1, help='cosine decay to this fraction of lr')
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--phases', default=None,
                   help='JSON list of {"until": step, "k": [...], "sparsity": [lo, hi]}; default k 1..3, sparsity 3..8')
    p.add_argument('--aux-pi', type=float, default=0.0,
                   help='LAST RESORT, uses transformation labels: weight of a cross-entropy on the posterior of the true t')
    p.add_argument('--compose', choices=('code', 'cell'), default='code',
                   help="code: multiply the whole word in code space; cell: clean up to cells after every generator step")
    p.add_argument('--closure', type=float, default=0.0,
                   help='weight of the label-free closure prior (StructureModel.closure_loss): transitions permute cells')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--out', required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phases = json.loads(args.phases) if args.phases else DEFAULT_PHASES

    split = D.make_split(seed=0)
    inst = D.load_instances()
    assert (inst['group'] == split['group']).all(), 'instance groups differ from make_split(0)'
    train_tasks = np.sort(split['train'])
    assert (split['group'][train_tasks] == 0).all() and len(train_tasks) == 2419
    assert not np.isin(train_tasks // G.NUM_C, split['heldout_t_elements']).any()
    heldout_t = split['heldout_t_elements'].tolist()

    cfg = dict(vars(args), phases=phases, n_train_tasks=len(train_tasks), heldout_t=heldout_t)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))

    model = StructureModel(mode=args.mode, d=args.d, generators=args.generators, cmap=args.cmap,
                           sinkhorn_iters=args.sinkhorn_iters, cmap_eps=args.cmap_eps, tau=args.tau, beta=args.beta,
                           gen_init=args.gen_init, seed=args.seed, compose=args.compose).to(device)
    fixed = {'log_tau'} if args.tau_sched else set()
    fixed |= {'log_beta'} if args.beta_sched else set()
    params = [p for name, p in model.named_parameters() if name not in fixed]
    n_params = sum(p.numel() for p in params)
    print(f'{args.mode} d={model.d} {args.generators} cmap={args.cmap}: {n_params} trainable parameters; out {out}')
    opt = torch.optim.Adam(params, lr=args.lr) if params else None
    sampler = TorchSampler(train_tasks, device, seed=10_000 + args.seed)
    train_set = torch.as_tensor(train_tasks, device=device)

    def run_eval(step):
        model.eval()
        res = evaluate(model, inst, device)
        diag = diagnostics(model, heldout_t)
        model.train()
        with open(out / 'evals.jsonl', 'a') as f:
            f.write(json.dumps(dict(step=step, eval=res, diag=diag)) + '\n')
        print(f'--- eval at step {step}\n{format_eval(res)}\n  {format_diag(diag)}', flush=True)
        return res, diag

    log = open(out / 'train.jsonl', 'a')
    acc, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        ph = phase_at(phases, step)
        if args.tau_sched:
            model.set_tau_beta(tau=schedule(args.tau_sched, step))
        if args.beta_sched:
            model.set_tau_beta(beta=schedule(args.beta_sched, step))
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        if step > args.warmup:
            frac = (step - args.warmup) / max(args.steps - args.warmup, 1)
            lr = args.lr * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
        b = sampler.batch(args.batch, ph['k'], tuple(ph['sparsity']))
        assert bool(torch.isin(b['task'], train_set).all()), 'a non-training task was drawn'
        res = model(b['demo_in'], b['demo_out'], b['demo_mask'], b['query_in'])
        loss = nll(res['probs'], b['query_out'])
        true_t = b['task'] // G.NUM_C
        lp_true = res['log_post'].gather(1, true_t[:, None]).squeeze(1)
        total = loss - args.aux_pi * lp_true.mean() if args.aux_pi else loss
        if args.closure:
            total = total + args.closure * model.closure_loss()
        if opt is not None:
            for gr in opt.param_groups:
                gr['lr'] = lr
            opt.zero_grad(set_to_none=True)
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            opt.step()
        else:
            gnorm = 0.0
        with torch.no_grad():
            pred = res['probs'].argmax(-1)
            cell = (pred == b['query_out']).float()
            post = res['log_post'].exp()
            ent = -(post * res['log_post']).sum(1)
            acc.append(torch.stack([loss.detach(), cell.min(1).values.mean(), cell.mean(), lp_true.exp().mean(),
                                    ent.mean()]))
        if step % args.log_every == 0 or step == args.steps:
            m = torch.stack(acc).mean(0).tolist()
            acc = []
            rec = dict(step=step, loss=m[0], exact=m[1], cell=m[2], post_true_t=m[3], post_entropy=m[4],
                       tau=model.tau.item(), beta=model.beta.item(), lr=lr, gnorm=gnorm, phase=ph,
                       sec=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(f'step {step:6d} loss {m[0]:.4f} exact {m[1]:.3f} cell {m[2]:.4f} pi(true t) {m[3]:.3f} '
                  f'H {m[4]:.2f} tau {rec["tau"]:.2f} beta {rec["beta"]:.3f} gnorm {gnorm:.3f} {rec["sec"]}s',
                  flush=True)
        if args.eval_every and step % args.eval_every == 0 and step != args.steps:
            run_eval(step)
    log.close()

    res, diag = run_eval(args.steps)
    counts = sampler.counts.cpu().numpy()
    assert counts[split['group'] != 0].sum() == 0, 'a group-1/2 task was drawn during training'
    (out / 'eval.json').write_text(json.dumps(res, indent=1))
    (out / 'diag.json').write_text(json.dumps(diag, indent=1))
    (out / 'train_tasks.json').write_text(json.dumps(dict(
        train_tasks=train_tasks.tolist(), heldout_t=heldout_t,
        groups_of_train_tasks=np.bincount(split['group'][train_tasks], minlength=3).tolist(),
        drawn_samples=int(counts.sum()), drawn_by_group=[int(counts[split['group'] == g].sum()) for g in range(3)],
        drawn_counts={int(t): int(c) for t, c in enumerate(counts) if c})))
    torch.save(dict(model=model.state_dict(), config=cfg), out / 'final.pt')
    print(f'done in {time.time() - t0:.0f}s; wrote {out}')
    return res, diag


if __name__ == '__main__':
    main()
