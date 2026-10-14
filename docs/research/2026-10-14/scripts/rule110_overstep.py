"""When the trained Rule 110 model degrades past its horizon, does its output drift toward the state after MORE than 4
Rule 110 steps?

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_overstep.py [--ckpt NAME] [--blocks 128] [--smax 40]

Checkpoint: the 1x8 leaky (rho 0.5) bounded hybrid model trained 10k steps with AdamATan2 (default). Held-out 4,096 rows.
At selected blocks b, the per-cell prediction is compared with evolve(x, s) for s = 0..smax:
  agreement  mean fraction of matching cells
  exact      number of rows whose 36 cells all match
Baseline: the same agreement between the true answer evolve(x, 4) and evolve(x, s) (what a model that kept the right
answer would show). Among rows wrong at block b, the step s with the highest agreement per row is tallied.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_hybrid as hy                                   # noqa: E402
import rule110_leaky_train as lk                              # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT, rp        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='rule110_k4_hybrid_wT0.34_lam0.5_bound0.14_leaky0.5_adamatan2_lr0.001_wd0.1_s10000_seed0')
    ap.add_argument('--blocks', type=int, default=128)
    ap.add_argument('--smax', type=int, default=40)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(OUT / 'checkpoints' / f'{opt.ckpt}.pt', map_location=DEVICE)
    hy.CTL.update(omega_t=ck['omega_t'], bound=ck['bound'], record=None)
    rp.advance_clock, rp.phase_clock_step = hy.advance, lk.step
    lk.ACC.update(on=True, rho=ck['rho'], W=None, norms=None)
    base, inner = lk.build(ck['seed'])
    base.load_state_dict(ck['state'])
    base.eval()
    x, y = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    steps = torch.stack([r110.evolve(x, s) for s in range(opt.smax + 1)])            # [S+1, B, T]
    show = [b for b in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128) if b <= opt.blocks]
    preds = {}
    hy.CTL['block'], lk.ACC['W'] = 0, None
    batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=DEVICE))
    with torch.no_grad():
        h = inner.init_hidden.float().expand(x.shape[0], T, HIDDEN).clone()
        inj = inner.injection(batch)
        mem = ek = ev = None
        for b in range(1, opt.blocks + 1):
            h, mem, ek, ev = inner.block(inner.layers[0], h, inj, mem, ek, ev, None)
            if b in show:
                preds[b] = inner.w_cls(h).argmax(-1) - 1
    base_agree = (steps == steps[4][None]).float().mean((-1, -2))                   # answer vs step s
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 40]
    cols = [s for s in cols if s <= opt.smax]
    print('agreement (fraction of matching cells) with the state after s Rule 110 steps')
    print('  s =                ' + ' '.join(f'{s:>6d}' for s in cols))
    print('  answer (baseline)  ' + ' '.join(f'{float(base_agree[s]):6.3f}' for s in cols))
    report = dict(baseline=base_agree.tolist(), blocks={})
    for b in show:
        p = preds[b]
        agree = (steps == p[None]).float().mean(-1)                                  # [S+1, B]
        exact = (steps == p[None]).all(-1)                                           # [S+1, B]
        wrong = ~exact[4]
        best = agree[:, wrong].argmax(0)                                             # best s per wrong row
        tally = Counter(int(v) for v in best.tolist())
        other_exact = exact[[s for s in range(opt.smax + 1) if s != 4]][:, wrong].any(0)
        per_s = {s: int(exact[s, wrong].sum()) for s in range(opt.smax + 1) if s != 4 and int(exact[s, wrong].sum()) > 0}
        print(f'  block {b:3d}          ' + ' '.join(f'{float(agree[s].mean()):6.3f}' for s in cols)
              + f'   | wrong rows {int(wrong.sum()):4d}; exact match to another s: {int(other_exact.sum())} {per_s if per_s else ""}')
        report['blocks'][b] = dict(agreement=agree.mean(1).tolist(), wrong=int(wrong.sum()), exact_other=int(other_exact.sum()),
                                   exact_other_by_s=per_s, best_s_tally=dict(tally.most_common(8)))
    print('\nbest-matching s among wrong rows (top 6, count):')
    for b in show:
        print(f'  block {b:3d}: ' + ', '.join(f's={s}: {c}' for s, c in report['blocks'][b]['best_s_tally'].items()))
    (OUT / 'results' / f'{opt.ckpt}_overstep.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
