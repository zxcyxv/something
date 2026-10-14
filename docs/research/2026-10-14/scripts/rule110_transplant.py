"""Timing transplant in the trained Rule 110 run-3 model: replace only the firing phases (tags fed to the write) of one
cell with those of another cell of the same row, keeping every activity unchanged. Does the decision follow the timing?

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_transplant.py [--seed 0] [--n 1024]

Rows: held-out rows unsolved after block 1. Target cell c in 8..27 (one run per c). Donors, same row, d != c:
  opposite  same input bit as c, opposite answer      (the test)
  same      same input bit as c, same answer           (control: any transplant)
Neuron sets: all 16 K + 16 V, or only the six neurons of the three answer-switching synapses found at block 8
(V1<-K14, V5<-K10, V3<-K11: K14, K10, K11, V1, V5, V3). Tags are replaced at every block from block 2 on.
Reported: P(final decision at c != its answer); per-block for c; decision changes at offsets +-1..+-4 vs the
unperturbed run; the shift of c's margin toward the donor's answer.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_dissect as rd                                  # noqa: E402
from rule110_ltp_ltd import T, BLOCKS, DEVICE, OUT            # noqa: E402


def donors(x, y, c, opposite, gen):
    """Per row, a random cell d != c with x[d] == x[c] and y[d] != / == y[c]; -1 when none."""
    B = x.shape[0]
    ok = (x == x[:, c:c + 1]) & ((y != y[:, c:c + 1]) if opposite else (y == y[:, c:c + 1]))
    ok[:, c] = False
    score = torch.rand(B, T, device=DEVICE, generator=gen).masked_fill(~ok, -1)
    d = score.argmax(-1)
    return torch.where(ok.any(-1), d, torch.full_like(d, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=1024)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    name = f'rule110_k{r110.K_STEPS}_run3_seed{opt.seed}'
    ck = torch.load(OUT / 'checkpoints' / f'{name}.pt', map_location=DEVICE)
    base, inner = r110.build('run3', opt.seed)
    base.load_state_dict(ck['state'])
    base.eval()
    ctl = {}
    rd.install(inner, ctl)
    xt, yt = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    lg = rd.run(inner, ctl, xt)
    uns = ~((lg[0].argmax(-1) - 1) == yt).all(-1)
    x, y = xt[uns][:opt.n], yt[uns][:opt.n]
    base_lg = rd.run(inner, ctl, x)
    base_dec = base_lg.argmax(-1) - 1                                             # [blocks, B, T]
    all_k = torch.ones(16, dtype=torch.bool, device=DEVICE)
    six_k = torch.zeros(16, dtype=torch.bool, device=DEVICE)
    six_v = torch.zeros(16, dtype=torch.bool, device=DEVICE)
    six_k[[13, 9, 10]] = True                                                     # K14, K10, K11
    six_v[[0, 4, 2]] = True                                                       # V1, V5, V3
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    results = {}
    for label, mk, mv in (('all 32 neurons', all_k, all_k), ('6 switching neurons', six_k, six_v)):
        for kind in ('opposite', 'same'):
            own_wrong, own_per_block, n_cases, shift = 0.0, torch.zeros(BLOCKS, device=DEVICE), 0, 0.0
            base_own_wrong = 0.0
            nb_change = {o: [0.0, 0] for o in (-4, -3, -2, -1, 1, 2, 3, 4)}
            for c in range(8, 28):
                d = donors(x, y, c, kind == 'opposite', gen)
                keep = d >= 0
                xs, ys, ds = x[keep], y[keep], d[keep]
                cc = torch.full_like(ds, c)
                ctl['transplant'] = (cc, ds, mk, mv)
                lgt = rd.run(inner, ctl, xs)
                ctl.pop('transplant')
                dec = lgt.argmax(-1) - 1
                n = int(keep.sum())
                n_cases += n
                own_wrong += float((dec[-1, :, c] != ys[:, c]).float().sum())
                base_own_wrong += float((base_dec[-1][keep][:, c] != ys[:, c]).float().sum())
                own_per_block += (dec[:, :, c] != ys[None, :, c]).float().sum(1)
                donor_ans = ys.gather(1, ds[:, None])[:, 0]
                m = lgt[-1, :, c]
                mb = base_lg[-1][keep][:, c]
                toward = lambda z: z.gather(1, (donor_ans + 1)[:, None])[:, 0] - z.gather(1, (2 - donor_ans)[:, None])[:, 0]
                shift += float((toward(m) - toward(mb)).sum())
                for o in nb_change:
                    t_ = c + o
                    nb_change[o][0] += float((dec[-1, :, t_] != base_dec[-1][keep][:, t_]).float().sum())
                    nb_change[o][1] += n
            key = f'{label} / {kind}'
            results[key] = dict(cases=n_cases, own_wrong=own_wrong / n_cases, own_wrong_unperturbed=base_own_wrong / n_cases,
                                own_wrong_per_block=(own_per_block / n_cases).tolist(), margin_shift_toward_donor=shift / n_cases,
                                neighbour_change={o: v[0] / v[1] for o, v in nb_change.items()})
            r = results[key]
            print(f'\n[{key}] cases {n_cases}')
            print(f'  cell c final decision != its answer: {r["own_wrong"]:.3f}   (unperturbed {r["own_wrong_unperturbed"]:.4f})')
            print('  per block (blocks 1..8):              ' + ' '.join(f'{v:.3f}' for v in r['own_wrong_per_block']))
            print(f'  c margin shift toward the donor answer: {r["margin_shift_toward_donor"]:+.2f}')
            print('  neighbours, decisions changed vs unperturbed:  '
                  + '  '.join(f'{o:+d}: {v:.3f}' for o, v in r['neighbour_change'].items()))
    (OUT / 'results' / f'{name}_transplant.json').write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
