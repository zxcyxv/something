"""Inference-only removal of LTD (or, as the mirror control, LTP) in the trained Rule 110 run-3 model.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_drop_ltd.py [--seed 0] [--n 1024] [--blocks 32]

LTD = the part of the same-block timing term whose window sin(phi_V - phi_K) is negative; it is subtracted from G
(G' = G - sum_n v_n k~_n^T * min(W, 0)); cross-block pairs and everything else are kept. Scale-changing and
out of distribution: it shows what this model depends on, not whether a model trained without LTD would do better.
Conditions: none, drop LTD at blocks 2..B, drop LTD at blocks 4..B (after the answers settle), and the same two for LTP.
Run to --blocks blocks (trained with 8) on held-out rows unsolved after block 1. Reported per block: exact, cell
accuracy, mean margin, state drift ||h_b - h_{b-1}|| / ||h_b||, and read norms (Hebbian read, timing read, total read).
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
from rule110_ltp_ltd import T, HIDDEN, DEVICE, OUT            # noqa: E402


def forward_states(inner, x, blocks):
    batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=x.device))
    h = inner.init_hidden.to(torch.float32).expand(x.shape[0], T, HIDDEN).clone()
    inj = inner.injection(batch)
    memory = e_k = e_v = None
    layer = inner.layers[0]
    logits, drift = [], []
    for _ in range(blocks):
        h_new, memory, e_k, e_v = inner.block(layer, h, inj, memory, e_k, e_v, None)
        drift.append(float(((h_new - h).norm(dim=-1) / h_new.norm(dim=-1)).mean()))
        h = h_new
        logits.append(inner.w_cls(h))
    return torch.stack(logits), drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--blocks', type=int, default=32)
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
    nb = opt.blocks
    conds = {'none': None,
             'drop LTD, blocks 2..': ('ltd', set(range(1, nb))), 'drop LTD, blocks 4..': ('ltd', set(range(3, nb))),
             'drop LTP, blocks 2..': ('ltp', set(range(1, nb))), 'drop LTP, blocks 4..': ('ltp', set(range(3, nb)))}
    show = [b for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if b <= nb]
    report = {}
    for label, drop in conds.items():
        ctl['block'], ctl['norms'] = 0, []
        if drop is not None:
            ctl['drop'] = drop
        with torch.no_grad():
            lgs, drift = forward_states(inner, x, nb)
        ctl.pop('drop', None)
        norms = ctl.pop('norms')
        pred = lgs.argmax(-1) - 1
        exact = (pred == y).all(-1).float().mean(-1).tolist()
        cell = (pred == y).float().mean((-1, -2)).tolist()
        margin = rd.margins(lgs, y).mean((-1, -2)).tolist()
        report[label] = dict(exact=exact, cell=cell, margin=margin, drift=drift, norms=norms)
        print(f'\n[{label}]')
        print('  block      ' + ' '.join(f'{b:>7d}' for b in show))
        print('  exact      ' + ' '.join(f'{exact[b - 1]:7.3f}' for b in show))
        print('  cell acc   ' + ' '.join(f'{cell[b - 1]:7.4f}' for b in show))
        print('  margin     ' + ' '.join(f'{margin[b - 1]:7.2f}' for b in show))
        print('  drift      ' + ' '.join(f'{drift[b - 1]:7.3f}' for b in show))
        print('  |Hebb read|' + ' '.join(f'{norms[b - 1]["hebb"]:7.2f}' for b in show))
        print('  |timing rd|' + ' '.join(f'{norms[b - 1]["timing"]:7.2f}' for b in show))
        print('  |total rd| ' + ' '.join(f'{norms[b - 1]["total"]:7.2f}' for b in show))
    (OUT / 'results' / f'{name}_drop_ltd.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
