"""Does the trained Rule 110 run-3 model simulate the automaton one step per block? Probe the intermediate states.

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_unfolding.py [--seed 0]

Target: the cell's bit after s Rule 110 steps, s = 0 (input) .. 4 (the answer). Intermediate bits are correlated
with the input bit and the answer, so the score is conditional: within each of the 4 strata (input bit, answer bit),
the probe's accuracy minus that stratum's majority rate, averaged over strata weighted by size. A probe that only
knows the input and the answer scores 0 for every s; s = 0 and s = 4 are 0 by construction.
Features per cell at block b: the firing phases (cos, sin of 16 K + 16 V tags), and, for reference, the hidden state h.
Logistic probes, held-out rows split 70/30 by row.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule110_ltp_ltd as r110                                # noqa: E402
import rule110_dissect as rd                                  # noqa: E402
from rule110_ltp_ltd import T, HIDDEN, BLOCKS, DEVICE, OUT    # noqa: E402


def probe_score(feat, tgt, strata, split):
    clf = torch.nn.Linear(feat.shape[-1], 2).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    xtr, ytr = feat[:split].reshape(-1, feat.shape[-1]), tgt[:split].reshape(-1)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(clf(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = clf(feat[split:].reshape(-1, feat.shape[-1])).argmax(-1)
    yt, st = tgt[split:].reshape(-1), strata[split:].reshape(-1)
    score, raw = 0.0, float((pred == yt).float().mean())
    for s in range(4):
        m = st == s
        if int(m.sum()) == 0:
            continue
        maj = float(torch.bincount(yt[m], minlength=2).max()) / int(m.sum())
        score += (float((pred[m] == yt[m]).float().mean()) - maj) * int(m.sum()) / yt.numel()
    return score, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    name = f'rule110_k{r110.K_STEPS}_run3_seed{opt.seed}'
    ck = torch.load(OUT / 'checkpoints' / f'{name}.pt', map_location=DEVICE)
    base, inner = r110.build('run3', opt.seed)
    base.load_state_dict(ck['state'])
    base.eval()
    ctl = {}
    rd.install(inner, ctl)
    x, y = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    steps = [r110.evolve(x, s) for s in range(r110.K_STEPS + 1)]
    strata = 2 * x + y
    print('share of cells whose step-s bit differs from both the input bit and the answer: '
          + ' '.join(f's{s} {float(((steps[s] != x) & (steps[s] != y)).float().mean()):.3f}' for s in range(5)))
    # forward with recorded phases and hidden states
    ctl['block'], ctl['record'] = 0, []
    hs = []
    with torch.no_grad():
        batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=DEVICE))
        h = inner.init_hidden.float().expand(x.shape[0], T, HIDDEN).clone()
        inj = inner.injection(batch)
        mem = ek = ev = None
        for _ in range(BLOCKS):
            h, mem, ek, ev = inner.block(inner.layers[0], h, inj, mem, ek, ev, None)
            hs.append(h.clone())
    recs = ctl.pop('record')
    split = int(0.7 * x.shape[0])
    report = {}
    for fname in ('phases', 'hidden'):
        print(f'\n[{fname}] conditional score (accuracy above the within-(input, answer) majority), rows = block, cols = step s')
        print('          ' + ' '.join(f'   s={s}' for s in range(5)))
        rows = []
        for b in range(BLOCKS):
            if fname == 'phases':
                pk, pv = recs[b]['pk'], recs[b]['pv']
                feat = torch.cat([pk.cos(), pk.sin(), pv.cos(), pv.sin()], -1)
            else:
                feat = hs[b]
            scores = [probe_score(feat, steps[s], strata, split) for s in range(5)]
            rows.append(dict(block=b + 1, score=[sc for sc, _ in scores], raw=[r_ for _, r_ in scores]))
            print(f'  block {b + 1} ' + ' '.join(f'{sc:+.3f}' for sc, _ in scores)
                  + '    raw acc ' + ' '.join(f'{r_:.3f}' for _, r_ in scores))
        report[fname] = rows
    (OUT / 'results' / f'{name}_unfolding.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
