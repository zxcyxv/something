"""Where does the correction of an early confident error come from in the trained Rule 110 run-3 model?

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_correction.py [--seed 0] [--thr 2]

Block 1 has zero tags, so its decision is Hebbian only. Cells:
  corrected  block-1 margin < -thr (confidently wrong) and correct at block 8
  control    block-1 margin > +thr (confidently right) and correct at block 8
(margin = correct logit - wrong logit of the per-block readout; held-out 4,096 rows.)
1. Margin and first-order evidence (Hebb / LTP / LTD / cross-block, summed over source cells, as in
   rule110_dissect.py) per block, mean over the cells of each group.
2. Interventions (blocks 2..8), success = the cell is correct at block 8:
   flip-self       LTP <-> LTD for the cell's own tags
   flip-neighbours LTP <-> LTD for the tags of cells at offsets +-1..+-3 (not the cell)
   transplant-wrong   the cell's tags <- those of a same-row cell with the same input bit whose final answer is
                      the cell's block-1 (wrong) answer
   transplant-right   ... whose final answer is the cell's correct answer (control)
   One target cell per row and pass; passes repeat until every selected cell has been a target once.
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
from rule110_ltp_ltd import T, BLOCKS, DEVICE, OUT            # noqa: E402


def evidence(recs, mg, b):
    """Per-cell first-order evidence at block index b: dict of [B, T] (summed over source cells)."""
    r = recs[b]
    g = torch.autograd.grad(mg[b].sum(), r['read'], retain_graph=True)[0][:, 0]
    qr, kr, v = r['qr'].detach(), r['kr'].detach(), r['v'].detach()
    out = dict(hebb=[], ltp=[], ltd=[])
    for s in range(0, g.shape[0], 512):
        sl = slice(s, s + 512)
        w = r['alpha'] * (1 - rd.rep2(r['lk'][sl]))[..., None, :] * torch.sin(
            rd.rep2(r['pv'][sl])[..., :, None] - rd.rep2(r['pk'][sl])[..., None, :])
        kq = torch.einsum('bni,bti->btn', kr[sl], qr[sl])
        out['hebb'].append((kq * torch.einsum('btd,bnd->btn', g[sl], v[sl])).sum(-1))
        for key, ww in (('ltp', w.clamp(min=0)), ('ltd', w.clamp(max=0))):
            gvw = torch.einsum('bta,bna,bnai->btni', g[sl], v[sl], ww)
            out[key].append(torch.einsum('btni,bni,bti->btn', gvw, kr[sl], qr[sl]).sum(-1))
    out = {k: torch.cat(vs) for k, vs in out.items()}
    out['cross'] = (g * r['timing_read']).sum(-1) - out['ltp'] - out['ltd']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--thr', type=float, default=2.0)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    name = f'rule110_k{r110.K_STEPS}_run3_seed{opt.seed}'
    ck = torch.load(OUT / 'checkpoints' / f'{name}.pt', map_location=DEVICE)
    base, inner = r110.build('run3', opt.seed)
    base.load_state_dict(ck['state'])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    ctl = {}
    rd.install(inner, ctl)
    x, y = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    # per-block margins and evidence, in row chunks (the recorded graph of all rows does not fit)
    m_parts, ev = [], {b: {k: [] for k in ('hebb', 'ltp', 'ltd', 'cross')} for b in range(1, BLOCKS)}
    for s in range(0, x.shape[0], 512):
        ctl['record'] = []
        lg = rd.run(inner, ctl, x[s:s + 512], grad=True)
        recs = ctl.pop('record')
        mg = rd.margins(lg, y[s:s + 512])
        for b in range(1, BLOCKS):
            for k, v_ in evidence(recs, mg, b).items():
                ev[b][k].append(v_.detach())
        m_parts.append(mg.detach())
        del recs, mg, lg
    m = torch.cat(m_parts, 1)
    ev = {b: {k: torch.cat(v_) for k, v_ in d.items()} for b, d in ev.items()}
    groups = dict(corrected=(m[0] < -opt.thr) & (m[-1] > 0), control=(m[0] > opt.thr) & (m[-1] > 0))
    stuck = (m[0] < -opt.thr) & (m[-1] <= 0)
    print(f'cells 4096 x 36 = {x.numel()}; block-1 margin quantiles 5/50/95%: '
          + ' '.join(f'{v:.2f}' for v in torch.quantile(m[0].flatten(), torch.tensor([0.05, 0.5, 0.95], device=DEVICE)).tolist()))
    print(f'corrected {int(groups["corrected"].sum())}   control {int(groups["control"].sum())}   '
          f'confidently wrong at block 1 and still wrong at block 8 {int(stuck.sum())}')

    # 1. margin and evidence per block
    report = dict(threshold=opt.thr, counts={k: int(v.sum()) for k, v in groups.items()}, stuck=int(stuck.sum()), trajectory={})
    print('\n[1] mean per block: margin | evidence Hebb  LTP  LTD  cross (first order, at that block\'s readout)')
    for gname, sel in groups.items():
        print(f'  {gname}')
        rows = []
        print(f'    block 1: margin {float(m[0][sel].mean()):+6.2f} | (no timing term)')
        for b in range(1, BLOCKS):
            e = {k: float(v_[sel].mean()) for k, v_ in ev[b].items()}
            rows.append(dict(block=b + 1, margin=float(m[b][sel].mean()), **e))
            print(f'    block {b + 1}: margin {float(m[b][sel].mean()):+6.2f} | {e["hebb"]:+6.2f} {e["ltp"]:+6.2f} {e["ltd"]:+6.2f} {e["cross"]:+6.2f}'
                  f'   share of cells with margin > 0: {float((m[b][sel] > 0).float().mean()):.3f}')
        report['trajectory'][gname] = rows
    del ev

    # 2. interventions
    print('\n[2] interventions at blocks 2..8; success = correct at block 8')
    early_wrong_answer = 1 - y                                                     # the block-1 (wrong) answer of corrected cells
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    report['interventions'] = {}
    for gname, sel in groups.items():
        res = {}
        for mode in ('flip-self', 'flip-neighbours', 'transplant-wrong', 'transplant-right'):
            remaining = sel.clone()
            ok_sum, n_sum = 0.0, 0
            while bool(remaining.any()):
                score = torch.rand(remaining.shape, device=DEVICE, generator=gen).masked_fill(~remaining, -1)
                c = score.argmax(-1)
                rows = remaining.any(-1)
                remaining[rows.nonzero()[:, 0], c[rows]] = False
                xs, ys, cs = x[rows], y[rows], c[rows]
                B = xs.shape[0]
                ar = torch.arange(B, device=DEVICE)
                if mode.startswith('flip'):
                    mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
                    if mode == 'flip-self':
                        mask[ar, cs] = True
                    else:
                        for o in (-3, -2, -1, 1, 2, 3):
                            t_ = cs + o
                            ok = (t_ >= 0) & (t_ < T)
                            mask[ar[ok], t_[ok]] = True
                    ctl.update(flip_blocks=set(range(1, BLOCKS)), flip_mask=mask)
                    keep = torch.ones(B, dtype=torch.bool, device=DEVICE)
                else:
                    want = early_wrong_answer[rows].gather(1, cs[:, None]) if gname == 'corrected' and mode == 'transplant-wrong' \
                        else ys.gather(1, cs[:, None])
                    if gname == 'control' and mode == 'transplant-wrong':
                        want = 1 - ys.gather(1, cs[:, None])
                    cand = (xs == xs.gather(1, cs[:, None])) & (ys == want)
                    cand[ar, cs] = False
                    sc = torch.rand(B, T, device=DEVICE, generator=gen).masked_fill(~cand, -1)
                    keep = cand.any(-1)
                    allk = torch.ones(16, dtype=torch.bool, device=DEVICE)
                    ctl['transplant'] = (cs[keep], sc.argmax(-1)[keep], allk, allk)
                xs, ys, cs = xs[keep], ys[keep], cs[keep]
                if 'flip_mask' in ctl:
                    ctl['flip_mask'] = ctl['flip_mask'][keep]
                lgt = rd.run(inner, ctl, xs)
                for k in ('flip_blocks', 'flip_mask', 'transplant'):
                    ctl.pop(k, None)
                dec = lgt[-1].argmax(-1) - 1
                ok_sum += float((dec.gather(1, cs[:, None])[:, 0] == ys.gather(1, cs[:, None])[:, 0]).float().sum())
                n_sum += int(cs.numel())
            res[mode] = dict(success=ok_sum / max(n_sum, 1), cases=n_sum)
            print(f'  {gname:9s} {mode:17s} success {ok_sum / max(n_sum, 1):.3f}   (cases {n_sum})')
        report['interventions'][gname] = res
    (OUT / 'results' / f'{name}_correction.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
