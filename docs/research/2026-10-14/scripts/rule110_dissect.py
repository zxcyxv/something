"""Dissect how LTP/LTD enter the decisions of the trained Rule 110 run-3 model (rule110_ltp_ltd.py checkpoint).

Usage (repository root):  python docs/research/2026-10-14/scripts/rule110_dissect.py [--seed 0] [--n 1024]

All analyses use held-out rows that the block-1 readout gets wrong.
1. Which block's LTP/LTD matters: flip the sign of every phase difference (LTP <-> LTD, magnitudes kept) at one block only.
2. Where a cell's timing acts: flip the tags of one source cell (all blocks) and measure the change of every cell's
   final decision margin, by offset target - source.
3. Evidence decomposition of each decision: at block b, the read of target cell t is a sum over source cells n of
     Hebb      (k~_n . q~_t) v_n
     LTP/LTD   same-block timing term  sum_i v_n,a k~_n,i q~_t,i W_n,a,i,  W = alpha (1 - lam_K,i) sin(phi_V,a - phi_K,i),
               split by the sign of W (W > 0: LTP, W < 0: LTD)
     cross     the rest of alpha Im S (pairs with events of earlier blocks)
   Each part is projected on g_t = d margin_t / d read_t (margin = correct logit - wrong logit of the block-b readout;
   cells do not interact after the read inside a block, so g_t is per cell). First-order attribution.
4. What the phases encode: logistic probes from (cos, sin) of the 32 tags of a cell to its input bit, its input
   neighbourhood (8 classes) and its target bit, on held-out rows (70/30 split).
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
from rule110_ltp_ltd import T, BLOCKS, DEVICE, OUT, rp         # noqa: E402


def rep2(x):
    return x.repeat_interleave(2, -1)


def install(inner, ctl):
    """Torch-path memory_step of the run-3 class with tag interventions and recording (one head)."""
    eps = inner.config.eps

    def memory_step(layer, q, k, v, memory=None, e_k=None, e_v=None, fresh=None, hidden=None):
        q, k, v = q.float(), k.float(), v.float()
        q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        tables = inner.rope_tables(layer)

        def rope(x):
            return inner.apply_rope(x, layer, tables)
        qr, kr = rope(q), rope(k)
        clk, traces = inner.unpack_state(memory, e_k, e_v, fresh, k)
        pk, pv, lk, lv, clk = inner.advance(layer, hidden, clk, torch.float32, k_unit=k, v=v)
        b = ctl['block']
        ctl['block'] += 1
        flip = torch.zeros(T, dtype=torch.bool, device=q.device)
        if b in ctl.get('flip_blocks', ()):
            flip = ctl.get('flip_cells', torch.ones(T, dtype=torch.bool, device=q.device))
        f = flip[None, None, :, None]
        if b in ctl.get('flip_blocks', ()) and 'flip_mask' in ctl:
            f = ctl['flip_mask'][:, None, :, None]                                  # per-row cells [B, T]
        pk_u, pv_u = torch.where(f, -pk, pk), torch.where(f, -pv, pv)
        tp = ctl.get('transplant')
        if tp is not None and b >= 1:
            # tags of cell c[r] replaced by those of donor cell d[r] in the same row (selected neurons only);
            # activities and the clock state are untouched
            c, d, mk, mv = tp
            rows = torch.arange(pk.shape[0], device=pk.device)
            pk_u, pv_u = pk_u.clone(), pv_u.clone()
            pk_u[rows, 0, c] = torch.where(mk, pk[rows, 0, d], pk_u[rows, 0, c])
            pv_u[rows, 0, c] = torch.where(mv, pv[rows, 0, d], pv_u[rows, 0, c])
        hebb, im_s, (ekr, eki, evr, evi) = rp.phase_clock_step(k, kr, v, rope, pk_u, pv_u, lk, lv, traces)
        alpha = layer.stdp_alpha.float()
        g_mat = hebb + alpha * im_s
        drop = ctl.get('drop')
        if drop is not None and b in drop[1]:
            # remove the LTD (sin < 0) or LTP (sin > 0) part of the same-block timing term; cross-block pairs kept
            w = alpha * (1 - rep2(lk))[..., None, :] * torch.sin(rep2(pv_u)[..., :, None] - rep2(pk_u)[..., None, :])
            part = w.clamp(max=0) if drop[0] == 'ltd' else w.clamp(min=0)
            g_mat = g_mat - torch.einsum('bhna,bhni,bhnai->bhai', v, kr, part)
        read = qr @ g_mat.transpose(-1, -2)
        if ctl.get('norms') is not None:
            hebb_read = qr @ hebb.transpose(-1, -2)
            ctl['norms'].append(dict(block=b + 1, hebb=float(hebb_read.norm(dim=-1).mean()),
                                     timing=float((read - hebb_read).norm(dim=-1).mean()), total=float(read.norm(dim=-1).mean())))
        if ctl.get('record') is not None:
            # leaf per block: d margin_b / d read_b is then the within-block (per-cell) derivative
            read = read.detach().requires_grad_(True)
            ctl['record'].append(dict(qr=qr[:, 0], kr=kr[:, 0], v=v[:, 0], pk=pk_u[:, 0], pv=pv_u[:, 0], lk=lk[:, 0],
                                      alpha=alpha.detach(), timing_read=(qr @ (alpha * im_s).transpose(-1, -2))[:, 0].detach(),
                                      read=read))
        return read, clk, torch.cat((ekr, eki), -1), torch.cat((evr, evi), -1)
    inner.memory_step = memory_step


def run(inner, ctl, x, grad=False):
    ctl['block'] = 0
    with torch.set_grad_enabled(grad):
        return r110.forward(inner, x)


def margins(logits, y):
    """correct logit - wrong logit, [blocks, B, T]; tokens are bit + 1."""
    correct = logits.gather(-1, (y + 1)[None, ..., None].expand(logits.shape[0], -1, -1, 1))[..., 0]
    wrong = logits.gather(-1, (2 - y)[None, ..., None].expand(logits.shape[0], -1, -1, 1))[..., 0]
    return correct - wrong


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
    for p in base.parameters():
        p.requires_grad_(False)
    ctl = {}
    install(inner, ctl)
    xt, yt = r110.sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    lg = run(inner, ctl, xt)
    unsolved = ~((lg[0].argmax(-1) - 1) == yt).all(-1)
    x, y = xt[unsolved][:opt.n], yt[unsolved][:opt.n]
    B = x.shape[0]
    base_lg = run(inner, ctl, x)
    base_m = margins(base_lg, y)
    report = dict(n_rows=B)
    print(f'rows (unsolved after block 1): {B}; final exact {float(((base_lg[-1].argmax(-1) - 1) == y).all(-1).float().mean()):.4f}')

    # 1. single-block flip
    print('\n[1] flip LTP<->LTD at one block only -> exact at that block / at block 8')
    rows = []
    for b in range(1, BLOCKS):                       # block index 0 has zero tags
        ctl.update(flip_blocks={b})
        lgb = run(inner, ctl, x)
        ex = ((lgb.argmax(-1) - 1) == y).all(-1).float().mean(-1)
        rows.append(dict(block=b + 1, exact_at_block=float(ex[b]), exact_final=float(ex[-1]),
                         exact_next=float(ex[min(b + 1, BLOCKS - 1)])))
        print(f'  block {b + 1}: at block {float(ex[b]):.3f}  next block {float(ex[min(b + 1, BLOCKS - 1)]):.3f}  final {float(ex[-1]):.3f}'
              f'   (unflipped at block: {float(((base_lg[b].argmax(-1) - 1) == y).all(-1).float().mean()):.3f})')
    report['single_block_flip'] = rows
    ctl.pop('flip_blocks', None)

    # 2. single-source-cell flip -> influence by offset
    print('\n[2] flip the tags of one source cell (all blocks) -> final margin change by offset (target - source)')
    sub = slice(0, min(B, 512))
    dm_sum = torch.zeros(2 * T - 1, device=DEVICE)
    change_sum = torch.zeros(2 * T - 1, device=DEVICE)
    count = torch.zeros(2 * T - 1, device=DEVICE)
    base_dec = (base_m[-1, sub] > 0)
    for n in range(8, 28):
        mask = torch.zeros(T, dtype=torch.bool, device=DEVICE)
        mask[n] = True
        ctl.update(flip_blocks=set(range(BLOCKS)), flip_cells=mask)
        mn = margins(run(inner, ctl, x[sub]), y[sub])[-1]
        offs = torch.arange(T, device=DEVICE) - n + T - 1
        dm_sum.index_add_(0, offs, (mn - base_m[-1, sub]).abs().mean(0))
        change_sum.index_add_(0, offs, ((mn > 0) != base_dec).float().mean(0))
        count.index_add_(0, offs, torch.ones(T, device=DEVICE))
    ctl.pop('flip_blocks', None)
    ctl.pop('flip_cells', None)
    infl = []
    for o in range(-8, 9):
        i = o + T - 1
        infl.append(dict(offset=o, mean_abs_dmargin=float(dm_sum[i] / count[i]), decision_change=float(change_sum[i] / count[i])))
        print(f'  offset {o:+d}: |d margin| {float(dm_sum[i] / count[i]):.3f}   decisions changed {float(change_sum[i] / count[i]):.4f}')
    far = [(o, i) for o, i in ((o, o + T - 1) for o in range(-(T - 1), T)) if abs(o) > 8 and count[i] > 0]
    print(f'  |offset| > 8 (mean): |d margin| {sum(float(dm_sum[i] / count[i]) for _, i in far) / len(far):.3f}   decisions changed '
          f'{sum(float(change_sum[i] / count[i]) for _, i in far) / len(far):.4f}')
    print(f'  baseline final |margin| mean {float(base_m[-1, sub].abs().mean()):.3f}')
    report['single_cell_flip'] = infl

    # 3. evidence decomposition
    print('\n[3] evidence for the correct answer at block b, by source offset (target - source); first-order, mean per decision')
    ctl['record'] = []
    lgr = run(inner, ctl, x, grad=True)
    mg = margins(lgr, y)
    recs = ctl.pop('record')
    decomp = []
    for b in (1, 2, 3, 7):
        r = recs[b]
        g = torch.autograd.grad(mg[b].sum(), r['read'], retain_graph=True)[0][:, 0]       # [B, T, D]
        qr, kr, v = r['qr'].detach(), r['kr'].detach(), r['v'].detach()
        w = r['alpha'] * (1 - rep2(r['lk']))[..., None, :] * torch.sin(rep2(r['pv'])[..., :, None] - rep2(r['pk'])[..., None, :])
        kq = torch.einsum('bni,bti->btn', kr, qr)                                          # [B, T_t, T_n]
        gv = torch.einsum('btd,bnd->btn', g, v)                                            # g_t . v_n
        hebb = kq * gv
        # sum_a g_t,a v_n,a sum_i W_n,a,i k~_n,i q~_t,i
        gvw = torch.einsum('bta,bna,bnai->btni', g, v, w.clamp(min=0))
        ltp = torch.einsum('btni,bni,bti->btn', gvw, kr, qr)
        gvw = torch.einsum('bta,bna,bnai->btni', g, v, w.clamp(max=0))
        ltd = torch.einsum('btni,bni,bti->btn', gvw, kr, qr)
        cross = (g * r['timing_read']).sum(-1) - ltp.sum(-1) - ltd.sum(-1)                # [B, T]
        mb = mg[b].detach()
        out = dict(block=b + 1, margin=float(mb.mean()), totals=dict(hebb=float(hebb.sum(-1).mean()), ltp=float(ltp.sum(-1).mean()),
                   ltd=float(ltd.sum(-1).mean()), cross=float(cross.mean())), by_offset=[])
        print(f'  block {b + 1}: mean margin {float(mb.mean()):.2f};  totals  Hebb {out["totals"]["hebb"]:+.2f}  LTP {out["totals"]["ltp"]:+.2f}'
              f'  LTD {out["totals"]["ltd"]:+.2f}  cross-block {out["totals"]["cross"]:+.3f}')
        idx = torch.arange(T, device=DEVICE)
        off = idx[None, :] - idx[:, None]                                                   # [t, n] -> n - t
        print('    source offset (n - t):  ' + ' '.join(f'{o:+6d}' for o in range(-5, 6)))
        for label, comp in (('Hebb', hebb), ('LTP', ltp), ('LTD', ltd)):
            vals = []
            for o in range(-5, 6):
                sel = (off == o)
                vals.append(float(comp[:, sel].mean()))
            far = float(comp[:, off.abs() > 5].sum() / (comp.shape[0] * T))
            out['by_offset'].append(dict(component=label, offsets=list(range(-5, 6)), mean=vals, far_sum_per_cell=far))
            print(f'    {label:5s}                    ' + ' '.join(f'{v_:+6.2f}' for v_ in vals) + f'   |offset|>5 total {far:+.2f}')
        decomp.append(out)
    report['evidence'] = decomp

    # 4. phase probes
    print('\n[4] logistic probes from a cell\'s tags (cos, sin of 16 K + 16 V tags) on held-out rows (70/30)')
    left, right = F.pad(x, (1, 0))[:, :-1], F.pad(x, (0, 1))[:, 1:]
    targets = dict(input_bit=x, neighbourhood=4 * left + 2 * x + right, target_bit=y)
    probes = []
    split = int(0.7 * B)
    for b in (1, 3, 7):
        r = recs[b]
        feat = torch.cat([r['pk'].cos(), r['pk'].sin(), r['pv'].cos(), r['pv'].sin()], -1).detach()   # [B, T, 64]
        for tname, tgt in targets.items():
            k = int(tgt.max()) + 1
            clf = torch.nn.Linear(feat.shape[-1], k).to(DEVICE)
            o = torch.optim.Adam(clf.parameters(), lr=0.05)
            xtr, ytr = feat[:split].reshape(-1, feat.shape[-1]), tgt[:split].reshape(-1)
            for _ in range(400):
                o.zero_grad()
                F.cross_entropy(clf(xtr), ytr).backward()
                o.step()
            with torch.no_grad():
                acc = float((clf(feat[split:].reshape(-1, feat.shape[-1])).argmax(-1) == tgt[split:].reshape(-1)).float().mean())
            chance = float(torch.bincount(tgt[split:].reshape(-1), minlength=k).max() / tgt[split:].numel())
            probes.append(dict(block=b + 1, target=tname, acc=acc, majority=chance))
            print(f'  block {b + 1}  {tname:13s} acc {acc:.3f}   (majority {chance:.3f})')
    report['phase_probes'] = probes
    (OUT / 'results' / f'{name}_dissect.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
