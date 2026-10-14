"""Does the solution propagate as a chain of decisions carried by the order (STDP) read?  Bounded phase-clock checkpoint.

Usage (repository root):  python docs/research/2026-10-14/scripts/stdp_wave.py RUN_DIR STEP [N=512] [BLOCKS=128] [BATCH=256]

Same forward pass and source decomposition as stdp_order_decomp.py (EMA weights, torch path, FP32, no grad; shadow traces
with reversed block spacing give the order part O_b of Im S_b).  Per cell t the decision block d_t is the block from which
its argmax stays at its final value.  For every cell that decided (5 <= d_t <= BLOCKS - 4) we look at the order read two
blocks before its decision (the peak found on 10-14) and ask where it came from, using the exact per-source shares
share_ts (sum_s share_ts = 1), the cell itself excluded:
  top source      s* = argmax_s |share_ts|; its lag d_t - d_{s*} (positive: the source decided earlier) and whether it is a
                  constraint neighbour (row / column / box, 20 cells)
  recent mass     sum of |share_ts| over sources that decided 1..4 blocks before t, divided by sum_s |share_ts|; chance = the
                  fraction of the other 80 cells with that lag
  weighted lag    histogram of d_t - d_s weighted by |share_ts|
Given cells never decide (d = 1), so the recent-source mass and the blank-source top are taken over blank sources only.
The Hebb read is decomposed the same way as a control (share_ts = (q~_t . k~_s)(v_s . r_t) / |r_t|^2).
Front structure: per lane, mean |d_t - d_s| over constraint-neighbour pairs against non-neighbour pairs (cells that changed
at least once), for solved and unsolved lanes.
Writes docs/research/2026-10-14/results/wave_<run>_step<STEP>_n<N>_b<BLOCKS>.json.
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from lt import train as t                                     # noqa: E402
from lt import research_phase_clock_stdp as rp                # noqa: E402
from lt.kv_stability import ORIGINAL_MODEL_ID                 # noqa: E402

run, step = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 256
LEAD = 2                                                       # blocks before the decision at which the read is examined
protocol = json.loads((run / 'protocol.json').read_text())
if protocol.get('clock') != 'bounded':
    raise ValueError('bounded clock only')
WT = float(protocol['omega_t'])
REC = dict(n=0, shadow=None, cap=None)
ORIG_STEP = rp.phase_clock_step


def step_hook(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = ORIG_STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    shift = 2 * REC['n'] * WT
    shadow = traces if REC['shadow'] is None else REC['shadow']
    _, im_s2, shadow_new = ORIG_STEP(k_unit, kr, v, rope, pk - shift, pv - shift, lk, lv, shadow)
    REC['shadow'] = shadow_new
    REC['cap'] = dict(hebb=hebb, O=(im_s - im_s2) / 2, kr=kr, v=v, pk=pk, pv=pv, lv=lv, old=traces, new=new,
                      shadow_old=shadow, shadow_new=shadow_new, shift=shift, rope=rope)
    return hebb, im_s, new


rp.phase_clock_step = step_hook


def neighbour_mask(device):
    idx = torch.arange(81, device=device)
    r, c = idx // 9, idx % 9
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (((r // 3)[:, None] == (r // 3)[None]) & ((c // 3)[:, None] == (c // 3)[None]))
    return same & ~torch.eye(81, dtype=torch.bool, device=device)


def timing_contrib(qr, v, kr, tau_v, tau_k, ekr_re, ekr_im, dv_re, dv_im, r, alpha):
    tr = r.transpose(-1, -2)
    z_r, z_i = qr @ ekr_re.transpose(-1, -2), qr @ ekr_im.transpose(-1, -2)
    A, Bm = ((v * tau_v.sin()) @ tr).transpose(-1, -2), ((v * tau_v.cos()) @ tr).transpose(-1, -2)
    w_c, w_s = qr @ (kr * tau_k.cos()).transpose(-1, -2), qr @ (kr * tau_k.sin()).transpose(-1, -2)
    Cm, Dm = (dv_im @ tr).transpose(-1, -2), (dv_re @ tr).transpose(-1, -2)
    return alpha * (z_r * A + z_i * Bm + w_c * Cm - w_s * Dm)


def main():
    ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'], global_batch_size=BATCH)
    cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    _, _, x, y, _, _ = t.load_data(cfg)
    x, y = x[:N], y[:N]
    began = time.monotonic()
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'], kernel='torch')
    inner = base.model.inner
    layer = inner.layers[0]
    H, D, T = inner.H, inner.dh, 81
    alpha = layer.stdp_alpha.detach().float()[None, :, None, None]
    nb = neighbour_mask(device)
    eye = torch.eye(T, dtype=torch.bool, device=device)
    lags = list(range(-8, 9))
    reads = ('order', 'hebb')
    tot = {r: dict(cells=0, top_nb=0.0, top_self_excluded=True, top_lag_hist=torch.zeros(len(lags) + 2, dtype=torch.float64),
                   recent_mass=0.0, recent_chance=0.0, top_given=0.0, topb_nb=0.0, topb_lag_hist=torch.zeros(len(lags) + 2, dtype=torch.float64), wlag_hist=torch.zeros(len(lags) + 2, dtype=torch.float64),
                   top_decided_before=0.0) for r in reads}
    front = dict(solved=[], unsolved=[])
    n_lanes = 0
    with torch.no_grad():
        for batch in t.eval_batches(x, y, BATCH, 0, 1):
            b = {k: v.to(device) for k, v in batch.items()}
            B = b['inputs'].shape[0]
            n_lanes += B
            inj = inner.injection(b)
            h = inner.init_hidden.float().expand(B, T, inner.d).clone()
            mem = ek = ev = None
            REC.update(n=0, shadow=None, cap=None)
            preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
            shares = {r: torch.zeros(BLOCKS, B, T, T, dtype=torch.float16, device=device) for r in reads}   # head-mean
            tables = inner.rope_tables(layer)
            for blk in range(1, BLOCKS + 1):
                REC['n'] = blk - 1
                u = h + inner.embed_scale * inj
                q = layer.q_proj(u).reshape(B, T, H, D).transpose(1, 2).float()
                q = q / (q.norm(dim=-1, keepdim=True) + inner.config.eps)
                qr = inner.apply_rope(q, layer, tables)
                h, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                cap = REC['cap']
                v, kr = cap['v'], cap['kr']
                r = dict(hebb=qr @ cap['hebb'].transpose(-1, -2), order=alpha * (qr @ cap['O'].transpose(-1, -2)))
                tau_v, tau_k = rp._rep2(cap['pv']), rp._rep2(cap['pk'])
                lv2 = rp._rep2(cap['lv'])
                plus = dict(tau_v=tau_v, tau_k=tau_k, ekr_re=cap['rope'](cap['new'][0]), ekr_im=cap['rope'](cap['new'][1]),
                            dv_re=lv2 * cap['old'][2], dv_im=lv2 * cap['old'][3])
                minus = dict(tau_v=tau_v - cap['shift'], tau_k=tau_k - cap['shift'], ekr_re=cap['rope'](cap['shadow_new'][0]),
                             ekr_im=cap['rope'](cap['shadow_new'][1]), dv_re=lv2 * cap['shadow_old'][2], dv_im=lv2 * cap['shadow_old'][3])
                contrib = {'hebb': (qr @ kr.transpose(-1, -2)) * (v @ r['hebb'].transpose(-1, -2)).transpose(-1, -2),
                           'order': (timing_contrib(qr, v, kr, r=r['order'], alpha=alpha, **plus)
                                     - timing_contrib(qr, v, kr, r=r['order'], alpha=alpha, **minus)) / 2}
                for name in reads:
                    share = contrib[name] / (r[name].norm(dim=-1, keepdim=True) ** 2 + 1e-12)
                    shares[name][blk - 1] = share.mean(1).to(torch.float16)
                preds[blk - 1] = inner.w_cls(h).argmax(-1).to(torch.int16)
            # decisions
            final = preds[-1]
            suffix_same = (preds == final[None]).flip(0).cumprod(0).flip(0).bool()
            dec = suffix_same.float().argmax(0) + 1                                     # [B,T]
            right = (preds.long() == b['labels'].long()[None]).all(-1)
            solved = right.flip(0).cumprod(0).flip(0).bool()[-1]                         # [B]
            given = (b['inputs'] - 1) != 0                                               # [B,T]
            ok = (dec >= 5) & (dec <= BLOCKS - 4)
            lane_idx, cell_idx = ok.nonzero(as_tuple=True)
            blk_idx = dec[lane_idx, cell_idx] - LEAD - 1                                 # 0-based block index
            lag_all = dec[lane_idx, cell_idx][:, None] - dec[lane_idx]                   # [M, T]: d_t - d_s
            self_mask = eye[cell_idx]                                                    # [M, T]
            given_src = given[lane_idx]                                                  # [M, T]
            blank_src = ~given_src & ~self_mask
            recent = (lag_all >= 1) & (lag_all <= 4) & blank_src
            chance = recent.float().sum(-1) / blank_src.float().sum(-1).clamp_min(1)
            def hist(vals, w):
                hcounts = torch.zeros(len(lags) + 2, dtype=torch.float64)
                for j, lg in enumerate(lags):
                    hcounts[j] = float(w[vals == lg].sum())
                hcounts[-2] = float(w[vals < lags[0]].sum())
                hcounts[-1] = float(w[vals > lags[-1]].sum())
                return hcounts
            for name in reads:
                row = shares[name][blk_idx, lane_idx, cell_idx].float()                 # [M, T]
                row = row.masked_fill(self_mask, 0)
                ab = row.abs()
                top = ab.argmax(-1)
                top_lag = lag_all.gather(1, top[:, None])[:, 0]
                st = tot[name]
                st['cells'] += int(ok.sum())
                st['top_nb'] += float(nb[cell_idx, top].float().sum())
                st['top_given'] += float(given_src.gather(1, top[:, None])[:, 0].float().sum())
                abb = ab * blank_src
                topb = abb.argmax(-1)
                st['topb_lag_hist'] += hist(lag_all.gather(1, topb[:, None])[:, 0], torch.ones_like(topb, dtype=torch.float))
                st['topb_nb'] += float(nb[cell_idx, topb].float().sum())
                st['recent_mass'] += float(((abb * recent).sum(-1) / (abb.sum(-1) + 1e-12)).sum())
                st['top_decided_before'] += float((top_lag >= 1).float().sum())
                st['top_lag_hist'] += hist(top_lag, torch.ones_like(top_lag, dtype=torch.float))
                st['recent_chance'] += float(chance.sum())
                st['wlag_hist'] += hist(lag_all.flatten(), (ab / (ab.sum(-1, keepdim=True) + 1e-12)).flatten())
            # front structure: |d_t - d_s| over neighbour vs non-neighbour pairs, cells that changed
            changed = dec >= 2
            dd = (dec[:, :, None] - dec[:, None, :]).abs().float()
            pair_ok = changed[:, :, None] & changed[:, None, :] & ~eye[None]
            nbp, nnp = pair_ok & nb[None], pair_ok & ~nb[None]
            for name, m in (('solved', solved), ('unsolved', ~solved)):
                for lane in m.nonzero()[:, 0].tolist():
                    if nbp[lane].sum() > 0 and nnp[lane].sum() > 0:
                        front[name].append((float(dd[lane][nbp[lane]].mean()), float(dd[lane][nnp[lane]].mean())))
            del shares
    out = dict(run=run.name, step=step, n=n_lanes, blocks=BLOCKS, lead=LEAD, lags=lags + ['<-8', '>8'], seconds=time.monotonic() - began)
    for name in reads:
        st = tot[name]
        m = max(st['cells'], 1)
        out[name] = dict(cells=st['cells'], top_neighbour=st['top_nb'] / m, top_decided_before=st['top_decided_before'] / m,
                         top_given=st['top_given'] / m, top_blank_neighbour=st['topb_nb'] / m, top_blank_lag_hist=(st['topb_lag_hist'] / m).tolist(),
                         recent_mass=st['recent_mass'] / m, recent_chance=st['recent_chance'] / m,
                         top_lag_hist=(st['top_lag_hist'] / m).tolist(), weighted_lag_hist=(st['wlag_hist'] / m).tolist())
    for name in front:
        if front[name]:
            a = torch.tensor(front[name])
            out[f'front_{name}'] = dict(lanes=len(front[name]), neighbour=float(a[:, 0].mean()), non_neighbour=float(a[:, 1].mean()),
                                        ratio=float((a[:, 0] / a[:, 1]).mean()))
    path = REPO / 'docs/research/2026-10-14/results' / f'wave_{run.name}_step{step}_n{n_lanes}_b{BLOCKS}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(f"step {step}: {n_lanes} lanes, {BLOCKS} blocks, read examined {LEAD} blocks before each cell's decision; {out['seconds']:.0f}s")
    for name in reads:
        o = out[name]
        print(f"[{name} read] cells {o['cells']}: top source is a constraint neighbour {o['top_neighbour']:.3f} (chance 0.250), a given {o['top_given']:.3f}; "
              f"top source decided before t {o['top_decided_before']:.3f}; among blank sources: top is a neighbour {o['top_blank_neighbour']:.3f}, "
              f"recent (lag 1..4) share mass {o['recent_mass']:.3f} vs chance {o['recent_chance']:.3f}")
        print('   lag d_t - d_s:    ' + ' '.join(f'{str(l):>5s}' for l in out['lags']))
        print('   top source      ' + ' '.join(f'{v:5.3f}' for v in o['top_lag_hist']))
        print('   |share|-weighted' + ' '.join(f'{v:5.3f}' for v in o['weighted_lag_hist']))
        print('   top blank source' + ' '.join(f'{v:5.3f}' for v in o['top_blank_lag_hist']))
    for name in ('solved', 'unsolved'):
        if f'front_{name}' in out:
            f = out[f'front_{name}']
            print(f"[front] {name}: mean |d_t - d_s| neighbours {f['neighbour']:.2f} vs non-neighbours {f['non_neighbour']:.2f} (ratio {f['ratio']:.3f}; {f['lanes']} lanes)")
    print('WROTE', path)


if __name__ == '__main__':
    main()
