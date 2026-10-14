"""Do the read edges into a cell, at the block it decides, point at the cells that logically justify the decision?

Usage (repository root):  python docs/research/2026-10-14/scripts/stdp_justification.py RUN_DIR STEP [N=512] [BLOCKS=128] [BATCH=256]

Solver side (L0 synchronous singles, sudoku_depth.py): a cell t fixed at round r with digit d is justified by
  naked single   every candidate but d is eliminated by a fixed peer: J = the peers fixed before round r (all of them carry
                 an eliminated digit)
  hidden single  in some unit U of t, no other cell can take d: J = the other 8 cells of U (the fixed ones cannot take d, the
                 unfixed ones must carry "d is excluded for me", which a one-hop model can only receive from them)
A cell can be both; both tests are run where they apply.
Model side (torch path, shadow traces): the exact per-source shares of four reads into t at the block t decides (argmax
final from there on, correct, L0 depth finite): Hebb q~ Hebb^T, order alpha q~ O^T, latency alpha q~ L^T, and the same-block
latency part alone alpha (1-lambda) q~ SB^T (SB = sum_s sin(a_V - a_K) (x) v_s k~_s^T; a = c sin phi).
For each read: |share| mass on the justification cells against the chance level (their number / 80), and the per-cell density
ratio against the other peers; the same two blocks earlier (the order read peaks there).
Writes docs/research/2026-10-14/results/justification_<run>_step<STEP>_n<N>_b<BLOCKS>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sudoku_depth as sd                                     # noqa: E402
from lt import train as t                                     # noqa: E402
from lt import research_phase_clock_stdp as rp                # noqa: E402
from lt.kv_stability import ORIGINAL_MODEL_ID                 # noqa: E402

REPO = sd.REPO
run, step = Path(sys.argv[1]).resolve(strict=True), int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 256
protocol = json.loads((run / 'protocol.json').read_text())
WT, LAM = float(protocol['omega_t']), float(protocol['lambda_init'])
REC = dict(n=0, shadow=None, cap=None)
ORIG_STEP = rp.phase_clock_step
READS = ('hebb', 'order', 'latency', 'sameblock')
LEADS = (0, 2)


def step_hook(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    hebb, im_s, new = ORIG_STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    shift = 2 * REC['n'] * WT
    shadow = traces if REC['shadow'] is None else REC['shadow']
    _, im_s2, shadow_new = ORIG_STEP(k_unit, kr, v, rope, pk - shift, pv - shift, lk, lv, shadow)
    REC['shadow'] = shadow_new
    REC['cap'] = dict(hebb=hebb, O=(im_s - im_s2) / 2, L=(im_s + im_s2) / 2, kr=kr, v=v, pk=pk, pv=pv, lv=lv, old=traces, new=new,
                      shadow_old=shadow, shadow_new=shadow_new, shift=shift, rope=rope)
    return hebb, im_s, new


rp.phase_clock_step = step_hook


def timing_contrib(qr, v, kr, tau_v, tau_k, ekr_re, ekr_im, dv_re, dv_im, r, alpha):
    tr = r.transpose(-1, -2)
    z_r, z_i = qr @ ekr_re.transpose(-1, -2), qr @ ekr_im.transpose(-1, -2)
    A, Bm = ((v * tau_v.sin()) @ tr).transpose(-1, -2), ((v * tau_v.cos()) @ tr).transpose(-1, -2)
    w_c, w_s = qr @ (kr * tau_k.cos()).transpose(-1, -2), qr @ (kr * tau_k.sin()).transpose(-1, -2)
    Cm, Dm = (dv_im @ tr).transpose(-1, -2), (dv_re @ tr).transpose(-1, -2)
    return alpha * (z_r * A + z_i * Bm + w_c * Cm - w_s * Dm)


def justifications(x, y):
    """For every cell fixed by L0 singles: naked flag and the units where it is a hidden single, from the previous round's state."""
    N = x.shape[0]
    cand = sd.initial_state(x)
    depth = np.where(x > 0, 0, -1)
    naked = np.zeros((N, 81), dtype=bool)
    hidden_units = np.zeros((N, 81, 27), dtype=bool)
    fixed_before = []                                                          # per round r: fixed mask before round r
    for r in range(1, 200):
        pc = sd.POP[cand]
        fixed = pc == 1
        fixed_before.append(fixed.copy())
        new, bad = sd.singles_pass(cand)
        if bad.any():
            raise RuntimeError('contradiction')
        newly = (sd.POP[new] == 1) & ~fixed
        if not newly.any():
            break
        b = sd.bits(cand)                                                      # [N,81,9]
        for u in range(27):
            cells = sd.UNITS[u]
            ub = b[:, cells, :] & ~fixed[:, cells, None]                       # unfixed cells' candidates
            cnt = ub.sum(1)                                                    # [N,9]
            for j, c in enumerate(cells):
                lanes = np.nonzero(newly[:, c])[0]
                if lanes.size == 0:
                    continue
                d = np.log2(new[lanes, c].astype(np.float64)).astype(int)
                ok = (cnt[lanes, d] == 1) & ub[lanes, j, d]
                hidden_units[lanes[ok], c, u] = True
        naked |= newly & ~hidden_units.any(-1)                                 # singleton by elimination, not set by a hidden single
        depth[newly] = r
        cand = new
    return depth, naked, hidden_units, fixed_before


def main():
    ck = torch.load(run / f'step_{step}.pt', map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'], global_batch_size=BATCH)
    cfg['data_npz'] = str(REPO / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol.get('precision', 'highest'))
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    _, _, x_np, y_np, _, _ = t.load_data(cfg)
    x_np, y_np = x_np[:N], y_np[:N]
    x, y = x_np.reshape(-1, 81).astype(int), y_np.reshape(-1, 81).astype(int)
    began = time.monotonic()
    depth, naked, hidden_units, fixed_before = justifications(x, y)
    print(f'solver {time.monotonic() - began:.0f}s: L0-fixed cells {int((depth > 0).sum())}, naked {int((naked & (depth > 0)).sum())}, '
          f'hidden {int((hidden_units.any(-1) & (depth > 0)).sum())}, both {int((naked & hidden_units.any(-1)).sum())}')
    peers = np.zeros((81, 81), dtype=bool)
    for u in sd.UNITS:
        peers[np.ix_(u, u)] = True
    np.fill_diagonal(peers, False)
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'], kernel='torch')
    inner = base.model.inner
    layer = inner.layers[0]
    H, D, T = inner.H, inner.dh, 81
    alpha = layer.stdp_alpha.detach().float()[None, :, None, None]
    peers_t = torch.from_numpy(peers).to(device)
    acc = {}                                                                   # (read, lead, test, key) -> [sum, count]

    def add(key, val, n=1):
        s = acc.setdefault(key, [0.0, 0])
        s[0] += float(val); s[1] += n
    lane0 = 0
    with torch.no_grad():
        for batch in t.eval_batches(x_np, y_np, BATCH, 0, 1):
            b = {k: v.to(device) for k, v in batch.items()}
            B = b['inputs'].shape[0]
            inj = inner.injection(b)
            h = inner.init_hidden.float().expand(B, T, inner.d).clone()
            mem = ek = ev = None
            REC.update(n=0, shadow=None, cap=None)
            preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
            shares = {r: torch.zeros(BLOCKS, B, T, T, dtype=torch.float16, device=device) for r in READS}
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
                a_k, a_v = rp._rep2(cap['pk'] - (blk - 1) * WT), rp._rep2(cap['pv'] - (blk - 1) * WT)
                sb = (v * a_v.sin()).transpose(-1, -2) @ (kr * a_k.cos()) - (v * a_v.cos()).transpose(-1, -2) @ (kr * a_k.sin())
                r = dict(hebb=qr @ cap['hebb'].transpose(-1, -2), order=alpha * (qr @ cap['O'].transpose(-1, -2)),
                         latency=alpha * (qr @ cap['L'].transpose(-1, -2)), sameblock=alpha * (1 - LAM) * (qr @ sb.transpose(-1, -2)))
                tau_v, tau_k = rp._rep2(cap['pv']), rp._rep2(cap['pk'])
                lv2 = rp._rep2(cap['lv'])
                plus = dict(tau_v=tau_v, tau_k=tau_k, ekr_re=cap['rope'](cap['new'][0]), ekr_im=cap['rope'](cap['new'][1]),
                            dv_re=lv2 * cap['old'][2], dv_im=lv2 * cap['old'][3])
                minus = dict(tau_v=tau_v - cap['shift'], tau_k=tau_k - cap['shift'], ekr_re=cap['rope'](cap['shadow_new'][0]),
                             ekr_im=cap['rope'](cap['shadow_new'][1]), dv_re=lv2 * cap['shadow_old'][2], dv_im=lv2 * cap['shadow_old'][3])
                contrib = {'hebb': (qr @ kr.transpose(-1, -2)) * (v @ r['hebb'].transpose(-1, -2)).transpose(-1, -2)}
                for name, sign in (('order', -1), ('latency', 1)):
                    contrib[name] = (timing_contrib(qr, v, kr, r=r[name], alpha=alpha, **plus)
                                     + sign * timing_contrib(qr, v, kr, r=r[name], alpha=alpha, **minus)) / 2
                # same-block latency read: read_t[d] = sum_s v_s[d] [sin a_V,s[d] C_ts - cos a_V,s[d] S_ts]
                tr = r['sameblock'].transpose(-1, -2)
                C = qr @ (kr * a_k.cos()).transpose(-1, -2)
                Sm = qr @ (kr * a_k.sin()).transpose(-1, -2)
                P1 = ((v * a_v.sin()) @ tr).transpose(-1, -2)
                P2 = ((v * a_v.cos()) @ tr).transpose(-1, -2)
                contrib['sameblock'] = alpha * (1 - LAM) * (C * P1 - Sm * P2)
                for name in READS:
                    share = contrib[name] / (r[name].norm(dim=-1, keepdim=True) ** 2 + 1e-12)
                    if blk == 3:
                        big = r[name].norm(dim=-1) > 1e-4
                        err = float(((share.sum(-1) - 1).abs() * big).max())
                        if err > 1e-2:
                            raise AssertionError(f'{name} shares do not sum to 1: {err:.2e}')
                    shares[name][blk - 1] = share.mean(1).to(torch.float16)
                preds[blk - 1] = inner.w_cls(h).argmax(-1).to(torch.int16)
            final = preds[-1]
            suffix_same = (preds == final[None]).flip(0).cumprod(0).flip(0).bool()
            dec = suffix_same.float().argmax(0) + 1                                     # [B,T]
            correct = final.long() == b['labels'].long()
            dep = torch.from_numpy(depth[lane0:lane0 + B]).to(device)
            nak = torch.from_numpy(naked[lane0:lane0 + B]).to(device)
            hid = torch.from_numpy(hidden_units[lane0:lane0 + B]).to(device)
            fb = [torch.from_numpy(f[lane0:lane0 + B]).to(device) for f in fixed_before]
            sel = (dep > 0) & correct & (dec >= 3) & (dec <= BLOCKS)
            lanes, cells = sel.nonzero(as_tuple=True)
            r_idx = dep[lanes, cells]                                                   # L0 round
            fixed_prev = torch.stack([fb[int(rr) - 1][l] for l, rr in zip(lanes.tolist(), r_idx.tolist())]) if lanes.numel() else None
            eye = torch.eye(T, dtype=torch.bool, device=device)
            for lead in LEADS:
                blk_idx = (dec[lanes, cells] - lead - 1).clamp(0, BLOCKS - 1)
                for name in READS:
                    row = shares[name][blk_idx, lanes, cells].float().abs()             # [M,T]
                    row = row.masked_fill(eye[cells], 0)
                    tot = row.sum(-1) + 1e-12
                    pm = peers_t[cells]                                                 # [M,T] peers of t
                    # naked: justification = fixed peers before round r
                    nk = nak[lanes, cells]
                    if nk.any():
                        J = pm & fixed_prev                                             # [M,T]
                        other_peer = pm & ~J
                        nJ, nO = J.float().sum(-1), other_peer.float().sum(-1)
                        okn = nk & (nJ > 0) & (nO > 0)
                        massJ = (row * J).sum(-1) / tot
                        chance = nJ / 80
                        dens = ((row * J).sum(-1) / nJ.clamp_min(1)) / (((row * other_peer).sum(-1) / nO.clamp_min(1)) + 1e-12)
                        add((name, lead, 'naked', 'mass_on_J'), massJ[okn].sum(), int(okn.sum()))
                        add((name, lead, 'naked', 'chance'), chance[okn].sum(), int(okn.sum()))
                        add((name, lead, 'naked', 'density_J_vs_other_peers'), dens[okn].sum(), int(okn.sum()))
                        add((name, lead, 'naked', 'mass_on_peers'), ((row * pm).sum(-1) / tot)[okn].sum(), int(okn.sum()))
                    # hidden: justification = the other 8 cells of the unit(s) where t is a hidden single
                    hu = hid[lanes, cells]                                              # [M,27]
                    hh = hu.any(-1)
                    if hh.any():
                        units_t = torch.from_numpy(sd.UNITS).to(device)                 # [27,9]
                        Jh = torch.zeros(lanes.numel(), T, dtype=torch.bool, device=device)
                        for uu in range(27):
                            m = hu[:, uu]
                            if m.any():
                                Jh[m.nonzero()[:, 0][:, None], units_t[uu][None]] = True
                        Jh &= ~eye[cells]
                        other_peer = pm & ~Jh
                        nJ, nO = Jh.float().sum(-1), other_peer.float().sum(-1)
                        okh = hh & (nO > 0)
                        massJ = (row * Jh).sum(-1) / tot
                        dens = ((row * Jh).sum(-1) / nJ.clamp_min(1)) / (((row * other_peer).sum(-1) / nO.clamp_min(1)) + 1e-12)
                        add((name, lead, 'hidden', 'mass_on_J'), massJ[okh].sum(), int(okh.sum()))
                        add((name, lead, 'hidden', 'chance'), (nJ / 80)[okh].sum(), int(okh.sum()))
                        add((name, lead, 'hidden', 'density_J_vs_other_peers'), dens[okh].sum(), int(okh.sum()))
                        unf = Jh & ~fixed_prev
                        fx = Jh & fixed_prev
                        add((name, lead, 'hidden', 'J_mass_unfixed_part'), ((row * unf).sum(-1) / tot)[okh].sum(), int(okh.sum()))
                        add((name, lead, 'hidden', 'J_mass_fixed_part'), ((row * fx).sum(-1) / tot)[okh].sum(), int(okh.sum()))
                        add((name, lead, 'hidden', 'J_unfixed_count'), unf.float().sum(-1)[okh].sum(), int(okh.sum()))
            lane0 += B
            del shares
    out = dict(run=run.name, step=step, n=N, blocks=BLOCKS, seconds=time.monotonic() - began, results={})
    for (name, lead, test, key), (s, n) in acc.items():
        out['results'].setdefault(f'{name}|lead{lead}|{test}', {})[key] = s / max(n, 1)
        out['results'][f'{name}|lead{lead}|{test}']['cells'] = n
    path = REPO / 'docs/research/2026-10-14/results' / f'justification_{run.name}_step{step}_n{N}_b{BLOCKS}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(f"step {step}: {out['seconds']:.0f}s.  Read into cell t at its decision block (lead 0) and two blocks earlier (lead 2).")
    for test in ('naked', 'hidden'):
        print(f"[{test} single] justification J = {'fixed peers before the round' if test == 'naked' else 'the other 8 cells of the unit'}")
        print('   read       lead  cells   mass on J  chance   density J / other peers' + ('   mass on peers' if test == 'naked' else '   J mass: unfixed part / fixed part (unfixed cells in J)'))
        for name in READS:
            for lead in LEADS:
                k = f'{name}|lead{lead}|{test}'
                if k not in out['results']:
                    continue
                v = out['results'][k]
                extra = f"   {v['mass_on_peers']:.3f}" if test == 'naked' else f"   {v['J_mass_unfixed_part']:.3f} / {v['J_mass_fixed_part']:.3f} ({v['J_unfixed_count']:.1f})"
                print(f"   {name:9s} {lead:4d}  {v['cells']:6d}   {v['mass_on_J']:.3f}      {v['chance']:.3f}    {v['density_J_vs_other_peers']:.2f}" + extra)
    print('WROTE', path)


if __name__ == '__main__':
    main()
