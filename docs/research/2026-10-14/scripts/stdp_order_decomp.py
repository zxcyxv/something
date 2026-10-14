"""Theorems 1 and 2 of 2026-10-14 on a bounded phase-clock checkpoint (EMA weights, torch path, FP32, no grad).

Usage (repository root):  python docs/research/2026-10-14/scripts/stdp_order_decomp.py RUN_DIR STEP [N=512] [BLOCKS=128] [BATCH=256]

Held-out puzzles, one lane per puzzle, BLOCKS blocks without segment boundaries (inference never detaches the carry),
all lanes fresh at block 1, so the block counter is n = b - 1 for every lane.

Theorem 1 (order / latency split).  Im S_b = O_b + L_b, where O_b is odd and L_b even under time reversal
omegaT -> -omegaT along the SAME trajectory.  Measured with shadow traces that see the reversed tags
tau' = tau - 2 n omegaT (same k^, v, phi):  O_b = (Im S(+) - Im S(-)) / 2,  L_b = (Im S(+) + Im S(-)) / 2.
At a fixed point of the recurrence O_b = 0 and L_b = kappa SB_b, where SB_b = sum_t sin(a_V - a_K) (x) v_t k~_t^T is
the same-block latency term without its (1 - lambda) weight (a = c sin phi) and
kappa = (1 - lambda)(1 - lambda^2) / (1 - 2 lambda cos omegaT + lambda^2).
Theorem 2 (velocity coupling).  For small per-block changes O_b ~ gamma Wr_b, Wr_b = sum_t (dv k~^T - v dk~^T),
gamma = sum_k k (1 - lambda) lambda^k sin(k omegaT).  The reads alpha q~ O^T (order read), alpha q~ L^T (latency read)
and q~ Hebb^T (Hebb read) are decomposed by source cell s exactly, per head:
share_ts = <contribution of s, r_t> / |r_t|^2, sum_s share_ts = 1 (checked).
Per block (mean over lanes and heads): |O|/|Hebb|, |L|/|Hebb|, |L|/|SB| and cos(L, SB), cos(O, gamma Wr), |O|/(gamma |Wr|),
state velocity |h_b - h_{b-1}|, read norms (Hebb, order, latency), the share of each read on the cell's 20 constraint
neighbours and on itself, and the |share|-weighted mean of the source change |dv_s| against the plain mean.
Per lane: solve block (all cells right from there on), settle block (|O|/|Hebb| below 10% of its own peak from there on);
per cell: decision block (argmax constant from there on) and the event-triggered order / latency / Hebb read norm around it,
against a control with the decision blocks shuffled across cells.
Writes docs/research/2026-10-14/results/order_decomp_<run>_step<STEP>_n<N>_b<BLOCKS>.json.
"""
import json
import math
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
protocol = json.loads((run / 'protocol.json').read_text())
if protocol.get('clock') != 'bounded':
    raise ValueError('bounded clock only')
WT, C, LAM = float(protocol['omega_t']), float(protocol['bound']), float(protocol['lambda_init'])
KAPPA = (1 - LAM) * (1 - LAM ** 2) / (1 - 2 * LAM * math.cos(WT) + LAM ** 2)
GAMMA = sum(k * (1 - LAM) * LAM ** k * math.sin(k * WT) for k in range(1, 200))
REC = dict(n=0, shadow=None, cap=None)
ORIG_STEP = rp.phase_clock_step


def step_hook(k_unit, kr, v, rope, pk, pv, lk, lv, traces):
    """Run the real step; run the time-reversed shadow step on the same events; capture everything for this block."""
    hebb, im_s, new = ORIG_STEP(k_unit, kr, v, rope, pk, pv, lk, lv, traces)
    shift = 2 * REC['n'] * WT
    shadow = traces if REC['shadow'] is None else REC['shadow']        # block 1: both zero
    _, im_s2, shadow_new = ORIG_STEP(k_unit, kr, v, rope, pk - shift, pv - shift, lk, lv, shadow)
    REC['shadow'] = shadow_new
    REC['cap'] = dict(hebb=hebb, im_s=im_s, O=(im_s - im_s2) / 2, L=(im_s + im_s2) / 2, k_unit=k_unit, kr=kr, v=v,
                      pk=pk, pv=pv, lv=lv, old=traces, new=new, shadow_old=shadow, shadow_new=shadow_new, shift=shift, rope=rope)
    return hebb, im_s, new


rp.phase_clock_step = step_hook


def fro(m):
    return m.flatten(-2).norm(dim=-1)


def cosine(a, b):
    return (a.flatten(-2) * b.flatten(-2)).sum(-1) / (fro(a) * fro(b) + 1e-12)


def neighbour_mask(device):
    idx = torch.arange(81, device=device)
    r, c = idx // 9, idx % 9
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (((r // 3)[:, None] == (r // 3)[None]) & ((c // 3)[:, None] == (c // 3)[None]))
    return same & ~torch.eye(81, dtype=torch.bool, device=device)


def spearman(a, b):
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def timing_contrib(qr, v, kr, tau_v, tau_k, ekr_re, ekr_im, dv_re, dv_im, r, alpha):
    """<contribution of source s to the timing read at target t, r_t>, [B,H,T_t,T_s] (exact; see phase_clock_step).
    Term 1: current v_s x RoPE(K trace)_s; term 2: decayed V trace_s x current k~_s."""
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
    H, D = inner.H, inner.dh
    alpha = layer.stdp_alpha.detach().float()[None, :, None, None]
    nb = neighbour_mask(device)
    T = 81
    reads = ('hebb', 'order', 'latency')
    keys = ['O_hebb', 'L_hebb', 'L_sb', 'cos_L_sb', 'cos_O_wr', 'O_gwr', 'dh', 'dv_plain'] + [f'read_{n}' for n in reads] \
        + [f'share_nb_{n}' for n in reads] + [f'share_self_{n}' for n in reads] + [f'dv_weighted_{n}' for n in reads]
    agg = {k: torch.zeros(BLOCKS, dtype=torch.float64) for k in keys}
    O_ratio_all, preds_all, labels_all = [], [], []
    read_all = {n: [] for n in reads}
    n_batches = 0
    with torch.no_grad():
        for batch in t.eval_batches(x, y, BATCH, 0, 1):
            n_batches += 1
            b = {k: v.to(device) for k, v in batch.items()}
            B = b['inputs'].shape[0]
            inj = inner.injection(b)
            h = inner.init_hidden.float().expand(B, T, inner.d).clone()
            mem = ek = ev = None
            REC.update(n=0, shadow=None, cap=None)
            prev_v = prev_kr = None
            O_ratio = torch.zeros(BLOCKS, B, device=device)
            preds = torch.zeros(BLOCKS, B, T, dtype=torch.int16, device=device)
            read_cell = {n: torch.zeros(BLOCKS, B, T, device=device) for n in reads}
            tables = inner.rope_tables(layer)
            for blk in range(1, BLOCKS + 1):
                REC['n'] = blk - 1
                u = h + inner.embed_scale * inj
                q = layer.q_proj(u).reshape(B, T, H, D).transpose(1, 2).float()
                q = q / (q.norm(dim=-1, keepdim=True) + inner.config.eps)
                qr = inner.apply_rope(q, layer, tables)
                h_new, mem, ek, ev = inner.block(layer, h, inj, mem, ek, ev, None)
                cap = REC['cap']
                O, L, hebb = cap['O'], cap['L'], cap['hebb']
                v, kr = cap['v'], cap['kr']
                a_k = rp._rep2(cap['pk'] - (blk - 1) * WT)
                a_v = rp._rep2(cap['pv'] - (blk - 1) * WT)
                sb = (v * a_v.sin()).transpose(-1, -2) @ (kr * a_k.cos()) - (v * a_v.cos()).transpose(-1, -2) @ (kr * a_k.sin())
                i = blk - 1
                agg['O_hebb'][i] += (fro(O) / fro(hebb)).mean().item()
                agg['L_hebb'][i] += (fro(L) / fro(hebb)).mean().item()
                agg['L_sb'][i] += (fro(L) / (fro(sb) + 1e-12)).mean().item()
                agg['cos_L_sb'][i] += cosine(L, sb).mean().item()
                agg['dh'][i] += (h_new - h).norm(dim=-1).mean().item()
                O_ratio[i] = (fro(O) / fro(hebb)).mean(1)
                if prev_v is not None:
                    wr = v.transpose(-1, -2) @ prev_kr - prev_v.transpose(-1, -2) @ kr
                    agg['cos_O_wr'][i] += cosine(O, wr).mean().item()
                    agg['O_gwr'][i] += (fro(O) / (GAMMA * fro(wr) + 1e-12)).mean().item()
                    dv = (v - prev_v).norm(dim=-1)                                              # [B,H,T_s]
                else:
                    dv = torch.zeros(B, H, T, device=device)
                agg['dv_plain'][i] += dv.mean().item()
                # reads per head and their exact source decompositions
                r = dict(hebb=qr @ hebb.transpose(-1, -2), order=alpha * (qr @ O.transpose(-1, -2)), latency=alpha * (qr @ L.transpose(-1, -2)))
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
                for name in reads:
                    norm2 = r[name].norm(dim=-1, keepdim=True) ** 2
                    share = contrib[name] / (norm2 + 1e-12)
                    if blk in (2, BLOCKS):
                        big = (norm2[..., 0] > 1e-4)
                        err = float(((share.sum(-1) - 1).abs() * big).max())
                        if err > 1e-2:
                            raise AssertionError(f'{name} shares do not sum to 1 at block {blk}: max error {err:.2e}')
                    read_cell[name][i] = r[name].norm(dim=-1).mean(1)
                    agg[f'read_{name}'][i] += read_cell[name][i].mean().item()
                    agg[f'share_nb_{name}'][i] += (share * nb).sum(-1).mean().item()
                    agg[f'share_self_{name}'][i] += share.diagonal(dim1=-2, dim2=-1).mean().item()
                    ab = share.abs()
                    agg[f'dv_weighted_{name}'][i] += ((ab * dv[:, :, None, :]).sum(-1) / (ab.sum(-1) + 1e-12)).mean().item()
                preds[i] = inner.w_cls(h_new).argmax(-1).to(torch.int16)
                prev_v, prev_kr, h = v, kr, h_new
            O_ratio_all.append(O_ratio.cpu()); preds_all.append(preds.cpu()); labels_all.append(b['labels'].cpu())
            for name in reads:
                read_all[name].append(read_cell[name].cpu())
    n_lanes = sum(p.shape[1] for p in preds_all)
    curves = {k: (v / n_batches).tolist() for k, v in agg.items()}                # means of per-batch means (equal batches)
    O_ratio = torch.cat(O_ratio_all, 1)                                              # [BLOCKS, N]
    preds = torch.cat(preds_all, 1).long()
    labels = torch.cat(labels_all, 0).long()
    read_cat = {n: torch.cat(v, 1) for n, v in read_all.items()}
    # solve block: all cells right from there on
    right = (preds == labels[None]).all(-1)                                          # [BLOCKS, N]
    suffix_ok = right.flip(0).cumprod(0).flip(0).bool()
    solved = suffix_ok[-1]
    solve_blk = torch.where(solved, suffix_ok.float().argmax(0) + 1, torch.full((n_lanes,), BLOCKS + 1))
    # settle block: |O|/|Hebb| below 10% of its own peak from there on
    thr = 0.1 * O_ratio.max(0).values
    below = O_ratio < thr[None]
    suffix_below = below.flip(0).cumprod(0).flip(0).bool()
    settled = suffix_below[-1]
    settle_blk = torch.where(settled, suffix_below.float().argmax(0) + 1, torch.full((n_lanes,), BLOCKS + 1))
    split = {}
    for name, m in (('solved', solved), ('unsolved', ~solved)):
        split[name] = dict(n=int(m.sum()), O_hebb=O_ratio[:, m].mean(1).tolist() if m.any() else [])
    # decision block per cell: argmax constant from there on
    final = preds[-1]
    same = preds == final[None]
    suffix_same = same.flip(0).cumprod(0).flip(0).bool()
    dec = suffix_same.float().argmax(0) + 1                                         # [N, T] (1 = never changed)
    offs = list(range(-4, 5))
    ok = (dec >= 5) & (dec <= BLOCKS - 4)
    gen = torch.Generator().manual_seed(0)
    shuffled = dec.flatten()[torch.randperm(dec.numel(), generator=gen)].reshape(dec.shape)
    ok_s = (shuffled >= 5) & (shuffled <= BLOCKS - 4) & ok
    eta = {}
    for name, src in read_cat.items():
        normed = src / (src.mean(0, keepdim=True) + 1e-12)                            # per-cell normalisation
        vals, ctrl = [], []
        for o in offs:
            idx = (dec + o - 1).clamp(0, BLOCKS - 1)
            vals.append(float(normed.gather(0, idx[None])[0][ok].mean()))
            idx2 = (shuffled + o - 1).clamp(0, BLOCKS - 1)
            ctrl.append(float(normed.gather(0, idx2[None])[0][ok_s].mean()))
        eta[name] = dict(offsets=offs, event=vals, shuffled_control=ctrl, cells=int(ok.sum()))
    lo = max(0, BLOCKS - 32)
    late = slice(lo, BLOCKS)
    mean_late = lambda k: float(sum(curves[k][late]) / (BLOCKS - lo))
    summary = dict(
        run=run.name, step=step, n=n_lanes, blocks=BLOCKS, kappa=KAPPA, gamma=GAMMA, lam=LAM, omega_t=WT, bound=C,
        alpha=[round(float(a), 4) for a in alpha.flatten()], seconds=time.monotonic() - began,
        solved=int(solved.sum()), settled=int(settled.sum()),
        solve_block_median=float(solve_blk[solved].float().median()) if solved.any() else None,
        settle_block_median_solved=float(settle_blk[solved & settled].float().median()) if (solved & settled).any() else None,
        settle_block_median_unsolved=float(settle_blk[~solved & settled].float().median()) if (~solved & settled).any() else None,
        unsolved_never_settled=int((~solved & ~settled).sum()), solved_never_settled=int((solved & ~settled).sum()),
        spearman_solve_vs_settle_solved=spearman(solve_blk[solved & settled].float(), settle_blk[solved & settled].float()) if int((solved & settled).sum()) > 2 else None,
        late_blocks=dict(first=lo + 1, O_hebb_solved=float(O_ratio[late][:, solved].mean()) if solved.any() else None,
                         O_hebb_unsolved=float(O_ratio[late][:, ~solved].mean()) if (~solved).any() else None,
                         L_sb=mean_late('L_sb'), L_sb_predicted=KAPPA, cos_L_sb=mean_late('cos_L_sb')),
        curves=curves, split=split, event_triggered=eta,
        solve_block=solve_blk.tolist(), settle_block=settle_blk.tolist())
    out = REPO / 'docs/research/2026-10-14/results' / f'order_decomp_{run.name}_step{step}_n{n_lanes}_b{BLOCKS}.json'
    out.write_text(json.dumps(summary, indent=1) + '\n')
    show = [b for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if b <= BLOCKS]
    print(f"step {step}: {n_lanes} lanes, {BLOCKS} blocks; solved {summary['solved']}, settled(O) {summary['settled']}; "
          f"alpha {summary['alpha']}; kappa {KAPPA:.3f} gamma {GAMMA:.3f}; {summary['seconds']:.0f}s")
    print('block                  ' + ' '.join(f'{b:7d}' for b in show))
    for k in keys:
        print(f'  {k:20s} ' + ' '.join(f'{curves[k][b - 1]:7.3f}' for b in show))
    for name in ('solved', 'unsolved'):
        if split[name]['O_hebb']:
            print(f"  O/Hebb {name:13s} " + ' '.join(f'{split[name]["O_hebb"][b - 1]:7.3f}' for b in show) + f"  (n={split[name]['n']})")
    lb = summary['late_blocks']
    print(f"late blocks ({lo + 1}-{BLOCKS}): O/Hebb solved {lb['O_hebb_solved']}, unsolved {lb['O_hebb_unsolved']}; "
          f"|L|/|SB| {lb['L_sb']:.3f} (fixed-point prediction kappa = {KAPPA:.3f}), cos(L,SB) {lb['cos_L_sb']:.3f}")
    print(f"solve block median {summary['solve_block_median']}, settle block median (solved) {summary['settle_block_median_solved']}, "
          f"(unsolved) {summary['settle_block_median_unsolved']}; never settled: unsolved {summary['unsolved_never_settled']}, solved {summary['solved_never_settled']}; "
          f"spearman(solve, settle | solved) {summary['spearman_solve_vs_settle_solved']}")
    for name in reads:
        e = eta[name]
        print(f"  event-triggered {name:8s} read (offset -4..+4, cell-normalised; {e['cells']} cells): " + ' '.join(f'{v:.3f}' for v in e['event'])
              + '  | shuffled: ' + ' '.join(f'{v:.3f}' for v in e['shuffled_control']))
    print('WROTE', out)


if __name__ == '__main__':
    main()
