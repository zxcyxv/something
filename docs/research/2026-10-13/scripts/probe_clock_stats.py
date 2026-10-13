"""Clock and write statistics of a phase-clock run, per block over a 16-segment held-out run.

Usage (repository root):  python docs/research/2026-10-13/scripts/probe_clock_stats.py RUN STEP [--n 128] [--segs 16]

Hooks the torch path's phase_clock_step on the EMA weights and records, per block (segment x block):
  dt mean; |dt w_K|, |dt w_V| mean (rad per block) and the fraction above 0.1 rad; lam_K, lam_V mean;
  ||Hebb||_F and alpha_h ||Im S||_F per head (batch mean) and their ratio; the phase spread at the end.
Writes docs/research/2026-10-13/results/clock_stats_<run>_step<STEP>_n<N>_seg<S>.json.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lt import train as t                                   # noqa: E402
from lt import research_phase_clock_stdp as rp              # noqa: E402
from lt.kv_stability import ORIGINAL_MODEL_ID               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('step', type=int)
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--segs', type=int, default=16)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(run / f'step_{opt.step}.pt', map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'])
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(ROOT / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    _, _, x, y, _, _ = t.load_data(cfg)
    x, y = x[:opt.n], y[:opt.n]
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'], kernel='torch')
    lt = base.model
    inner = lt.inner
    alpha = inner.layers[0].stdp_alpha.detach().float()
    records = []
    original_step, original_clock = rp.phase_clock_step, rp.advance_clock
    pending = {}

    def hooked_clock(clock, dt, freq_k, freq_v, rate_k, rate_v, state, **extra):
        out = original_clock(clock, dt, freq_k, freq_v, rate_k, rate_v, state, **extra)
        phase_k, phase_v, lam_k, lam_v, new_state = out
        if clock == 'displacement':             # per-block tag increments = observed pair-angle displacements
            wk, wv = rp.wrap_phase(new_state[0] - state[0]).abs(), rp.wrap_phase(new_state[1] - state[1]).abs()
        else:
            wk, wv = (dt * freq_k).abs(), (dt * freq_v).abs()
        pending.update(dt=dt.mean().item(), w_k=wk.mean().item(), w_v=wv.mean().item(),
                       w_k_gt01=(wk > 0.1).float().mean().item(), w_v_gt01=(wv > 0.1).float().mean().item(),
                       lam_k=lam_k.mean().item(), lam_v=lam_v.mean().item(),
                       phase_k_std=phase_k.std().item(), phase_v_std=phase_v.std().item(),
                       rel_phase_std=rp.wrap_phase(phase_v - phase_k).std().item())
        if clock == 'leaky':
            psi_k, psi_v, carrier = new_state
            pending.update(carrier_step=(dt * extra['omega_bar']).abs().mean().item(), psi_k_abs=psi_k.abs().mean().item(),
                           psi_v_abs=psi_v.abs().mean().item(), psi_diff_std=(psi_v - psi_k).std().item())
        return out

    def hooked_step(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces):
        hebb, im_s, new_traces = original_step(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces)
        hn = hebb.flatten(2).norm(dim=-1).mean(0)                      # [H]
        sn = im_s.flatten(2).norm(dim=-1).mean(0) * alpha.to(im_s.device).abs()
        records.append(dict(pending, hebb=hn.tolist(), timing=sn.tolist(), ratio=(sn / hn).tolist()))
        return hebb, im_s, new_traces
    rp.phase_clock_step, rp.advance_clock = hooked_step, hooked_clock
    exact_per_seg = []
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in next(t.eval_batches(x, y, len(x), 0, 1)).items()}
        carry = lt.initial_carry(batch)
        for seg in range(opt.segs):
            lt.config.blocks_per_seg = cfg['blocks_per_seg']
            carry, out = lt(carry, batch)
            pred = out['logits'].argmax(-1)
            exact_per_seg.append(int(((pred == batch['labels']) | (batch['labels'] == t.IGNORE_LABEL_ID)).all(-1).sum()))
    rp.phase_clock_step, rp.advance_clock = original_step, original_clock
    H = alpha.numel()
    bps = cfg['blocks_per_seg']
    summary = dict(run=run.name, step=opt.step, n=opt.n, segs=opt.segs, blocks_per_seg=bps,
                   alpha=alpha.tolist(), exact_per_segment=exact_per_seg, blocks=records)
    out = ROOT / 'docs/research/2026-10-13/results' / f'clock_stats_{run.name}_step{opt.step}_n{opt.n}_seg{opt.segs}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + '\n')
    print('alpha', [round(a, 3) for a in alpha.tolist()])
    print('exact per segment', exact_per_seg)
    print('block   dt    |dtwK| |dtwV|  >0.1K >0.1V  lamK  lamV   ratio alpha|ImS|/|Hebb| (mean over heads, min, max)  phase std K/V  rel std' + ('  carrier |psiK| |psiV| psidiff' if 'carrier_step' in records[0] else ''))
    for i in (0, 1, 3, 7, 15, 31, 63, 127):
        if i >= len(records):
            continue
        r = records[i]
        rt = r['ratio']
        print(f"{i + 1:5d} {r['dt']:5.3f} {r['w_k']:6.3f} {r['w_v']:6.3f}  {r['w_k_gt01']:5.2f} {r['w_v_gt01']:5.2f} "
              f"{r['lam_k']:5.3f} {r['lam_v']:5.3f}   {sum(rt) / H:5.3f} ({min(rt):5.3f}, {max(rt):5.3f})   "
              f"{r['phase_k_std']:5.2f} / {r['phase_v_std']:5.2f}  {r['rel_phase_std']:5.2f}"
              + (f"  {r['carrier_step']:5.2f} {r['psi_k_abs']:5.2f} {r['psi_v_abs']:5.2f} {r['psi_diff_std']:5.2f}" if 'carrier_step' in r else ''))
    print('WROTE', out)


if __name__ == '__main__':
    main()
