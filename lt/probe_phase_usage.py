"""Does a trained address-angle model use its per-token phases?

Three held-out evaluations from one checkpoint (EMA weights): phases as computed;
phases permuted across tokens inside each puzzle; phases replaced by their
per-channel circular mean over tokens. Then, on a few puzzles, the K-pair
magnitude distribution and the share of write mass carried by weak channels.
Works for the pairangle (atan2 phases) and softangle (soft phasor) windows.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from . import train as t
from .experiment_free_phase_windows import model_class


def build(run, checkpoint, batch_size, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    protocol = json.loads((run / 'protocol.json').read_text())
    t.KVSTDPInner = model_class(protocol['window'], protocol['phase_dynamic'], protocol['modes'], protocol['epsilon'],
                               protocol['generator'], protocol.get('feature_precision', 'float32'),
                               protocol.get('window_scale_factor', 1.0), protocol.get('tie_qk', False),
                               protocol.get('tie_vo', False), protocol.get('qk_l2', False),
                               protocol.get('write_sum', False), protocol.get('tau_phi', 2.0),
                               protocol.get('phase_floor', 0.5), protocol.get('v_norm', 'none'),
                               protocol.get('tie_all', False), protocol.get('phase_kappa', 1.0), protocol.get('phase_omega', 0.0),
                               protocol.get('phase_frame', 'rotated'),
                               protocol.get('dc_hebbian', False),
                               protocol.get('dc_alpha_init', 0.0))
    t._resolve_precision(cfg, device)
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    with torch.device(device):
        base = t.ACTLossHead(t.LT(dict(cfg, batch_size=batch_size, seq_len=cfg['grid'] ** 2,
                                       num_puzzle_identifiers=1)), q_weight=cfg['q_weight'])
    base.load_state_dict(ck['model_state_dict'])
    base.eval()
    return base, cfg, protocol, int(ck['step'])


def install_phase_mode(inner, window, mode):
    """Patch the phase source so that mode in {'normal','shuffle','channel_mean'} applies."""
    def transform(*phases):
        if mode == 'normal':
            return phases
        if mode == 'shuffle':
            perm = torch.randperm(phases[0].shape[-2], device=phases[0].device)
            return tuple(p[..., perm, :] for p in phases)
        if mode == 'channel_mean':
            return tuple(torch.atan2(p.sin().mean(-2, keepdim=True), p.cos().mean(-2, keepdim=True)).expand_as(p)
                         for p in phases)
        raise ValueError(mode)
    if window == 'pairangle':
        original = inner.phases
        def phases(layer, rk=None, v=None):
            return transform(*original(layer, rk, v))
        inner.phases = phases
    elif window == 'softangle':
        original = inner.soft_phasor
        def soft_phasor(x):
            cx, sx = original(x)
            rho = torch.sqrt(cx * cx + sx * sx)
            angle, = transform(torch.atan2(sx, cx))
            return angle.cos() * rho, angle.sin() * rho
        inner.soft_phasor = soft_phasor
    else:
        raise ValueError(window)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--puzzles', type=int, default=1024)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    device = torch.device('cuda')
    torch.set_float32_matmul_precision('highest')
    base, cfg, protocol, step = build(run, opt.checkpoint, cfg_bs := 128, device)
    inner = base.model.inner if hasattr(base.model, 'inner') else base.model
    _, _, tx, ty, *_ = t.load_data(cfg)
    tx, ty = tx[:opt.puzzles], ty[:opt.puzzles]
    results = {}
    for mode in ('normal', 'shuffle', 'channel_mean'):
        fresh, _, _, _ = build(run, opt.checkpoint, cfg_bs, device)
        fresh_inner = fresh.model.inner if hasattr(fresh.model, 'inner') else fresh.model
        install_phase_mode(fresh_inner, protocol['window'], mode)
        torch.manual_seed(0)
        m = t.evaluate(fresh, tx, ty, dict(cfg, test_size=len(tx)), 0, 1, device, step, ema=None)
        results[mode] = dict(accuracy=m['accuracy'], exact=round(m['exact_accuracy'] * m['count']), lm_loss=m['lm_loss'])
        print(f"{mode:13s} acc {m['accuracy']*100:.2f}%  exact {results[mode]['exact']}/{len(tx)}  loss {m['lm_loss']:.3f}", flush=True)
        del fresh
    # Weak-channel write mass on 8 held-out puzzles, last block of the second segment.
    rec = []
    original = inner.window_write
    def window_write(layer, kr, v):
        rec.append((kr.detach(), v.detach()))
        return original(layer, kr, v)
    inner.window_write = window_write
    small, _, _, _ = build(run, opt.checkpoint, 8, device)
    small_inner = small.model.inner if hasattr(small.model, 'inner') else small.model
    small_inner.window_write = window_write
    batch = {k: v.to(device) for k, v in next(t.eval_batches(tx[:8], ty[:8], 8, 0, 1)).items()}
    with torch.no_grad():
        carry = small.initial_carry(batch)
        for _ in range(2):
            carry, *_ = small(return_keys=(), carry=carry, batch=batch)
    kr, v = rec[-1]
    P = kr.shape[-1] // 2
    kc, vc = (z.reshape(*z.shape[:-1], P, 2) for z in (kr, v))
    mk, mv = kc.norm(dim=-1), vc.norm(dim=-1)
    rk = mk.square().mean(-1, keepdim=True).sqrt()
    quant = torch.tensor([.1, .25, .5, .75, .9], device=mk.device)
    phk = torch.atan2(kc[..., 1], kc[..., 0]).repeat_interleave(2, -1)
    phv = torch.atan2(vc[..., 1], vc[..., 0]).repeat_interleave(2, -1)
    d = phv[..., :, None] - phk[..., None, :]
    c = inner.window_coefficients
    W = sum(c[r] * torch.sin((r + 1) * d) for r in range(c.shape[0]))
    if protocol['window'] == 'softangle':
        rho = lambda m: (m.square() / (m.square() + protocol['phase_floor'] ** 2 * m.square().mean(-1, keepdim=True))).sqrt()
        coh = (rho(mv).repeat_interleave(2, -1)[..., :, None] * rho(mk).repeat_interleave(2, -1)[..., None, :])
        W = sum(c[r] * coh ** (r + 1) * torch.sin((r + 1) * d) for r in range(c.shape[0]))
    term = (v[..., :, None] * kr[..., None, :] * W).abs()
    relk = (mk / rk).repeat_interleave(2, -1)[..., None, :].expand_as(term)
    mass = {}
    for lo, hi in ((0, .5), (.5, 1), (1, 1.5), (1.5, 99)):
        sel = (relk >= lo) & (relk < hi)
        mass[f'[{lo},{hi})'] = dict(pair_fraction=float(sel.float().mean()), write_mass_fraction=float(term[sel].sum() / term.sum()))
    stats = dict(k_pair_rel_magnitude_quantiles=[round(x, 3) for x in torch.quantile((mk / rk).flatten(), quant).tolist()],
                 write_mass_by_k_magnitude=mass,
                 window_mean=float(W.mean()), window_abs_mean=float(W.abs().mean()), window_positive_fraction=float((W > 0).float().mean()))
    print(json.dumps(stats, indent=1), flush=True)
    out = run / 'diagnostics' / f'phase_usage_step{step}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(step=step, puzzles=len(tx), ablation=results, **stats), indent=2) + '\n')
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
