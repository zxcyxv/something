"""Sign and size of dL/d alpha_h under variants of the timing term (torch path, EMA weights).

Usage (repository root): python docs/research/2026-10-13/scripts/probe_alpha_gradient.py RUN STEP [--n 128] [--batch 64] [--segs 16]

Variants of Im S inside phase_clock_step (weights, clock and traces otherwise unchanged):
  full          the model as trained
  same_block    only this block's (post, pre) pairs: (1 - lam_K) * Im[(v e^{i phi_V})^T RoPE(k_hat e^{-i phi_K})]
  cross_block   only pairs with a past event (the lam_K E_K and D_V terms)
  content_phase phases replaced by the pair angles atan2(v pair), atan2(k_hat pair) (the 10-09 window's phases)
For each variant the 16 segments of each batch run as in training (loss per segment, carry detached) and the
per-head alpha gradients are summed. A consistently negative gradient means the term raises the loss on average.
"""
import argparse, json, math, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lt import train as t
from lt import research_phase_clock_stdp as rp
from lt.kv_stability import ORIGINAL_MODEL_ID

VARIANTS = ('full', 'same_block', 'cross_block', 'content_phase')


def pair_angle(x):
    pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    return torch.atan2(pair[..., 1], pair[..., 0])


def make_step(variant):
    rep2 = rp._rep2

    def step(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces):
        ek_re, ek_im, ev_re, ev_im = traces
        tag_k, tag_v = (pair_angle(k_unit), pair_angle(v)) if variant == 'content_phase' else (phase_k, phase_v)
        ck, sk, cv, sv = rep2(tag_k.cos()), rep2(tag_k.sin()), rep2(tag_v.cos()), rep2(tag_v.sin())
        lam_k, lam_v = rep2(lam_k), rep2(lam_v)
        new_re, new_im = (1 - lam_k) * k_unit * ck, -(1 - lam_k) * k_unit * sk
        old_re, old_im = lam_k * ek_re, lam_k * ek_im
        if variant == 'same_block':
            use_re, use_im, dv_re, dv_im = new_re, new_im, torch.zeros_like(ev_re), torch.zeros_like(ev_im)
        elif variant == 'cross_block':
            use_re, use_im, dv_re, dv_im = old_re, old_im, lam_v * ev_re, lam_v * ev_im
        else:
            use_re, use_im, dv_re, dv_im = old_re + new_re, old_im + new_im, lam_v * ev_re, lam_v * ev_im
        vs, vc = (v * sv).transpose(-1, -2), (v * cv).transpose(-1, -2)
        im_s = (vs @ rope(use_re) + vc @ rope(use_im)
                + dv_im.transpose(-1, -2) @ (kr * ck) - dv_re.transpose(-1, -2) @ (kr * sk))
        hebb = v.transpose(-1, -2) @ kr
        ek_re, ek_im = old_re + new_re, old_im + new_im
        dv_re_t, dv_im_t = lam_v * ev_re, lam_v * ev_im
        ev_re, ev_im = dv_re_t + (1 - lam_v) * v * cv, dv_im_t + (1 - lam_v) * v * sv
        return hebb, im_s, (ek_re, ek_im, ev_re, ev_im)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path); ap.add_argument('step', type=int)
    ap.add_argument('--n', type=int, default=128); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--segs', type=int, default=16)
    ap.add_argument('--split', choices=('held_out', 'train'), default='held_out',
                    help="'train': the original (unaugmented) training puzzles")
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(run / f'step_{opt.step}.pt', map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'], global_batch_size=opt.batch, activation_checkpoint=False)
    cfg['data_npz'] = str(ROOT / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    t.model_id_of = lambda c: ORIGINAL_MODEL_ID(c) + ':research-' + c['research_variant']
    tr_x, tr_y, x, y, _, _ = t.load_data(cfg)
    if opt.split == 'train':
        x, y = tr_x, tr_y
    x, y = x[:opt.n], y[:opt.n]
    base = rp.make_model(protocol, cfg, device, state=ck['model_state_dict'], kernel='torch')
    base.train()
    alpha = base.model.inner.layers[0].stdp_alpha
    original = rp.phase_clock_step
    out = dict(run=run.name, step=opt.step, n=opt.n, segs=opt.segs, split=opt.split, alpha=alpha.detach().tolist(), variants={})
    for variant in VARIANTS:
        rp.phase_clock_step = make_step(variant)
        grad = torch.zeros_like(alpha)
        per_seg = []
        losses = []
        for batch in t.eval_batches(x, y, opt.batch, 0, 1):
            b = {k: v.to(device) for k, v in batch.items()}
            carry = base.initial_carry(b)
            for s in range(opt.segs):
                base.zero_grad(set_to_none=True)
                carry, loss, metrics, *_ = base(carry=carry, batch=b, return_keys=set())
                (loss / opt.batch).backward()
                grad += alpha.grad.detach()
                per_seg.append(alpha.grad.detach().clone())
                losses.append(float(loss.detach()) / opt.batch)
        per_seg = torch.stack(per_seg)                                   # [batches*segs, H]
        frac_neg = (per_seg < 0).float().mean(0)
        out['variants'][variant] = dict(grad=grad.tolist(), frac_negative=frac_neg.tolist(),
                                        mean_loss=sum(losses) / len(losses))
        print(f"{variant:14s} loss {sum(losses) / len(losses):.4f}  dL/dalpha per head "
              + ' '.join(f'{g:+.3f}' for g in grad.tolist()) + '   frac<0 ' + ' '.join(f'{f:.2f}' for f in frac_neg.tolist()), flush=True)
    rp.phase_clock_step = original
    path = ROOT / 'docs/research/2026-10-13/results' / f'alpha_gradient_{run.name}_step{opt.step}_{opt.split}_n{opt.n}_seg{opt.segs}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print('alpha', [round(a, 3) for a in alpha.detach().tolist()])
    print('WROTE', path)


if __name__ == '__main__':
    main()
