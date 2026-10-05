"""Replicate the phase-boundary diagnostic on independent saved puzzle pairs.

CPU FP64 uses the algebraic replica; CUDA BF16 calls the production block.
Only isolated checkpoint copies are changed, never the live training process.
Sign replay is a diagnostic control with the same baseline, not a new model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import train as t
from .kv_stability import install
from .audit_puzzle_phase_feedback import fd_updates, rms, trace


def production_rollout(inner, h, inj, labels, replay=None):
    records = []
    original_window = inner.phase_window

    def capture_window(layer, dtype, *, phases):
        pk, pv = (x.to(dtype) for x in phases)
        delta = pv[..., :, None] - pk[..., None, :]
        natural_sign = delta.sign()
        applied = natural_sign if replay is None else replay[len(records)]
        records.append(natural_sign.detach())
        return applied * torch.exp(-delta.abs() / inner.phase_tau)

    inner.phase_window = capture_window
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            for _ in range(8):
                h = inner.block(inner.layers[0], h, inj, None, None, None, None)[0]
            logits = inner.w_cls(h).float()
        valid = labels != t.IGNORE_LABEL_ID
        value = (t.stablemax_cross_entropy(logits, labels) /
                 valid.sum(-1).clamp_min(1)[:, None]).sum() / labels.shape[0]
        return value, records
    finally:
        del inner.phase_window
        assert inner.phase_window == original_window


def bf16_updates(inner, h, inj, labels, adam_states, cfg):
    layer = inner.layers[0]
    params = (layer.phase_k_proj.weight, layer.phase_v_proj.weight)
    originals = [p.detach().clone() for p in params]
    value, replay = production_rollout(inner, h, inj, labels)
    grads = torch.autograd.grad(value, params)
    baseline = float(value.detach())
    with torch.no_grad():
        baseline_replay, _ = production_rollout(inner, h, inj, labels, replay)
        torch.testing.assert_close(baseline_replay, value.detach(), rtol=0, atol=0)
    b1, b2 = cfg['beta1'], cfg['beta2']
    lr, wd = cfg['lr'], cfg['weight_decay']
    adam_direction = []
    for p, g, st in zip(params, grads, adam_states):
        age = int(st['step']) + 1
        m = b1 * st['m'].to(p) + (1-b1)*g
        v = b2 * st['v'].to(p) + (1-b2)*g.square()
        adam_direction.append(-lr*torch.atan2(m/(1-b1**age),
                              (v/(1-b2**age)).sqrt()) - lr*wd*p.detach())
    length = torch.cat([d.flatten() for d in adam_direction]).norm()
    glength = torch.cat([g.flatten() for g in grads]).norm().clamp_min(1e-30)
    descent = [-g * length/glength for g in grads]
    report = dict(baseline_loss=baseline, baseline_replay_abs_error=0.,
                  phase_gradient_norm=float(glength), directions={})
    try:
        for name, direction in [('negative_gradient', descent),
                                ('adam_with_saved_moments', adam_direction)]:
            slope = sum(float((g*d).double().sum()) for g,d in zip(grads,direction))
            entry = dict(predicted_loss_change_per_unit=slope,
                         parameter_update_rms=rms(torch.cat([d.flatten() for d in direction])),
                         measurements=[])
            for alpha in (.1, 1.):
                row = dict(alpha=alpha, predicted_plus_loss_change=alpha*slope)
                with torch.no_grad():
                    for p, original, d in zip(params, originals, direction):
                        p.copy_(original + alpha*d)
                    for mode, masks in [('true', None), ('replayed_sign', replay)]:
                        new_value, signs = production_rollout(inner, h, inj, labels, masks)
                        row[mode] = dict(plus_loss=float(new_value),
                            plus_loss_change=float(new_value)-baseline,
                            plus_branch_crossings_by_block=[int((s!=old).sum()) for s,old in zip(signs,replay)])
                entry['measurements'].append(row)
                print('BF16', name, alpha, 'pred', alpha*slope,
                      'true', row['true']['plus_loss_change'],
                      'replay', row['replayed_sign']['plus_loss_change'], flush=True)
            report['directions'][name] = entry
    finally:
        with torch.no_grad():
            for p, original in zip(params, originals):
                p.copy_(original)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', default='docs/research/2026-10-05/phase_gradient_audit/replication.json')
    args = ap.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    install('phase_puzzle_exp_current_only')
    cfg = dict(ck['cfg'], batch_size=128, seq_len=81, num_puzzle_identifiers=1,
               amp=False, compile=False, activation_checkpoint=False)
    base = t.ACTLossHead(t.LT(cfg))
    base.load_state_dict(ck['raw_model_state_dict'])
    base.eval()
    opts, _ = t.create_optimizers(base, cfg, 1)
    names = {id(p):n for n,p in base.named_parameters()}
    saved_adam = {}
    for now, saved in zip(opts[-1].param_groups, ck['optimizer_states'][-1]['param_groups']):
        assert len(now['params']) == len(saved['params'])
        for p, pid in zip(now['params'], saved['params']):
            if 'phase_' in names[id(p)]:
                saved_adam[names[id(p)]] = ck['optimizer_states'][-1]['state'][pid]
    for p in base.parameters():
        p.requires_grad_(False)
    inner = base.model.inner
    inner.layers[0].phase_k_proj.weight.requires_grad_()
    inner.layers[0].phase_v_proj.weight.requires_grad_()
    adam_states = [saved_adam['model.inner.layers.0.phase_k_proj.weight'],
                   saved_adam['model.inner.layers.0.phase_v_proj.weight']]
    carry = ck['rank_states'][0]['carry']
    batch = {k:v[:6].clone() for k,v in carry['current_data'].items()}
    h = carry['current_hidden'][:6].clone()
    with torch.no_grad():
        inj = inner.injection(batch).detach()
    labels = batch['labels']
    report = dict(step=ck['step'], checkpoint_argument=args.checkpoint,
        source=f"runs/kv_phase_puzzle_exp_fresh_20261005/step_{ck['step']}.pt",
        convention='Three disjoint pairs from saved carry; not three independent training runs. Phase-only hypothetical updates; all other weights frozen. BF16 is eager production arithmetic, not compiled execution.',
        traces={}, fp64=[], production_bf16=[])
    out = Path(args.out)
    def save():
        out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    # Correct distinction: frozen-first-sign has zero applied mask flips even
    # when the natural ordering of the underlying phases continues to change.
    for mode in ('dynamic', 'frozen_first_sign', 'no_attention', 'no_ffn'):
        report['traces'][mode] = trace(inner, h, inj, labels, mode=mode)
    assert report['traces']['frozen_first_sign']['averages']['sign_flip_fraction'] == 0.
    save()
    inner.double()
    for start in (0, 2, 4):
        sl = slice(start, start+2)
        result = fd_updates(inner, h[sl].double(), inj[sl].double(), labels[sl],
            adam_states, cfg['lr'], cfg['weight_decay'],
            (cfg['beta1'], cfg['beta2']), [1e-6, .1, 1.])
        report['fp64'].append(dict(puzzle_indices=[start,start+1], result=result))
        save()
    inner.float().cuda()
    batch_gpu = {k:v.cuda() for k,v in batch.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        inj_gpu = inner.injection(batch_gpu).detach()
    for start in (0, 2, 4):
        sl = slice(start, start+2)
        result = bf16_updates(inner, h[sl].cuda(), inj_gpu[sl], labels[sl].cuda(),
                              adam_states, cfg)
        report['production_bf16'].append(dict(puzzle_indices=[start,start+1], result=result))
        save()
    report['elapsed_seconds'] = time.monotonic()-started
    save()
    print('Saved', out, 'seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
