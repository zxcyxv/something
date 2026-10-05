"""Frozen-checkpoint audit of puzzle-phase feedback and discontinuous updates.

Hypotheses and tests are registered in the output protocol before this script
runs. All interventions are on an isolated CPU model. The FP64 replica retains
the equations but removes production FP32 rounding for finite differences.
Replay of baseline sign masks is a diagnostic smooth extension, not a proposed
training window. A frozen-subset counterfactual does not prove retraining gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from . import train as t
from .kv_stability import install


def rms(x):
    return float(x.detach().double().square().mean().sqrt())


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


def block(inner, h, inj, *, gain=1., frozen_summary=None, sign_override=None,
          detach_summary=False, no_attention=False, no_ffn=False):
    layer = inner.layers[0]
    b, n, d = h.shape
    x = h + inner.embed_scale * inj
    mean = x.mean(1)
    c = norm(mean) if frozen_summary is None else frozen_summary
    if detach_summary:
        c = c.detach()
    ak, av = layer.phase_k_proj(c), layer.phase_v_proj(c)
    rk = layer.theta_k_raw[None] + gain * ak.reshape(b, inner.H, inner.dh)
    rv = layer.theta_v_raw[None] + gain * av.reshape(b, inner.H, inner.dh)
    pk, pv = inner.phase_limit * rk.tanh(), inner.phase_limit * rv.tanh()
    delta = pv[..., :, None] - pk[..., None, :]
    natural_sign = delta.sign()
    sign = natural_sign if sign_override is None else sign_override
    envelope = (-delta.abs() / inner.phase_tau).exp()
    window = sign * envelope
    heads = lambda z: z.reshape(b, n, inner.H, inner.dh).transpose(1, 2)
    q, k, v = [heads(p(x)) for p in (layer.q_proj, layer.k_proj, layer.v_proj)]
    tables = inner.rope_tables(layer)
    q, k = [inner.apply_rope(z, layer, tables) for z in (q, k)]
    raw_g = v.transpose(-1, -2) @ k / n
    g = raw_g * window
    read = (q @ g.transpose(-1, -2)).transpose(1, 2).reshape(b, n, d)
    attention = layer.out_proj(read)
    if no_attention:
        attention = attention * 0
    u = norm(x + attention)
    gate, value = layer.b_gate_up(u).chunk(2, -1)
    ffn = layer.b_down(.5 * gate * value)
    if no_ffn:
        ffn = ffn * 0
    hn = norm(u + ffn)
    return hn, dict(h=h, x=x, mean=mean, c=c, ak=ak, av=av, rk=rk, rv=rv,
                    delta=delta, sign=natural_sign, applied_sign=sign,
                    envelope=envelope, window=window,
                    raw_g=raw_g, g=g, read=read, attention=attention, u=u, ffn=ffn)


def loss(inner, h, labels):
    logits = inner.w_cls(h)
    valid = labels != t.IGNORE_LABEL_ID
    return (t.stablemax_cross_entropy(logits, labels) /
            valid.sum(-1).clamp_min(1)[:, None]).sum() / labels.shape[0]


def rollout(inner, h, inj, labels, blocks=8, *, replay=None, **kwargs):
    records, states = [], [h]
    for i in range(blocks):
        local = dict(kwargs)
        if replay is not None:
            local['sign_override'] = replay[i]
        h, info = block(inner, h, inj, **local)
        records.append(info)
        states.append(h)
    return loss(inner, h, labels), records, states


@torch.no_grad()
def trace(inner, h, inj, labels, blocks=16, mode='dynamic'):
    initial_c = norm((h + inner.embed_scale * inj).mean(1))
    initial_sign = None
    prev, prev2 = None, None
    records = []
    for i in range(blocks):
        kw = {}
        if mode == 'gain_0.1': kw['gain'] = .1
        if mode == 'gain_0': kw['gain'] = 0.
        if mode == 'frozen_summary': kw['frozen_summary'] = initial_c
        if mode == 'frozen_first_sign' and initial_sign is not None:
            kw['sign_override'] = initial_sign
        if mode == 'no_attention': kw['no_attention'] = True
        if mode == 'no_ffn': kw['no_ffn'] = True
        hn, info = block(inner, h, inj, **kw)
        if initial_sign is None:
            initial_sign = info['sign']
        m = info['mean']
        after_attention_mean = (info['u'] + inner.embed_scale * inj).mean(1)
        next_mean = (hn + inner.embed_scale * inj).mean(1)
        row = dict(block=i, hidden_change_rms=rms(hn-h), hidden_cos_next=cosine(h, hn),
                   mean_rms=rms(m), summary_rms=rms(info['c']),
                   summary_norm_inverse_rms=float(torch.rsqrt(m.square().mean(-1)+1e-5).mean()),
                   hidden_mean_rms=rms(h.mean(1)), injection_mean_rms=rms(inner.embed_scale*inj.mean(1)),
                   mean_change_attention_stage_rms=rms(after_attention_mean-m),
                   mean_change_ffn_stage_rms=rms(next_mean-after_attention_mean),
                   attention_rms=rms(info['attention']), ffn_rms=rms(info['ffn']),
                   dynamic_raw_rms=rms(torch.cat([info['ak'], info['av']], -1)),
                   pre_tanh_saturation_fraction=float((torch.cat([info['rk'], info['rv']], -1).abs()>3).float().mean()),
                   loss=float(loss(inner, hn, labels)))
        if prev is not None:
            flip = info['applied_sign'] * prev['applied_sign'] < 0
            natural_flip = info['sign'] * prev['sign'] < 0
            energy = info['g'].square() + prev['g'].square()
            row.update(sign_flip_fraction=float(flip.float().mean()),
                       natural_order_flip_fraction=float(natural_flip.float().mean()),
                       g_energy_on_L_flips=float((energy*flip).sum()/energy.sum().clamp_min(1e-30)),
                       summary_cos_previous=cosine(info['c'], prev['c']),
                       summary_change_rms=rms(info['c']-prev['c']),
                       raw_phase_change_rms=rms(torch.cat([info['ak']-prev['ak'], info['av']-prev['av']], -1)),
                       delta_cos_previous=cosine(info['delta'], prev['delta']),
                       g_cos_previous=cosine(info['g'], prev['g']),
                       read_cos_previous=cosine(info['read'], prev['read']))
        if prev2 is not None:
            row['summary_cos_lag2'] = cosine(info['c'], prev2['c'])
        prev2, prev = prev, info
        records.append(row)
        h = hn
    pred = inner.w_cls(h).argmax(-1)
    valid = labels != t.IGNORE_LABEL_ID
    aggregate = {key: sum(r[key] for r in records if key in r) / sum(key in r for r in records)
                 for key in records[-1] if key != 'block'}
    return dict(mode=mode, averages=aggregate, final_loss=records[-1]['loss'],
                cell_accuracy=float(((pred==labels)&valid).sum()/valid.sum()),
                exact_count=int(((pred==labels)|~valid).all(-1).sum()), records=records)


def gradient_audit(inner, h0, inj, labels, *, detach_summary=False, gain=1.):
    h0 = h0.detach().requires_grad_()
    value, infos, states = rollout(inner, h0, inj, labels, gain=gain, detach_summary=detach_summary)
    layer = inner.layers[0]
    params = (layer.phase_k_proj.weight, layer.phase_v_proj.weight)
    targets = list(params) + states + [r['ak'] for r in infos] + [r['av'] for r in infos]
    grads = torch.autograd.grad(value, targets)
    gp = grads[:2]
    nh = len(states)
    gh = grads[2:2+nh]
    gak, gav = grads[2+nh:2+nh+8], grads[2+nh+8:]
    contributions = [torch.cat([(gk.T @ r['c'].detach()).flatten(),
                                (gv.T @ r['c'].detach()).flatten()])
                     for gk, gv, r in zip(gak, gav, infos)]
    total = torch.cat([g.flatten() for g in gp])
    reconstructed = torch.stack(contributions).sum(0)
    torch.testing.assert_close(reconstructed, total, rtol=5e-5, atol=5e-7)
    denom = sum(float(g.norm()) for g in contributions)
    report = dict(loss=float(value.detach()),
        state_gradient_norms=[float(g.norm()) for g in gh],
        phase_gradient_norm=float(total.norm()),
        per_block_phase_gradient_norms=[float(g.norm()) for g in contributions],
        adjacent_phase_gradient_cosines=[cosine(a,b) for a,b in zip(contributions,contributions[1:])],
        temporal_gradient_retention_ratio=float(total.norm())/max(denom,1e-30),
        gradient_sum_max_abs_error=float((reconstructed-total).abs().max()),
        gradient_sum_relative_l2_error=float((reconstructed-total).norm()/total.norm().clamp_min(1e-30)))
    return report, [g.detach() for g in gp], [r['sign'].detach() for r in infos], states[-1].detach()


def fd_updates(inner, h, inj, labels, adam_states, lr, wd, betas, factors):
    """Phase-only directions; other weights remain fixed. Uses FP64 replica."""
    report, grads, replay, _ = gradient_audit(inner, h, inj, labels)
    layer = inner.layers[0]
    params = (layer.phase_k_proj.weight, layer.phase_v_proj.weight)
    originals = [p.detach().clone() for p in params]
    b1, b2 = betas
    adam_direction = []
    for p, g, st in zip(params, grads, adam_states):
        age = int(st['step']) + 1
        m = b1*st['m'].double() + (1-b1)*g
        v = b2*st['v'].double() + (1-b2)*g.square()
        update = -lr * torch.atan2(m/(1-b1**age), (v/(1-b2**age)).sqrt()) - lr*wd*p.detach()
        adam_direction.append(update)
    length = torch.cat([d.flatten() for d in adam_direction]).norm()
    glength = torch.cat([g.flatten() for g in grads]).norm().clamp_min(1e-30)
    descent = [-g*length/glength for g in grads]
    directions = dict(negative_gradient=descent, adam_with_saved_moments=adam_direction)
    answer = dict(baseline=report, directions={},
                  convention='hypothetical phase-only update on this subset; saved optimizer moments, current small-subset gradients')
    for name, direction in directions.items():
        predicted = sum(float((g*d).sum()) for g,d in zip(grads,direction))
        row = dict(predicted_loss_change_per_unit=predicted,
                   parameter_update_rms=rms(torch.cat([d.flatten() for d in direction])), measurements=[])
        for alpha in factors:
            measures = {}
            for mode, masks in [('true',None),('replayed_sign',replay)]:
                evaluations = []
                for sign in (1,-1):
                    with torch.no_grad():
                        for p, original, direction_i in zip(params, originals, direction):
                            p.copy_(original + sign*alpha*direction_i)
                        val, infos, _ = rollout(inner,h,inj,labels,replay=masks)
                        flips = [int((r['sign'] != original_mask).sum()) for r,original_mask in zip(infos,replay)]
                        evaluations.append((float(val),flips))
                plus, minus = evaluations
                slope = (plus[0]-minus[0])/(2*alpha)
                measures[mode] = dict(plus_loss=plus[0],minus_loss=minus[0],
                    plus_loss_change=plus[0]-report['loss'],
                    central_slope=slope,relative_slope_error=abs(slope-predicted)/max(abs(predicted),1e-12),
                    plus_branch_crossings_by_block=plus[1],minus_branch_crossings_by_block=minus[1])
            row['measurements'].append(dict(alpha=alpha,predicted_plus_loss_change=alpha*predicted,**measures))
            print('FD',name,alpha,'pred',alpha*predicted,'true',measures['true']['plus_loss_change'],
                  'replay',measures['replayed_sign']['plus_loss_change'],flush=True)
        answer['directions'][name] = row
        with torch.no_grad():
            for p, original in zip(params, originals): p.copy_(original)
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/kv_phase_puzzle_exp_fresh_20261005')
    ap.add_argument('--out', default='docs/research/2026-10-05/phase_gradient_audit')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(51005)
    out = Path(args.out)
    assert (out/'protocol.json').exists(), 'Register hypotheses before running'
    path = max(Path(args.run).glob('step_*.pt'), key=lambda p:int(p.stem[5:]))
    ck = torch.load(path,map_location='cpu',weights_only=False)
    install('phase_puzzle_exp_current_only')
    cfg = dict(ck['cfg'],batch_size=128,seq_len=81,num_puzzle_identifiers=1,
               amp=False,compile=False,activation_checkpoint=False)
    base = t.ACTLossHead(t.LT(cfg));base.load_state_dict(ck['raw_model_state_dict']);base.eval()
    # Match saved optimizer parameter ids before freezing unrelated parameters.
    opts,_ = t.create_optimizers(base,cfg,1)
    names = {id(p):n for n,p in base.named_parameters()}
    saved_adam = {}
    for now, saved in zip(opts[-1].param_groups,ck['optimizer_states'][-1]['param_groups']):
        assert len(now['params']) == len(saved['params'])
        for p,pid in zip(now['params'],saved['params']):
            if 'phase_' in names[id(p)]: saved_adam[names[id(p)]] = ck['optimizer_states'][-1]['state'][pid]
    for p in base.parameters(): p.requires_grad_(False)
    inner,layer = base.model.inner,base.model.inner.layers[0]
    layer.phase_k_proj.weight.requires_grad_();layer.phase_v_proj.weight.requires_grad_()
    carry = ck['rank_states'][0]['carry']
    batch = {k:v[:8] for k,v in carry['current_data'].items()}
    h0 = carry['current_hidden'][:8].clone()
    with torch.no_grad(): inj=inner.injection(batch).detach()
    labels=batch['labels']
    report=dict(checkpoint=str(path),step=ck['step'],device='CPU',trace_puzzles=8,gradient_puzzles=2,
        initial_state='saved raw training carry, continued at frozen weights; fresh-state trace separately',
        phase_weight_norms={n:float(p.norm()) for n,p in layer.named_parameters() if 'phase_' in n},
        traces={},gradient_audits={})
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    started=time.monotonic()
    with torch.no_grad():
        actual=inner.block(layer,h0[:2],inj[:2],None,None,None,None)[0]
        replica=block(inner,h0[:2],inj[:2])[0]
        torch.testing.assert_close(actual,replica,rtol=0,atol=0)
        report['production_fp32_replica_max_abs_error']=float((actual-replica).abs().max())
    print('checkpoint',path,'replica parity passed',flush=True)
    for mode in ('dynamic','gain_0.1','gain_0','frozen_summary','frozen_first_sign','no_attention','no_ffn'):
        report['traces'][mode]=trace(inner,h0,inj,labels,mode=mode)
        print('TRACE',mode,report['traces'][mode]['averages'],flush=True)
        save()
    with torch.no_grad():
        fresh=inner.init_hidden.expand_as(h0).clone()
    report['traces']['dynamic_fresh']=trace(inner,fresh,inj,labels)
    # Keep the forward unchanged while removing only the summary feedback gradient.
    full,gfull,_,outfull=gradient_audit(inner,h0[:2],inj[:2],labels[:2])
    detach,gdetach,_,outdetach=gradient_audit(inner,h0[:2],inj[:2],labels[:2],detach_summary=True)
    torch.testing.assert_close(outfull,outdetach,rtol=0,atol=0)
    small,_,_,_=gradient_audit(inner,h0[:2],inj[:2],labels[:2],gain=.1)
    report['gradient_audits'].update(full=full,detached_summary=detach,gain_0_1=small,
        full_vs_detached_phase_gradient_cosine=cosine(torch.cat([g.flatten() for g in gfull]),torch.cat([g.flatten() for g in gdetach])),
        same_forward_max_abs_error=float((outfull-outdetach).abs().max()))
    print('GRADIENT',report['gradient_audits'],flush=True)
    save()
    # Double precision is for derivative/finite-step consistency, not speed or
    # faithful reproduction of the BF16 training trajectory at long horizons.
    inner.double()
    states=[saved_adam['model.inner.layers.0.phase_k_proj.weight'],saved_adam['model.inner.layers.0.phase_v_proj.weight']]
    report['finite_updates_fp64']=fd_updates(inner,h0[:2].double(),inj[:2].double(),labels[:2],
        states,cfg['lr'],cfg['weight_decay'],(cfg['beta1'],cfg['beta2']),[1e-6,1e-3,.1,1.])
    report['elapsed_seconds']=time.monotonic()-started
    report['cuda_initialized']=torch.cuda.is_initialized()
    save()
    print('Saved',out/'results.json','elapsed',report['elapsed_seconds'],flush=True)


if __name__=='__main__': main()
