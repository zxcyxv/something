"""Exact STDP events induced by one recorded K/V optimizer update.

The last AdamATan2 update can be inverted from its saved moments. We hold hidden
activity and spatial rotation fixed, so this isolates the change of K/V slow
projections. Checkpoints are never modified. A stationary-activity construction
tests the analytical boundary term; it is not a replay of the training history.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import train as t
from .kv_stability import install


def rms(x):
    return float(x.double().square().mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(2)
    results = {'scope': 'one actual final optimizer K/V update, fixed hidden/input and fixed current RoPE',
               'caveat': 'The final saved segment is terminal, so forced continuation is diagnostic, not an event that occurred after that checkpoint in training. No earlier optimizer history is reconstructed.',
               'runs': {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in args.checkpoints:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        variant = ck['cfg'].get('research_variant', 'original')
        install(variant)
        cfg = dict(ck['cfg'], batch_size=args.batch, seq_len=81, num_puzzle_identifiers=1,
                   amp=False, activation_checkpoint=False, nograd_blocks=0)
        # The sparse embedding's local trainable buffer must be created on the
        # target device, so it remains a leaf when building the original optimizer.
        with torch.device('cuda'):
            base = t.ACTLossHead(t.LT(cfg), q_weight=cfg['q_weight'])
        base.eval()
        base.load_state_dict(ck['raw_model_state_dict'])
        opts, _ = t.create_optimizers(base, cfg, 1)
        for opt, sd in zip(opts, ck['optimizer_states']):
            opt.load_state_dict(sd)
        inner, layer = base.model.inner, base.model.inner.layers[0]
        previous = {}
        with torch.no_grad():
            for name in ('k_proj', 'v_proj'):
                p = getattr(layer, name).weight
                group = next(g for g in opts[-1].param_groups if any(x is p for x in g['params']))
                st = opts[-1].state[p]
                b1, b2 = group['betas']
                correction = group['lr']*torch.atan2(st['m']/(1-b1**st['step']),
                                                    (st['v']/(1-b2**st['step'])).sqrt())
                old = (p+correction)/(1-group['lr']*group['weight_decay'])
                previous[name] = old
            saved = ck['rank_states'][0]['carry']
            batch = {k:v[:args.batch].cuda() for k,v in saved['current_data'].items()}
            hp = saved['current_hidden'][:args.batch].cuda()+inner.embed_scale*inner.injection(batch)
            def heads(x):
                return x.reshape(args.batch, 81, inner.H, inner.dh).transpose(1, 2).double()
            kn, vn = heads(layer.k_proj(hp)), heads(layer.v_proj(hp))
            ko, vo = heads(F.linear(hp, previous['k_proj'])), heads(F.linear(hp, previous['v_proj']))
            krn, kro = inner.apply_rope(kn, layer), inner.apply_rope(ko, layer)
            q = inner.apply_rope(heads(layer.q_proj(hp)), layer)
            lam = layer.trace_decay_channels.double()[None, :, None, :]
            def mat(v,k):
                return v.transpose(-1,-2) @ k / k.shape[-2]
            # Incoming trace is the fully equilibrated OLD constant activity.
            ek, ev = ko.clone(), vo.clone()
            pulse_memory = torch.zeros(args.batch, inner.H, inner.dh, inner.dh, dtype=torch.float64, device='cuda')
            writes = []
            for _ in range(8):
                g = mat(vn, inner.apply_rope(ek, layer))-mat(ev, krn)
                pulse_memory += g
                writes.append(rms(g))
                ek, ev = lam*ek+(1-lam)*kn, lam*ev+(1-lam)*vn
            c = (1-lam**8)/(1-lam)
            predicted = mat(vn, inner.apply_rope(c*(ko-kn), layer))-mat(c*(vo-vn), krn)
            current = mat(vn, krn)
            record = {
                'checkpoint': str(Path(path).resolve()),
                'projection_relative_changes': {
                    name: float((getattr(layer,name).weight-previous[name]).norm()/previous[name].norm())
                    for name in previous},
                'constant_hidden_optimizer_event': {
                    'eight_writes_rms': writes,
                    'integrated_memory_rms': rms(pulse_memory),
                    'integrated_memory_over_current_KV_norm': float(pulse_memory.norm()/current.norm()),
                    'read_from_integrated_event_rms': rms(q @ pulse_memory.transpose(-1,-2)),
                    'formula_max_abs_error': float((pulse_memory-predicted).abs().max())}}
            if saved['key_trace'] is not None:
                ek, ev = saved['key_trace'][:args.batch].cuda().double(), saved['value_trace'][:args.batch].cuda().double()
                ekr = inner.apply_rope(ek, layer)
                gn, go = mat(vn,ekr)-mat(ev,krn), mat(vo,ekr)-mat(ev,kro)
                predicted_delta = mat(vn-vo, ekr)-mat(ev,krn-kro)
                record['saved_state_forced_continuation'] = {
                    'new_write_rms': rms(gn), 'old_projection_write_rms': rms(go),
                    'update_induced_write_rms': rms(gn-go),
                    'update_induced_over_new_write_norm': float((gn-go).norm()/gn.norm()),
                    'update_induced_read_rms': rms(q @ (gn-go).transpose(-1,-2)),
                    'formula_max_abs_error': float((gn-go-predicted_delta).abs().max())}
        results['runs'][variant] = record
        out.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
        print(variant, json.dumps(record), flush=True)
        del ck, base, inner, layer, opts
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
