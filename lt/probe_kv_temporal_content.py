"""Inspect the information selected by STDP on existing frozen trajectories.

No model, optimizer, checkpoint or actual trajectory is modified. Alternate
time-window calculations replay captured activities, so they measure operator
content and do not claim counterfactual training or inference accuracy.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from . import train as t
from .kv_stability import install


def rms(x):
    return float(x.double().square().mean().sqrt())


def comparison(a, b):
    af, bf = a.double().flatten(), b.double().flatten()
    return {'relative_l2': float((af-bf).norm()/bf.norm().clamp_min(1e-30)),
            'cosine': float((af @ bf)/(af.norm()*bf.norm()).clamp_min(1e-30))}


def mat(v, k):
    return v.transpose(-1, -2) @ k / k.shape[-2]


def replay(keys, values, lam):
    ek, ev = torch.zeros_like(keys[0]), torch.zeros_like(values[0])
    writes = []
    for k, v in zip(keys, values):
        writes.append(mat(v, ek)-mat(ev, k))
        ek, ev = lam*ek+(1-lam)*k, lam*ev+(1-lam)*v
    writes = torch.stack(writes)
    return writes, writes.cumsum(0)


def phase_summary(xs):
    even, odd = xs[1::2], xs[::2]
    mean_even, mean_odd = even.mean(0), odd.mean(0)
    mean, delta = (mean_even+mean_odd)/2, (mean_even-mean_odd)/2
    residual = xs.clone()
    residual[1::2] -= mean_even
    residual[::2] -= mean_odd
    total_centered = xs-xs.mean(0)
    return mean, delta, {
        'mean_rms': rms(mean), 'alternating_rms': rms(delta),
        'residual_rms': rms(residual),
        'period_two_fraction_of_centered_energy':
            float(1-residual.double().square().sum()/total_centered.double().square().sum().clamp_min(1e-30))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--blocks', type=int, default=128)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--tail', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.blocks % 2 or args.tail % 2 or not 2 <= args.tail <= args.blocks:
        ap.error('blocks and tail must be even; 2 <= tail <= blocks')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {'method': 'frozen original trajectories, same puzzles, fresh states; alternate writes replay the same captured activities',
               'precision': 'FP32 trajectory, FP64 algebra replay',
               'blocks': args.blocks, 'batch': args.batch, 'tail': args.tail,
               'caveat': 'Operator-content measurements do not identify the cause of training reversal and give no accuracy for a changed model.',
               'runs': {}}
    common_batch = None
    for path in args.checkpoints:
        started = time.monotonic()
        ck = torch.load(path, map_location='cpu', weights_only=False)
        variant = ck['cfg'].get('research_variant', 'original')
        install(variant)
        cfg = dict(ck['cfg'], batch_size=args.batch, seq_len=81, num_puzzle_identifiers=1,
                   amp=False, activation_checkpoint=False, nograd_blocks=0)
        model = t.LT(cfg).to(args.device)
        model.load_state_dict({k.removeprefix('model.'): v for k, v in ck['raw_model_state_dict'].items()})
        model.eval()
        inner, layer = model.inner, model.inner.layers[0]
        assert len(inner.layers) == 1
        assert not inner.config.kv_qk_l2norm and not inner.config.kv_qk_rmsnorm
        assert inner.config.kv_write_reduction == 'mean'
        if common_batch is None:
            common_batch = {k: v[:args.batch].to(args.device) for k, v in ck['rank_states'][0]['carry']['current_data'].items()}
        keys, values, queries, memories, actual_reads, rows = [], [], [], [], [], []
        original_memory_step = inner.memory_step

        def capture(L, q, k, v, *rest):
            answer = original_memory_step(L, q, k, v, *rest)
            keys.append(inner.apply_rope(k.float(), L).double())
            values.append(v.double())
            queries.append(inner.apply_rope(q.float(), L).double())
            memories.append(answer[1].double())
            actual_reads.append(answer[0].double())
            return answer

        inner.memory_step = capture
        with torch.no_grad():
            inj = inner.injection(common_batch)
            h = inner.init_hidden[None, None, :].expand(args.batch, 81, -1)
            state = h, None, None, None
            previous_h = h
            before_previous = None
            for r in range(1, args.blocks+1):
                state = inner.block(layer, state[0], inj, *state[1:], None)
                logits = inner.w_cls(state[0])
                pred = logits.argmax(-1)
                row = {'block': r, 'accuracy': float((pred==common_batch['labels']).float().mean()),
                       'loss': float(t.stablemax_cross_entropy(logits, common_batch['labels']).mean()),
                       'hidden_one_step_rms': rms(state[0]-previous_h)}
                if before_previous is not None:
                    row['hidden_two_step_rms'] = rms(state[0]-before_previous)
                rows.append(row)
                before_previous, previous_h = previous_h, state[0]

            k, v, q = map(torch.stack, (keys, values, queries))
            actual_m, actual_y = map(torch.stack, (memories, actual_reads))
            if hasattr(layer, 'trace_decay_channels'):
                lam = layer.trace_decay_channels.double()[None, :, None, :]
            else:
                # Read-only comparison of the chosen STDP window on a B-only trajectory.
                lam = torch.full((1, inner.H, 1, inner.dh), .1, device=args.device, dtype=torch.float64)
            common_lam = lam.mean(-1, keepdim=True).expand_as(lam)
            g, m = replay(k, v, lam)
            gc, mc = replay(k, v, common_lam)
            b = v.transpose(-1, -2) @ k / k.shape[-2]

            tail_k, tail_v = k[-args.tail:], v[-args.tail:]
            km, kd, ks = phase_summary(tail_k)
            vm, vd, vs = phase_summary(tail_v)
            # Even-minus-odd convention matches phase_summary. Mean is unaffected.
            a = (1-lam)/(1+lam)
            gdc = mat(a*vd, kd)-mat(vd, a*kd)
            galt = mat((1+a)*vd, km)-mat(vm, (1+a)*kd)

            def heads(x):
                return x.reshape(args.batch, 81, inner.H, inner.dh).transpose(1, 2)

            k_inj = inner.apply_rope(heads(layer.k_proj(inner.embed_scale*inj)), layer).double()
            v_inj = heads(layer.v_proj(inner.embed_scale*inj)).double()
            kh, vh = k-k_inj, v-v_inj
            _, mh = replay(kh, vh, common_lam)
            # Closed-form endpoint contribution at every prefix.
            boundary_errors = []
            endpoints = []
            for n in range(1, args.blocks+1):
                powers = torch.arange(n, device=args.device)[:, None, None, None, None]
                coeff = common_lam.pow(n-1-powers)-common_lam.pow(powers)
                dk = (coeff*kh[:n]).sum(0)
                dv = (coeff*vh[:n]).sum(0)
                endpoint = mat(dv, k_inj)-mat(v_inj, dk)
                endpoints.append(endpoint)
                boundary_errors.append(float((mc[n-1]-mh[n-1]-endpoint).abs().max()))
            endpoints = torch.stack(endpoints)
            y_common = q @ mc.transpose(-1, -2)
            y_dynamic = q @ mh.transpose(-1, -2)
            y_boundary = q @ endpoints.transpose(-1, -2)
            tail_g = g[-args.tail:]
            observed_gdc = tail_g.mean(0)
            observed_galt = (tail_g[1::2].mean(0)-tail_g[::2].mean(0))/2
            tail_rows = rows[-args.tail:]
            supervision_rows = [row for row in tail_rows if row['block'] % 8 == 0]

            record = {
                'checkpoint': str(Path(path).resolve()), 'step': ck['step'],
                'seconds': time.monotonic()-started,
                'lambda_min': float(lam.min()), 'lambda_max': float(lam.max()),
                'trajectory_at_supervised_blocks': {
                    key: sum(row[key] for row in supervision_rows)/len(supervision_rows)
                    for key in ('accuracy', 'loss', 'hidden_one_step_rms', 'hidden_two_step_rms')},
                'key_tail': ks, 'value_tail': vs,
                'operator_replay': {
                    'tail_current_KV_rms': rms(b[-args.tail:]),
                    'tail_STDP_write_rms': rms(tail_g),
                    'final_heterogeneous_vs_common_memory': comparison(m[-1], mc[-1]),
                    'final_memory_difference_rms': rms(m[-1]-mc[-1]),
                    'actual_memory_vs_replayed_M': comparison(actual_m[-1], m[-1]),
                    'note': 'Actual M equivalence is only expected for accumulated-STDP variants. On B-only or G-only trajectories this is a counterfactual operator calculation.'},
                'period_two_prediction': {
                    'observed_cycle_average_write_rms': rms(observed_gdc),
                    'predicted_heterogeneous_cycle_average_write_rms': rms(gdc),
                    'mean_prediction_vs_observed': comparison(gdc, observed_gdc),
                    'predicted_alternating_write_rms': rms(galt),
                    'alternating_prediction_vs_observed': comparison(galt, observed_galt),
                    'shared_window_observed_cycle_average_write_rms': rms(gc[-args.tail:].mean(0)),
                    'note': 'Formula is exact for a prescribed exact period-2 orbit with equilibrated traces. Residual temporal drift is visible in the prediction error.'},
                'common_window_static_input_decomposition': {
                    'max_endpoint_identity_error': max(boundary_errors),
                    'final_M_rms': rms(mc[-1]),
                    'final_dynamic_history_M_rms': rms(mh[-1]),
                    'final_static_input_boundary_M_rms': rms(endpoints[-1]),
                    'final_boundary_vs_total_M': comparison(endpoints[-1], mc[-1]),
                    'tail_total_read_rms': rms(y_common[-args.tail:]),
                    'tail_dynamic_history_read_rms': rms(y_dynamic[-args.tail:]),
                    'tail_static_input_boundary_read_rms': rms(y_boundary[-args.tail:]),
                    'tail_boundary_vs_total_read': comparison(y_boundary[-args.tail:], y_common[-args.tail:]),
                    'note': 'The components can interfere; their norms are not additive fractions. Common-window replay does not change the trajectory.'},
                'rows': rows}
            results['runs'][variant] = record
            out.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
            print(variant, json.dumps({k: record[k] for k in
                ('seconds', 'trajectory_at_supervised_blocks', 'key_tail', 'value_tail', 'period_two_prediction',
                 'common_window_static_input_decomposition')}, allow_nan=False), flush=True)
        inner.memory_step = original_memory_step
        del ck, model, inner, layer, k, v, q, g, m, gc, mc, b, kh, vh, mh
        del keys, values, queries, memories, actual_reads, actual_m, actual_y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
