"""Exact decomposition of current/past and two-phase reads on frozen paths.

All decompositions use the unchanged trajectory. They are not counterfactual
inference or training, and cannot establish which component causes collapse.
"""
import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .kv_stability import install


def rms(x):
    return float(x.square().mean().sqrt())


def component(x, total):
    denominator = total.square().sum().clamp_min(1e-30)
    return dict(rms=rms(x),
                energy_relative_to_total=float(x.square().sum()/denominator),
                signed_projection_on_total=float((x*total).sum()/denominator))


def decomposition(a, b, total):
    return dict(total_rms=rms(total), a=component(a,total), b=component(b,total),
                a_b_cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)),
                relative_identity_error=float((a+b-total).norm()/total.norm().clamp_min(1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--blocks', type=int, default=128)
    ap.add_argument('--tail', type=int, default=32)
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()
    if args.blocks % 2 or args.tail % 2 or not 2 <= args.tail < args.blocks:
        ap.error('Even blocks/tail and 2 <= tail < blocks required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(method='Unchanged fixed-parameter trajectories, FP32; FP64 read decomposition after output projection',
                  blocks=args.blocks, tail=args.tail, batch=args.batch,
                  caveat='Same saved training puzzles, no optimizer or parameter modification. Contributions on one trajectory are not causal interventions.', runs={})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    common = None
    for path in args.checkpoints:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        variant = ck['cfg'].get('research_variant', 'original')
        if variant not in ('original','complex_current_only'):
            raise ValueError('Only accumulated STDP M and G-only are supported')
        install(variant)
        cfg = dict(ck['cfg'], batch_size=args.batch, seq_len=81,
                   num_puzzle_identifiers=1, amp=False, activation_checkpoint=False,
                   nograd_blocks=0)
        model = t.LT(cfg).cuda().eval()
        model.load_state_dict({k.removeprefix('model.'):v for k,v in ck['raw_model_state_dict'].items()})
        inner, layer = model.inner, model.inner.layers[0]
        assert len(inner.layers)==1
        assert not cfg['kv_qk_l2norm'] and not cfg['kv_qk_rmsnorm']
        batch = {k:v[:args.batch].cuda() for k,v in ck['rank_states'][0]['carry']['current_data'].items()}
        if common is None:
            common = batch
        else:
            assert all(torch.equal(v,common[k]) for k,v in batch.items())
        queries, operators, writes, actual_reads = [], [], [], []
        original_step, original_update = inner.memory_step, inner.update_memory
        pending = {}
        block = 0

        def capture_update(memory, write):
            pending['write'] = write
            return original_update(memory,write)

        def capture_step(L,q,k,v,*rest):
            answer = original_step(L,q,k,v,*rest)
            # One preceding block also supports the exact first-difference identity.
            if block >= args.blocks-args.tail:
                queries.append(inner.apply_rope(q.float(),L).double())
                operators.append(answer[1].double())
                writes.append(pending['write'].double())
                actual_reads.append(answer[0].double())
            return answer

        inner.update_memory, inner.memory_step = capture_update, capture_step
        with torch.no_grad():
            inj = inner.injection(common)
            state = inner.init_hidden[None,None,:].expand(args.batch,81,-1),None,None,None
            for block in range(1,args.blocks+1):
                state = inner.block(layer,state[0],inj,*state[1:],None)
            q, op, g, actual = map(torch.stack,(queries,operators,writes,actual_reads))
            assert len(q)==args.tail+1
            weight = layer.out_proj.weight.double()

            def project(heads):
                joined = heads.transpose(-3,-2).reshape(*heads.shape[:-3],81,inner.d)
                return joined @ weight.T

            def read(matrix, query):
                return project(query @ matrix.transpose(-1,-2))

            y = read(op,q)
            g_read = read(g,q)
            current_past = decomposition(g_read[1:],y[1:]-g_read[1:],y[1:])
            qo, qe = q[1::2], q[2::2]
            mo, me = op[1::2], op[2::2]
            qm, qd = (qe+qo)/2,(qe-qo)/2
            mm, md = (me+mo)/2,(me-mo)/2
            mean_y, delta_y = (y[2::2]+y[1::2])/2,(y[2::2]-y[1::2])/2
            record = dict(checkpoint=str(Path(path).resolve()),
                production_read_reconstruction_error=float((y-project(actual)).norm()/y.norm().clamp_min(1e-30)),
                current_and_past=current_past,
                paired_mean=decomposition(read(mm,qm),read(md,qd),mean_y),
                paired_alternation=decomposition(read(mm,qd),read(md,qm),delta_y))
            # Keep expression labels separate from measured component data.
            for key, labels in (('current_and_past',('G_r q_r','(operator_r-G_r) q_r')),
                                ('paired_mean',('operator_mean q_mean','operator_delta q_delta')),
                                ('paired_alternation',('operator_mean q_delta','operator_delta q_mean'))):
                record[key]['a_expression'],record[key]['b_expression']=labels
            if variant=='original':
                delta_y = y[1:]-y[:-1]
                record['read_first_difference'] = decomposition(
                    g_read[1:],read(op[:-1],q[1:]-q[:-1]),delta_y)
                record['read_first_difference']['a_expression']='G_r q_r'
                record['read_first_difference']['b_expression']='M_previous (q_r-q_previous)'
            # These identities hold on every pair, without assuming periodicity.
            assert record['production_read_reconstruction_error'] < 1e-5
            for key in ('current_and_past','paired_mean','paired_alternation'):
                assert record[key]['relative_identity_error'] < 1e-10
            if variant=='original':
                assert record['read_first_difference']['relative_identity_error'] < 1e-4
            result['runs'][variant] = record
            out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(variant,json.dumps(record),flush=True)
        del ck, model, inner, layer, q, op, g, y, actual, state
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
