"""Fixed-window causal diagnostics of preselected late correction events."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from probe_late_puzzles import EventBlocks, setup, write_json


MODES = ('normal', 'freeze_agree', 'freeze_window', 'freeze_both',
         'freeze_memory', 'instant_memory_read', 'freeze_source_directions')


def incident(destination, source, cell):
    destination[:, cell, :] = source[:, cell, :]
    destination[:, :, cell] = source[:, :, cell]


def stable_start(correct, start, width=32):
    if len(correct) < width:
        return None
    at = np.flatnonzero(np.convolve(correct.astype(int), np.ones(width, dtype=int), 'valid') == width)
    return int(start + at[0]) if len(at) else None


def describe(pred, margins, labels, cell, start, event_block):
    target = pred[:, cell] == labels[cell]
    whole = (pred == labels).all(-1)
    t = event_block - start
    return {'prediction_digit_at_T': int(pred[t, cell])-1,
            'target_correct_at_T': bool(target[t]), 'puzzle_correct_at_T': bool(whole[t]),
            'gold_minus_old_margin_at_T': float(margins[t]),
            'first_32block_stable_target': stable_start(target, start),
            'first_32block_stable_puzzle': stable_start(whole, start),
            'target_correct_fraction_after_T': float(target[t+1:].mean()),
            'final_target_correct': bool(target[-1]), 'final_puzzle_correct': bool(whole[-1])}


def restore(root, b, start, baseline_pred):
    k0 = ((start - 1) // 128) * 128
    state = torch.load(root/'snapshots'/f'block_{k0:05d}.pt', map_location='cuda', weights_only=True)
    h, w = state['h'], state['w']
    for k in range(k0+1, start):
        h, w = b.block(h, w)
    got = b.inner.w_cls(h).argmax(-1).cpu().numpy()
    assert np.array_equal(got, baseline_pred[start-1]), 'Snapshot replay did not reproduce baseline.'
    return h, w


def score_cell(b, hidden, gold, old):
    logits = b.inner.w_cls(hidden)
    return {'digit': int(logits.argmax())-1, 'gold_minus_old_margin': float(logits[gold]-logits[old])}


def local_diagnostics(b, p, pre, event, before_pred, labels, inputs):
    i, c = event['batch_index'], event['cell']
    gold, old = event['gold_digit']+1, event['old_digit']+1
    decoder = b.inner.w_cls.weight[gold]-b.inner.w_cls.weight[old]
    denominator = torch.sqrt(1+pre[i,c].square().sum()/b.inner.d)
    direction = torch.einsum('hcd,d->hc', b.layer.w_sh, decoder)
    evidence = (p['v'][i]*direction[None]).sum(-1).T/denominator
    lam = b.lam[:,0,0,None]
    coeffs = {'instant_read': (1-lam)*p['a'][i,:,c],
              'current_write': lam*p['target'][i,:,c],
              'history_vs_current_G': lam*(p['w'][i,:,c]-p['target'][i,:,c]),
              'total_memory': lam*p['w'][i,:,c]}
    base = score_cell(b, b.inner.phi(pre[i,c]), gold, old)
    def remove(coefficient):
        delta = torch.einsum('hn,nhc,hcd->d', coefficient, p['v'][i], b.layer.w_sh)
        return score_cell(b, b.inner.phi(pre[i,c]-delta), gold, old)
    report = {'normal': base, 'remove_component': {name: remove(value) for name,value in coeffs.items()},
              'algebraic_margin_components': {name: float((value*evidence).sum()) for name,value in coeffs.items()}}
    report['algebraic_margin_components']['prepared_hidden'] = float((p['q'][i,c]*decoder).sum()/denominator)
    report['algebraic_margin_components']['classifier_bias'] = float(b.inner.w_cls.bias[gold]-b.inner.w_cls.bias[old])
    parts = report['algebraic_margin_components']
    reconstructed = sum(parts[key] for key in ('prepared_hidden','classifier_bias','instant_read','current_write','history_vs_current_G'))
    report['algebraic_margin_reconstruction_error'] = reconstructed-base['gold_minus_old_margin']
    assert abs(report['algebraic_margin_reconstruction_error']) < 2e-3
    edges = (coeffs['history_vs_current_G']*evidence).cpu().numpy()
    edges[:, c] = -np.inf
    ranked = np.argsort(edges.ravel())[::-1][:5]
    report['top_history_edges'] = []
    combined = torch.zeros_like(coeffs['history_vs_current_G'])
    for rank, index in enumerate(ranked):
        head, source = map(int, np.unravel_index(index, edges.shape))
        one = torch.zeros_like(combined)
        one[head,source] = coeffs['history_vs_current_G'][head,source]
        if rank < 3:
            combined += one
        r, col = divmod(c,9); sr, sc = divmod(source,9)
        entry = {'head':head,'source':source,'source_rc':[sr+1,sc+1],
                 'source_digit_before':int(before_pred[source])-1,
                 'source_correct_before':bool(before_pred[source] == labels[source]),
                 'source_given':bool(inputs[source] != 1),
                 'sudoku_peer':bool(r==sr or col==sc or (r//3==sr//3 and col//3==sc//3)),
                 'agree':float(p['agree'][i,head,c,source]),
                 'write_window':float(p['window'][i,head,c,source]),
                 'G':float(p['target'][i,head,c,source]),'w':float(p['w'][i,head,c,source]),
                 'history_margin_contribution':float(edges[head,source]),
                 'removed_history_edge':remove(one)}
        report['top_history_edges'].append(entry)
    report['remove_top3_history_edges'] = remove(combined)
    return report


def analyze_event(root, b, event, normal_pred, x, y):
    i, c, t = event['batch_index'], event['cell'], event['stable_complete_block']
    gold, old = event['gold_digit']+1, event['old_digit']+1
    start, stop = t-128, t+128
    initial_h, initial_w = restore(root,b,start,normal_pred)
    p0 = b.parts(initial_h,initial_w)
    fixed = {name:p0[name][i].clone() for name in ('agree','window','v')}
    fixed['w'] = initial_w[i].clone()
    result = {'event':event,'window':[start,stop],'active_intervention_blocks':[start,t],
              'continuations':{},'normal_replay_all_predictions_exact':True}
    saved = {'blocks':np.arange(start,stop+1),'labels':y[i]+1,'inputs':x[i]+1}
    trace = {}
    before = None
    for mode in MODES:
        h,w = initial_h.clone(),initial_w.clone()
        predictions, margins = [],[]
        before = normal_pred[start-1,i]
        for k in range(start,stop+1):
            p = b.parts(h,w)
            if k<=t:
                if mode in ('freeze_agree','freeze_both'):
                    incident(p['agree'][i],fixed['agree'],c)
                if mode in ('freeze_window','freeze_both'):
                    incident(p['window'][i],fixed['window'],c)
                if mode in ('freeze_agree','freeze_window','freeze_both'):
                    p['target'] = b.gain*(p['window']*p['agree'])
                    p['w'] = (1-b.eta)*w+b.eta*p['target']
                if mode=='freeze_memory':
                    incident(p['w'][i],fixed['w'],c)
            read_w = None
            if k<=t and mode=='instant_memory_read':
                read_w = p['w'].clone()
                read_w[i,:,c] = p['target'][i,:,c]
            hn,pre,effective = b.read(p,read_w)
            if k<=t and mode=='freeze_source_directions' and k>start:
                old_v,current_v = fixed['v'],p['v'][i]
                alt = old_v/old_v.norm(dim=-1,keepdim=True).clamp_min(1e-12)*current_v.norm(dim=-1,keepdim=True)
                alt[c] = current_v[c]
                delta = torch.einsum('hn,nhc,hcd->d',effective[i,:,c],current_v-alt,b.layer.w_sh)
                hn[i,c] = b.inner.phi(pre[i,c]-delta)
            logits = b.inner.w_cls(hn)
            pred = logits.argmax(-1).cpu().numpy().astype(np.uint8)
            predictions.append(pred[i]); margins.append(float(logits[i,c,gold]-logits[i,c,old]))
            if mode=='normal':
                assert np.array_equal(pred,normal_pred[k]), f'Normal replay mismatch at block {k}'
                decoder = b.inner.w_cls.weight[gold]-b.inner.w_cls.weight[old]
                denominator = torch.sqrt(1+pre[i,c].square().sum()/b.inner.d)
                direction = torch.einsum('hcd,d->hc',b.layer.w_sh,decoder)
                evidence = (p['v'][i]*direction[None]).sum(-1).T/denominator
                lam = b.lam[:,0,0,None]
                fields = {name:p[name][i,:,c] for name in ('agree','window','target','w','a')}
                fields.update(current_write_contribution=lam*p['target'][i,:,c]*evidence,
                              history_contribution=lam*(p['w'][i,:,c]-p['target'][i,:,c])*evidence,
                              instant_read_contribution=(1-lam)*p['a'][i,:,c]*evidence)
                for name,value in fields.items():
                    trace.setdefault(name,[]).append(value.cpu().numpy().copy())
                if k==t:
                    result['local_at_T'] = local_diagnostics(b,p,pre,event,before,y[i]+1,x[i]+1)
            h,w = hn,p['w']
            before = pred[i]
        predictions,margins = np.stack(predictions),np.asarray(margins)
        saved[mode+'_predictions'] = predictions
        saved[mode+'_margin'] = margins
        result['continuations'][mode] = describe(predictions,margins,y[i]+1,c,start,t)
        print('CASE',event['puzzle'],mode,json.dumps(result['continuations'][mode]),flush=True)
    for name,values in trace.items():
        saved['normal_'+name] = np.stack(values)
    case_dir = root/f"case_{event['puzzle']}_{c}"
    case_dir.mkdir(exist_ok=True)
    np.savez_compressed(case_dir/'traces.npz',**saved)
    write_json(case_dir/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/late_puzzle_probe_v11')
    args=p.parse_args();root=Path(args.root)
    with np.load(root/'baseline.npz') as data:
        ids,x,y,pred = (data[key] for key in ('indices','X','Y','predictions'))
    events=json.loads((root/'events.json').read_text())['selected_events']
    model,batch,x2,y2=setup(ids)
    assert np.array_equal(x,x2) and np.array_equal(y,y2)
    b=EventBlocks(model,batch)
    results=[]
    for event in events:
        results.append(analyze_event(root,b,event,pred,x,y))
    metadata={'checkpoint_sha256':hashlib.sha256(Path('checkpoints/v1.1_step160000.npz').read_bytes()).hexdigest(),
              'data_sha256':hashlib.sha256(Path('data/sudoku_lt_1k.npz').read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256(Path('docs/late_puzzle_probe_plan_v11.md').read_bytes()).hexdigest(),
              'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(),
              'precision':'FP32; autocast/TF32 off; fp16-compressed checkpoint',
              'case_selection':'outcome-selected mechanistic cases; not an estimate of population frequency'}
    write_json(root/'analysis_summary.json',{'metadata':metadata,'cases':results})


if __name__=='__main__':
    main()
