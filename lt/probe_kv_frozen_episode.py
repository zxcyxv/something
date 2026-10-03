"""Compare saved online-training carry with a fresh fixed-parameter episode.

Use each checkpoint's exact saved training puzzles and all saved lanes. This is
read-only; a difference includes the entire history of slow parameter updates
and is not by itself a causal isolation of STDP or truncation.
"""
import argparse
import json
from pathlib import Path

import torch

from . import train as t
from .kv_stability import install


def metrics(logits, labels):
    valid = labels != t.IGNORE_LABEL_ID
    counts = valid.sum(-1).clamp_min(1)
    correct = ((logits.argmax(-1)==labels)&valid)
    return {'accuracy': float((correct.sum(-1)/counts).mean()),
            'exact': float((correct.sum(-1)==counts).float().mean()),
            'loss': float((t.stablemax_cross_entropy(logits, labels)/counts[:, None]).sum(-1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'precision': 'BF16 projections / FP32 states, same as original training',
              'method': 'each checkpoint, all saved training puzzles, fresh hidden/M/traces, frozen final raw parameters',
              'caveat': 'Saved carry uses earlier evolving parameters; new classifier is used on both. This is a training/evaluation trajectory difference, not its attribution to any single mechanism.',
              'runs': {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in args.checkpoints:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        variant = ck['cfg'].get('research_variant', 'original')
        install(variant)
        cfg = dict(ck['cfg'], batch_size=args.chunk, seq_len=81, num_puzzle_identifiers=1,
                   amp=True, activation_checkpoint=False, nograd_blocks=0)
        model = t.LT(cfg).cuda().eval()
        model.load_state_dict({k.removeprefix('model.'): v for k, v in ck['raw_model_state_dict'].items()})
        inner = model.inner
        saved = ck['rank_states'][0]['carry']
        labels = saved['current_data']['labels']
        n = len(labels)
        steps = saved['steps'].unique().tolist()
        assert len(steps) == 1
        blocks = int(steps[0])*cfg['blocks_per_seg']
        saved_outputs, fresh_outputs, hidden_errors, per_segment = [], [], [], []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            for start in range(0, n, args.chunk):
                batch = {k: v[start:start+args.chunk].cuda() for k, v in saved['current_data'].items()}
                count = len(batch['labels'])
                old_h = saved['current_hidden'][start:start+args.chunk].cuda()
                saved_outputs.append(inner.w_cls(old_h).float().cpu())
                inj = inner.injection(batch)
                state = inner.init_hidden[None, None, :].expand(count, 81, -1), None, None, None
                segment_logits = []
                for r in range(1, blocks+1):
                    for layer in inner.layers:
                        state = inner.block(layer, state[0], inj, *state[1:], None)
                    if r % cfg['blocks_per_seg'] == 0:
                        segment_logits.append(inner.w_cls(state[0]).float().cpu())
                fresh_outputs.append(segment_logits[-1])
                per_segment.append(torch.stack(segment_logits))
                hidden_errors.append((state[0]-old_h).float().cpu())
        online_logits = torch.cat(saved_outputs)
        frozen_logits = torch.cat(fresh_outputs)
        segments = torch.cat(per_segment, dim=1)
        record = {'checkpoint': str(Path(path).resolve()), 'puzzles': n, 'blocks': blocks,
                  'saved_online_carry': metrics(online_logits, labels),
                  'fresh_frozen_episode': metrics(frozen_logits, labels),
                  'prediction_disagreement': float((online_logits.argmax(-1)!=frozen_logits.argmax(-1)).float().mean()),
                  'hidden_difference_rms': float(torch.cat(hidden_errors).square().mean().sqrt()),
                  'frozen_segments': [{'segment': i+1, **metrics(x, labels)} for i, x in enumerate(segments)]}
        result['runs'][variant] = record
        out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(variant, json.dumps({k:v for k,v in record.items() if k!='frozen_segments'}), flush=True)
        del ck, model, inner, saved
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
