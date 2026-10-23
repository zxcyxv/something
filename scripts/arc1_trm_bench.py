"""Speed benchmark bookkeeping for scripts/arc1_trm.sh (one GPU instance per JSON; two instances compared by decide).

    python scripts/arc1_trm_bench.py collect BENCH_DIR --price 12.5 --gpu-mem 79.6 --target-steps 402600
    python scripts/arc1_trm_bench.py decide bench/4xH100.json bench/4xPRO6000.json
    python scripts/arc1_trm_bench.py decide H100:0.42:16.0 PRO6000:0.66:8.0        # label:s_per_step:price_per_hour

collect reads BENCH_DIR/<candidate>/{config.json,train.jsonl,stdout.log} (one candidate = precision x grad_accum x
act_ckpt), keeps the steady-state steps (after WARM), and writes BENCH_DIR/bench.json: per candidate the wall-clock
seconds per optimizer step (data loading included), the step-time median (GPU step only) and the peak memory; per
precision the fastest candidate; and hours / cost for 1,000 steps, the target and 24 hours.
decide: per precision, the cost of 1,000 steps on each instance; the instance whose speed advantage does not cover its
price ratio loses (the break-even price ratio equals the speed ratio).
"""
import argparse
import json
import math
import socket
import subprocess
import sys
from pathlib import Path

WARM = 10                      # steps dropped: compilation (step 1), carry/ACT recompiles, allocator warm-up


def _rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _median(x):
    x = sorted(x)
    n = len(x)
    return math.nan if not n else (x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2]))


def precision(cfg):
    if cfg.get('band_bf16'):
        return 'bf16'
    return 'tf32' if cfg.get('matmul_precision', 'highest') != 'highest' else 'fp32'


def candidate(d):
    d = Path(d)
    out = dict(name=d.name, ok=False)
    cfg_path, log = d / 'config.json', d / 'stdout.log'
    text = log.read_text(errors='replace') if log.exists() else ''
    if 'OutOfMemoryError' in text or 'CUDA out of memory' in text:
        out['error'] = 'OOM'
    if not cfg_path.exists() or not (d / 'train.jsonl').exists():
        out.setdefault('error', 'no steps (see stdout.log)')
        return out
    cfg = json.loads(cfg_path.read_text())
    rows = _rows(d / 'train.jsonl')
    out.update(precision=precision(cfg), grad_accum=cfg['grad_accum'], act_ckpt=cfg['act_ckpt'],
               global_batch_size=cfg['global_batch_size'], loops=cfg['loops'], steps=len(rows))
    if len(rows) < WARM + 5:
        out.setdefault('error', f'only {len(rows)} steps')
        return out
    steady = rows[WARM:]
    wall = (steady[-1]['elapsed'] - steady[0]['elapsed']) / (len(steady) - 1)
    finite = all(math.isfinite(r['lm_loss']) for r in rows)
    out.update(ok=finite and 'error' not in out, wall_s_per_step=wall,
               step_s_median=_median([r['seconds'] for r in steady]),
               mem_gb=max(r.get('mem_gb', 0.0) for r in rows), lm_loss_last=rows[-1]['lm_loss'],
               compile_s=rows[0]['seconds'])
    if not finite:
        out['error'] = 'nonfinite loss'
    return out


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ''


def collect(a):
    root = Path(a.bench_dir)
    cands = [candidate(d) for d in sorted(root.iterdir()) if d.is_dir()]
    try:
        import torch
        env = dict(torch=torch.__version__, cuda=torch.version.cuda, ngpu=torch.cuda.device_count(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    except Exception as exc:                                   # report without torch
        env = dict(error=str(exc))
    env['driver'] = _sh('nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')
    env['host'] = socket.gethostname()
    label = a.label or f"{env.get('ngpu', '?')}x{env.get('gpu', 'gpu').replace('NVIDIA ', '').replace(' ', '_')}"
    best = {}
    for c in cands:
        if c['ok'] and (c['precision'] not in best or c['wall_s_per_step'] < best[c['precision']]['wall_s_per_step']):
            best[c['precision']] = c
    proj = {}
    for p, c in best.items():
        h1k = c['wall_s_per_step'] * 1000 / 3600
        proj[p] = dict(setting=dict(grad_accum=c['grad_accum'], act_ckpt=c['act_ckpt'], precision=p),
                       s_per_step=c['wall_s_per_step'], hours_per_1k_steps=h1k,
                       usd_per_1k_steps=h1k * a.price if a.price else None,
                       target_steps=a.target_steps, target_hours=h1k * a.target_steps / 1000,
                       target_usd=h1k * a.target_steps / 1000 * a.price if a.price else None,
                       steps_in_24h=int(24 * 3600 / c['wall_s_per_step']))
    res = dict(label=label, price_per_hour=a.price, gpu_mem_gib=a.gpu_mem, env=env, candidates=cands, best=proj)
    (root / 'bench.json').write_text(json.dumps(res, indent=1) + '\n')
    print(f"\n== {label}  ({env.get('torch')}, CUDA {env.get('cuda')}, driver {env.get('driver')})"
          + (f", ${a.price}/h" if a.price else ''))
    for c in cands:
        if c.get('wall_s_per_step'):
            print(f"  {c['name']:28s} {c['wall_s_per_step']:.3f} s/step (GPU step {c['step_s_median']:.3f}), "
                  f"peak {c['mem_gb']:.1f} GiB" + ('' if c['ok'] else f"  [{c.get('error')}]"))
        else:
            print(f"  {c['name']:28s} failed: {c.get('error')}")
    for p, r in proj.items():
        s = r['setting']
        print(f"  best {p}: grad_accum {s['grad_accum']} act_ckpt {s['act_ckpt']} -> {r['s_per_step']:.3f} s/step, "
              f"{r['hours_per_1k_steps']:.3f} h/1k steps"
              + (f", ${r['usd_per_1k_steps']:.2f}/1k steps" if r['usd_per_1k_steps'] is not None else '')
              + f"; {r['target_steps']:,} steps = {r['target_hours']:.1f} h"
              + (f" (${r['target_usd']:,.0f})" if r['target_usd'] is not None else '')
              + f"; 24 h = {r['steps_in_24h']:,} steps")
    print(f"BENCH_JSON {json.dumps(dict(label=label, price=a.price, best={p: dict(s_per_step=r['s_per_step'], **r['setting']) for p, r in proj.items()}))}")
    print(f"wrote {root / 'bench.json'}")


def _load(spec):
    p = Path(spec)
    if p.exists():
        d = json.loads(p.read_text())
        if 'label' in d and 'best' in d and 'price' in d and 'env' not in d:          # a pasted BENCH_JSON line
            return d['label'], d.get('price'), {k: v['s_per_step'] for k, v in d['best'].items()}
        return d['label'], d.get('price_per_hour'), {k: v['s_per_step'] for k, v in d['best'].items()}
    if spec.startswith('{'):
        d = json.loads(spec)
        return d['label'], d.get('price'), {k: v['s_per_step'] for k, v in d['best'].items()}
    label, s, price = spec.split(':')                         # label:s_per_step:price (one precision: 'any')
    return label, float(price) if price else None, {'any': float(s)}


def decide(a):
    runs = [_load(s) for s in a.specs]
    if len(runs) != 2:
        sys.exit('decide takes two instances')
    (la, pa, sa), (lb, pb, sb) = runs
    common = [p for p in ('fp32', 'tf32', 'bf16', 'any') if p in sa and p in sb]
    if not common:
        sys.exit(f'no common precision: {sorted(sa)} vs {sorted(sb)}')
    print(f'{"precision":9s} {la:>22s} {lb:>22s}  speed {la}/{lb}  price {la}/{lb}  cheaper per step')
    for p in common:
        speed = sb[p] / sa[p]                                  # > 1: A is faster
        line = f'{p:9s} {sa[p]:18.3f} s/st {sb[p]:18.3f} s/st  {speed:9.2f}x'
        if pa and pb:
            ca, cb = sa[p] * pa / 3.6, sb[p] * pb / 3.6        # $ per 1,000 steps
            win = la if ca < cb else lb
            line += (f'  {pa / pb:9.2f}x  {win} (${min(ca, cb):.2f} vs ${max(ca, cb):.2f} per 1k steps, '
                     f'{100 * (1 - min(ca, cb) / max(ca, cb)):.0f}% less)')
        else:
            line += f'  break-even price ratio {speed:.2f} (A is worth it if its price is under {speed:.2f}x B)'
        print(line)
    if pa and pb:
        p = min(common, key=lambda q: min(sa[q] * pa, sb[q] * pb))
        win = la if sa[p] * pa < sb[p] * pb else lb
        print(f'-> keep {win} (precision {p}); release the other instance.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('bench_dir')
    c.add_argument('--price', type=float, default=None, help='instance $/hour (all GPUs)')
    c.add_argument('--gpu-mem', type=float, default=None, help='GiB per GPU (recorded only)')
    c.add_argument('--target-steps', type=int, default=519200, help='TRM ARC-AGI-1: 100,000 epochs x 3,988 / 768')
    c.add_argument('--label', default=None)
    d = sub.add_parser('decide')
    d.add_argument('specs', nargs='+', help='bench.json files, BENCH_JSON payloads, or label:s_per_step:price')
    a = ap.parse_args(argv)
    collect(a) if a.cmd == 'collect' else decide(a)


if __name__ == '__main__':
    main()
