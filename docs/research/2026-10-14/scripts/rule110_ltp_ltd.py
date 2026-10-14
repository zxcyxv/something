"""Rule 110 with the run-3 model (displacement clock): on test inputs not solved after one block,
does the trained model actually use LTP/LTD?

Usage (repository root):
  python docs/research/2026-10-14/scripts/rule110_ltp_ltd.py train --variant run3|bonly --steps N [--seed S]
  python docs/research/2026-10-14/scripts/rule110_ltp_ltd.py probe --variant run3 [--seed S]

Task: a row of 36 cells (bits, zero outside the row); target = the row after K_STEPS Rule 110 updates.
Per-cell output (2 classes), like the sudoku harness. Inputs are fresh random rows every step (no fixed train set);
the test set is 4,096 rows from a separate seed.

Model: the actual run-3 class (lt/research_phase_clock_stdp.py, clock 'displacement', torch path, FP32, CUDA):
hidden 32, one head (32 channels = 16 K pairs and 16 V pairs), lambda_init 0.1, alpha_init 1, SwiGLU,
two residual RMSNorms. Positions are 1D (pos_u = cell index, pos_w = 0) instead of the 2D sudoku grid.
'bonly' is the same model with alpha fixed at 0 (no timing term).
Training differs from the sudoku harness: one segment of BLOCKS blocks, full backprop, a cross-entropy loss on the
readout after every block (so 'solved after one block' is defined), AdamW lr 1e-3, weight decay 0.01 except
alpha, lambda, RoPE theta and biases.

probe: on the test rows whose block-1 readout is wrong, final-block accuracy with scale-preserving interventions on
the phase tags fed to the write (the clock state itself keeps running):
  flip     phi -> -phi for K and V: every sin(phi_V - phi_K) changes sign (LTP <-> LTD), magnitudes kept
  shuffle  the tags of all neurons are permuted across cells (same permutation for K and V), distribution kept
  freeze   tags fixed at their block-2 values from block 3 on
and, as a scale-changing reference, alpha = 0. Also the LTP/LTD shares and ||alpha Im S|| / ||Hebb|| (G level).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lt import train as t                                   # noqa: E402
from lt import research_phase_clock_stdp as rp              # noqa: E402

T, GRID, K_STEPS, BLOCKS, HIDDEN = 36, 6, 4, 8, 32
RULE110 = torch.tensor([0, 1, 1, 1, 0, 1, 1, 0])           # output for neighbourhood 4l + 2c + r
OUT = ROOT / 'docs/research/2026-10-14'
DEVICE = torch.device('cuda')


def evolve(x, steps):
    table = RULE110.to(x.device)
    for _ in range(steps):
        left, right = F.pad(x, (1, 0))[:, :-1], F.pad(x, (0, 1))[:, 1:]
        x = table[4 * left + 2 * x + right]
    return x


def sample(n, gen):
    x = torch.randint(0, 2, (n, T), generator=gen, device=DEVICE)
    return x, evolve(x, K_STEPS)


def build(variant, seed):
    t.KVSTDPInner = rp.model_class(clock='displacement', kernel='torch', alpha_init=1.0)
    cfg = dict(t.DEFAULT_CFG, **t.PRESETS['kv_stdp'])
    cfg.update(hidden_size=HIDDEN, num_heads=1, puzzle_emb_ndim=0, global_batch_size=512, batch_size=512,
               seq_len=T, grid=GRID, vocab_size=3, num_puzzle_identifiers=1, blocks_per_seg=BLOCKS, loops=1,
               amp=False, amp_dtype='float32', forward_dtype='float32', activation_checkpoint=False,
               compile=False, kv_write_reduction='mean', nograd_blocks=0)
    torch.manual_seed(seed)
    with torch.device(DEVICE):
        base = t.ACTLossHead(t.LT(cfg), q_weight=cfg['q_weight'])
    inner = base.model.inner
    inner.pos_u = torch.arange(T, dtype=torch.float32, device=DEVICE)
    inner.pos_w = torch.zeros(T, dtype=torch.float32, device=DEVICE)
    if variant == 'bonly':
        inner.layers[0].stdp_alpha.data.zero_()
        inner.layers[0].stdp_alpha.requires_grad_(False)
    return base, inner


def forward(inner, x, blocks=BLOCKS):
    """Per-block logits [blocks, B, T, 3]; tokens are bit + 1 (0 is unused)."""
    batch = dict(inputs=x + 1, puzzle_identifiers=torch.zeros(x.shape[0], dtype=torch.int32, device=x.device))
    h = inner.init_hidden.to(torch.float32).expand(x.shape[0], T, HIDDEN).clone()
    inj = inner.injection(batch)
    memory = e_k = e_v = None
    layer = inner.layers[0]
    logits = []
    for _ in range(blocks):
        h, memory, e_k, e_v = inner.block(layer, h, inj, memory, e_k, e_v, None)
        logits.append(inner.w_cls(h))
    return torch.stack(logits)


def accuracy(logits, y):
    pred = logits.argmax(-1) - 1
    cell = (pred == y).float().mean((-1, -2))
    exact = (pred == y).all(-1).float().mean(-1)
    return cell, exact


def train(opt):
    base, inner = build(opt.variant, opt.seed)
    no_decay = lambda n: n.endswith('bias') or any(k in n for k in ('stdp_alpha', 'clock_lam_raw', 'theta'))
    params = [(n, p) for n, p in base.named_parameters() if p.requires_grad]
    optim = torch.optim.AdamW([dict(params=[p for n, p in params if not no_decay(n)], weight_decay=0.01),
                               dict(params=[p for n, p in params if no_decay(n)], weight_decay=0.0)], lr=1e-3)
    gen = torch.Generator(device=DEVICE).manual_seed(1000 + opt.seed)
    xt, yt = sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    name = f'rule110_k{K_STEPS}_{opt.variant}_seed{opt.seed}'
    log = []
    began = time.monotonic()
    for step in range(1, opt.steps + 1):
        x, y = sample(512, gen)
        logits = forward(inner, x)
        loss = F.cross_entropy(logits.reshape(-1, 3), (y + 1).repeat(BLOCKS, 1).reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        optim.step()
        if step % 500 == 0 or step == opt.steps:
            with torch.no_grad():
                cell, exact = accuracy(forward(inner, xt), yt)
            layer = inner.layers[0]
            row = dict(step=step, loss=float(loss), seconds=time.monotonic() - began,
                       test_cell=[round(float(v), 4) for v in cell], test_exact=[round(float(v), 4) for v in exact],
                       alpha=float(layer.stdp_alpha), lam=float(torch.exp(-F.softplus(layer.clock_lam_raw)).mean()))
            log.append(row)
            print(json.dumps(row), flush=True)
    (OUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=base.state_dict(), variant=opt.variant, seed=opt.seed, steps=opt.steps),
               OUT / 'checkpoints' / f'{name}.pt')
    (OUT / 'results').mkdir(parents=True, exist_ok=True)
    (OUT / 'results' / f'{name}_train.json').write_text(json.dumps(log, indent=1) + '\n')


def probe(opt):
    name = f'rule110_k{K_STEPS}_{opt.variant}_seed{opt.seed}'
    ck = torch.load(OUT / 'checkpoints' / f'{name}.pt', map_location=DEVICE)
    base, inner = build(opt.variant, opt.seed)
    base.load_state_dict(ck['state'])
    base.eval()
    layer = inner.layers[0]
    xt, yt = sample(4096, torch.Generator(device=DEVICE).manual_seed(7))
    with torch.no_grad():
        logits = forward(inner, xt)
    pred = logits.argmax(-1) - 1
    unsolved = ~(pred[0] == yt).all(-1)
    xs, ys = xt[unsolved], yt[unsolved]
    print(f'test rows 4096: exact after block 1 {float((~unsolved).float().mean()):.4f}, after block {BLOCKS} '
          f'{float((pred[-1] == yt).all(-1).float().mean()):.4f}; unsolved after block 1: {int(unsolved.sum())}')

    original_clock, original_step = rp.advance_clock, rp.phase_clock_step
    state = dict(mode='none', block=0, frozen=None, perm=None)

    def clock(*a, **kw):
        pk, pv, lk, lv, st = original_clock(*a, **kw)
        state['block'] += 1
        mode = state['mode']
        if mode == 'flip':
            pk, pv = -pk, -pv
        elif mode == 'shuffle':
            idx = state['perm'][:, None, :, None].expand_as(pk)
            pk, pv = pk.gather(2, idx), pv.gather(2, idx)
        elif mode == 'freeze':
            if state['block'] == 2:
                state['frozen'] = (pk.clone(), pv.clone())
            elif state['block'] > 2:
                pk, pv = state['frozen']
        return pk, pv, lk, lv, st

    stats = []

    def step_hook(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces):
        hebb, im_s, new = original_step(k_unit, kr, v, rope, phase_k, phase_v, lam_k, lam_v, traces)
        if state['mode'] == 'none':
            alpha = layer.stdp_alpha.detach()
            factor = 1 + alpha * (1 - lam_k[..., None, :]) * torch.sin(phase_v[..., :, None] - phase_k[..., None, :])
            stats.append(dict(ltp=float((factor > 1 + 1e-6).float().mean()), ltd=float((factor < 1 - 1e-6).float().mean()),
                              q=torch.quantile(factor.flatten()[:1_000_000], torch.tensor([0.05, 0.5, 0.95], device=factor.device)).tolist(),
                              hebb=hebb, timing=alpha * im_s))
        return hebb, im_s, new
    rp.advance_clock, rp.phase_clock_step = clock, step_hook
    gen = torch.Generator(device=DEVICE).manual_seed(11)
    results = {}
    try:
        for mode in ('none', 'flip', 'shuffle', 'freeze', 'alpha0'):
            state.update(mode=mode, block=0, frozen=None,
                         perm=torch.argsort(torch.rand(xs.shape[0], T, device=DEVICE, generator=gen), -1))
            saved_alpha = layer.stdp_alpha.data.clone()
            if mode == 'alpha0':
                layer.stdp_alpha.data.zero_()
            with torch.no_grad():
                lg = forward(inner, xs)
            layer.stdp_alpha.data.copy_(saved_alpha)
            cell, exact = accuracy(lg, ys)
            results[mode] = dict(final_cell=float(cell[-1]), final_exact=float(exact[-1]),
                                 per_block_exact=[round(float(v), 4) for v in exact])
            print(f'{mode:8s} final exact {float(exact[-1]):.4f} cell {float(cell[-1]):.4f}   per block exact '
                  + ' '.join(f'{float(v):.3f}' for v in exact))
    finally:
        rp.advance_clock, rp.phase_clock_step = original_clock, original_step

    # G-level size of the timing term relative to the Hebbian term, per block, on the unsolved rows
    read_ratio = []
    with torch.no_grad():
        for b, s in enumerate(stats, 1):
            read_ratio.append(float(s['timing'].norm() / s['hebb'].norm()))
    print('block | LTP share LTD share | factor 5/50/95% | ||alpha Im S||/||Hebb|| (G level)')
    for b, s in enumerate(stats, 1):
        print(f"{b:5d} | {s['ltp']:.3f} {s['ltd']:.3f} | {s['q'][0]:.3f} {s['q'][1]:.3f} {s['q'][2]:.3f} | {read_ratio[b - 1]:.3f}")
    (OUT / 'results' / f'{name}_probe.json').write_text(json.dumps(dict(
        n_test=4096, n_unsolved_block1=int(unsolved.sum()), alpha=float(layer.stdp_alpha),
        lam=float(torch.exp(-F.softplus(layer.clock_lam_raw)).mean()), interventions=results,
        per_block=[dict(block=b, ltp=s['ltp'], ltd=s['ltd'], factor_q=s['q'], g_ratio=read_ratio[b - 1])
                   for b, s in enumerate(stats, 1)]), indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('train', 'probe'))
    ap.add_argument('--variant', choices=('run3', 'bonly'), default='run3')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    opt = ap.parse_args()
    torch.set_num_threads(4)
    (train if opt.cmd == 'train' else probe)(opt)


if __name__ == '__main__':
    main()
