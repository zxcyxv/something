"""One solved puzzle, block by block: predictions, logical structure and LTP/LTD.

Logic level of a blank (from the puzzle alone): 1 = fixed by naked/hidden singles from the
clues, 2 = needs one pass of one-assumption contradiction elimination first, 3 = two passes,
and so on; 0 = clue; -1 = still open after that (search). `step` orders cells within the
propagation. Predictions are argmax w_cls(h) after every block (exactly the model output
at the last block of a segment). Per cell n and block, over all heads and channel pairs
(A, B) with co-firing weight m = |v_nA| |k_nB|: LTP = mass with sin(delta) > 0, the mean
gain 1 + alpha W(delta), and the reversed mass (gain < 0).

python -m lt.case_stdp_puzzle --run runs/<run> --checkpoint runs/<run>/step_20000.pt --solve-segment 6
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from . import train as t
from .analyze_stdp_direct import make_model, predict
from .sudoku_solver import PEERS, UNITS


def singles(g, cand):
    """Naked/hidden singles until stuck. Returns (rounds, contradiction)."""
    rounds = []
    while True:
        for U in UNITS:
            placed = [g[i] for i in U if g[i]]
            if len(placed) != len(set(placed)):
                return rounds, True
            for d in range(1, 10):
                if not any(g[i] == d or (g[i] == 0 and d in cand[i]) for i in U):
                    return rounds, True
        if any(g[i] == 0 and not cand[i] for i in range(81)):
            return rounds, True
        found = {}
        for i in range(81):
            if g[i] == 0 and len(cand[i]) == 1:
                found[i] = (next(iter(cand[i])), 'naked')
        for U in UNITS:
            for d in range(1, 10):
                spots = [i for i in U if g[i] == 0 and d in cand[i]]
                if len(spots) == 1 and spots[0] not in found:
                    found[spots[0]] = (d, 'hidden')
        if not found:
            return rounds, False
        for i, (d, _) in found.items():
            g[i], cand[i] = d, {d}
            for p in PEERS[i]:
                if g[p] == 0:
                    cand[p].discard(d)
        rounds.append([(i, d, m) for i, (d, m) in found.items()])


def logic_levels(grid):
    g = np.array(grid, dtype=int)
    cand = [set(range(1, 10)) if g[i] == 0 else {g[i]} for i in range(81)]
    for i in range(81):
        if g[i]:
            for p in PEERS[i]:
                cand[p].discard(g[i])
    level, step = np.where(g > 0, 0, -1), np.zeros(81, dtype=int)
    lvl, counter = 1, 0
    while True:
        rounds, bad = singles(g, cand)
        assert not bad, 'invalid puzzle'
        for r in rounds:
            counter += 1
            for i, _, _ in r:
                level[i], step[i] = lvl, counter
        if not (g == 0).any():
            break
        removed = 0
        for i in np.flatnonzero(g == 0):
            for d in sorted(cand[i]):
                g2, c2 = g.copy(), copy.deepcopy(cand)
                g2[i], c2[i] = d, {d}
                for p in PEERS[i]:
                    if g2[p] == 0:
                        c2[p].discard(d)
                if singles(g2, c2)[1]:
                    cand[i].discard(d)
                    removed += 1
        if not removed:
            break
        lvl += 1
    return level, step, g


class CaseProbe:
    def __init__(self, inner, idx):
        self.inner, self.idx = inner, idx
        self.wpairs = list(zip(inner.window_frequencies.tolist(), inner.window_coefficients.tolist()))
        self.rows = []

    def install(self):
        inner, i = self.inner, self.idx
        self.orig_block, self.orig_memory = inner.block, inner.memory_step

        def block(L, h, inj, *args):
            out = self.orig_block(L, h, inj, *args)
            with torch.autocast('cuda', enabled=False):
                self.rows[-1]['pred'] = inner.w_cls(out[0][i].float()).argmax(-1).cpu()
            return out

        def memory_step(L, q, k, v, *args, **kwargs):
            out = self.orig_memory(L, q, k, v, *args, **kwargs)
            with torch.autocast('cuda', enabled=False):
                self.rows.append(self.measure(L, q[i:i + 1].float(), k[i:i + 1].float(), v[i:i + 1].float()))
            return out
        inner.block, inner.memory_step = block, memory_step

    def remove(self):
        del self.inner.block, self.inner.memory_step

    def measure(self, L, q, k, v):
        inner = self.inner
        _, H, T, D = v.shape
        P = D // 2
        eps = inner.config.eps
        qU = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        kU = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        tables = inner.rope_tables(L)
        qr, kr = inner.apply_rope(qU, L, tables), inner.apply_rope(kU, L, tables)
        vp, kp, krp = (x.reshape(1, H, T, P, 2) for x in (v, kU, kr))
        vmag, kmag = vp.norm(dim=-1), kp.norm(dim=-1)
        delta = torch.atan2(vp[..., 1], vp[..., 0])[..., :, None] - torch.atan2(kp[..., 1], kp[..., 0])[..., None, :]
        W = sum(c * torch.sin(f * delta) for f, c in self.wpairs)
        alpha = L.stdp_alpha.detach().float()
        gain = 1 + alpha.view(1, H, 1, 1, 1) * W
        m = vmag[..., :, None] * kmag[..., None, :]
        ltp = torch.sin(delta) > 0
        cell = lambda x: ((m * x).sum((1, 3, 4)) / m.sum((1, 3, 4)))[0].cpu()
        GH = torch.einsum('bhna,bhnc->bhac', v, kr)
        GS = torch.einsum('bhnAi,bhnAB,bhnBj->bhAiBj', vp, W, krp).reshape(1, H, D, D)
        rH = (qr @ GH.transpose(-1, -2)).pow(2).sum((1, 3))
        rS = (alpha.view(1, H, 1, 1) * (qr @ GS.transpose(-1, -2))).pow(2).sum((1, 3))
        return dict(ltp=cell(ltp.float()), gain=cell(gain), rev=cell((gain < 0).float()),
                    m=m.sum((1, 3, 4))[0].cpu(), read_ratio=(rS / rH).sqrt()[0].cpu(),
                    head_ltp=((m * ltp).sum((2, 3, 4)) / m.sum((2, 3, 4)))[0].cpu())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--solve-segment', type=int, default=6, help='pick a puzzle first fully solved at this segment')
    ap.add_argument('--puzzle', type=int, default=None, help='held-out index; overrides --solve-segment')
    ap.add_argument('--out', type=Path, default=None)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'])
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    _, _, x, y, _, _ = t.load_data(cfg)
    base = make_model(protocol, cfg, device, ck['model_state_dict'])
    preds, _ = predict(base, x, y, cfg, device)
    labels = y.reshape(-1, 81).astype(np.int64) + 1
    ok = (preds.numpy().astype(np.int64) == labels[None]).all(-1)            # [S, N]
    stays = np.flip(np.logical_and.accumulate(np.flip(ok, 0), 0), 0)        # solved from s to the end
    first = np.where(stays.any(0), stays.argmax(0) + 1, 0)                  # 0: not solved at s16
    if opt.puzzle is None:
        pool = np.flatnonzero(first == opt.solve_segment)
        assert len(pool), f'no puzzle first solved at segment {opt.solve_segment}'
        idx = int(pool[0])
    else:
        idx = opt.puzzle
    gbs = cfg['global_batch_size']
    b0, i = idx // gbs * gbs, idx % gbs
    level, step, solution = logic_levels(x.reshape(-1, 81)[idx])
    assert (solution[level >= 0] == y.reshape(-1, 81)[idx][level >= 0]).all()
    probe = CaseProbe(base.model.inner, i)
    probe.install()
    seg_pred = []
    try:
        batch = {k: v.to(device) for k, v in next(t.eval_batches(x[b0:b0 + gbs], y[b0:b0 + gbs], gbs, 0, 1)).items()}
        with torch.no_grad():
            carry = base.initial_carry(batch)
            for s in range(cfg['loops']):
                carry, _, _, out, _ = base(carry=carry, batch=batch, return_keys={'preds'})
                seg_pred.append(out['preds'][i].cpu())
    finally:
        probe.remove()
    blocks = cfg['blocks_per_seg']
    rows = probe.rows
    for s, p in enumerate(seg_pred):                                       # logit readout = output at segment ends
        assert torch.equal(rows[(s + 1) * blocks - 1]['pred'], p), 'readout differs from the model output'
        assert torch.equal(p, preds[s, idx].long()), 'batch replay differs from the evaluation'
    result = dict(puzzle=idx, first_solved_segment=int(first[idx]), inputs=x.reshape(-1, 81)[idx].tolist(),
                  solution=y.reshape(-1, 81)[idx].tolist(), logic_level=level.tolist(), logic_step=step.tolist(),
                  alpha=base.model.inner.layers[0].stdp_alpha.tolist(),
                  blocks=[{k: (v.tolist() if torch.is_tensor(v) else v) for k, v in r.items()} for r in rows])
    out = opt.out or run / 'diagnostics' / f'case_puzzle{idx}_step{int(ck["step"])}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + '\n')
    print(f'puzzle {idx}: first solved at segment {first[idx]}; levels', np.bincount(level[level > 0]).tolist(),
          'open', int((level < 0).sum()), f'-> {out}')


if __name__ == '__main__':
    main()
