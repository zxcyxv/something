"""Is the STDP timing of the general-window pairangle model token-specific or channel-wide?

Within one block every token n gives each complex channel B a phase phi_nB = arg z_nB.
If a channel's phase is nearly the same for all 81 tokens (a channel-preferred phase),
the window W(phiV_A - phiK_B) is nearly the same for every token and the STDP write
reduces to a channel-pair mask on the Hebbian write, G_S ~ G_H * Wbar. Measured at the
last block of selected segments, split by solved / unsolved puzzles:

  rho        phase concentration of a channel over the tokens, |sum_n z_n| / sum_n |z_n|
             (1: same phase in every cell); reference: the same magnitudes with random phases
  across     consistency of a channel's preferred phase across puzzles
  common     energy fraction of the token mean in the hidden state, |mean_n h|^2 / mean_n |h|^2
  residual   |G_S - G_H * Wbar| / |G_S| for the token-mean window and for the least-squares
             best channel-pair mask, and the share of the total read carried by that residual

python -m lt.probe_stdp_token_timing --run runs/<run> --checkpoint runs/<run>/step_20000.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import train as t
from .analyze_stdp_direct import make_model, predict

SEGMENTS = (1, 2, 4, 8, 16)
CHUNK = 32
SUMS = ('rhoK_num', 'rhoK_den', 'rhoKr_num', 'rhoV_num', 'rhoV_den', 'rhoK_rand_num', 'rhoV_rand_num',
        'GS_sq', 'res_mean_sq', 'res_ls_sq', 'read_sq', 'read_S_sq', 'read_res_mean_sq', 'read_res_ls_sq')


class TimingProbe:
    def __init__(self, inner, device):
        self.inner, self.device = inner, device
        H, P = inner.H, inner.dh // 2
        f64 = dict(dtype=torch.float64, device=device)
        self.sums = {n: torch.zeros(2, len(SEGMENTS), H, **f64) for n in SUMS}
        self.common = torch.zeros(2, len(SEGMENTS), 2, **f64)          # sum of ratios, count
        self.pref = {r: torch.zeros(2, len(SEGMENTS), H, P, dtype=torch.complex128, device=device) for r in 'KV'}
        self.pref_w = {r: torch.zeros(2, len(SEGMENTS), H, P, **f64) for r in 'KV'}
        self.wpairs = list(zip(inner.window_frequencies.tolist(), inner.window_coefficients.tolist()))
        self.gen = torch.Generator(device=device).manual_seed(0)

    def install(self):
        inner = self.inner
        self.orig_block, self.orig_memory = inner.block, inner.memory_step

        def block(L, h, inj, *args):
            self.h_pre = h + inner.embed_scale * inj
            return self.orig_block(L, h, inj, *args)

        def memory_step(L, q, k, v, *args, **kwargs):
            out = self.orig_memory(L, q, k, v, *args, **kwargs)
            if self.k == inner.config.blocks_per_seg - 1 and self.seg in SEGMENTS:
                with torch.autocast('cuda', enabled=False):
                    self.measure(L, q.float(), k.float(), v.float(), SEGMENTS.index(self.seg))
            self.k += 1
            return out
        inner.block, inner.memory_step = block, memory_step

    def remove(self):
        del self.inner.block, self.inner.memory_step

    def window(self, delta):
        return sum(c * torch.sin(f * delta) for f, c in self.wpairs)

    def measure(self, L, q, k, v, j):
        h = self.h_pre.float()
        ratio = h.mean(1).pow(2).sum(-1) / h.pow(2).sum(-1).mean(1)
        self.common[:, j, 0].index_add_(0, self.groups, ratio.double())
        self.common[:, j, 1].index_add_(0, self.groups, torch.ones_like(ratio, dtype=torch.float64))
        for s0 in range(0, q.shape[0], CHUNK):
            sl = slice(s0, s0 + CHUNK)
            self.chunk(L, q[sl], k[sl], v[sl], self.groups[sl], j)

    def chunk(self, L, q, k, v, g, j):
        inner = self.inner
        b, H, T, D = v.shape
        P = D // 2
        eps = inner.config.eps
        qU = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + eps)
        kU = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + eps)
        tables = inner.rope_tables(L)
        qr, kr = inner.apply_rope(qU, L, tables), inner.apply_rope(kU, L, tables)
        cplx = lambda x: torch.view_as_complex(x.reshape(b, H, T, P, 2).contiguous())
        zK, zKr, zV = cplx(kU), cplx(kr), cplx(v)
        add = lambda name, value: self.sums[name][:, j].index_add_(0, g, value.double())
        for name, z in (('K', zK), ('Kr', zKr), ('V', zV)):
            add(f'rho{name}_num', z.sum(2).abs().sum(-1))
            if name != 'Kr':
                add(f'rho{name}_den', z.abs().sum((2, 3)))
                phase = torch.rand(z.shape, generator=self.gen, device=z.device) * (2 * torch.pi)
                add(f'rho{name}_rand_num', (z.abs() * torch.exp(1j * phase)).sum(2).abs().sum(-1))
                self.pref[name][:, j].index_add_(0, g, z.sum(2).to(torch.complex128))
                self.pref_w[name][:, j].index_add_(0, g, z.sum(2).abs().double())
        # STDP write against the best token-independent channel-pair mask on the Hebbian write
        vp, krp = v.reshape(b, H, T, P, 2), kr.reshape(b, H, T, P, 2)
        delta = torch.angle(zV)[..., :, None] - torch.angle(zK)[..., None, :]
        W = self.window(delta)
        GH = torch.einsum('bhna,bhnc->bhac', v, kr)
        GS = torch.einsum('bhnAi,bhnAB,bhnBj->bhAiBj', vp, W, krp).reshape(b, H, D, D)
        GHb, GSb = GH.reshape(b, H, P, 2, P, 2), GS.reshape(b, H, P, 2, P, 2)
        mask = lambda M: (GHb * M[:, :, :, None, :, None]).reshape(b, H, D, D)
        res_mean = GS - mask(W.mean(2))
        res_ls = GS - mask((GSb * GHb).sum((3, 5)) / GHb.pow(2).sum((3, 5)).clamp_min(1e-30))
        alpha = L.stdp_alpha.detach().float().view(1, H, 1, 1)
        rd = lambda G: (qr @ G.transpose(-1, -2)).pow(2).sum((2, 3))
        add('GS_sq', GS.pow(2).sum((2, 3)))
        add('res_mean_sq', res_mean.pow(2).sum((2, 3)))
        add('res_ls_sq', res_ls.pow(2).sum((2, 3)))
        add('read_sq', rd(GH + alpha * GS))
        add('read_S_sq', rd(alpha * GS))
        add('read_res_mean_sq', rd(alpha * res_mean))
        add('read_res_ls_sq', rd(alpha * res_ls))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    opt = ap.parse_args()
    run = opt.run.resolve(strict=True)
    ck = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    protocol = json.loads((run / 'protocol.json').read_text())
    cfg = dict(ck['cfg'])
    if not Path(cfg['data_npz']).is_file():
        cfg['data_npz'] = str(Path(__file__).resolve().parents[1] / 'data/sudoku_lt_1k.npz')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(protocol['precision'])
    device = torch.device('cuda')
    t._resolve_precision(cfg, device)
    _, _, x, y, _, fingerprint = t.load_data(cfg)
    assert fingerprint == cfg['data_fingerprint']
    base = make_model(protocol, cfg, device, ck['model_state_dict'])
    preds, official = predict(base, x, y, cfg, device)
    labels = torch.from_numpy(y.reshape(-1, 81).astype(np.int64) + 1)
    groups = (~(preds[-1].long() == labels).all(-1)).long()
    probe = TimingProbe(base.model.inner, device)
    probe.install()
    try:
        start = 0
        for batch in t.eval_batches(x, y, cfg['global_batch_size'], 0, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            n = batch['inputs'].shape[0]
            probe.groups = groups[start:start + n].to(device)
            carry = base.initial_carry(batch)
            for s in range(cfg['loops']):
                probe.seg, probe.k = s + 1, 0
                carry, *_ = base(carry=carry, batch=batch, return_keys=set())
            start += n
    finally:
        probe.remove()
    S = {k: v.sum(-1) for k, v in probe.sums.items()}                 # sum over heads
    ratio = lambda a, b: (S[a] / S[b]).tolist()
    root = lambda a, b: (S[a] / S[b]).sqrt().tolist()
    across = {r: ((probe.pref[r].abs().sum(-1) / probe.pref_w[r].sum(-1))).tolist() for r in 'KV'}
    result = dict(checkpoint=str(opt.checkpoint), segments=SEGMENTS, groups=('solved', 'unsolved'),
                  solved=int((groups == 0).sum()), official_exact=official['exact'],
                  rho=dict(K=ratio('rhoK_num', 'rhoK_den'), K_rotated=ratio('rhoKr_num', 'rhoK_den'),
                           V=ratio('rhoV_num', 'rhoV_den'), K_random_phase=ratio('rhoK_rand_num', 'rhoK_den'),
                           V_random_phase=ratio('rhoV_rand_num', 'rhoV_den')),
                  rho_per_head_K=(probe.sums['rhoK_num'] / probe.sums['rhoK_den']).tolist(),
                  rho_per_head_V=(probe.sums['rhoV_num'] / probe.sums['rhoV_den']).tolist(),
                  across_puzzles=across,
                  hidden_common_mode=(probe.common[..., 0] / probe.common[..., 1]).tolist(),
                  residual_GS=dict(token_mean_mask=root('res_mean_sq', 'GS_sq'), best_mask=root('res_ls_sq', 'GS_sq')),
                  read_share=dict(stdp=root('read_S_sq', 'read_sq'), residual_token_mean_mask=root('read_res_mean_sq', 'read_sq'),
                                  residual_best_mask=root('read_res_ls_sq', 'read_sq')))
    out = opt.out or run / 'diagnostics' / f"stdp_token_timing_step{int(ck['step'])}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
