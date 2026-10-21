"""Which path produces the answers of a cmap checkpoint (Mini-ARC test pairs, version 0)?

(a) attribution: for every query-output cell, the colour-source term (rule read, same-position copy, background, cell-frame
colours) that contributes most to the correct colour's logit; (b) ablations: drop one fast term of the rule synapse
(A, Pi, Gam, Bh; in every read, so also in the write) or the whole rule read, or one output source, or skip
the write (synapse frozen at its learned initial value), and count exact answers.

    python docs/evidence/2026-10-21/interp_paths.py runs/cmap_noemb_sf_s0/last.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lt.cmap import Cmap, CmapConfig, NC  # noqa: E402
from lt.cmap_data import CmapEpisodes  # noqa: E402

ck = torch.load(sys.argv[1], map_location='cuda', weights_only=False)
data = CmapEpisodes(ROOT / 'data/miniarc_pairs.npz', max_demos=ck['meta']['cfg']['D'])
m = Cmap(CmapConfig(**ck['meta']['cfg'])).cuda()
m.load_state_dict(ck['ema'] if ck.get('ema') is not None else ck['model'])
m.eval()
items = [(t, p) for t in range(data.n_tasks) for p in data.test_pairs[t]]
Kc = m.cfg.Kc
SRC = ['rule read', 'same-position copy', 'background'] + [f'cell-frame {k} colour' for k in range(Kc)]

state = dict(drop_term=None, drop_read=False, drop_src=None, no_write=False)
orig_read = m.read
orig_prox = m.prox_step


def prox_step(st, *a):
    return st if state['no_write'] else orig_prox(st, *a)


m.prox_step = prox_step


def read(st, M, Uo, kap):
    if state['drop_read']:
        return orig_read(st, M, Uo, kap) * 0                        # stays in the graph (the write differentiates it)
    if state['drop_term'] is not None:
        st = dict(st)
        st[state['drop_term']] = st[state['drop_term']] * 0          # keeps the block in the graph (the write differentiates it)
    out = orig_read(st, M, Uo, kap)
    cap['chat'] = out
    return out


m.read = read
cap = {}


def src_hook(mod, inp, out):
    if state['drop_src'] is not None:
        out = out.clone()
        out[..., state['drop_src']] = 0
    cap['w'] = out
    return out


m.src_w.register_forward_hook(src_hook)
orig_frames = m.frames


def frames(*a):
    out = orig_frames(*a)
    cap['kap'] = out[1]
    return out


m.frames = frames


def run(attribute=False):
    solved, wins = 0, np.zeros(len(SRC))
    with torch.no_grad():
        for i in range(0, len(items), 32):
            chunk = items[i:i + 32]
            b = {k: v.cuda() for k, v in data.collate([data.episode(t, data.train_pairs[t], p, 0)
                                                       for t, p in chunk]).items()}
            lg, _, _, _ = m(b, train=False)
            lab = b['label']
            ok = ((lg.argmax(-1) == lab) | (lab < 0)).flatten(1).all(1)
            solved += int(ok.sum())
            if attribute:
                B = lab.shape[0]
                w = cap['w'][:, -1]                                            # [B,O,3+Kc]
                xpair = torch.nn.functional.one_hot(b['color'][:, m.in_grids[-1]].reshape(B, -1).clamp_min(0), NC).float()
                xpair = xpair * (b['color'][:, m.in_grids[-1]].reshape(B, -1, 1) >= 0)
                bg = torch.zeros_like(xpair)
                bg[..., 0] = 1
                src = torch.stack([cap['chat'][:, -1], xpair, bg], 2)
                src = torch.cat([src, cap['kap'][:, m.out_grids[-1], :, 1:1 + Kc]], 2)   # [B,O,S,10]
                contrib = w[..., None] * src * m.temp                          # [B,O,S,10]
                lab_f = lab.reshape(B, -1)
                cell = lab_f >= 0
                c_lab = torch.gather(contrib, 3, lab_f.clamp_min(0)[:, :, None, None].expand(-1, -1, src.shape[2], 1))[..., 0]
                top = c_lab.argmax(-1)                                         # [B,O] main source of the correct logit
                for j in range(B):
                    if ok[j]:
                        wins += np.bincount(top[j][cell[j]].cpu().numpy(), minlength=len(SRC))
    return solved, wins


base, wins = run(attribute=True)
print(f'baseline exact {base}/149')
print('\n(a) main source of the correct colour logit, query-output cells of the solved test pairs:')
for nm, v in zip(SRC, wins):
    print(f'  {nm:22s} {v / wins.sum():.2f}')
print('\n(b) ablations (exact / 149):')
for term in ('A', 'Pi', 'Gam', 'Bh'):
    state['drop_term'] = term
    print(f'  rule term {term:3s} = 0 in every read (write and query): {run()[0]}')
state['drop_term'] = None
state['drop_read'] = True
print(f'  whole rule read = 0 (no information from the demonstrations reaches the query): {run()[0]}')
state['drop_read'] = False
state['no_write'] = True
print(f'  no write (synapse stays at its learned initial value; reads it, demonstrations never written): {run()[0]}')
state['no_write'] = False
for s, nm in enumerate(SRC):
    state['drop_src'] = s
    print(f'  output source "{nm}" off: {run()[0]}')
