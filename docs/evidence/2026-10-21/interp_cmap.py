"""Interpretability of a cmap checkpoint on the Mini-ARC test pairs (version 0): (1) does the inferred rule synapse hold the
true rule (colour map Pi on pure recolouring tasks; the spatial read map on D4 tasks), (2) what do the learned frames pick
(grid masks vs simple cell features; where cell-frame anchors land), (3) does a longer recurrence at inference
(extrapolation) solve more.

    python docs/evidence/2026-10-21/interp_cmap.py runs/cmap_noemb_sf_s0/last.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lt.cmap import Cmap, CmapConfig, NC  # noqa: E402
from lt.cmap_data import CmapEpisodes, op_apply  # noqa: E402

ck = torch.load(sys.argv[1], map_location='cuda', weights_only=False)
data = CmapEpisodes(ROOT / 'data/miniarc_pairs.npz', max_demos=ck['meta']['cfg']['D'])
m = Cmap(CmapConfig(**ck['meta']['cfg'])).cuda()
m.load_state_dict(ck['ema'] if ck.get('ema') is not None else ck['model'])
m.eval()
names = np.load(ROOT / 'data/miniarc_pairs.npz')['task_names']
items = [(t, p) for t in range(data.n_tasks) for p in data.test_pairs[t]]
N = data.N


def batches(bs=32):
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        yield chunk, {k: v.cuda() for k, v in data.collate([data.episode(t, data.train_pairs[t], p, 0)
                                                             for t, p in chunk]).items()}


def solved(steps=None):
    out = {}
    with torch.no_grad():
        for chunk, b in batches():
            lg, _, _, _ = m(b, train=False, steps=steps)
            ok = ((lg.argmax(-1) == b['label']) | (b['label'] < 0)).flatten(1).all(1)
            for j, (t, _) in enumerate(chunk):
                out[t] = bool(ok[j])
    return out


# ---------------------------------------------------------------- (3) extrapolation
print('== (3) recurrence extrapolation (trained with', m.cfg.steps, 'steps)')
base = solved()
for s in (8, 12, 16, 20, 24, 32, 48, 64):
    r = solved(s)
    gain = [names[t] for t in r if r[t] and not base[t]]
    loss = [names[t] for t in r if base[t] and not r[t]]
    print(f'  steps {s:3d}: exact {sum(r.values()):3d}/149  (+{len(gain)} -{len(loss)} vs 16)'
          + (f'  gained: {", ".join(g[:40] for g in gain[:4])}' if gain else ''))

# ---------------------------------------------------------------- capture the last step's internals
cap = {}
orig_frames, orig_read = m.frames, m.read


def frames_hook(h, cv, live, color1h):
    cap['frames_in'] = (h, cv, live, color1h)
    out = orig_frames(h, cv, live, color1h)
    cap['U'] = out[0]
    return out


def read_hook(st, M, Uo, kap):
    cap['read'] = (st, M, Uo, kap)
    return orig_read(st, M, Uo, kap)


m.frames, m.read = frames_hook, read_hook


def task_kind(t):
    """('d4', k) if every pair satisfies out = op_k(in); ('recolor', f) if out = f(in) cellwise; else None."""
    pairs = [data.grids(p) for p in list(data.train_pairs[t]) + list(data.test_pairs[t])]
    for k in range(1, 8):
        if all(op_apply(a, k).shape == b.shape and (op_apply(a, k) == b).all() for a, b in pairs):
            return ('d4', k)
    f = {}
    for a, b in pairs:
        if a.shape != b.shape:
            return None
        for x, y in zip(a.ravel(), b.ravel()):
            if f.setdefault(int(x), int(y)) != int(y):
                return None
    return ('recolor', f) if any(x != y for x, y in f.items()) else None


kinds = {t: task_kind(t) for t, _ in items}
P = data.D + 1
pi_rows, read_rows = [], []
frame_stats = []
with torch.no_grad():
    for chunk, b in batches():
        lg, _, st, _ = m(b, train=False)
        st_, M, Uo, kap = cap['read']
        U = cap['U']                                           # [B,G,O,F,C2]
        ok = ((lg.argmax(-1) == b['label']) | (b['label'] < 0)).flatten(1).all(1)
        om = m.omega / (m.omega.abs().sum((1, 2), keepdim=True) + 1e-6)
        for j, (t, p) in enumerate(chunk):
            kind = kinds[t]
            a, y = data.grids(p)
            if kind and kind[0] == 'recolor':
                Pi = st_['Pi'][j].cpu().numpy()
                changed = [(x, z) for x, z in kind[1].items() if x != z and x in set(a.ravel().tolist())]
                hit = sum(int(np.argmax(Pi[:, x]) == z) for x, z in changed)
                pi_rows.append((names[t], hit, len(changed), bool(ok[j])))
            if kind and kind[0] == 'd4':
                # read map w(o, s) = (1/n) sum_{j,k,q} om[j,k,q] U_k(s)^T A_j U_q(o) on the query pair
                Ui = U[j, m.in_grids[-1]]                      # [O,F,C2] query input cells
                Uq = Uo[j, -1]                                 # [O,F,C2] query output cells
                A = st_['A'][j]                                # [J,C2,C2]
                AU = torch.einsum('jde,oqe->joqd', A, Uq)
                w = torch.einsum('jkq,skd,joqd->os', om, Ui, AU)          # [O_out, O_in]
                live_in = (b['color'][j, m.in_grids[-1]].reshape(-1) >= 0)
                w = w.masked_fill(~live_in[None], float('-inf'))
                src = op_apply(np.arange(a.size).reshape(a.shape), kind[1])   # out cell -> index into a
                hits, tot = 0, 0
                for r in range(src.shape[0]):
                    for c in range(src.shape[1]):
                        sr, sc = divmod(int(src[r, c]), a.shape[1])
                        hits += int(int(w[r * N + c].argmax()) == sr * N + sc)
                        tot += 1
                read_rows.append((names[t], kind[1], hits, tot, bool(ok[j])))
        # (2) frames on this batch: grid masks and cell-frame anchors from the last step's hidden state
        h, cv, live, col1h = cap['frames_in']
        _, grid = m.masks(h, cv, live, cells=False)            # [B,G,N,N,Kg]
        colk = col1h.reshape(h.shape[0], m.G, m.O, NC)[:, m.key_grid]
        cmean, _ = m.cell_frames_sdpa(h, cv, live, colk)      # [B,G,O,Kc,2]
        frame_stats.append((b['color'].cpu(), grid.cpu(), cmean.cpu()))

print('\n== (1a) colour map Pi on pure recolouring tasks (argmax Pi[:, a] == f(a) for every changed colour a)')
for nm, hit, n, okk in pi_rows:
    print(f'  {"SOLVED" if okk else "      "} {hit}/{n}  {nm[:70]}')
print('\n== (1b) spatial read map on D4 tasks (argmax_s w(o, s) == true source of o)')
for nm, k, hit, tot, okk in read_rows:
    print(f'  {"SOLVED" if okk else "      "} op {k}: {hit}/{tot} cells  {nm[:60]}')

# ---------------------------------------------------------------- (2) frames
print('\n== (2) learned frames (input grids of all test episodes)')
cols = torch.cat([f[0] for f in frame_stats])                 # [E,G,N,N]
grid = torch.cat([f[1] for f in frame_stats])                 # [E,G,N,N,Kg]
cmean = torch.cat([f[2] for f in frame_stats])                # [E,G,O,Kc,2]
ins = m.in_grids
feat = {}
c_in = cols[:, ins]                                            # [E,P,N,N]
live = c_in >= 0
cnt = torch.stack([(c_in == v).flatten(2).sum(-1) for v in range(NC)], -1)   # [E,P,10]
nb = cnt.clone()
nb[..., 0] = -1
top = nb.argmax(-1)                                            # most frequent non-background colour
feat['background'] = (c_in == 0)
feat['non-background'] = (c_in > 0)
feat['most frequent colour'] = (c_in == top[..., None, None]) & (c_in > 0)
sing = torch.gather(cnt, 2, c_in.clamp_min(0).flatten(2)).view_as(c_in) == 1
feat['colour appears once'] = sing & (c_in > 0)
pad = torch.nn.functional.pad(c_in.float(), (1, 1, 1, 1), value=-1)
diff = torch.zeros_like(live)
for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
    diff |= (pad[..., 1 + dr:1 + dr + N, 1 + dc:1 + dc + N] != c_in.float())
feat['object edge (different neighbour)'] = diff & (c_in > 0)
gm = grid[:, ins]                                              # [E,P,N,N,Kg]
for k in range(gm.shape[-1]):
    x = gm[..., k][live].float()
    best = []
    for nm, fz in feat.items():
        y = fz[live].float()
        r = torch.corrcoef(torch.stack([x, y]))[0, 1].item() if y.std() > 0 and x.std() > 0 else 0.0
        best.append((r, nm))
    best.sort(reverse=True)
    print(f'  grid mask {k}: mean {x.mean():.2f}; corr ' + ', '.join(f'{nm} {r:+.2f}' for r, nm in best[:3]))
pos = m.pos.cpu()
cm = cmean[:, ins]                                             # [E,P,O,Kc,2]
lv = live.flatten(2)                                           # [E,P,O]
cflat = c_in.flatten(2)
for k in range(cm.shape[-2]):
    a = cm[..., k, :][lv]                                      # [n,2]
    s_pos = pos[None, None].expand(cm.shape[0], cm.shape[1], -1, -1)[lv]
    d = (a - s_pos).norm(dim=-1)
    nearest = torch.cdist(a, pos).argmin(-1)                   # nearest cell to the anchor (within the same grid)
    e_idx, p_idx, _ = torch.nonzero(lv, as_tuple=True)
    tgt_col = cflat[e_idx, p_idx, nearest]
    src_col = cflat[lv]
    print(f'  cell frame {k}: mean |anchor - cell| {d.mean():.2f}; anchor on self {(d < 0.5).float().mean():.2f}; '
          f'anchor on non-background {(tgt_col > 0).float().mean():.2f}; anchor on same colour as the cell '
          f'{(tgt_col == src_col).float().mean():.2f}')
