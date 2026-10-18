"""Per-puzzle end state after SEGS segments: frozen vs moving prediction, cycles, hidden-state drift, Sudoku validity.
Usage: python -I classify.py CKPT SEGS N OUT_JSON"""
import json, sys, time, torch
sys.path.insert(0, '/workspace/something')
from lt.analysis.common import load_model
from lt.data import load_data, load_task_ids, eval_batches
ck, segs, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
W = 200                                                           # tail window (segments)
torch.set_float32_matmul_precision('highest')
dev = torch.device('cuda')
model, cfg = load_model(ck, dev, weights='ema', kernel='auto', amp=True)
_, _, x, y, path, _ = load_data(cfg.get('data_npz'))
_, ids = load_task_ids(path)
x, y = x[:n], y[:n]; ids = None if ids is None else ids[:n]
model.eval(); model.config.loops = segs + 1
b = {k: v.to(dev) for k, v in next(eval_batches(x, y, n, 0, 1, ids)).items()}
lab = b['labels']
weights = torch.randint(1, 2 ** 61, (lab.shape[1],), device=dev, generator=torch.Generator(dev).manual_seed(0))
solved_hist = torch.zeros(segs, n, dtype=torch.bool, device=dev)
hashes, dh = [], []
carry, prev_h = model.initial_carry(b), None
t0 = time.monotonic()
with torch.no_grad():
    for s in range(segs):
        carry, logits = model(carry, b)
        pred = logits.argmax(-1)
        solved_hist[s] = (pred == lab).all(-1)
        if s >= segs - W:
            hashes.append((pred.long() * weights).sum(-1))
            h = carry.hidden.double()
            dh.append(((h - prev_h).norm(dim=(1, 2)) / h.norm(dim=(1, 2))) if prev_h is not None else torch.zeros(n, device=dev, dtype=torch.double))
        if s >= segs - W - 1:
            prev_h = carry.hidden.double().clone()
        if (s + 1) % 100 == 0:
            print(f'[{time.monotonic() - t0:5.0f}s] seg {s + 1}: solved {int(solved_hist[s].sum())}', flush=True)
H = torch.stack(hashes).T.cpu()                                   # [n, W]
DH = torch.stack(dh).T.cpu()
final = pred.cpu(); lab_c = lab.cpu(); inp = b['inputs'].cpu()
sh = solved_hist.cpu()
g = 9
def violations(p):
    d = (p - 1).view(g, g)                                        # digits 1..9 (token - 1)
    v = 0
    for u in list(d) + list(d.T) + [d[r:r + 3, c:c + 3].reshape(-1) for r in (0, 3, 6) for c in (0, 3, 6)]:
        v += len(u) - len(set(u.tolist()))
    return v
rows = []
for i in range(n):
    seq = H[i].tolist()
    changes = sum(a != b_ for a, b_ in zip(seq, seq[1:]))
    distinct = len(set(seq))
    period = 0
    if changes:
        for p in range(1, W // 2):
            if all(seq[t] == seq[t - p] for t in range(p, W)):
                period = p; break
    clue = inp[i] > 1
    rows.append(dict(i=i, solved_final=bool(sh[-1, i]), first_solved=int(sh[:, i].float().argmax()) + 1 if sh[:, i].any() else -1,
                     ever_solved=bool(sh[:, i].any()), lost_after_solve=bool(sh[:, i].any() and not sh[-1, i]),
                     changes_tail=changes, distinct_tail=distinct, period=period,
                     dh_last=float(DH[i, -1]), dh_tail_mean=float(DH[i, 1:].mean()),
                     wrong_cells=int((final[i] != lab_c[i]).sum()), violations=violations(final[i]),
                     clue_broken=int((final[i][clue] != inp[i][clue]).sum())))
json.dump(dict(ckpt=ck, segs=segs, n=n, tail=W, puzzles=rows), open(out, 'w'))
def summ(name, rs):
    if not rs: print(f'{name}: 0'); return
    fz = [r for r in rs if r['changes_tail'] == 0]
    cyc = [r for r in rs if r['changes_tail'] and r['period']]
    wan = [r for r in rs if r['changes_tail'] and not r['period']]
    med = lambda v: sorted(v)[len(v) // 2]
    print(f'{name}: {len(rs)} | prediction frozen over last {W} segs {len(fz)}, periodic {len(cyc)} '
          f'(periods {sorted(set(r["period"] for r in cyc))[:10]}), wandering {len(wan)}', flush=True)
    for nm, grp in (('frozen', fz), ('periodic', cyc), ('wandering', wan)):
        if grp:
            print(f'   {nm:9s}: median |dh|/|h| last {med([r["dh_last"] for r in grp]):.2e}, median wrong cells '
                  f'{med([r["wrong_cells"] for r in grp])}, median violations {med([r["violations"] for r in grp])}, '
                  f'median distinct preds {med([r["distinct_tail"] for r in grp])}, clue broken in {sum(r["clue_broken"] > 0 for r in grp)}', flush=True)
summ('SOLVED at the end', [r for r in rows if r['solved_final']])
summ('UNSOLVED at the end', [r for r in rows if not r['solved_final']])
print(f"solved at some segment but not at the end: {sum(r['lost_after_solve'] for r in rows)}", flush=True)
