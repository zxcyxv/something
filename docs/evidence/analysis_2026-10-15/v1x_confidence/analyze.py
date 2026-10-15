"""Confidence calibration at the training horizon vs extrapolation, by model and by puzzle difficulty.

Inputs: R_*.pt (P argmax [128,N,81] int8, H digit entropy [128,N,81], y, given) and difficulty.pt.
Blank cells only. At seg16 (index 15): correct = argmax == y. H+ = mean entropy of correct blanks, H- = of wrong blanks,
conf = H < h0 (h0 = 0.2 nats ~ p_max >= 0.95). AUC = P(H_wrong > H_correct) pooled over the group's cells.
Groups among puzzles unsolved at seg16: late = solved at some segment 17..128, never = not.
"""
import sys, torch, math
import os; S = os.environ.get('CONF_DIR', os.path.dirname(os.path.abspath(__file__)))   # R_*.pt and difficulty.pt live here
H0 = 0.2
names = sys.argv[1:] if len(sys.argv) > 1 else ['v1', 'v11', 'v17', 'b40k']
diff = torch.load(f'{S}/difficulty.pt')
l1n = diff['l1_needed']
q = torch.quantile(l1n.float(), torch.tensor([1 / 3, 2 / 3]))
dbin = torch.bucketize(l1n.float(), q)                       # 0 easy, 1 medium, 2 hard (tertiles of L1-needed blanks)
BIN = ['easy', 'medium', 'hard']
print(f'difficulty: L1-needed blanks tertile edges {q.tolist()}, clues mean {diff["clues"].float().mean():.1f}, '
      f'L0-solvable {int((diff["l0_rounds"] >= 0).sum())}/512')


def auc(h_wrong, h_correct, n=4000):
    if len(h_wrong) == 0 or len(h_correct) == 0:
        return float('nan')
    g = torch.Generator().manual_seed(0)
    a = h_wrong[torch.randint(len(h_wrong), (n,), generator=g)]
    b = h_correct[torch.randint(len(h_correct), (n,), generator=g)]
    return float((a > b).float().mean() + 0.5 * (a == b).float().mean())


def stats(P, H, y, given, idx, seg=15):
    """Per-group means at segment index `seg` for puzzle indices idx."""
    blank = ~given[idx]
    pred, ent = P[seg, idx].long(), H[seg, idx].float()
    corr = (pred == y[idx]) & blank
    wrong = (pred != y[idx]) & blank
    hp = ent[corr]; hm = ent[wrong]
    # precision@k: among each puzzle's k highest-entropy blanks (k = its number of wrong blanks), the fraction that are wrong
    pk = []
    e_masked = torch.where(blank, ent, torch.full_like(ent, -1.0))
    for j in range(len(idx)):
        k = int(wrong[j].sum())
        if k == 0: continue
        top = e_masked[j].topk(k).indices
        pk.append(float(wrong[j][top].float().mean()))
    return dict(n=len(idx), Hp=float(hp.mean()) if len(hp) else float('nan'), Hm=float(hm.mean()) if len(hm) else float('nan'),
                conf_corr=float((corr & (ent < H0)).sum(-1).float().mean()), conf_wrong=float((wrong & (ent < H0)).sum(-1).float().mean()),
                tight_wrong=float((wrong & (ent < 0.02)).sum(-1).float().mean()), tight_corr=float((corr & (ent < 0.02)).sum(-1).float().mean()),
                wrong_cells=float(wrong.sum(-1).float().mean()), auc=auc(hm, hp), p_at_k=(sum(pk) / len(pk)) if pk else float('nan'))


rows = []
for nm in names:
    R = torch.load(f'{S}/R_{nm}.pt')
    P, H, y, given = R['P'], R['H'], R['y'], R['given']
    y8 = y.to(torch.int8)
    solved = (P == y8[None]).all(-1)                          # [128,N]
    s16 = solved[15]; ever_late = solved[16:].any(0) & ~s16; never = ~solved.any(0)
    uns = ~s16
    print(f'\n## {R["name"]}: solved@16 {int(s16.sum())}/512, late-solved {int(ever_late.sum())} of {int(uns.sum())} unsolved '
          f'({100 * ever_late.sum() / max(1, uns.sum()):.1f}%), never {int(never.sum())}')
    print(f"{'group':>22} {'n':>4} {'H+':>6} {'H-':>6} {'conf✓':>6} {'conf✗':>6} {'tight✓':>6} {'tight✗':>6} {'wrong':>6} {'AUC':>5} {'P@k':>5}")
    groups = [('solved@16', s16), ('unsolved@16', uns), ('  late-solved', ever_late), ('  never', never)]
    for g, mask in groups:
        idx = mask.nonzero().flatten()
        if len(idx) == 0:
            continue
        st = stats(P, H, y, given, idx)
        print(f"{g:>22} {st['n']:4d} {st['Hp']:6.3f} {st['Hm']:6.3f} {st['conf_corr']:6.1f} {st['conf_wrong']:6.2f} {st['tight_corr']:6.1f} {st['tight_wrong']:6.2f} {st['wrong_cells']:6.1f} {st['auc']:5.2f} {st['p_at_k']:5.2f}")
    # by difficulty
    print(f"{'difficulty':>10} {'n':>4} {'solved@16':>9} {'late/unsolved':>13} {'H+ uns':>7} {'H- uns':>7} {'conf✗ uns':>9} {'AUC uns':>7} {'H+ late':>7} {'H+ never':>8} {'conf✗ late':>10} {'conf✗ never':>11}")
    for b in range(3):
        m = dbin == b
        idx_u = (m & uns).nonzero().flatten(); idx_l = (m & ever_late).nonzero().flatten(); idx_n = (m & never).nonzero().flatten()
        su = stats(P, H, y, given, idx_u) if len(idx_u) else None
        sl = stats(P, H, y, given, idx_l) if len(idx_l) else None
        sn = stats(P, H, y, given, idx_n) if len(idx_n) else None
        late_rate = 100 * len(idx_l) / max(1, len(idx_u))
        rows.append((nm, BIN[b], late_rate, su['Hp'] if su else float('nan'), su['auc'] if su else float('nan'), su['conf_wrong'] if su else float('nan'), su['p_at_k'] if su else float('nan')))
        print(f"{BIN[b]:>10} {int(m.sum()):4d} {int((m & s16).sum()):9d} {late_rate:12.1f}% "
              f"{su['Hp'] if su else float('nan'):7.3f} {su['Hm'] if su else float('nan'):7.3f} {su['conf_wrong'] if su else float('nan'):9.2f} {su['auc'] if su else float('nan'):7.2f} "
              f"{sl['Hp'] if sl else float('nan'):7.3f} {sn['Hp'] if sn else float('nan'):8.3f} {sl['conf_wrong'] if sl else float('nan'):10.2f} {sn['conf_wrong'] if sn else float('nan'):11.2f}")
    # progressive commitment: confident-correct / confident-wrong counts over segments, late vs never
    print(f"{'seg':>5} {'late conf✓':>10} {'late conf✗':>10} {'late wrong':>10} {'never conf✓':>11} {'never conf✗':>11} {'never wrong':>11}")
    for seg in (16, 24, 32, 48, 64, 96, 128):
        sl = stats(P, H, y, given, ever_late.nonzero().flatten(), seg - 1) if ever_late.any() else None
        sn = stats(P, H, y, given, never.nonzero().flatten(), seg - 1) if never.any() else None
        print(f"{seg:5d} {sl['conf_corr'] if sl else float('nan'):10.1f} {sl['conf_wrong'] if sl else float('nan'):10.2f} {sl['wrong_cells'] if sl else float('nan'):10.1f} "
              f"{sn['conf_corr'] if sn else float('nan'):11.1f} {sn['conf_wrong'] if sn else float('nan'):11.2f} {sn['wrong_cells'] if sn else float('nan'):11.1f}")
    # do correct cells stay correct? per-segment flips correct->wrong (per puzzle) for late and never groups, seg17..128
    for g, mask in (('late-solved', ever_late), ('never', never)):
        idx = mask.nonzero().flatten()
        if len(idx) == 0: continue
        blank = ~given[idx]; yy = y[idx]
        Pg = P[:, idx].long()
        c2w = ((Pg[15:-1] == yy[None]) & (Pg[16:] != yy[None]) & blank[None]).sum(-1).float()     # [112, n]
        w2c = ((Pg[15:-1] != yy[None]) & (Pg[16:] == yy[None]) & blank[None]).sum(-1).float()
        print(f"{g}: per segment (17..128) correct->wrong flips {float(c2w.mean()):.3f}/puzzle, wrong->correct {float(w2c.mean()):.3f}/puzzle; "
              f"puzzles with any correct->wrong flip {int((c2w.sum(0) > 0).sum())}/{len(idx)}")
    # what changes between seg16 and the solve: entropy at seg16 of the cells that get corrected vs cells kept
    if ever_late.any():
        idx = ever_late.nonzero().flatten()
        first = torch.stack([solved[16:, i].nonzero()[0, 0] + 16 for i in idx])           # solve segment index
        blank = ~given[idx]; p16 = P[15, idx].long(); e16 = H[15, idx].float()
        wrong16 = (p16 != y[idx]) & blank; corr16 = (p16 == y[idx]) & blank
        psol = torch.stack([P[first[j], idx[j]] for j in range(len(idx))]).long()
        changed = (psol != p16) & blank
        print(f"late-solved: wrong@16 cells {wrong16.sum(-1).float().mean():.1f}/puzzle, entropy@16 of wrong cells {e16[wrong16].mean():.3f}, "
              f"of correct cells that stayed {e16[corr16 & ~changed].mean():.3f}, of correct cells that changed {e16[corr16 & changed].mean() if (corr16 & changed).any() else float('nan'):.3f} "
              f"(count {float((corr16 & changed).sum(-1).float().mean()):.2f}/puzzle); solve segment median {float(first.float().median()) + 1:.0f}")

print('\n## summary: late-solve rate vs calibration (model x difficulty)')
print(f"{'model':>8} {'bin':>7} {'late%':>6} {'H+ uns':>7} {'AUC':>5} {'conf✗':>6} {'P@k':>5}")
for r in rows:
    print(f"{r[0]:>8} {r[1]:>7} {r[2]:6.1f} {r[3]:7.3f} {r[4]:5.2f} {r[5]:6.2f} {r[6]:5.2f}")
import statistics
def spearman(a, b):
    ra = sorted(range(len(a)), key=lambda i: a[i]); rb = sorted(range(len(b)), key=lambda i: b[i])
    ia = [0] * len(a); ib = [0] * len(b)
    for k, i in enumerate(ra): ia[i] = k
    for k, i in enumerate(rb): ib[i] = k
    n = len(a); return 1 - 6 * sum((ia[i] - ib[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
ok = [r for r in rows if not any(math.isnan(v) for v in r[2:])]
if len(ok) > 3:
    print('Spearman(late%%, H+ unsolved) = %.2f; (late%%, AUC) = %.2f; (late%%, conf✗) = %.2f; (late%%, P@k) = %.2f  over %d model×bin cells' % (
        spearman([r[2] for r in ok], [r[3] for r in ok]), spearman([r[2] for r in ok], [r[4] for r in ok]),
        spearman([r[2] for r in ok], [r[5] for r in ok]), spearman([r[2] for r in ok], [r[6] for r in ok]), len(ok)))
