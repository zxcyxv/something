"""Static interpretability measurements of a trained Rule 110 checkpoint -> npz for plotting."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1]); from r110lib import *
torch.set_grad_enabled(False)
model, cfg = build(sys.argv[2]); out = sys.argv[3]
BL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128]
NB = [1, 2, 4, 8, 16, 32, 64, 128]
PB = [1, 2, 3, 4, 8, 16, 32, 64, 128]
KB = [1, 2, 4, 8, 32, 128]
RMAX, NROW, BATCH, C = 6, 32768, 8192, 20
nb = len(BL)
S1 = {r: torch.zeros(nb, 2 ** (2 * r + 1), device=dev, dtype=torch.float64) for r in range(RMAX + 1)}
S2 = {r: torch.zeros_like(S1[r]) for r in S1}; CNT = {r: torch.zeros(2 ** (2 * r + 1), device=dev, dtype=torch.float64) for r in S1}
TOT = torch.zeros(nb, 3, device=dev, dtype=torch.float64)                # n, sum, sumsq
cell_acc = torch.zeros(nb, device=dev, dtype=torch.float64); row_acc = torch.zeros(nb, device=dev, dtype=torch.float64)
FS1 = torch.zeros(len(NB), 512, 256, device=dev, dtype=torch.float64); FS2 = torch.zeros_like(FS1)
KER = torch.zeros(len(KB), 17, device=dev, dtype=torch.float64); KERA = torch.zeros_like(KER)
wnorm = torch.zeros(128, device=dev, dtype=torch.float64)
probe_h = {b: [] for b in PB}; probe_y = []
gen = torch.Generator(device=dev).manual_seed(123)
for bi in range(NROW // BATCH):
    x, y = sample(BATCH, gen)
    widx = {r: windows(x, r) for r in range(RMAX + 1)}
    for r in range(RMAX + 1):
        CNT[r] += torch.bincount(widx[r].flatten(), minlength=2 ** (2 * r + 1)).double()
    states = evolve_all(x)
    if bi == 0:
        # targets: x_t[i + d] for t 0..4, d -4..4 at the centre cells of the first 4096 rows
        tg = torch.stack([s.unfold(1, 9, 1)[:, 4:24] for s in states], 2)[:4096]   # [4096, C, 5, 9]
        probe_y.append(tg.reshape(-1, 45).cpu())
    def each(b, h, logits, box, state):
        wnorm[b - 1] += state[3].norm(dim=(-1, -2)).mean().double()
        if b in BL:
            k = BL.index(b)
            z = (logits[..., 2] - logits[..., 1])[:, 8:28].double()
            pred = logits.argmax(-1) - 1
            cell_acc[k] += (pred == y).double().mean(); row_acc[k] += (pred == y).all(-1).double().mean()
            TOT[k] += torch.stack([torch.tensor(float(z.numel()), device=dev, dtype=torch.float64), z.sum(), (z * z).sum()])
            for r in range(RMAX + 1):
                S1[r][k].index_add_(0, widx[r].flatten(), z.flatten()); S2[r][k].index_add_(0, widx[r].flatten(), (z * z).flatten())
        if b in NB:
            k = NB.index(b); a = box['ffn'][:, 8:28].double().reshape(-1, 256); w4 = widx[4].flatten()
            FS1[k].index_add_(0, w4, a); FS2[k].index_add_(0, w4, a * a)
        if b in PB and bi == 0:
            probe_h[b].append(h[:4096, 8:28].reshape(-1, 32).cpu())
        if b in KB:
            k = KB.index(b); A = (box['qr'] @ box['kr'].transpose(-1, -2))[:, 0]            # [B, T, T] read score
            for j, d in enumerate(range(-8, 9)):
                v = A[:, 8:28].diagonal(offset=8 + d, dim1=1, dim2=2)
                KER[k, j] += v.double().mean(); KERA[k, j] += v.abs().double().mean()
    run(model, x, 128, each)
    print('batch', bi, flush=True)
nbat = NROW // BATCH
# --- locality R^2 (ANOVA-adjusted) per block and radius
N = TOT[:, 0]; var_tot = (TOT[:, 2] - TOT[:, 1] ** 2 / N) / (N - 1)
R2 = np.zeros((nb, RMAX + 1))
for r in range(RMAX + 1):
    n = CNT[r].clamp_min(1); G = int((CNT[r] > 0).sum())
    within = (S2[r] - S1[r] ** 2 / n).sum(1) / (N - G)
    R2[:, r] = (1 - within / var_tot).cpu().numpy()
# --- Walsh spectra of the radius-4 table (model mean z per window) vs truth
H = hadamard(); f = truth9().double(); Ft = (2 * f - 1).float()
g = (S1[4] / CNT[4].clamp_min(1)).float()                                # [nb, 512]
spec_truth = (H @ Ft) / 512; spec_model = (g @ H.T) / 512                 # [512], [nb, 512]
table_acc = ((g > 0).float() == f.float()).float().mean(1)
# --- FFN neurons: mean activation per window, locality, spectra
FM = (FS1 / CNT[4].clamp_min(1)[None, :, None]).float()                   # [nb2, 512, 256]
Ntot = CNT[4].sum()
fvar_tot = (FS2.sum(1) - FS1.sum(1) ** 2 / Ntot) / (Ntot - 1)
fwithin = (FS2 - FS1 ** 2 / CNT[4].clamp_min(1)[None, :, None]).sum(1) / (Ntot - 512)
f_loc = (1 - fwithin / fvar_tot.clamp_min(1e-12)).float()
f_std = fvar_tot.sqrt().float()
f_spec = torch.einsum('sw,kwn->ksn', H, FM) / 512                         # [nb2, 512 S, 256]
# --- probes: ridge least squares on +-1 targets, train/test split
Y = 2 * torch.cat(probe_y).double() - 1; ntr = int(0.75 * Y.shape[0]); PR = np.zeros((len(PB), 5, 9))
for k, b in enumerate(PB):
    X = torch.cat(probe_h[b]).double(); X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
    Wp = torch.linalg.solve(X[:ntr].T @ X[:ntr] + 1e-3 * torch.eye(33, dtype=torch.float64), X[:ntr].T @ Y[:ntr])
    acc = ((X[ntr:] @ Wp).sign() == Y[ntr:]).double().mean(0)
    PR[k] = acc.reshape(5, 9).numpy()
theta = model.block.rope_theta.detach()[0, :, 0].cpu().numpy()          # [16] angle per cell (1D positions)
np.savez(out, BL=BL, NB=NB, PB=PB, KB=KB, R2=R2, cell_acc=(cell_acc / nbat).cpu().numpy(), row_acc=(row_acc / nbat).cpu().numpy(),
         spec_truth=spec_truth.cpu().numpy(), spec_model=spec_model.cpu().numpy(), table_acc=table_acc.cpu().numpy(),
         g=g.cpu().numpy(), f=f.cpu().numpy(), FM=FM.cpu().numpy(), f_loc=f_loc.cpu().numpy(), f_std=f_std.cpu().numpy(),
         f_spec=f_spec.cpu().numpy(), PR=PR, theta=theta, KER=(KER / nbat).cpu().numpy(), KERA=(KERA / nbat).cpu().numpy(),
         wnorm=(wnorm / nbat).cpu().numpy(), alpha=float(model.block.alpha), var_tot=var_tot.cpu().numpy())
print('table_acc', [round(float(v), 3) for v in table_acc]); print('R2 r=4', np.round(R2[:, 4], 3)); print('row_acc', np.round((row_acc / nbat).cpu().numpy(), 3))
print('probe block2\n', np.round(PR[1], 3)); print('probe block8\n', np.round(PR[4], 3))
