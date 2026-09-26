"""Identify or reject native common-drive damped Fourier traces on real time.

For z_j(k+1)=lambda_j*z_j(k)+b_j*x(k), eliminating x between channels
j,l gives a linear dependence of [z'_j,z_j,z'_l,z_l]. This is necessary
even for a complex scalar x, unknown b and unknown lambda. Time differences
obey the same identity and avoid a large stationary component hiding error.
No definition of x, frequency, phase pulse, or hidden intervention is needed.

An additional low-rank-innovation fit quantifies approximate correspondence.
It is an oracle-input reconstruction, not a forecast of unknown activities.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def top_eigen(matrix, rank):
    values, vectors = np.linalg.eigh(matrix)
    return values[-rank:], vectors[:, -rank:]


def realify(z):
    return np.concatenate((z.real, z.imag), axis=-1)


def matrix_for(lam):
    p = len(lam)
    d = np.arange(p)
    m = np.zeros((2*p, 2*p))
    m[d, d] = m[d+p, d+p] = lam.real
    m[d, d+p] = lam.imag
    m[d+p, d] = -lam.imag
    return m


def covariance(x, y):
    x, y = realify(x), realify(y)
    return x.T@x/len(x), x.T@y/len(x), y.T@y/len(y)


def residual_covariance(stats, lam):
    xx, xy, yy = stats
    m = matrix_for(lam)
    r = yy-m.T@xy-xy.T@m+m.T@xx@m
    return (r+r.T)/2


def fit_trace(stats, rank, iterations=500, real_lambda=False):
    xx, xy, yy = stats
    p = len(xx)//2
    d = np.arange(p)
    denom = xx[d, d]+xx[d+p, d+p]
    ordinary = ((xy[d, d]+xy[d+p, d+p])+
                1j*(xy[d, d+p]-xy[d+p, d])) / np.maximum(denom, 1e-30)
    seeds = [ordinary, np.zeros(p, complex), np.ones(p, complex)*.9]
    fits = []
    for seed in seeds:
        lam = seed.copy()
        if real_lambda:
            lam = lam.real.astype(complex)
        lam /= np.maximum(1, np.abs(lam))
        previous = np.inf
        for iteration in range(iterations):
            r = residual_covariance(stats, lam)
            eig, basis = top_eigen(r, rank)
            proj = basis@basis.T
            m = matrix_for(lam)
            target = xy-(xy-xx@m)@proj
            nxt = ((target[d, d]+target[d+p, d+p])+
                   1j*(target[d, d+p]-target[d+p, d])) / np.maximum(denom, 1e-30)
            if real_lambda:
                nxt = nxt.real.astype(complex)
            nxt /= np.maximum(1, np.abs(nxt))
            loss = max(0., float(np.trace(r)-eig.sum()))
            change = float(np.max(np.abs(nxt-lam)))
            lam = nxt
            if change < 1e-10 or (iteration > 20 and abs(previous-loss) < 1e-12*max(1., np.trace(yy))):
                break
            previous = loss
        r = residual_covariance(stats, lam)
        eig, basis = top_eigen(r, rank)
        loss = max(0., float(np.trace(r)-eig.sum()))
        fits.append(dict(lam=lam, basis=basis, loss=loss, iterations=iteration+1))
    return min(fits, key=lambda x: x["loss"])


def score_fit(stats, fit):
    r = residual_covariance(stats, fit["lam"])
    b = fit["basis"]
    error = max(0., float(np.trace(r)-np.trace(b.T@r@b)))
    target = float(np.trace(stats[2]))
    return dict(relative_residual_norm=float(np.sqrt(error/max(target, 1e-30))),
                residual_energy_fraction=float(error/max(target, 1e-30)))


def pair_test(z, channel_pairs=None):
    """z [time, channel], return independent four-column certificates."""
    z = np.asarray(z, np.complex128)
    dz = np.diff(z, axis=0)
    x, y = dz[:-1], dz[1:]
    raw_rms = np.sqrt(np.mean(np.abs(z)**2, axis=0))
    diff_rms = np.sqrt(np.mean(np.abs(dz)**2, axis=0))
    active = diff_rms > np.maximum(1e-9, 1e-4*raw_rms)
    if channel_pairs is None:
        ii, jj = np.triu_indices(z.shape[-1], 1)
        keep = active[ii] & active[jj]
        ii, jj = ii[keep], jj[keep]
    else:
        ii, jj = np.asarray(channel_pairs).T
    if len(ii) == 0:
        return dict(active_channels=0, pair_count=0), None
    m = np.stack((y[:, ii], x[:, ii], y[:, jj], x[:, jj]), axis=-1).transpose(1, 0, 2)
    norms = np.linalg.norm(m, axis=1)
    m /= np.maximum(norms[:, None], 1e-30)
    _, singular, vh = np.linalg.svd(m, full_matrices=False)
    ratio = singular[:, -1]/singular[:, 0]
    # Conservative storage/differencing noise scale, not a statistical CI.
    # Generated FP32 trace controls below calibrate this threshold as well.
    noise = 16*np.finfo(np.float32).eps*np.sqrt(
        (raw_rms[ii]/np.maximum(diff_rms[ii], 1e-30))**2 +
        (raw_rms[jj]/np.maximum(diff_rms[jj], 1e-30))**2)
    threshold = np.maximum(1e-3, 10*noise)
    violation = ratio > threshold
    # m @ null = 0; undo column scaling before extracting recurrence lambdas.
    null = vh[:, -1].conj()/norms
    inferred_i = -null[:, 1]/np.where(np.abs(null[:, 0]) > 1e-30, null[:, 0], np.nan)
    inferred_j = -null[:, 3]/np.where(np.abs(null[:, 2]) > 1e-30, null[:, 2], np.nan)
    result = dict(active_channels=int(active.sum()), pair_count=len(ii),
                  pairs_violating_necessary_identity=int(violation.sum()),
                  sigma4_over_sigma1_quantiles=dict(zip(
                      ["min", "p10", "median", "p90", "max"],
                      map(float, np.quantile(ratio, [0, .1, .5, .9, 1])))),
                  threshold_max=float(threshold.max()))
    arrays = dict(channel_i=ii, channel_j=jj, singular_values=singular,
                  ratio=ratio, violation=violation, threshold=threshold,
                  inferred_i=inferred_i, inferred_j=inferred_j)
    return result, arrays


def synthetic_control(out):
    rng = np.random.default_rng(9042)
    p, n = 52, 4096
    lam = np.linspace(.65, .98, p)*np.exp(1j*np.linspace(.035, 1.45, p))
    b = rng.normal(size=p)+1j*rng.normal(size=p)
    activity = rng.normal(size=n)
    z = np.zeros((n, p), complex)
    for k in range(1, n):
        z[k] = lam*z[k-1]+b*activity[k]
    z = z.astype(np.complex64).astype(np.complex128)
    pair, arrays = pair_test(z)
    dz = np.diff(z, axis=0)
    x, y = dz[:-1], dz[1:]
    scale = np.sqrt(np.mean(np.abs(x[:2048])**2, axis=0))
    train = covariance(x[:2048]/scale, y[:2048]/scale)
    valid = covariance(x[2048:]/scale, y[2048:]/scale)
    fit = fit_trace(train, 1, iterations=1200)
    scores = score_fit(valid, fit)
    frequency_error = float(np.max(np.abs(np.angle(fit["lam"]/lam))))
    lambda_error = float(np.max(np.abs(fit["lam"]-lam)))
    assert pair["pairs_violating_necessary_identity"] == 0
    assert scores["relative_residual_norm"] < 1e-4, scores
    assert lambda_error < 1e-3, lambda_error
    report = dict(pair_test=pair, fit_holdout=scores, max_lambda_error=lambda_error,
                  max_frequency_error_radians_per_block=frequency_error,
                  purpose="Verify the procedure recovers a true common-drive damped rotation from FP32 traces; this is not an untrained-model comparison")
    dump(out/"synthetic_control.json", report)
    np.savez_compressed(out/"synthetic_control.npz", true_lambda=lam,
                        fitted_lambda=fit["lam"], **arrays)
    return report


def load_xy(paths, head, max_per_file=12000):
    xs, ys = [], []
    for path in paths:
        raw = np.load(path, mmap_mode="r")[:, :, head].astype(np.complex128)
        diff = np.diff(raw, axis=0)
        x, y = diff[:-1].reshape(-1, 52), diff[1:].reshape(-1, 52)
        # Deterministic coverage across time and cells; never select on errors.
        keep = np.linspace(0, len(x)-1, min(max_per_file, len(x)), dtype=int)
        xs.append(x[keep]); ys.append(y[keep])
    return np.concatenate(xs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/phase_timing_v11"))
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()
    out = args.root/"time_analysis"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        hypothesis="Native address channels are fixed damped Fourier traces of a common scalar activity: z_j(k+1)=lambda_j*z_j(k)+b_j*x(k+1). lambda and b may be unknown, complex, and learned.",
        necessary_identity="For every pair j,l, [dz_j(k+1),dz_j(k),dz_l(k+1),dz_l(k)] has complex rank <=3. This condition even allows complex x and arbitrary lambda, without fitting either.",
        no_intervention="Use actual model trajectories only; compare with a generated known trace as a measurement control.",
        pair_scope="Test each cell separately, allowing different hypothetical lambda/b across cells; fixed across its recorded time window.",
        pair_rejection="sigma4/sigma1 > max(1e-3,10*conservative_FP32_noise_scale); exclude channels with RMS increment below 1e-4 of RMS level",
        target_cells=[26, 19, 28], diagnostic_cells=[0, 8, 40, 72, 80],
        approximate_fit="Fixed lambda and real rank-r innovations across training puzzles; fresh activity coefficients allowed freely on every holdout step. This is structural reconstruction, not forecasting.",
        ranks=[1, 2, 4, 8, 16, 32],
        normalization="RMS of complex increments in training data, per channel. This fixed rescaling leaves lambda invariant.",
        exact_acceptance="Necessary rank identities and holdout residual <=1e-4; synthetic control must recover frequency to 1e-3 rad/block.",
        approximate_threshold="Report error continuously. <=0.1 relative residual norm is a predeclared useful-approximation target, not mathematical identity.",
        limitation="An unconstrained independent complex activity per channel permits any chosen lambda by definition; it does not identify a temporal frequency from v1.1 alone.",
    )
    dump(out/"protocol.json", protocol)
    tic = time.monotonic()
    control = synthetic_control(out)
    print("CONTROL", json.dumps(control), flush=True)
    if args.control_only:
        return
    data = args.root/"time_data"
    capture = json.loads((data/"summary.json").read_text())
    assert len(capture["windows"]) == 4
    train_paths = [data/f"early_puzzle_{p}_raw.npy" for p in (38, 72, 111, 128)]
    valid_paths = [data/f"early_puzzle_{p}_raw.npy" for p in (58, 209, 230)]
    late_paths = [data/f"late_{p}_puzzle_{p}_raw.npy" for p in (58, 209, 230)]
    pairs = []
    for path in valid_paths+late_paths:
        raw = np.load(path, mmap_mode="r")
        puzzle = int(path.stem.split("_puzzle_")[1].split("_")[0])
        target = {58: 26, 209: 19, 230: 28}[puzzle]
        cells = [target]+protocol["diagnostic_cells"]
        for cell in cells:
            for head in range(8):
                result, arrays = pair_test(raw[:, cell, head])
                result.update(file=path.name, puzzle=puzzle, cell=cell, head_1based=head+1)
                pairs.append(result)
                if cell == target:
                    np.savez_compressed(out/f"pair_{path.stem}_cell{cell}_head{head+1}.npz", **arrays)
        print("PAIR", path.name, json.dumps(pairs[-8:]), flush=True)
    dump(out/"pair_rank_summary.json", dict(protocol=protocol, control=control, cases=pairs))
    fits = []
    for head in range(8):
        datasets = {}
        x, y = load_xy(train_paths, head)
        scale = np.maximum(np.sqrt(np.mean(np.abs(x)**2, axis=0)), 1e-10)
        datasets["train"] = covariance(x/scale, y/scale)
        for name, paths in (("holdout_early", valid_paths), ("holdout_late", late_paths)):
            x, y = load_xy(paths, head)
            datasets[name] = covariance(x/scale, y/scale)
        head_report = dict(head_1based=head+1, fits=[])
        save = dict(channel_scale=scale)
        for rank in protocol["ranks"]:
            fit = fit_trace(datasets["train"], rank)
            scores = {name: score_fit(stats, fit) for name, stats in datasets.items()}
            row = dict(rank_real_innovation=rank, iterations=fit["iterations"], scores=scores,
                       median_abs_frequency=float(np.median(np.abs(np.angle(fit["lam"])))),
                       median_rho=float(np.median(np.abs(fit["lam"]))))
            head_report["fits"].append(row)
            save[f"rank{rank}_lambda"] = fit["lam"]
            save[f"rank{rank}_basis"] = fit["basis"]
            print("FIT", head+1, json.dumps(row), flush=True)
        real_fit = fit_trace(datasets["train"], 1, real_lambda=True)
        head_report["rank1_real_lambda_only"] = {name: score_fit(stats, real_fit) for name, stats in datasets.items()}
        save["rank1_real_only_lambda"] = real_fit["lam"]
        np.savez_compressed(out/f"fitted_head{head+1}.npz", **save)
        fits.append(head_report)
        dump(out/"fit_summary.json", dict(protocol=protocol, control=control, heads=fits,
                                          elapsed_seconds=time.monotonic()-tic))
    print("DONE", time.monotonic()-tic, flush=True)


if __name__ == "__main__":
    main()
