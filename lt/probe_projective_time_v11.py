"""A normalization-independent necessary test for temporal Fourier addresses.

Allow the more general identity Y_j(s)=a_s*lambda_j*X_j(s)+b_j*c_s,
where a_s and c_s are arbitrary COMPLEX scalars at every time step.
For every three channels, det([Y,lambda*X,b])=0. Expansion gives a
nonzero constant null vector for the six columns Y_i*X_j, i!=j.
Consequently that six-column matrix has rank <=5. Degenerate zero-input
or zero-retention cases also have dependent columns.

This remains necessary under any nonzero time-varying common complex
rescaling of the address, including unit normalization and scalar value
components. Taking differences of the product columns preserves every
constant null vector and suppresses stationary offsets.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n")


def triples(p=52):
    chosen = {tuple(sorted((j, (j+17) % p, (j+34) % p))) for j in range(p)}
    chosen |= {(0, 1, 2), (5, 7, 11), (16, 30, 45), (49, 50, 51)}
    return np.asarray(sorted(chosen))


def evaluate(z, storage_epsilon=np.finfo(np.float32).eps):
    z = np.asarray(z, np.complex128)
    x, y = z[:-1], z[1:]
    tri = triples(z.shape[-1])
    i, j, k = tri.T
    columns = np.stack((y[:, i]*x[:, j], -y[:, i]*x[:, k],
                        -y[:, j]*x[:, i], y[:, j]*x[:, k],
                        y[:, k]*x[:, i], -y[:, k]*x[:, j]), axis=-1)
    # [triple, time, six products]
    m0 = columns.transpose(1, 0, 2)
    m = np.diff(m0, axis=1)
    levels = np.sqrt(np.mean(np.abs(m0)**2, axis=1))
    changes = np.sqrt(np.mean(np.abs(m)**2, axis=1))
    active = (changes > np.maximum(1e-12, 1e-4*levels)).all(-1)
    norms = np.linalg.norm(m, axis=1)
    normalized = m/np.maximum(norms[:, None], 1e-30)
    singular = np.linalg.svd(normalized, compute_uv=False)
    ratio = singular[:, -1]/np.maximum(singular[:, 0], 1e-30)
    noise = 32*storage_epsilon*np.sqrt(np.sum(
        (levels/np.maximum(changes, 1e-30))**2, axis=-1))
    threshold = np.maximum(1e-3, 10*noise)
    failed = active & (ratio > threshold)
    good = ratio[active]
    report = dict(active_triples=int(active.sum()), total_triples=len(tri),
                  triples_violating_necessary_identity=int(failed.sum()),
                  sigma6_over_sigma1_quantiles=dict(zip(
                      ["min", "p10", "median", "p90", "max"],
                      map(float, np.quantile(good, [0, .1, .5, .9, 1])))) if len(good) else {},
                  threshold_max=float(threshold[active].max()) if active.any() else None)
    arrays = dict(triples=tri, singular_values=singular, ratio=ratio,
                  active=active, violation=failed, threshold=threshold,
                  witness_triple=tri[0], witness_normalized_matrix=normalized[0])
    return report, arrays


def control():
    rng = np.random.default_rng(29043)
    n, p = 2048, 52
    lam = np.linspace(.67, .97, p)*np.exp(1j*np.linspace(.023, 1.41, p))
    b = rng.normal(size=p)+1j*rng.normal(size=p)
    x = rng.normal(size=n)+1j*rng.normal(size=n)
    z = np.zeros((n, p), complex)
    for t in range(1, n):
        z[t] = lam*z[t-1]+b*x[t]
    # This deliberately destroys the ordinary constant-coefficient recurrence
    # while preserving the projective temporal-Fourier property being tested.
    gauge = np.exp(.5*rng.normal(size=n)+1j*rng.uniform(-np.pi, np.pi, n))
    z = (z*gauge[:, None]).astype(np.complex64)
    result, _ = evaluate(z)
    assert result["active_triples"] == len(triples())
    assert result["triples_violating_necessary_identity"] == 0, result
    shuffled, _ = evaluate(z[rng.permutation(n)])
    # A necessary condition rejects the shared-clock hypothesis as soon as
    # one group violates it. Nearby filters can remain almost collinear even
    # after shuffling, so rejection of every triple is not a valid demand.
    assert shuffled["triples_violating_necessary_identity"] > 0, shuffled
    result["time_shuffled_control"] = shuffled
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/phase_timing_v11"))
    args = parser.parse_args()
    out = args.root/"projective_time"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        hypothesis="z_next = a(k)*diag(lambda)*z + b*c(k), arbitrary complex scalars a(k),c(k), fixed complex lambda,b within each cell/time window",
        rationale="A common-activity damped Fourier bank under arbitrary common time-dependent amplitude and phase gauges has this form. Unit address normalization is included.",
        necessary_condition="For any three channels, the six time series Y_i*X_j (i!=j) have a constant linear dependence. Their temporal differences must therefore have rank <=5.",
        cell_selection="Each late target plus fixed cells 0,8,40,72,80; same as the four-column test",
        triples_zero_based=triples().tolist(),
        criterion="Active product differences exceed 1e-4 of level RMS; reject if sigma6/sigma1 > max(1e-3,10*conservative FP32 noise scale)",
        normalization="Normalize each product-difference column to unit L2; no fitting to choose a favorable triple",
        scope="No assumed frequency, input activity, gauge scale, or decay. Tests the native frequency coordinates; independent unconstrained inputs for every frequency are a different, unidentifiable model.",
    )
    dump(out/"protocol.json", protocol)
    positive = control()
    print("CONTROL", json.dumps(positive), flush=True)
    data = args.root/"time_data"
    files = [data/f"early_puzzle_{p}_raw.npy" for p in (58, 209, 230)]
    files += [data/f"late_{p}_puzzle_{p}_raw.npy" for p in (58, 209, 230)]
    reports = []
    for path in files:
        raw = np.load(path, mmap_mode="r")
        puzzle = int(path.stem.split("_puzzle_")[1].split("_")[0])
        target = {58: 26, 209: 19, 230: 28}[puzzle]
        for cell in (target, 0, 8, 40, 72, 80):
            for head in range(8):
                row, arrays = evaluate(raw[:, cell, head])
                row.update(file=path.name, puzzle=puzzle, cell=cell, head_1based=head+1)
                reports.append(row)
                if cell == target:
                    np.savez_compressed(out/f"{path.stem}_cell{cell}_head{head+1}.npz", **arrays)
        print("CASE", path.name, json.dumps(reports[-8:]), flush=True)
        dump(out/"summary.json", dict(protocol=protocol, positive_control=positive, cases=reports))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
