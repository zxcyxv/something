"""Extend the projective frequency test to several latent activity streams.

For Y=a(k)D X+B c(k), with diagonal fixed D and rank(B)<=m-2, choose
two independent row annihilators u,v of B. Then
(uY)(vDX)-(vY)(uDX)=0, a constant relation between Y_i X_j, i!=j.
Thus full column rank of these m(m-1) series excludes <=m-2 complex
drives even with arbitrary shared complex normalization at every step.
"""

import json
from pathlib import Path

import numpy as np


def test(z, storage_epsilon=np.finfo(np.float32).eps):
    z = np.asarray(z, np.complex128)
    m = z.shape[1]
    i, j = np.where(~np.eye(m, dtype=bool))
    products = z[1:, i]*z[:-1, j]
    delta = np.diff(products, axis=0)
    level = np.sqrt(np.mean(np.abs(products)**2, axis=0))
    change = np.sqrt(np.mean(np.abs(delta)**2, axis=0))
    active = bool((change > np.maximum(1e-12, 1e-4*level)).all())
    normalized = delta/np.maximum(np.linalg.norm(delta, axis=0), 1e-30)
    s = np.linalg.svd(normalized, compute_uv=False)
    ratio = float(s[-1]/s[0])
    noise = 32*storage_epsilon*np.linalg.norm(level/np.maximum(change, 1e-30))
    threshold = float(max(1e-3, 10*noise))
    return dict(channels=m, excluded_complex_drive_rank=m-2,
                active=active, sigma_min_over_max=ratio, threshold=threshold,
                violates_necessary_identity=bool(active and ratio > threshold)), s


def control(m, rank, rng):
    n = 2048
    lam = np.linspace(.6, .95, m)*np.exp(1j*np.linspace(.12, 1.2, m))
    b = rng.normal(size=(m, rank))+1j*rng.normal(size=(m, rank))
    activity = rng.normal(size=(n, rank))+1j*rng.normal(size=(n, rank))
    z = np.zeros((n, m), complex)
    for t in range(1, n):
        z[t] = lam*z[t-1]+b@activity[t]
    gauge = np.exp(.4*rng.normal(size=n)+1j*rng.uniform(-np.pi, np.pi, n))
    z = (z*gauge[:, None]).astype(np.complex64)
    row, _ = test(z)
    assert not row["violates_necessary_identity"], row
    assert row["sigma_min_over_max"] < 1e-6, row
    return row


def main():
    root = Path("runs/phase_timing_v11")
    out = root/"multi_drive_time"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        hypothesis="Y=a(k)*diag(lambda)*X+B*c(k), arbitrary complex activity vector c(k), fixed B/lambda within the cell/window, arbitrary complex scalar a(k)",
        selections="Late windows of puzzles58/209/230, original target cell, all8 heads, fixed channel groups starting at0,17,35",
        group_sizes=[4, 6, 10],
        meaning="Full column rank on m channels excludes up to m-2 complex input activities for that group; no activity identity or frequency is chosen",
        validation="Known synthetic filters with exactly m-2 independent complex drives must pass",
        scope="Finite native coordinate groups and fixed frequency dynamics. This does not exclude freely independent input innovations in every channel, which cannot identify frequencies from state observations alone.",
    )
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    rng = np.random.default_rng(98235)
    controls = [control(m, m-2, rng) for m in protocol["group_sizes"]]
    rows, arrays = [], {}
    for puzzle, cell in ((58, 26), (209, 19), (230, 28)):
        raw = np.load(root/"time_data"/f"late_{puzzle}_puzzle_{puzzle}_raw.npy", mmap_mode="r")
        for head in range(8):
            for m in protocol["group_sizes"]:
                for start in (0, 17, 35):
                    z = raw[:, cell, head, start:start+m]
                    row, s = test(z)
                    row.update(puzzle=puzzle, cell=cell, head_1based=head+1,
                               channels_zero_based=list(range(start, start+m)))
                    rows.append(row)
                    arrays[f"p{puzzle}_h{head+1}_m{m}_s{start}"] = s
    report = dict(protocol=protocol, controls=controls, cases=rows)
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(out/"singular_values.npz", **arrays)
    print("CONTROLS", json.dumps(controls), flush=True)
    for m in protocol["group_sizes"]:
        selected = [r for r in rows if r["channels"] == m]
        print(json.dumps(dict(channels=m, cases=len(selected),
                              violated=sum(r["violates_necessary_identity"] for r in selected),
                              ratio_quantiles=np.quantile([r["sigma_min_over_max"] for r in selected], [0,.5,1]).tolist(),
                              threshold_range=[min(r["threshold"] for r in selected), max(r["threshold"] for r in selected)])), flush=True)


if __name__ == "__main__":
    main()
