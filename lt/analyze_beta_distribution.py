"""CPU-only circular beta diagnostics for the stored LT checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def stats(beta):
    x = wrap(beta).ravel()
    zero = np.abs(x)
    even = np.minimum(zero, np.pi - zero)
    odd = np.abs(zero - np.pi / 2)
    c2, s2 = np.mean(np.cos(x)**2), np.mean(np.sin(x)**2)
    return dict(count=len(x), circular_mean_deg=float(np.degrees(np.angle(np.mean(np.exp(1j*x))))),
                circular_concentration=float(np.abs(np.mean(np.exp(1j*x)))),
                min_deg=float(np.degrees(x.min())), max_deg=float(np.degrees(x.max())),
                median_abs_deg=float(np.degrees(np.median(zero))),
                abs_deg_p90=float(np.degrees(np.quantile(zero, .9))),
                even_coefficient_energy=float(c2), odd_coefficient_energy=float(s2),
                odd_even_rms_ratio=float(np.sqrt(s2/c2)),
                nearer_even_axis=int((even < odd).sum()), nearer_odd_axis=int((odd < even).sum()),
                near={str(d): dict(zero=int((zero <= np.radians(d)).sum()),
                                  pi=int((np.pi-zero <= np.radians(d)).sum()),
                                  even_axis=int((even <= np.radians(d)).sum()),
                                  odd_axis=int((odd <= np.radians(d)).sum())) for d in (15, 30, 45)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/beta_distribution_v11"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {"v1.1_160k": "checkpoints/v1.1_step160000.npz",
             "v1_310527": "checkpoints/v1_step310527.npz",
             "v1_repro_120k": "checkpoints/v1_repro_step120000.npz"}
    result, values = {}, {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            key, = [k for k in z.files if k.startswith("ema/") and k.endswith(".beta")]
            beta = z[key].astype(np.float64)
            dtype = str(z[key].dtype)
        assert beta.shape == (8, 52)
        values[name] = wrap(beta)
        result[name] = dict(checkpoint=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            key=key, storage_dtype=dtype, overall=stats(beta),
                            heads=[dict(head=i+1, **stats(b)) for i, b in enumerate(beta)])
        np.savetxt(args.out / f"{name}_beta_degrees.csv", np.degrees(wrap(beta)), delimiter=",")
    result["interpretation"] = {
        "angles": "wrapped to [-pi, pi); EMA weights; heads numbered 1..8",
        "even_odd": "cos(delta+beta)=cos(beta)*cos(delta)-sin(beta)*sin(delta)",
        "energy": "unweighted parameter coefficient squares, NOT measured memory/message energy",
        "initialization": "LTLayer initializes beta ~ Normal(0, 0.5^2), already centered on zero",
        "initial_expected_odd_energy": float((1-np.exp(-2*.5**2))/2),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(-180, 180, 49)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax, (name, beta) in zip(axes, values.items()):
        ax.hist(np.degrees(beta.ravel()), bins=bins, color="#2878b5", edgecolor="white")
        for d in (-90, 90):
            ax.axvline(d, color="#cf4446", linestyle="--", alpha=.8)
        ax.axvline(0, color="#333333", linewidth=1)
        s = result[name]["overall"]
        ax.set_title(f"{name}: {s['near']['30']['zero']}/416 within 30 deg of 0; "
                     f"{s['near']['30']['odd_axis']}/416 within 30 deg of +/-90")
        ax.set_ylabel("Components")
    axes[-1].set_xticks([-180, -90, 0, 90, 180])
    axes[-1].set_xlabel("Wrapped beta (degrees); dashed lines = odd axes")
    fig.savefig(args.out / "beta_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        ax.hist(np.degrees(values["v1.1_160k"][i]), bins=bins, color="#2878b5")
        for d in (-90, 90):
            ax.axvline(d, color="#cf4446", linestyle="--", alpha=.8)
        ax.axvline(0, color="#333333", linewidth=1)
        s = result["v1.1_160k"]["heads"][i]
        ax.set_title(f"Head {i+1}: odd/even RMS = {s['odd_even_rms_ratio']:.3f}")
        ax.set_xticks([-180, -90, 0, 90, 180])
    fig.suptitle("v1.1 160k EMA: 52 beta components per head")
    fig.savefig(args.out / "beta_v11_heads.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v["overall"] for k, v in result.items() if "overall" in v}, indent=2))


if __name__ == "__main__":
    main()
