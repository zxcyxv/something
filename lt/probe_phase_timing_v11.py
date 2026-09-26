"""Direct phase interventions on the trained v1.1 write operator.

No hidden perturbation, free rollout, or assumed temporal frequency is used.
Positions, beta, amplitudes, values, agree and gain stay fixed. Conjugating
the unpositioned complex addresses reverses every intrinsic relative phase.
The position rotations in the production operator are deliberately retained.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research_cpu import CPUBlocks, load_cpu, load_snapshots, puzzle_batch


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def classify(plus, minus, scale):
    tol = 1e-9 * np.maximum(1.0, scale)
    return {
        "observed_positive_mirrored_negative": int(((plus > tol) & (minus < -tol)).sum()),
        "observed_negative_mirrored_positive": int(((plus < -tol) & (minus > tol)).sum()),
        "both_positive": int(((plus > tol) & (minus > tol)).sum()),
        "both_negative": int(((plus < -tol) & (minus < -tol)).sum()),
        "near_zero": int(((np.abs(plus) <= tol) | (np.abs(minus) <= tol)).sum()),
    }


def analyze_snapshot(blocks, state, block):
    h, _ = state
    p = blocks.parts(h)
    x, y = p["u"]
    original = p["G"][0].numpy()
    mirror = (blocks.gain *
              (blocks.inner.attn_xy((x, -y), blocks.kcb) * p["agree"]))[0].numpy()
    z = torch.complex(x, y)[0].permute(1, 0, 2).numpy()
    phase = np.angle(z[:, :, None] * z[:, None].conj())
    amplitude = np.abs(z[:, :, None] * z[:, None].conj())
    pos = np.stack((blocks.inner.pos_u.numpy(), blocks.inner.pos_w.numpy()), -1)
    chi = np.einsum("hjd,tnd->htnj", blocks.layer.theta.numpy(),
                    pos[:, None] - pos[None])
    offset = chi + blocks.layer.beta.numpy()[:, None, None]
    # gain has shape [H,1,1] in this checkpoint.
    scale = (blocks.gain.numpy() * p["agree"][0].numpy())
    scale *= blocks.kcb[0].numpy()
    coefficients = scale[..., None] * amplitude * np.exp(1j * offset)
    direct = np.real((coefficients * np.exp(1j * phase)).sum(-1))
    direct_mirror = np.real((coefficients * np.exp(-1j * phase)).sum(-1))
    error = max(float(np.max(np.abs(direct-original))),
                float(np.max(np.abs(direct_mirror-mirror))))
    assert error < 1e-10, error
    mask = ~np.eye(81, dtype=bool)
    even, odd = (original+mirror)/2, (original-mirror)/2
    beta = blocks.layer.beta.numpy()[:, None, None]
    odd_beta = -(scale[..., None]*amplitude*np.sin(phase)*np.sin(beta)*np.cos(chi)).sum(-1)
    odd_position = -(scale[..., None]*amplitude*np.sin(phase)*np.cos(beta)*np.sin(chi)).sum(-1)
    np.testing.assert_allclose(odd, odd_beta+odd_position, atol=1e-12, rtol=1e-12)
    heads = []
    for head in range(8):
        gp, gm = original[head][mask], mirror[head][mask]
        e, o = even[head][mask], odd[head][mask]
        bound = np.abs(coefficients[head]).sum(-1)[mask]
        counts = classify(gp, gm, bound)
        heads.append(dict(head_1based=head+1, pairs=int(mask.sum()), **counts,
                          odd_over_even_l2=float(np.linalg.norm(o)/np.linalg.norm(e)),
                          odd_energy_fraction=float(np.sum(o*o)/np.sum(o*o+e*e)),
                          odd_beta_l2=float(np.linalg.norm(odd_beta[head][mask])),
                          odd_position_l2=float(np.linalg.norm(odd_position[head][mask])),
                          both_directions_opposite_in_both_phase_orders=int((
                              (original[head]*original[head].T < 0) &
                              (mirror[head]*mirror[head].T < 0) &
                              (original[head]*mirror[head] < 0) & mask).sum())))
    selected = dict(puzzle=1, block=block, source=66, target=48,
                    phase=phase[:, 48, 66], amplitude=amplitude[:, 48, 66],
                    chi=chi[:, 48, 66], coefficients=coefficients[:, 48, 66],
                    agree=p["agree"][0, :, 48, 66].numpy(),
                    observed=original[:, 48, 66], mirrored=mirror[:, 48, 66])
    arrays = dict(original_G=original, mirrored_G=mirror, even_G=even, odd_G=odd,
                  odd_from_beta=odd_beta, odd_from_position=odd_position,
                  selected_coefficients=selected["coefficients"],
                  selected_phase=selected["phase"], selected_chi=selected["chi"],
                  selected_amplitude=selected["amplitude"], selected_agree=selected["agree"])
    return dict(block=block, analytical_vs_production_max_error=error,
                heads=heads), arrays, selected


def cached_case(root, puzzle, beta, gain):
    path = next(root.glob(f"case_{puzzle}_*/read_kernel_decomposition.json"))
    data = json.loads(path.read_text())
    normal = data["branches"]["normal"]
    kd = normal["kernel_decomposition"]
    timeline = next(r for r in normal["timeline"] if r["offset"] == 0)
    channels = kd["channels"]
    amplitude = np.array([r["pair_amplitude"] for r in channels])
    phase = np.deg2rad([r["content_phase_degrees"] for r in channels])
    chi = np.deg2rad([r["position_phase_degrees"] for r in channels])
    cbeta = np.deg2rad([r["beta_degrees"] for r in channels])
    head = data["head_1based"]-1
    np.testing.assert_allclose(np.exp(1j*cbeta), np.exp(1j*beta[head]), atol=1e-12)
    agree = timeline["agree"]
    scale = gain[head] * kd["distance_decay"] * agree
    coefficient = scale * amplitude * np.exp(1j*(beta[head]+chi))
    original = np.real((coefficient*np.exp(1j*phase)).sum())
    mirrored = np.real((coefficient*np.exp(-1j*phase)).sum())
    error = abs(original-timeline["G"])
    assert error < 3e-6, error
    return dict(puzzle=puzzle, source_rc=data["source_cell_rc"],
                target_rc=data["event"]["target_rc"],
                block=data["event"]["stable_complete_block"], head_1based=head+1,
                agree=agree, gain=float(gain[head]), distance_decay=kd["distance_decay"],
                phase=phase, amplitude=amplitude, chi=chi, beta=beta[head],
                coefficients=coefficient, observed=float(original), mirrored=float(mirrored),
                cached_G_reconstruction_error=float(error))


def summarize_case(case):
    c = case["coefficients"]
    angle = np.linspace(-np.pi, np.pi, 721)
    # A known one-dimensional phase cross-section, NOT a time-delay curve.
    curve = np.real(c.sum() * np.exp(1j*angle))
    sample_degrees = [0, 15, 30, 45, 60, 90, 120, 150, 180]
    samples = []
    for degree in sample_degrees:
        d = np.deg2rad(degree)
        samples.append(dict(relative_phase_degrees=degree,
                            G_positive=float(np.real(c.sum()*np.exp(1j*d))),
                            G_negative=float(np.real(c.sum()*np.exp(-1j*d)))))
    clean = {k: v for k, v in case.items()
             if k not in {"phase", "amplitude", "chi", "beta", "coefficients"}}
    clean.update(observed_phase_even=(case["observed"]+case["mirrored"])/2,
                 observed_phase_odd=(case["observed"]-case["mirrored"])/2,
                 common_angle_cos_coefficient=float(c.sum().real),
                 common_angle_sin_coefficient=float(-c.sum().imag),
                 common_angle_samples=samples,
                 fixed_position_channel_odd_coefficient_l2=float(np.linalg.norm(c.imag)),
                 fixed_position_channel_even_coefficient_l2=float(np.linalg.norm(c.real)))
    scale = case["gain"]*case["distance_decay"]*case["agree"]
    amp, chi, beta, phase = (case[k] for k in ("amplitude", "chi", "beta", "phase"))
    clean["observed_phase_odd_from_beta"] = float(-scale*np.sum(amp*np.sin(phase)*np.sin(beta)*np.cos(chi)))
    clean["observed_phase_odd_from_position"] = float(-scale*np.sum(amp*np.sin(phase)*np.cos(beta)*np.sin(chi)))
    reverse_c = scale*amp*np.exp(1j*(beta-chi))
    clean["same_physical_plus90_phase_G_forward"] = float(np.real(c.sum()*1j))
    clean["same_physical_plus90_phase_G_reverse"] = float(np.real(reverse_c.sum()*-1j))
    np.testing.assert_allclose(clean["observed_phase_odd"],
                               clean["observed_phase_odd_from_beta"]+clean["observed_phase_odd_from_position"], atol=1e-12)
    return clean, angle, curve


def plot(out, cases, snapshots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for row, case in enumerate(cases):
        c = case["coefficients"]
        angle = np.linspace(-np.pi, np.pi, 721)
        ax = axes[row, 0]
        ax.plot(np.rad2deg(angle), np.real(c.sum()*np.exp(1j*angle)), label="G(phi)")
        ax.plot(np.rad2deg(angle), np.real(c.sum()*np.exp(-1j*angle)), "--", label="G(-phi)")
        ax.axhline(0, color="black", lw=.6)
        ax.axvline(0, color="black", lw=.6)
        ax.set(xlabel="Common relative address phase (degrees; NOT block delay)",
               ylabel="G, values / amplitudes / positions fixed",
               title=f"Puzzle {case['puzzle']}, head {case['head_1based']}, "
                     f"r{case['source_rc'][0]}c{case['source_rc'][1]} -> "
                     f"r{case['target_rc'][0]}c{case['target_rc'][1]}")
        ax.legend(fontsize=8)
        channel = np.real(c[:, None]*np.exp(1j*angle[None]))
        vmax = np.max(np.abs(channel))
        im = axes[row, 1].imshow(channel, aspect="auto", origin="lower",
                                extent=[-180, 180, .5, 52.5], cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
        axes[row, 1].set(xlabel="This channel's relative phase (degrees)",
                         ylabel="Address channel", title="Individual channel contributions to G")
        fig.colorbar(im, ax=axes[row, 1], shrink=.8)
    fig.savefig(out/"phase_response.png", dpi=160)
    fig.savefig(out/"phase_response.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for s in snapshots:
        ax.plot(range(1, 9), [h["odd_energy_fraction"] for h in s["heads"]],
                marker="o", label=f"Puzzle 1 after block {s['block']}")
    ax.set(xlabel="Head", ylabel="Odd / (even + odd) squared response",
           title="Intrinsic address-phase reversal, spatial phase fixed", ylim=(0, 1))
    ax.legend()
    fig.savefig(out/"phase_parity_heads.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/phase_timing_v11"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        checkpoint="checkpoints/v1.1_step160000.npz", precision="CPU float64; cached FP32 cases",
        primary="Keep positions, beta, gain, amplitudes and all values/agree fixed; conjugate the unpositioned addresses. Compare G(delta) with G(-delta) on the same directed edges.",
        equation="G(delta)=g*D*agree*sum_j(r_tj*r_nj*cos(delta_j+beta_j+theta_j.dot(pos_t-pos_n)))",
        observable="The signed write target G; eta is positive and does not change its sign. No leak, reading or subsequent feedback in this measurement.",
        snapshots="Puzzle 1, after blocks 152 and 192, all 8 heads, all 6480 directed off-diagonal edges",
        cases="Pre-existing selected edges in late puzzles 58, 209, 230; no reselection based on phase result",
        curves="Complete channelwise phase response and a clearly labeled common-angle cross-section. The latter is not a temporal-frequency assignment.",
        sign_tolerance="1e-9 * max(1, absolute coefficient sum)",
        temporal_scope="No omega_j is assumed; a phase-mirror is not automatically a reversal of actual event time. Temporal identification is a separate part of this research.",
    )
    dump(out/"phase_protocol.json", protocol)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    model, _ = load_cpu()
    blocks = CPUBlocks(model, puzzle_batch(1))
    beta = blocks.layer.beta.numpy()
    gain = blocks.gain.numpy().reshape(-1)
    results = []
    snapshots = load_snapshots()
    for block in (152, 192):
        result, arrays, _ = analyze_snapshot(blocks, snapshots[block], block)
        np.savez_compressed(out/f"phase_snapshot_{block}.npz", **arrays)
        results.append(result)
        print(json.dumps(result), flush=True)
    root = Path("runs/late_puzzle_probe_v11")
    cases = [cached_case(root, puzzle, beta, gain) for puzzle in (58, 209, 230)]
    case_reports = []
    for case in cases:
        report, angle, curve = summarize_case(case)
        case_reports.append(report)
        np.savez_compressed(out/f"phase_case_{case['puzzle']}.npz",
                            coefficients=case["coefficients"], observed_phase=case["phase"],
                            amplitude=case["amplitude"], position_phase=case["chi"],
                            beta=case["beta"], phase_grid=angle, common_angle_G=curve)
        print(json.dumps(report), flush=True)
    report = dict(protocol=protocol, snapshots=results, cases=case_reports,
                  checkpoint_sha256=hashlib.sha256(Path(protocol["checkpoint"]).read_bytes()).hexdigest())
    dump(out/"phase_summary.json", report)
    plot(out, cases, results)


if __name__ == "__main__":
    main()
