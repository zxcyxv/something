"""Test a necessary condition for a shared-scalar temporal trace in v1.1.

At fixed h, the current raw address is fixed.  If, throughout a neighborhood,
z_next = Lambda z + b x(h, w), with real scalar x and fixed complex b, then
the real Jacobian d z_next / d w has rank at most one.  No x or Lambda is fit.

Use the complete orthonormal basis of symmetric, off-diagonal memory changes
incident to one cell, over all heads.  Differentiate the real model in FP64
and independently check the result with full-block JVPs and finite differences.
This is a test of a state-space identity, not of reachability on a natural
trajectory, arbitrary latent coordinates, or general multivariate histories.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev, jvp

from research_cpu import CPUBlocks, load_cpu, load_snapshots, puzzle_batch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectrum(matrix):
    s = torch.linalg.svdvals(matrix)
    energy = s.square().sum()
    return dict(
        singular_values=s.tolist(),
        sigma_2_over_sigma_1=float(s[1] / s[0]),
        rank1_energy_fraction=float(s[0].square() / energy),
        rank1_relative_frobenius_error=float((s[1:].square().sum() / energy).sqrt()),
        ranks_by_relative_threshold={str(t): int((s > t * s[0]).sum())
                                     for t in (1e-2, 1e-3, 1e-6, 1e-10)},
    )


def addresses(blocks, h, cell):
    x, y = blocks.inner.addr_raw(blocks.prepare(h), blocks.ab)
    ux, uy = blocks.inner._unit(x, y)
    return (torch.cat((x[0, cell], y[0, cell]), dim=-1),
            torch.cat((ux[0, cell], uy[0, cell]), dim=-1))


def memory_direction(coefficients, template, cell, sources):
    heads = template.shape[1]
    values = coefficients.reshape(heads, len(sources)) / math.sqrt(2)
    direction = torch.zeros_like(template)
    direction[0, :, cell, sources] = values
    direction[0, :, sources, cell] = values
    torch.testing.assert_close(direction.norm(), coefficients.norm(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(direction, direction.transpose(-1, -2), atol=0, rtol=0)
    assert torch.count_nonzero(direction.diagonal(dim1=-2, dim2=-1)) == 0
    return direction


def analyze(blocks, h, memory, start, cell, reference_predictions):
    tic = time.monotonic()
    inner, layer = blocks.inner, blocks.layer
    heads, tokens = memory.shape[1:3]
    sources = [n for n in range(tokens) if n != cell]
    parts = blocks.parts(h)
    memory_next = blocks.rho * memory + blocks.eta * parts["G"]
    effective = (1 - blocks.lam) * parts["a"] + blocks.lam * memory_next
    message = torch.einsum("bhtn,bnhc->bthc", effective, parts["v"])
    message = torch.einsum("bthc,hcd->btd", message, layer.w_sh)
    pre = parts["q"] + message
    h_next = inner.phi(pre)

    # Compare the factored implementation to the production one-block step.
    production_h, production_w, _ = inner.step(
        layer, parts["q"], blocks.ab, blocks.kc,
        w=memory, kcb=blocks.kcb)
    torch.testing.assert_close(h_next, production_h, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(memory_next, production_w, atol=2e-11, rtol=2e-11)
    baseline = torch.stack((inner.w_cls(h).argmax(-1)[0],
                            inner.w_cls(h_next).argmax(-1)[0])).numpy()
    disagreements = int((baseline != reference_predictions[start:start + 2]).sum())
    assert disagreements == 0, "FP64 one-block replay changed baseline predictions"

    # With h fixed, G, a_psi, v, and q are fixed; only the old-memory term moves.
    # Each column is (E_h,i,n + E_h,n,i)/sqrt(2).  The reverse edge affects
    # another cell and cannot reach this cell until a later block.
    lift = torch.einsum("nhc,hcd->hnd", parts["v"][0, sources], layer.w_sh)
    lift = lift * (blocks.rho * blocks.lam).reshape(heads, 1, 1) / math.sqrt(2)
    lift = lift.reshape(heads * len(sources), inner.d).T

    def raw_from_pre(local_pre):
        local_h = inner.phi(local_pre)
        next_q = inner.boundary(layer, local_h) + blocks.inj[0, cell]
        real = torch.einsum("hjd,d->hj", blocks.ab[0], next_q)
        imag = torch.einsum("hjd,d->hj", blocks.ab[1], next_q)
        return torch.cat((real, imag), dim=-1)

    raw, unit = addresses(blocks, h_next, cell)
    torch.testing.assert_close(raw_from_pre(pre[0, cell]), raw, atol=2e-11, rtol=2e-11)
    local_jacobian = jacrev(raw_from_pre, chunk_size=104)(pre[0, cell])
    jac_raw = local_jacobian @ lift
    radius = raw.norm(dim=-1, keepdim=True)
    eps = inner.config.eps
    radial = torch.einsum("hi,him->hm", raw, jac_raw)
    jac_unit = (jac_raw / (radius + eps)[..., None]
                - raw[..., None] * radial[:, None, :]
                / (radius * (radius + eps).square())[..., None])

    head_results = []
    for head in range(heads):
        own = slice(head * len(sources), (head + 1) * len(sources))
        head_results.append(dict(
            head_zero_based=head,
            raw_all_memory_heads=spectrum(jac_raw[head]),
            unit_all_memory_heads=spectrum(jac_unit[head]),
            raw_same_memory_head=spectrum(jac_raw[head, :, own]),
            unit_same_memory_head=spectrum(jac_unit[head, :, own]),
        ))

    # Use the two leading right singular vectors as explicit independent
    # witnesses, plus a fixed single edge to validate the symmetric lift.
    witness_head = 2
    _, _, right = torch.linalg.svd(jac_raw[witness_head], full_matrices=False)
    edge = torch.zeros_like(right[0])
    edge[witness_head * len(sources) + sources.index(66)] = 1
    directions = [("top_singular_1", right[0]),
                  ("top_singular_2", right[1]), ("head_2_source_66", edge)]

    def actual_next_addresses(w):
        nh, _ = blocks.block(h, w)
        return addresses(blocks, nh, cell)

    checks = []
    witness_arrays = {}
    for name, coefficient in directions:
        direction = memory_direction(coefficient, memory, cell, sources)
        _, actual_tangent = jvp(actual_next_addresses, (memory,), (direction,))
        calculated = (jac_raw @ coefficient, jac_unit @ coefficient)
        jvp_errors = {}
        for kind, found, expected in zip(("raw", "unit"), calculated, actual_tangent):
            torch.testing.assert_close(found, expected, atol=2e-10, rtol=2e-10)
            jvp_errors[kind] = float((found - expected).norm() / expected.norm())
        finite = []
        for step in (1e-4, 5e-5):
            plus = actual_next_addresses(memory + step * direction)
            minus = actual_next_addresses(memory - step * direction)
            row = dict(epsilon=step)
            for kind, positive, negative, expected in zip(
                    ("raw", "unit"), plus, minus, calculated):
                fd = (positive - negative) / (2 * step)
                relative = float((fd - expected).norm() / expected.norm())
                row[kind + "_relative_error"] = relative
                assert relative < 1e-5, (start, name, kind, step, relative)
            finite.append(row)
        checks.append(dict(direction=name, memory_frobenius_norm=float(direction.norm()),
                           full_block_jvp_relative_errors=jvp_errors,
                           central_finite_differences=finite))
        witness_arrays[name + "_coefficients"] = coefficient.numpy()
        witness_arrays[name + "_raw_response"] = calculated[0].numpy()
        witness_arrays[name + "_unit_response"] = calculated[1].numpy()

    targets = blocks.batch["labels"][0]
    result = dict(
        snapshot_after_block=start, next_block=start + 1,
        target_cell_zero_based=cell, target_rc_one_based=[cell // 9 + 1, cell % 9 + 1],
        current_wrong_cells=int((torch.from_numpy(baseline[0]) != targets).sum()),
        current_target_digit=int(baseline[0, cell]) - 1,
        target_gold_digit=int(targets[cell]) - 1,
        current_address_depends_on_memory=False,
        basis_columns=len(sources) * heads,
        real_output_dimensions_per_head=int(raw.shape[-1]),
        baseline_prediction_disagreements=disagreements,
        production_hidden_max_absolute_error=float((h_next - production_h).abs().max()),
        production_memory_max_absolute_error=float((memory_next - production_w).abs().max()),
        heads=head_results, derivative_checks=checks,
        witness_head_zero_based=witness_head, elapsed_seconds=time.monotonic() - tic,
    )
    arrays = dict(jac_raw=jac_raw.numpy(), jac_unit=jac_unit.numpy(),
                  current_raw=addresses(blocks, h, cell)[0].numpy(),
                  next_raw=raw.numpy(), next_unit=unit.numpy(),
                  sources=np.asarray(sources), **witness_arrays)
    print(json.dumps(dict(
        block=start, seconds=result["elapsed_seconds"],
        sigma2_over_sigma1=[r["raw_all_memory_heads"]["sigma_2_over_sigma_1"] for r in head_results],
        rank_at_1e6=[r["raw_all_memory_heads"]["ranks_by_relative_threshold"]["1e-06"]
                     for r in head_results],
        baseline_disagreements=disagreements)), flush=True)
    return result, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    ap.add_argument("--out", default="runs/address_trace_rank_v11")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        checkpoint=args.checkpoint, checkpoint_sha256=sha256(args.checkpoint),
        puzzle_test_index=1, target_cell_zero_based=48,
        snapshots_after_blocks=[152, 192], precision="FP64 CPU",
        hypothesis="raw address z_next = Lambda z + b x(h,w), one real scalar x per cell/head, fixed Lambda and b, throughout an open neighborhood",
        necessary_condition="at fixed h, rank_R(d z_next / d w) <= 1",
        interventions="all 640 orthonormal symmetric off-diagonal incident-memory directions, 80 per head; current h fixed",
        scope="Native address coordinates and a state-space identity. Does not rule out restricted natural trajectories, approximate filters, arbitrary nonlinear latent coordinates, or multivariate activity histories.",
        note_on_normalization="Raw address is primary; unit address is an additional check. Coordinate order is real[0:p], imag[0:p].",
        rank_thresholds_relative_to_sigma1=[1e-2, 1e-3, 1e-6, 1e-10],
        finite_difference_epsilons=[1e-4, 5e-5],
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model, _ = load_cpu(args.checkpoint)
    blocks = CPUBlocks(model, puzzle_batch(1))
    assert blocks.inner.config.stdp_gain_fixed < 0 and blocks.inner.config.stdp_lam_fixed < 0
    snapshots = load_snapshots()
    with np.load("runs/relation_transport_v11/normal.npz") as reference:
        predictions = reference["pred"][:, 1].copy()
    results, arrays = [], {}
    for start in protocol["snapshots_after_blocks"]:
        result, values = analyze(blocks, *snapshots[start], start, 48, predictions)
        results.append(result)
        arrays.update({f"block_{start}_{k}": v for k, v in values.items()})
    report = dict(protocol=protocol, states=results,
                  torch_version=torch.__version__, cuda_initialized=torch.cuda.is_initialized())
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "jacobians.npz", **arrays)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
