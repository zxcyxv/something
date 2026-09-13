"""Verify a phase-population plasticity construction and its structural predictions.

Run on CPU. Quadrature, an explicitly lifted activity, the production kernel,
and exact finite-history sums are independent checks of the stated identities.
This is not an accuracy or biological-equivalence experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_cpu import CPUBlocks, load_cpu, load_snapshots, puzzle_batch


def phase_quadrature(seed=0):
    rng = np.random.default_rng(seed)
    n, frequencies, contents, samples = 4, 3, 5, 32
    z = rng.normal(size=(n, frequencies))+1j*rng.normal(size=(n, frequencies))
    v = rng.normal(size=(n, contents))
    beta = rng.uniform(-np.pi, np.pi, frequencies)
    angles = 2*np.pi*np.arange(samples)/samples
    # i = target, n = source. Phase angle is an auxiliary cycle coordinate,
    # not the recurrent block index or an observed spike time.
    activity = np.sqrt(2)*np.real(z[:, :, None]*np.exp(1j*angles))[:, :, None, :]*v[:, None, :, None]
    window = 2*np.cos(angles[None, None, :]-angles[None, :, None]+beta[:, None, None])
    integrated = np.einsum("ipca,pab,npcb->in", activity, window, activity,
                           optimize=True)/samples**2
    kernel = np.real(np.einsum("ip,p,np->in", z, np.exp(1j*beta), z.conj()))
    direct = kernel*(v@v.T)
    X = z[:, :, None]*v[:, None, :]
    lifted = np.real(np.einsum("ipc,p,npc->in", X, np.exp(1j*beta), X.conj()))
    np.testing.assert_allclose(integrated, direct, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(lifted, direct, atol=2e-13, rtol=2e-13)
    return dict(quadrature_max_error=float(np.max(abs(integrated-direct))),
                lifted_max_error=float(np.max(abs(lifted-direct))))


def double_window_check():
    # A pair at (r,s) is counted by every overlapping inner window ending at k.
    T, r, s, gamma, rho, eta = 11, 2, 5, .8, .93, .07
    m = max(r, s)
    explicit = sum(eta*rho**(T-k)*gamma**(k-r)*gamma**(k-s)
                   for k in range(m, T+1))
    formula = eta*gamma**abs(r-s)*rho**(T-m)*sum(
        (gamma**2/rho)**j for j in range(T-m+1))
    single_pair_once = eta*rho**(T-m)
    np.testing.assert_allclose(explicit, formula, atol=1e-15)
    return dict(overlapping_windows_pair_weight=explicit, closed_form=formula,
                single_pair_once_weight=single_pair_once,
                ratio_to_single_count=explicit/single_pair_once)


def even_window_learning_geometry():
    # For a purely even target kernel, a beta=0 model can be stationary in
    # beta and common gain while still having a nonzero gradient in individual
    # real spectral amplitudes. This checks a representation property, not a
    # claim about the Sudoku training objective or its actual beta values.
    gen = torch.Generator().manual_seed(29)
    z = torch.complex(torch.randn(7, 3, generator=gen, dtype=torch.float64),
                      torch.randn(7, 3, generator=gen, dtype=torch.float64))
    v = torch.randn(7, 4, generator=gen, dtype=torch.float64)
    agree = v@v.T
    cross = torch.einsum("ip,np->pin", z, z.conj())*agree
    target_coefficients = torch.tensor([.25, 1.3, -.4], dtype=torch.float64)
    target = torch.einsum("p,pin->in", target_coefficients, cross.real)
    common = cross.real.sum(0)
    best_gain = float((common*target).sum()/common.square().sum())
    with torch.enable_grad():
        beta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        gain = torch.tensor(best_gain, dtype=torch.float64, requires_grad=True)
        pred = gain*torch.einsum("p,pin->in", torch.exp(1j*beta), cross).real
        angle_loss = (pred-target).square().mean()
        beta_grad, gain_grad = torch.autograd.grad(angle_loss, (beta, gain))
        c = torch.full((3,), best_gain, dtype=torch.float64, requires_grad=True)
        direct = torch.einsum("p,pin->in", c, cross.real)
        direct_loss = (direct-target).square().mean()
        c_grad, = torch.autograd.grad(direct_loss, (c,))
    torch.testing.assert_close(pred, direct, atol=2e-14, rtol=2e-14)
    assert beta_grad.norm() < 1e-12 and gain_grad.abs() < 1e-12 and c_grad.norm() > .01
    return dict(common_gain=best_gain, same_initial_loss=float(direct_loss.detach()),
                angle_gradient_norm=float(beta_grad.norm()), gain_gradient=float(gain_grad),
                real_spectrum_gradient_norm=float(c_grad.norm()),
                target_coefficients=target_coefficients.tolist(),
                scope="synthetic symmetric-kernel fitting objective, not checkpoint training")


def inspect_checkpoint(blocks, snapshots):
    layer, inner = blocks.layer, blocks.inner
    position = layer.theta[..., 0, None]*inner.pos_u+layer.theta[..., 1, None]*inner.pos_w
    rotation = torch.exp(1j*position.permute(2, 0, 1))
    report = dict(beta_cos_min_by_head=layer.beta.cos().amin(-1).tolist(), states={})
    for k in (152, 192, 256):
        h, memory = snapshots[k]
        parts = blocks.parts(h)
        z = torch.complex(*parts["u"])[0]*rotation
        v = parts["vn"][0]
        results = []
        for head in range(inner.H):
            # Lift only one head at a time. No T*T*p*c intermediate.
            X = z[:, head, :, None]*v[:, head, None, :]
            lifted = torch.einsum("ipc,p,npc->in", X, torch.exp(1j*layer.beta[head]), X.conj()).real
            distance = blocks.kcb[0][head]
            actual = parts["window"][0, head]*parts["agree"][0, head]
            error = float((distance*lifted-actual).abs().max())
            torch.testing.assert_close(distance*lifted, actual, atol=3e-13, rtol=3e-13)
            zero = distance*torch.einsum("ipc,npc->in", X, X.conj()).real
            beta_sym = (actual+actual.T)/2
            memory_sym = (memory[0, head]+memory[0, head].T)/2
            expected_diag = z[:, head].abs().square().sum(-1)*v[:, head].square().sum(-1)
            torch.testing.assert_close(zero.diag(), expected_diag, atol=3e-13, rtol=3e-13)
            results.append(dict(lifted_kernel_max_error=error,
                                zero_beta_min_eigenvalue=float(torch.linalg.eigvalsh(zero).min()),
                                actual_write_sym_min_eigenvalue=float(torch.linalg.eigvalsh(beta_sym).min()),
                                actual_memory_sym_min_eigenvalue=float(torch.linalg.eigvalsh(memory_sym).min()),
                                zero_beta_diag_max_deviation_from_one=float((zero.diag()-1).abs().max()),
                                zero_beta_diag_spread=float(zero.diag().max()-zero.diag().min()),
                                actual_memory_diag_relative_spread=float(memory_sym.diag().std()/memory_sym.diag().mean().abs()),
                                actual_memory_negative_offdiag_fraction=float((memory_sym[~torch.eye(inner.config.seq_len, dtype=torch.bool)]<0).double().mean())))
        report["states"][str(k)] = results
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/phase_activity_theory_v11")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    report = dict(quadrature=phase_quadrature(), double_window=double_window_check(),
                  even_window_learning_geometry=even_window_learning_geometry())
    model, meta = load_cpu()
    blocks = CPUBlocks(model, puzzle_batch())
    report["checkpoint_step"] = meta["step"]
    report["checkpoint"] = inspect_checkpoint(blocks, load_snapshots())
    report["cuda_initialized"] = torch.cuda.is_initialized()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
