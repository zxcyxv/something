"""CPU validation of the even-window architectural hypothesis.

Checks the production read/normalization equations, multi-block equivalence to
symmetric plastic transmission, episode reset, packed-state gradients, and a
small end-to-end training smoke run. No checkpoint accuracy claim is made.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import train
from even_plasticity import EvenLT, convert_memory, from_v11
from research_cpu import CPUBlocks, load_cpu, load_snapshots, puzzle_batch


def conversion_check(model, snapshots):
    blocks = CPUBlocks(model, puzzle_batch())
    candidate = from_v11(model).requires_grad_(False)
    inner, layer = candidate.inner, candidate.inner.layers[0]
    AB, kc = inner.W_C(layer), inner.kernel(layer)
    results = {}
    for k in (152, 192):
        h, w = (x.clone() for x in snapshots[k])
        hc = h.clone()
        pc = convert_memory(model, candidate, w)
        errors, target_errors = [], []
        for step in range(8):
            parts = blocks.parts(h)
            wn = blocks.rho*w+blocks.eta*parts["G"]
            hs = blocks.transmit_parts(parts, (wn+wn.transpose(-1, -2))/2)
            qc = inner.boundary(layer, hc)+blocks.inj
            u = inner.addr(qc, AB)
            v = torch.einsum("btd,hcd->bthc", qc, layer.w_sh)
            vn = v/(v.norm(dim=-1, keepdim=True)+inner.config.eps)
            Gc = inner.write_target(layer, u, vn, kc[0])
            Gs = blocks.lam*(parts["G"]+parts["G"].transpose(-1, -2))/2
            target_errors.append(float((Gc-Gs).abs().max()))
            hc, pc, _ = inner.step(layer, qc, AB, kc, pc)
            expected_memory = blocks.lam*(wn+wn.transpose(-1, -2))/2
            torch.testing.assert_close(inner.unpack(pc), expected_memory, atol=2e-11, rtol=2e-11)
            torch.testing.assert_close(hc, hs, atol=2e-10, rtol=2e-10)
            errors.append(float((hc-hs).abs().max()))
            h, w = hs, wn
        results[str(k)] = dict(blocks=8, hidden_max_error=max(errors),
                               write_target_max_error=max(target_errors))
    results["state_elements"] = dict(original=int(w.numel()), packed=int(pc.numel()),
                                      fraction=pc.numel()/w.numel())
    results["parameter_counts"] = dict(original=sum(p.numel() for p in model.parameters()),
                                        candidate=sum(p.numel() for p in candidate.parameters()))
    # Nonzero psi remains capable of asymmetric instantaneous transmission.
    results["retained_fast_kernel_skew_norm"] = float(
        (blocks.parts(snapshots[152][0])["a"]-
         blocks.parts(snapshots[152][0])["a"].transpose(-1, -2)).norm()/2)
    return results


def training_smoke():
    torch.manual_seed(71)
    cfg = dict(train.CFG, batch_size=2, seq_len=9, grid=3, hidden_size=32,
               num_heads=2, num_layers=1, num_puzzle_identifiers=1,
               puzzle_emb_ndim=0, legacy_gauge=False, use_trace=False,
               block_order="pre", amp=False, blocks_per_seg=3, loops=2)
    model = EvenLT(cfg).double()
    batch = dict(inputs=torch.randint(1, 11, (2, 9)),
                 labels=torch.randint(1, 11, (2, 9)),
                 puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
    def loss():
        carry = model.initial_carry(batch)
        carry, out = model(carry, batch)
        return F.cross_entropy(out["logits"].reshape(-1, 11), batch["labels"].flatten()), carry
    before, carry = loss()
    before.backward()
    norms = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
    for suffix in ("plastic_spectrum", "base_raw", "eta_raw", "psi", "w_sh", "wc_raw"):
        matching = [value for name, value in norms.items() if name.endswith(suffix)]
        if not matching or not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in matching):
            raise AssertionError(f"missing/invalid end-to-end gradient for {suffix}: {matching}")
    # A directional finite difference crosses pack/unpack, the recurrent write,
    # and the final classifier; it is not a test of a copied implementation.
    param = model.inner.layers[0].plastic_spectrum
    direction = torch.randn_like(param)
    direction /= direction.norm()
    analytic = float((param.grad*direction).sum())
    epsilon = 1e-5
    with torch.no_grad():
        saved = param.clone()
        param.copy_(saved+epsilon*direction)
        plus = float(loss()[0])
        param.copy_(saved-epsilon*direction)
        minus = float(loss()[0])
        param.copy_(saved)
    numerical = (plus-minus)/(2*epsilon)
    torch.testing.assert_close(torch.tensor(analytic), torch.tensor(numerical), atol=2e-6, rtol=2e-5)
    opt = torch.optim.SGD(model.parameters(), lr=.02)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        value, _ = loss()
        value.backward()
        opt.step()
    after, _ = loss()
    if not float(after.detach()) < float(before.detach()):
        raise AssertionError("tiny fitting smoke run did not lower its loss")
    # A halted episode must reset both h and the packed plastic memory.
    carry.halted.fill_(True)
    reset, out_reset = model(carry, batch)
    clean, out_clean = model(model.initial_carry(batch), batch)
    torch.testing.assert_close(reset.coupling, clean.coupling)
    torch.testing.assert_close(out_reset["logits"], out_clean["logits"])
    return dict(initial_loss=float(before.detach()), final_loss=float(after.detach()),
                optimizer_steps=20, directional_gradient=analytic,
                finite_difference=numerical, gradient_abs_error=abs(analytic-numerical),
                gradient_norms=norms, episode_reset_equal=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/even_plasticity_v11")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, _ = load_cpu()
    with torch.no_grad():
        report = dict(conversion=conversion_check(model, load_snapshots()))
    report["training_smoke"] = training_smoke()
    report["cuda_initialized"] = torch.cuda.is_initialized()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
