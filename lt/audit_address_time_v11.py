"""FP64 audit and an exact structural recurrence for the native address.

q_k=h_k+MLP(h_k)+I, u_k=C q_k, h_{k+1}=s_k(q_k+f_k) imply
u_{k+1}=s_k u_k+s_k C f_k+C MLP(h_{k+1})+C I.
The explicit carry term has positive real gain, with no complex rotation.
This identity alone does NOT rule out rotation learned inside the other paths.
"""

import json
from pathlib import Path

import numpy as np
import torch

from analyze_late_puzzles import restore
from probe_late_puzzles import EventBlocks, setup
from research_cpu import CPUBlocks, load_cpu, puzzle_batch
from probe_projective_time_v11 import evaluate
from probe_multi_drive_time_v11 import test


def main():
    root = Path("runs/phase_timing_v11")
    out = root/"fp64_audit"
    out.mkdir(parents=True, exist_ok=True)
    old = Path("runs/late_puzzle_probe_v11")
    with np.load(old/"baseline.npz") as archive:
        ids, reference = archive["indices"], archive["predictions"]
    events = json.loads((old/"events.json").read_text())["selected_events"]
    gpu_model, gpu_batch, _, _ = setup(ids)
    gb = EventBlocks(gpu_model, gpu_batch)
    states = []
    for event in events:
        start = event["stable_complete_block"]-128
        h, w = restore(old, gb, start, reference)
        idx = event["batch_index"]
        states.append((event, start, h[idx:idx+1].cpu().double(), w[idx:idx+1].cpu().double()))
    del h, w, gb, gpu_model, gpu_batch
    torch.cuda.empty_cache()
    torch.set_num_threads(2)
    model, _ = load_cpu()
    torch.set_grad_enabled(False)
    report = dict(
        precision="FP64 replay from matched FP32 snapshots; model weights unchanged",
        recurrence="u_next=s*u+s*C*f+C*MLP(h_next)+C*I; s=1/sqrt(1+||q+f||^2/d)",
        caveat="Positive real explicit carry is not a proof against an effective rotation encoded by learned feedback. The rank tests evaluate that separate hypothesis.",
        cases=[])
    for event, start, h, w in states:
        puzzle, cell = event["puzzle"], event["cell"]
        b = CPUBlocks(model, puzzle_batch(puzzle))
        addresses, scales, component_norms = [], [], []
        max_error = 0.
        predictions_differ = 0
        for tick in range(256):
            p = b.parts(h)
            ux, uy = b.inner.addr_raw(p["q"], b.ab)
            z = torch.complex(ux, uy)
            addresses.append(z[0, cell].numpy().copy())
            wn = b.rho*w+b.eta*p["G"]
            effective = (1-b.lam)*p["a"]+b.lam*wn
            f = torch.einsum("bhtn,bnhc->bthc", effective, p["v"])
            f = torch.einsum("bthc,hcd->btd", f, b.layer.w_sh)
            pre = p["q"]+f
            s = torch.rsqrt(1+pre.square().sum(-1)/b.inner.d)
            hn = b.inner.phi(pre)

            def project(value):
                x, y = b.inner.addr_raw(value, b.ab)
                return torch.complex(x, y)

            terms = [s[..., None, None]*z,
                     s[..., None, None]*project(f),
                     project(b.inner.boundary(b.layer, hn)-hn),
                     project(b.inj)]
            nxt = project(b.prepare(hn))
            error = float((nxt-sum(terms)).abs().max())
            max_error = max(max_error, error)
            assert error < 1e-10, error
            scales.append(float(s[0, cell]))
            component_norms.append([term[0, cell].abs().square().sum(-1).sqrt().numpy() for term in terms])
            got = b.inner.w_cls(hn).argmax(-1)[0].numpy()
            predictions_differ += int(np.count_nonzero(got != reference[start+tick, event["batch_index"]]))
            h, w = hn, wn
        z = np.stack(addresses)
        np.savez_compressed(out/f"puzzle_{puzzle}.npz", raw_address=z,
                            carry_scale=np.array(scales), component_norms=np.asarray(component_norms))
        heads = []
        multi = []
        for head in range(8):
            row, _ = evaluate(z[:, head], storage_epsilon=np.finfo(np.float64).eps)
            row["head_1based"] = head+1
            heads.append(row)
            for m in (4, 6, 10):
                for first in (0, 17, 35):
                    row, _ = test(z[:, head, first:first+m], storage_epsilon=np.finfo(np.float64).eps)
                    row.update(head_1based=head+1, first_channel_zero_based=first)
                    multi.append(row)
        item = dict(puzzle=puzzle, cell=cell, start_block=start,
                    recurrence_max_absolute_error=max_error,
                    positive_real_carry_range=[min(scales), max(scales)],
                    FP64_vs_FP32_prediction_disagreements=predictions_differ,
                    projective_heads=heads, multi_drive=multi)
        report["cases"].append(item)
        print(json.dumps(dict(puzzle=puzzle, error=max_error, carry=item["positive_real_carry_range"],
                              prediction_disagreements=predictions_differ,
                              rank6_violations=sum(r["triples_violating_necessary_identity"] for r in heads),
                              multi_violations={m:sum(r["violates_necessary_identity"] for r in multi if r["channels"]==m) for m in (4,6,10)})), flush=True)
        (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    eta = b.eta.numpy().reshape(-1)
    report["outer_memory"] = dict(eta=eta.tolist(),
        half_life_blocks=(np.log(.5)/np.log1p(-eta)).tolist(),
        alternating_signal_gain=(eta/(2-eta)).tolist(),
        constant_signal_gain=[1.]*len(eta))
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
