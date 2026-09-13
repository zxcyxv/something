"""Resolve an actual symmetric memory correction into its past write episodes.

The original batch-128 FP32 trajectory is captured under a 1 GiB allocator cap.
All subsequent temporal attribution and one-step interventions run on CPU.
Replacing an episode's writes by the current write preserves its EMA mass.
Decoded candidates are diagnostic labels on a trajectory, not inference rules
or controlled assignments of a hypothesis to the hidden state.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train
from analyze_memory_correction import CorrectionBlocks
from ckpt_npz import load_data, load_lt


class CaptureBlocks(CorrectionBlocks):
    def components_with_row(self, h, w, puzzle, cell):
        q = self.prepare(h)
        u = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(u, self.kc)
        v = torch.einsum("btd,hcd->bthc", q, self.layer.w_sh)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.inner.config.eps)
        agree = torch.einsum("bthc,bnhc->bhtn", vv, vv)
        window = self.inner.attn_xy(u, self.kcb)
        target = self.gain * (window * agree)
        wn = target if w is None else (1-self.eta)*w+self.eta*target
        def symmetric_row(x):
            return (x[puzzle, :, cell, :] + x[puzzle, :, :, cell])/2
        row = dict(agree=agree[puzzle, :, cell].cpu(),
                   phase=(self.gain[:, 0, 0, None]*symmetric_row(window)).cpu(),
                   write=symmetric_row(target).cpu(), memory=symmetric_row(wn).cpu())
        return (q, a, v, target, wn), row


def capture(args):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = 1024**3
    torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
    model, _, _ = load_lt(args.checkpoint, mod=train, batch_size=128, loops=129, amp=False)
    _, _, batch = load_data(n=128)
    b = CaptureBlocks(model, batch)
    normal = np.load(Path(args.root)/"normal.npz")["pred"]
    h = b.inner.init_hidden.expand(128, 81, -1).clone()
    w = None
    rows, states = [], {}
    for k in range(1, 161):
        parts, row = b.components_with_row(h, w, args.puzzle, args.cell)
        q, a, v, target, wn = parts
        effective = (1-b.lam)*a+b.lam*wn
        o = torch.einsum("bhtn,bnhc->bthc", effective, v)
        pre = q+torch.einsum("bthc,hcd->btd", o, b.layer.w_sh)
        hn = b.inner.phi(pre)
        pred = b.inner.w_cls(hn).argmax(-1).cpu().numpy()
        np.testing.assert_array_equal(pred, normal[k])
        row.update(before=torch.from_numpy(normal[k-1, args.puzzle].copy()),
                   after=torch.from_numpy(pred[args.puzzle].copy()),
                   prepared=b.inner.w_cls(q[args.puzzle]).argmax(-1).cpu())
        rows.append(row)
        if k >= 148:
            states[k] = dict(pre=pre[args.puzzle, args.cell].cpu(),
                             values=v[args.puzzle].cpu(),
                             logits=b.inner.w_cls(hn[args.puzzle, args.cell]).cpu())
        h, w = hn, wn
        # Avoid retaining last block's large temporaries during the next block.
        del parts, q, a, v, target, wn, effective, o, pre, hn
        if k % 32 == 0:
            print("capture", k, "all 128 predictions matched", flush=True)
    result = dict(rows=rows, states=states,
                  eta=b.eta[:, 0, 0].cpu(), lam=b.lam[:, 0, 0].cpu(),
                  w_sh=b.layer.w_sh.cpu(), decoder=b.inner.w_cls.weight.cpu(),
                  bias=b.inner.w_cls.bias.cpu(), d=b.inner.d,
                  labels=batch["labels"][args.puzzle].cpu(),
                  allocator_cap_bytes=cap, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  puzzle=args.puzzle, cell=args.cell, head=args.head, source=args.source,
                  checkpoint=args.checkpoint, prediction_disagreements=0)
    torch.save(result, Path(args.out)/"capture.pt")
    print("capture saved; peak MiB", result["peak_allocated_bytes"]/1024**2, flush=True)


def analyze(args):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    out = Path(args.out)
    c = torch.load(out/"capture.pt", map_location="cpu", weights_only=True)
    arrays = {key: torch.stack([r[key] for r in c["rows"]]).double().numpy()
              for key in ("agree", "phase", "write", "memory", "before", "after", "prepared")}
    eta, lam = (c[key].double().numpy() for key in ("eta", "lam"))
    cell, head, source = (c[key] for key in ("cell", "head", "source"))
    gold, old = int(c["labels"][cell]), 4  # Original 1-versus-3 case; token = digit+1.
    assert gold == 2
    report = dict(puzzle=c["puzzle"], cell=cell, source=source, head=head,
                  gold_digit=gold-1, alternative_digit=old-1,
                  allocator_cap_bytes=c["allocator_cap_bytes"],
                  peak_allocated_bytes=c["peak_allocated_bytes"],
                  prediction_disagreements=c["prediction_disagreements"], blocks={})
    residuals = []
    output_arrays = {}
    for k, state in c["states"].items():
        age = k-np.arange(1, k+1)
        alpha = eta[None, :]*(1-eta[None, :])**age[:, None]
        alpha[0] = (1-eta)**(k-1)
        np.testing.assert_allclose(alpha.sum(0), 1, atol=1e-12)
        g = arrays["write"][:k]
        phase, agree = arrays["phase"][:k], arrays["agree"][:k]
        pieces = alpha[:, :, None]*(g-g[-1])
        actual = arrays["memory"][k-1]-g[-1]
        residual = float(np.max(np.abs(pieces.sum(0)-actual)))
        residuals.append(residual)
        np.testing.assert_allclose(pieces.sum(0), actual, atol=2e-5, rtol=2e-5)
        # Per-time product split assigns half of the former covariance to each
        # factor. It is exact algebra, not a unique causal factor assignment.
        per_phase = alpha[:, :, None]*(phase-phase[-1])*(agree+agree[-1])/2
        per_agree = alpha[:, :, None]*(agree-agree[-1])*(phase+phase[-1])/2
        np.testing.assert_allclose(per_phase+per_agree, pieces, atol=2e-7, rtol=2e-5)
        pre = state["pre"].double().numpy()
        decoder = (c["decoder"][gold]-c["decoder"][old]).double().numpy()
        denom = np.sqrt(1+np.square(pre).sum()/c["d"])
        grad = decoder/denom-pre*np.dot(decoder, pre)/(c["d"]*denom**3)
        # All transport uses CURRENT values, exactly as the real memory does.
        v = state["values"].double().numpy()
        wsh = c["w_sh"].double().numpy()
        lift = np.einsum("nhc,hcd->hnd", v, wsh, optimize=True)
        evidence = lam[:, None]*np.einsum("hnd,d->hn", lift, grad)
        algebraic_evidence = lam[:, None]*np.einsum("hnd,d->hn", lift, decoder/denom)
        linear = pieces*evidence[None]
        def score(delta):
            cf_pre = state["pre"]-torch.from_numpy(np.asarray(delta)).float()
            cf_h = cf_pre/torch.sqrt(1+cf_pre.square().sum()/c["d"])
            logits = torch.nn.functional.linear(cf_h, c["decoder"], c["bias"])
            others = logits.clone(); others[gold] = -torch.inf
            return dict(digit=int(logits.argmax())-1, fixed_margin=float(logits[gold]-logits[old]),
                        gold_margin=float(logits[gold]-others.max()),
                        gold_logit=float(logits[gold]), alternative_logit=float(logits[old]))
        normal = score(np.zeros_like(pre))
        np.testing.assert_allclose(normal["fixed_margin"], float(state["logits"][gold]-state["logits"][old]), atol=2e-5)
        def summarize(mask):
            coefficient = pieces[mask].sum(0)
            actual_share = (alpha[mask, :, None]*g[mask]).sum(0)
            reference_share = alpha[mask].sum(0)[:, None]*g[-1]
            edge_coefficient = coefficient[head, source]
            delta = np.einsum("hn,hnd->d", lam[:, None]*coefficient, lift, optimize=True)
            edge_delta = lam[head]*edge_coefficient*lift[head, source]
            mass = float(alpha[mask, head].sum())
            result = dict(blocks=(np.flatnonzero(mask)+1).tolist(), edge_mass=mass,
                          edge_agree_contribution=float((alpha[mask, head]*agree[mask, head, source]).sum()),
                          edge_actual_memory_share=float((alpha[mask, head]*g[mask, head, source]).sum()),
                          edge_current_reference_share=float(mass*g[-1, head, source]),
                          edge_correction=float(edge_coefficient),
                          edge_linear_margin=float(linear[mask, head, source].sum()),
                          all_incoming_linear_margin=float(linear[mask].sum()),
                          edge_actual_linear_margin=float(actual_share[head, source]*evidence[head, source]),
                          edge_reference_linear_margin=float(reference_share[head, source]*evidence[head, source]),
                          incoming_actual_linear_margin=float((actual_share*evidence).sum()),
                          incoming_reference_linear_margin=float((reference_share*evidence).sum()),
                          incoming_actual_algebraic_margin=float((actual_share*algebraic_evidence).sum()),
                          incoming_reference_algebraic_margin=float((reference_share*algebraic_evidence).sum()),
                          incoming_correction_algebraic_margin=float((coefficient*algebraic_evidence).sum()),
                          incoming_phase_linear_margin=float((per_phase[mask]*evidence[None]).sum()),
                          incoming_agree_linear_margin=float((per_agree[mask]*evidence[None]).sum()),
                          edge_removed=score(edge_delta), incoming_removed=score(delta),
                          edge_phase_correction=float(per_phase[mask, head, source].sum()),
                          edge_agree_correction=float(per_agree[mask, head, source].sum()))
            result["edge_mean_agree"] = result["edge_agree_contribution"]/mass if mass else None
            result["edge_mean_write"] = result["edge_actual_memory_share"]/mass if mass else None
            return result
        time = np.arange(1, k+1)
        target_before = arrays["before"][:k, cell].astype(int)-1
        source_before = arrays["before"][:k, source].astype(int)-1
        target_prepared = arrays["prepared"][:k, cell].astype(int)-1
        groups = {"all": np.ones(k, bool),
                  "target_before_1": target_before == 1,
                  "target_before_3": target_before == 3,
                  "target_before_other": (target_before != 1)&(target_before != 3),
                  "target_prepared_1": target_prepared == 1,
                  "target_prepared_3": target_prepared == 3,
                  "target_prepared_other": (target_prepared != 1)&(target_prepared != 3),
                  "both_before_3": (target_before == 3)&(source_before == 3),
                  "before_target3_source4": (target_before == 3)&(source_before == 4),
                  "before_target1_source3": (target_before == 1)&(source_before == 3)}
        for start, end in ((1, 91), (92, 128), (129, 148), (149, 152), (153, 154), (155, 155), (156, 160)):
            groups[f"time_{start}_{end}"] = (time >= start)&(time <= end)
        for start, end in ((1, 128), (129, 155), (149, 155), (152, 153), (153, 154)):
            groups[f"window_{start}_{end}"] = (time >= start)&(time <= end)
        groups["window_130_148"] = (time >= 130)&(time <= 148)
        entries = {name: summarize(mask) for name, mask in groups.items()}
        def compare_memory_rows(removed_coefficient):
            delta = np.einsum("hn,hnd->d", lam[:, None]*removed_coefficient, lift, optimize=True)
            edge_delta = lam[head]*removed_coefficient[head, source]*lift[head, source]
            effect = removed_coefficient*evidence
            ranks = np.argsort(effect.ravel())[::-1]
            ranked = []
            for index in list(ranks[:8])+list(ranks[-4:]):
                hh, nn = np.unravel_index(index, effect.shape)
                edge_change = lam[hh]*removed_coefficient[hh, nn]*lift[hh, nn]
                ranked.append(dict(head=int(hh), source=int(nn),
                                   source_before_digit=int(arrays["before"][k-1, nn])-1,
                                   source_gold_digit=int(c["labels"][nn])-1,
                                   removed_coefficient=float(removed_coefficient[hh, nn]),
                                   linear_margin=float(effect[hh, nn]), changed=score(edge_change)))
            top_groups = {}
            for count in (1, 3, 6):
                selected = np.zeros_like(effect)
                selected.ravel()[ranks[:count]] = removed_coefficient.ravel()[ranks[:count]]
                change = np.einsum("hn,hnd->d", lam[:, None]*selected, lift, optimize=True)
                top_groups[count] = score(change)
            return dict(incoming_changed=score(delta), edge_changed=score(edge_delta),
                        incoming_linear_margin=float((removed_coefficient*evidence).sum()),
                        edge_coefficient_removed=float(removed_coefficient[head, source]),
                        incoming_algebraic_margin=float((removed_coefficient*algebraic_evidence).sum()),
                        ranked_edges=ranked, top_groups_changed=top_groups)
        # Does updating during the episode add useful information relative to
        # RETAINING the memory already present when it began? This reference
        # does not substitute the later, potentially misleading current write.
        # All h, values and later writes are replayed from normal: a controlled
        # memory-path intervention, not a free-running counterfactual trajectory.
        alternatives = {}
        for start, end in ((129, 148), (130, 148), (131, 148), (149, 152), (153, 154)):
            if end > k:
                continue
            decay_after = (1-eta)**(k-end)
            removed = decay_after[:, None]*(arrays["memory"][end-1]-arrays["memory"][start-2])
            replayed = arrays["memory"][start-2].copy()
            for rr in range(end, k):
                replayed = (1-eta[:, None])*replayed+eta[:, None]*g[rr]
            np.testing.assert_allclose(arrays["memory"][k-1]-removed, replayed, atol=2e-5, rtol=2e-5)
            alternatives[f"freeze_memory_{start}_{end}"] = compare_memory_rows(removed)
            mask = (time >= start)&(time <= end)
            frozen_write = g[start-2]
            removed = (alpha[mask, :, None]*(g[mask]-frozen_write)).sum(0)
            replayed = arrays["memory"][start-2].copy()
            for rr in range(start-1, k):
                write = frozen_write if rr < end else g[rr]
                replayed = (1-eta[:, None])*replayed+eta[:, None]*write
            np.testing.assert_allclose(arrays["memory"][k-1]-removed, replayed, atol=2e-5, rtol=2e-5)
            alternatives[f"repeat_write_from_{start-1}_through_{end}"] = compare_memory_rows(removed)
            mean_write = g[mask].mean(0)
            removed = (alpha[mask, :, None]*(g[mask]-mean_write)).sum(0)
            alternatives[f"constant_episode_mean_{start}_{end}"] = compare_memory_rows(removed)
        # Connections through which each selected episode acts at this fixed
        # current state. Keep every edge for reproducible, non-cherry-picked sums.
        episode_edges = {}
        for name in ("all", "target_before_1", "target_before_3", "time_129_148"):
            mask = groups[name]
            coeff = pieces[mask].sum(0)
            effect = linear[mask].sum(0)
            ranks = np.argsort(effect.ravel())[::-1]
            ranked = []
            for index in ranks:
                hh, nn = np.unravel_index(index, effect.shape)
                ranked.append(dict(head=int(hh), source=int(nn),
                                   source_before_digit=int(arrays["before"][k-1, nn])-1,
                                   source_gold_digit=int(c["labels"][nn])-1,
                                   correction=float(coeff[hh, nn]), linear_margin=float(effect[hh, nn]),
                                   current_write=float(g[-1, hh, nn]),
                                   full_memory=float(arrays["memory"][k-1, hh, nn])))
            episode_edges[name] = ranked
        per_block = []
        for r in range(k):
            mask = time == r+1
            item = summarize(mask)
            item.update(block=r+1, before_target=int(target_before[r]), before_source=int(source_before[r]),
                        prepared_target=int(target_prepared[r]),
                        after_target=int(arrays["after"][r, cell])-1,
                        after_source=int(arrays["after"][r, source])-1,
                        phase=float(phase[r, head, source]), agree=float(agree[r, head, source]),
                        write=float(g[r, head, source]))
            per_block.append(item)
        report["blocks"][k] = dict(normal=normal, edge_current_agree=float(agree[-1, head, source]),
                                  edge_agree_mean=float((alpha[:, head]*agree[:, head, source]).sum()),
                                  edge_current_write=float(g[-1, head, source]),
                                  edge_memory=float(arrays["memory"][k-1, head, source]),
                                  reconstruction_max_error=residual, groups=entries,
                                  per_block=per_block, episode_edges=episode_edges,
                                  memory_path_alternatives=alternatives)
        output_arrays[f"linear_{k}"] = linear
        output_arrays[f"correction_{k}"] = pieces
        output_arrays[f"alpha_{k}"] = alpha
        print("analyzed", k, "normal", normal["digit"], "margin", round(normal["fixed_margin"], 5), flush=True)
    report["max_reconstruction_error"] = max(residuals)
    (out/"summary.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out/"temporal_contributions.npz", **output_arrays)
    print("saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/relation_transport_v11")
    ap.add_argument("--out", default="runs/history_episodes_v11")
    ap.add_argument("--checkpoint", default="checkpoints/v1.1_step160000.npz")
    ap.add_argument("--puzzle", type=int, default=1)
    ap.add_argument("--cell", type=int, default=48)
    ap.add_argument("--head", type=int, default=2)
    ap.add_argument("--source", type=int, default=66)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--capture-only", action="store_true")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        capture(args)
    if not args.capture_only:
        analyze(args)


if __name__ == "__main__":
    main()
