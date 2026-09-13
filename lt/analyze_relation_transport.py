"""Identify which history factors and current messages support late corrections.

No task constraints or label-dependent inference rules are introduced. Whole-run
ablations start from the same segment-16 state. A separate, outcome-selected
event analysis describes stable corrections on the normal trajectory; its local
effects must not be interpreted as a deployable gate or a long-run causal score.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

import train
from ckpt_npz import load_data, load_lt
from analyze_memory_correction import CorrectionBlocks


MODES = ("drop_phase", "drop_agree", "drop_covariance", "drop_symmetric",
         "history_values_lag8", "history_directions_lag8")


@dataclass
class State:
    w: torch.Tensor
    phase_mean: torch.Tensor
    agree_mean: torch.Tensor
    values: tuple
    normalized_mean: torch.Tensor


class RelationBlocks(CorrectionBlocks):
    def parts(self, h, state):
        q = self.prepare(h)
        u = self.inner.addr(q, self.ab)
        a = self.inner.attn_xy(u, self.kc)
        v = torch.einsum("btd,hcd->bthc", q, self.layer.w_sh)
        vv = v / (v.norm(dim=-1, keepdim=True) + self.inner.config.eps)
        agree = torch.einsum("bthc,bnhc->bhtn", vv, vv)
        window = self.inner.attn_xy(u, self.kcb)
        target = self.gain * (window * agree)
        w = target if state is None else (1-self.eta)*state.w + self.eta*target
        phase = self.gain * (window + window.transpose(-1, -2)) / 2
        phase_mean = phase if state is None else (1-self.eta)*state.phase_mean + self.eta*phase
        agree_mean = agree if state is None else (1-self.eta)*state.agree_mean + self.eta*agree
        value_eta = self.eta[:, 0, 0][None, None, :, None]
        normalized_mean = vv if state is None else (1-value_eta)*state.normalized_mean+value_eta*vv
        mixed = torch.einsum("bthc,bnhc->bhtn", normalized_mean, vv)
        d = w-target
        symmetric = (d+d.transpose(-1, -2))/2
        # Symmetric allocation of the product-of-means change between its factors.
        phase_history = (phase_mean-phase)*(agree_mean+agree)/2
        agree_history = (agree_mean-agree)*(phase_mean+phase)/2
        agree_target = (agree_mean-agree+mixed-mixed.transpose(-1,-2))*(phase_mean+phase)/4
        agree_source = agree_history-agree_target
        covariance = symmetric-phase_history-agree_history
        cov_formula = (w+w.transpose(-1, -2))/2-phase_mean*agree_mean
        if state is None:
            torch.testing.assert_close(covariance, cov_formula, atol=2e-6, rtol=2e-5)
        old_values = state.values[0] if state is not None else v
        history = (() if state is None else state.values) + (v,)
        ns = State(w, phase_mean, agree_mean, history[-8:], normalized_mean)
        return dict(q=q, a=a, v=v, target=target, w=w, phase=phase,
                    phase_mean=phase_mean, agree=agree, agree_mean=agree_mean,
                    phase_history=phase_history, agree_history=agree_history,
                    agree_target=agree_target, agree_source=agree_source,
                    covariance=covariance, symmetric=symmetric, old_values=old_values,
                    state=ns)

    def read_parts(self, p, mode="normal", return_pre=False):
        w = p["w"]
        field = {"drop_phase": "phase_history", "drop_agree": "agree_history",
                 "drop_covariance": "covariance", "drop_agree_target": "agree_target",
                 "drop_agree_source": "agree_source"}.get(mode)
        if field:
            w = w-p[field]
        elif mode == "drop_half_agree":
            w = w-p["agree_history"]/2
        elif mode == "drop_symmetric":
            d = w-p["target"]
            w = p["target"]+(d-d.transpose(-1, -2))/2
        elif mode == "only_agree":
            w = w-p["symmetric"]+p["agree_history"]
        elif mode in ("no_memory_read", "read_only"):
            w = torch.zeros_like(w)
        effective = (1-self.lam)*p["a"]+self.lam*w
        if mode == "read_only":
            effective = p["a"]
        o = torch.einsum("bhtn,bnhc->bthc", effective, p["v"])
        if mode in ("history_values_lag8", "history_directions_lag8"):
            old = p["old_values"]
            if mode == "history_directions_lag8":
                old = old * (p["v"].norm(dim=-1, keepdim=True) /
                             old.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            o = o-torch.einsum("bhtn,bnhc->bthc", self.lam*p["symmetric"], p["v"]-old)
        update = torch.einsum("bthc,hcd->btd", o, self.layer.w_sh)
        pre = p["q"]+update
        h = self.inner.phi(pre)
        return (h, pre) if return_pre else h


def rollout(b, batch, total, mode="normal", snapshot=None):
    if snapshot is None:
        h = b.inner.init_hidden.expand(len(batch["inputs"]), 81, -1).clone()
        state, start = None, 0
    else:
        h, state = snapshot
        start = 128
    pred = [b.inner.w_cls(h).argmax(-1).cpu().numpy().astype(np.int8)]
    save = None
    for k in range(start+1, total+1):
        p = b.parts(h, state)
        hn = b.read_parts(p, mode)
        if k == 1:
            original = b.components(h, None)
            torch.testing.assert_close(hn, b.read(original, "normal"), atol=0, rtol=0)
            torch.testing.assert_close(p["w"], original[-1], atol=0, rtol=0)
        h, state = hn, p["state"]
        pred.append(b.inner.w_cls(h).argmax(-1).cpu().numpy().astype(np.int8))
        if k == 128:
            save = (h, state)
        if k % 256 == 0:
            print(mode, "seg", k//8, flush=True)
    return np.stack(pred), save


def classify_run(pred, labels, baseline):
    correct = (pred == labels[None]).all(-1)
    base = (baseline == labels[None]).all(-1)
    initial = base[128]
    return {"final_exact": int(correct[-1].sum()),
            "new_final_correct": int((correct[-1] & ~initial).sum()),
            "lost_initial_correct": int((~correct[-1] & initial).sum()),
            "normal_only_final": np.flatnonzero(base[-1] & ~correct[-1]).tolist(),
            "intervention_only_final": np.flatnonzero(~base[-1] & correct[-1]).tolist(),
            "final_cell_errors": int((pred[-1] != labels).sum())}


def select_events(pred, inputs, labels):
    """Last wrong->correct transition, with >=32 subsequent correct blocks."""
    solved = (pred[-1] == labels).all(-1)
    events = []
    for puzzle in np.flatnonzero(solved):
        for cell in np.flatnonzero((inputs[puzzle] == 1) & (pred[128, puzzle] != labels[puzzle])):
            wrong_times = np.flatnonzero(pred[:, puzzle, cell] != labels[puzzle, cell])
            k = int(wrong_times[-1])+1
            if 137 <= k <= len(pred)-33:
                events.append(dict(puzzle=int(puzzle), cell=int(cell), block=k,
                                   previous_choice=int(pred[k-1, puzzle, cell]),
                                   gold=int(labels[puzzle, cell])))
    return events


def event_analysis(b, batch, pred, out):
    inputs, labels = (batch[key].cpu().numpy() for key in ("inputs", "labels"))
    events = select_events(pred, inputs, labels)
    schedule = {}
    for ei, ev in enumerate(events):
        for offset in (-8, -4, -1, 0):
            schedule.setdefault(ev["block"]+offset, []).append((ei, offset))
    if not schedule:
        return {"events": [], "note": "No eligible stable correction."}
    h = b.inner.init_hidden.expand(len(inputs), 81, -1).clone()
    state = None
    rows = []
    names = ("all_symmetric", "phase_history", "agree_history", "covariance",
             "other_clue", "other_correct_blank", "other_wrong_blank", "self",
             "fresh_values_lag8", "fresh_directions_lag8", "agree_target", "agree_source")
    edge_rows = []
    for k in range(1, max(schedule)+1):
        p = b.parts(h, state)
        hn, pre = b.read_parts(p, return_pre=True)
        # Event analysis is on the identical normal rollout, not ablated states.
        if k in schedule:
            torch.testing.assert_close(b.inner.w_cls(hn).argmax(-1).cpu(), torch.from_numpy(pred[k]).long(), atol=0, rtol=0)
            entries = schedule[k]
            ids = torch.tensor([events[e]["puzzle"] for e, _ in entries], device="cuda")
            cells = torch.tensor([events[e]["cell"] for e, _ in entries], device="cuda")
            gold = batch["labels"][ids, cells]
            previous_choice = torch.tensor([events[e]["previous_choice"] for e, _ in entries], device="cuda")
            sv = p["v"][ids].permute(0, 2, 1, 3)
            old_sv = p["old_values"][ids].permute(0, 2, 1, 3)
            old_dir = old_sv * (sv.norm(dim=-1, keepdim=True) / old_sv.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            current_correct = torch.as_tensor(pred[k-1], device="cuda")[ids] == batch["labels"][ids]
            clue = batch["inputs"][ids] != 1
            eye = torch.arange(81, device="cuda")[None] == cells[:, None]
            s = p["symmetric"][ids, :, cells]
            coeffs = [s, p["phase_history"][ids, :, cells], p["agree_history"][ids, :, cells],
                      p["covariance"][ids, :, cells], s*(clue & ~eye)[:, None],
                      s*(~clue & current_correct & ~eye)[:, None],
                      s*(~clue & ~current_correct & ~eye)[:, None], s*eye[:, None]]
            def transport(c, values):
                return torch.einsum("mhn,mhnc,hcd->md", self_lam*c, values, b.layer.w_sh)
            self_lam = b.lam[:, 0, 0][None, :, None]
            deltas = [transport(c, sv) for c in coeffs]
            deltas += [transport(s, sv-old_sv), transport(s, sv-old_dir)]
            deltas += [transport(p[field][ids, :, cells], sv) for field in ("agree_target", "agree_source")]
            target_pre = pre[ids, cells]
            normal_logits = b.inner.w_cls(hn[ids, cells])
            def metrics(logits):
                loss = train.stablemax_cross_entropy(logits, gold).float()
                g = logits.gather(-1, gold[:, None]).squeeze(-1)
                competitors = logits.clone().scatter_(-1, gold[:, None], -torch.inf).amax(-1)
                fixed = logits.gather(-1, previous_choice[:, None]).squeeze(-1)
                return loss, g-competitors, g-fixed, (logits.argmax(-1) == gold).float()
            normal = metrics(normal_logits)
            effect = []
            cf_predictions = []
            for delta in deltas:
                cf_logits = b.inner.w_cls(b.inner.phi(target_pre-delta))
                cf = metrics(cf_logits)
                cf_predictions.append(cf_logits.argmax(-1))
                effect.append(torch.stack((cf[0]-normal[0], normal[1]-cf[1], normal[2]-cf[2], normal[3]-cf[3]), -1))
            effect = torch.stack(effect, 1).cpu().numpy()
            cf_predictions = torch.stack(cf_predictions, 1).cpu().numpy()
            normal_logits_np = normal_logits.cpu().numpy()
            for j, (ei, offset) in enumerate(entries):
                rows.append((ei, offset, effect[j], normal_logits_np[j], cf_predictions[j]))

            # At the actual transition only, decompose the first-order fixed
            # gold-vs-previous-choice margin into all head/source contributions.
            take = [j for j, (_, offset) in enumerate(entries) if offset == 0]
            if take:
                decoder = b.inner.w_cls.weight[gold]-b.inner.w_cls.weight[previous_choice]
                scale = (1+target_pre.square().sum(-1, keepdim=True)/b.inner.d).sqrt()
                grad = decoder/scale-target_pre*(decoder*target_pre).sum(-1, keepdim=True)/(b.inner.d*scale.pow(3))
                direction = torch.einsum("md,hcd->mhc", grad, b.layer.w_sh)
                evidence = torch.einsum("mhc,mhnc->mhn", direction, sv)
                contribution = self_lam*s*evidence
                fresh_contribution = self_lam*s*torch.einsum("mhc,mhnc->mhn", direction, sv-old_dir)
                gold_decoder = b.inner.w_cls.weight[gold]
                gold_grad = gold_decoder/scale-target_pre*(gold_decoder*target_pre).sum(-1,keepdim=True)/(b.inner.d*scale.pow(3))
                gold_direction = torch.einsum("md,hcd->mhc",gold_grad,b.layer.w_sh)
                gold_contribution = self_lam*s*torch.einsum("mhc,mhnc->mhn",gold_direction,sv)
                w_sym = (p["w"]+p["w"].transpose(-1,-2))/2
                g_sym = (p["target"]+p["target"].transpose(-1,-2))/2
                for j in take:
                    ei = entries[j][0]
                    edge_rows.append(dict(event=ei, contribution=contribution[j].cpu().numpy(),
                                          gold_logit_contribution=gold_contribution[j].cpu().numpy(),
                                          previous_logit_contribution=(gold_contribution[j]-contribution[j]).cpu().numpy(),
                                          fresh_direction_contribution=fresh_contribution[j].cpu().numpy(),
                                          phase_contribution=(self_lam*coeffs[1]*evidence)[j].cpu().numpy(),
                                          agree_contribution=(self_lam*coeffs[2]*evidence)[j].cpu().numpy(),
                                          covariance_contribution=(self_lam*coeffs[3]*evidence)[j].cpu().numpy(),
                                          agree_target_contribution=(self_lam*p["agree_target"][ids,:,cells]*evidence)[j].cpu().numpy(),
                                          agree_source_contribution=(self_lam*p["agree_source"][ids,:,cells]*evidence)[j].cpu().numpy(),
                                          source_correct=current_correct[j].cpu().numpy(),
                                          source_clue=clue[j].cpu().numpy(),
                                          source_prediction_changed=(torch.as_tensor(pred[k-1, ids[j].item()],device="cuda") != torch.as_tensor(pred[k-9, ids[j].item()],device="cuda")).cpu().numpy(),
                                          source_direction_cosine=((sv[j]*old_sv[j]).sum(-1)/(sv[j].norm(dim=-1)*old_sv[j].norm(dim=-1)).clamp_min(1e-12)).cpu().numpy(),
                                          symmetric_history=s[j].cpu().numpy(),
                                          memory=w_sym[ids[j], :, cells[j]].cpu().numpy(),
                                          current_write=g_sym[ids[j], :, cells[j]].cpu().numpy(),
                                          phase=p["phase"][ids[j], :, cells[j]].cpu().numpy(),
                                          phase_mean=p["phase_mean"][ids[j], :, cells[j]].cpu().numpy(),
                                          agree=p["agree"][ids[j], :, cells[j]].cpu().numpy(),
                                          agree_mean=p["agree_mean"][ids[j], :, cells[j]].cpu().numpy()))
        h, state = hn, p["state"]
        if k % 256 == 0:
            print("events seg", k//8, flush=True)
    ei = np.array([r[0] for r in rows]); offsets = np.array([r[1] for r in rows]); values = np.stack([r[2] for r in rows])
    np.savez_compressed(out/"event_effects.npz", event_ids=ei, offsets=offsets, effects=values,
                        normal_logits=np.stack([r[3] for r in rows]), cf_predictions=np.stack([r[4] for r in rows]))
    np.savez_compressed(out/"event_edges.npz", **{key: np.stack([row[key] for row in edge_rows]) for key in edge_rows[0]})
    report = {"events": events, "puzzles": len(set(e["puzzle"] for e in events)), "conditions": names,
              "metrics": ["removal_loss_increase", "history_gold_margin_gain", "history_fixed_margin_gain", "correctness_lost_on_removal"],
              "selection": "Wrong at block128; eventually fully solved puzzle; target remains correct from event until end, for >=32 blocks.",
              "by_offset": {}}
    for offset in (-8,-4,-1,0):
        mask = offsets == offset
        puzzle_ids = np.array([events[e]["puzzle"] for e in ei[mask]])
        per_puzzle = np.stack([values[mask][puzzle_ids == pp].mean(0) for pp in np.unique(puzzle_ids)])
        report["by_offset"][str(offset)] = dict(puzzle_mean=per_puzzle.mean(0).tolist(),
                                                changed_target_counts=(values[mask,:,-1] != 0).sum(0).tolist(),
                                                correctness_lost=(values[mask,:,-1] > 0).sum(0).tolist(),
                                                correctness_gained=(values[mask,:,-1] < 0).sum(0).tolist())
    return report


def summarize_edges(out, event_report):
    """Problem-weighted first-order message evidence, kept apart from exact CFs."""
    a = np.load(Path(out)/"event_edges.npz")
    ev = [event_report["events"][int(i)] for i in a["event"]]
    puzzles = np.array([e["puzzle"] for e in ev])
    cells = np.array([e["cell"] for e in ev])
    unique = np.unique(puzzles)
    def mean_by_puzzle(values):
        per = np.stack([values[puzzles == p].mean(0) for p in unique])
        return {"mean": np.mean(per, axis=0).tolist(), "per_puzzle": per.tolist()}
    fields = ("contribution", "gold_logit_contribution", "previous_logit_contribution",
              "phase_contribution", "agree_contribution", "covariance_contribution",
              "agree_target_contribution", "agree_source_contribution", "fresh_direction_contribution")
    result = {"definition": "First-order gold-vs-previous-choice margin contribution at the normal pre-phi state. "
                            "Average events within each selected puzzle, then weight puzzles equally.",
              "puzzles": unique.tolist(), "events": len(ev),
              "components": {key: mean_by_puzzle(a[key].sum((1,2))) for key in fields},
              "heads": mean_by_puzzle(a["contribution"].sum(-1)), "groups": {}}
    same = a["memory"]*a["current_write"] >= 0
    self_mask = np.arange(81)[None] == cells[:,None]
    masks = {
        "same_sign_old_stronger": same & (abs(a["memory"]) > abs(a["current_write"])),
        "same_sign_current_stronger": same & (abs(a["memory"]) <= abs(a["current_write"])),
        "opposite_memory_current_sign": ~same,
        "positive_history_coefficient": a["symmetric_history"] > 0,
        "negative_history_coefficient": a["symmetric_history"] < 0,
        "clue": (a["source_clue"] & ~self_mask)[:,None],
        "correct_blank": (~a["source_clue"] & a["source_correct"] & ~self_mask)[:,None],
        "wrong_blank": (~a["source_clue"] & ~a["source_correct"] & ~self_mask)[:,None],
        "self": self_mask[:,None],
        "source_prediction_changed_in_8_blocks": a["source_prediction_changed"][:,None],
    }
    for name, mask in masks.items():
        result["groups"][name] = {key: mean_by_puzzle((a[key]*mask).sum((1,2)))
                                    for key in fields[:3]}
    (Path(out)/"edge_summary.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--segs", type=int, default=128)
    ap.add_argument("--out", default="runs/relation_transport_v11")
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--events-only", action="store_true")
    ap.add_argument("--reuse-baseline", action="store_true")
    ap.add_argument("--no-events", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, _ = load_lt("checkpoints/v1.1_step160000.npz", mod=train, batch_size=args.n, loops=args.segs+1, amp=False)
    _, _, batch = load_data(n=args.n)
    b = RelationBlocks(model, batch)
    labels = batch["labels"].cpu().numpy()
    report_path = out/"summary.json"
    if args.events_only:
        report = json.loads(report_path.read_text())
        pred = np.load(out/"normal.npz")["pred"]
    else:
        start = time.monotonic()
        if args.reuse_baseline:
            report = json.loads(report_path.read_text())
            pred = np.load(out/"normal.npz")["pred"]
            prefix, snapshot = rollout(b, batch, 128)
            np.testing.assert_array_equal(prefix, pred[:129])
        else:
            pred, snapshot = rollout(b, batch, args.segs*8)
            np.savez_compressed(out/"normal.npz", pred=pred)
            report = dict(args=vars(args), precision="FP32; autocast/TF32 off; EMA checkpoint",
                          initial_exact=int((pred[128] == labels).all(-1).sum()),
                          runs={"normal": classify_run(pred, labels, pred)})
        report_path.write_text(json.dumps(report, indent=2))
        print("RESULT normal", report["runs"]["normal"], flush=True)
        for mode in args.modes:
            cf, _ = rollout(b, batch, args.segs*8, mode, snapshot)
            np.savez_compressed(out/(mode+".npz"), pred=cf)
            report["runs"][mode] = classify_run(cf, labels, pred)
            report_path.write_text(json.dumps(report, indent=2))
            print("RESULT", mode, report["runs"][mode], flush=True)
        report["rollout_seconds"] = time.monotonic()-start
    if not args.no_events:
        report["events"] = event_analysis(b, batch, pred, out)
        if report["events"].get("puzzles"):
            summarize_edges(out, report["events"])
    report_path.write_text(json.dumps(report, indent=2))
    if "events" in report:
        print("EVENTS", len(report["events"]["events"]), "puzzles", report["events"].get("puzzles"), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
