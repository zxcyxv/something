"""Reproduce the October 2 KV model without changing its training equations.

Diagnostics run outside the compiled forward and never draw random numbers.
Use a separate output directory for every independent experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from . import train as t
from .kv_stability import VARIANTS, install


def append(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def scalar(x):
    value = float(x)
    return value if math.isfinite(value) else str(value)


@torch.no_grad()
def diagnostics(base, carry):
    result = {}
    for name in ("current_hidden", "coupling", "key_trace", "value_trace"):
        x = getattr(carry, name, None)
        if x is not None:
            result[name] = dict(rms=scalar(x.float().square().mean().sqrt()),
                                absmax=scalar(x.abs().max()))
    if carry.coupling is not None:
        m = carry.coupling[:8].float()
        sv = torch.linalg.svdvals(m)
        result["memory_spectrum"] = dict(
            sigma_max=scalar(sv[..., 0].max()),
            sigma_mean=scalar(sv[..., 0].mean()),
            effective_rank=scalar((sv.square().sum(-1) / sv[..., 0].square().clamp_min(1e-30)).mean()))
    for i, layer in enumerate(base.model.inner.layers):
        result[f"layer_{i}"] = {
            name: dict(rms=scalar(p.float().square().mean().sqrt()),
                       norm=scalar(p.float().norm()))
            for name, p in layer.named_parameters()
        }
        if hasattr(layer, "trace_lam_raw"):
            lam = layer.trace_decay
            result[f"layer_{i}"]["trace_decay"] = dict(
                min=scalar(lam.min()), max=scalar(lam.max()), mean=scalar(lam.mean()))
        if hasattr(layer,"read_lam_raw"):
            lam=layer.read_lam_raw.sigmoid()
            result[f"layer_{i}"]["read_interpolation"]=dict(
                min=scalar(lam.min()),max=scalar(lam.max()),mean=scalar(lam.mean()),
                per_head_mean=lam.flatten(1).mean(-1).tolist())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="Overrides to the original top-level CFG")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--diagnostic-every", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-last", type=int, default=20)
    ap.add_argument("--variant", choices=VARIANTS, default="original")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(t.CFG)
    cfg.update(data_npz=str(Path("data/sudoku_lt_1k.npz").resolve()),
               out_dir=str(out), max_steps=args.steps, max_hours=None,
               run_selftests=False, num_processes=1, log_every=16,
               save_every_steps=args.save_every, keep_last=args.keep_last, milestone_every=0)
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    cfg["research_variant"] = args.variant
    install(args.variant)
    source = Path(t.__file__).read_bytes()
    (out / "trainer_snapshot.py").write_bytes(source)
    metadata = dict(commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        trainer_sha256=hashlib.sha256(source).hexdigest(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        variant=args.variant,
        protocol="original CFG; same data/seed/batch/depth/optimizer; explicit model control only",
        selftest_note="Original selftest fails bitwise resume equality for q_proj on CPU (max difference 7.45e-9, both finite); not altered.")
    if args.variant == "read_gain_quarter":
        metadata.update(
            only_equation_change="read = 0.25 * RoPE(Q) @ (M_old + G).T",
            write_equation="Original G = (V.T @ RoPE(eK_old) - eV_old.T @ RoPE(K)) / T; M_new = M_old + G.",
            read_gain=0.25, recurrent_memory=True, eligibility_traces=True, interpolation=False,
            rationale="Frozen-checkpoint gain interventions reduced two-block oscillation in both phases. Training checks whether that observation transfers; it is not a presumed fix.",
            parameter_note="No new parameters; original initialization, raw-key traces, optimizer/data/depth retained.",
            primary_criterion="Training loss/accuracy, not frozen-probe or EMA test accuracy.")
    elif args.variant == "historical_key_trace":
        metadata.update(
            key_trace_coordinates="Event-time RoPE(K), not raw K; incompatible with original raw-key carry.",
            write_equation="G = (V.T @ eK_rotated_previous - eV_previous.T @ RoPE(K)) / T",
            key_trace_equation="eK_rotated_new = lambda*eK_rotated_previous + (1-lambda)*RoPE_theta_now(K_now)",
            read_equation="RoPE(Q) @ (M_old + G).T",
            recurrent_memory=True, eligibility_traces=True, interpolation=False,
            comparison="Original baseline_ng0; same initialization, data, optimizer, depth, temporal window and additive M.",
            distinction="Same fixed-theta forward operator, but preserves event-time key coordinates across optimizer updates and detaches their past theta dependence at segment boundaries.",
            primary_criterion="Training loss/accuracy and whether sustained deterioration returns; EMA test performance is secondary.")
    elif args.variant == "phase_unit_gaussian_current_only":
        metadata.update(
            write_equation="G_ij=mean_tokens(V_i*RoPE(K)_j*L(phiV_i-phiK_j)); window applied before the token mean",
            read_equation="RoPE(Q) @ G.T (column convention: Gq)",
            window="L(delta)=sign(sin(delta))*exp(-delta^2/tau)",
            phase_shape="independent K/V [batch,heads,tokens,head_dim] per layer and recurrence",
            phase_parameterization="phi=(pi/2-1e-4)*tanh(W_phase h+theta_raw); h includes current input injection",
            phase_representation="Independent real signed activity and bounded phase projections (polar factors), not independent Cartesian real/imaginary projections",
            phase_projection_initialization="nn.Linear default Kaiming-uniform initialization; bias=False",
            phase_offset_initialization="independent uniform angles [-pi/4,pi/4] converted to raw tanh coordinates",
            phase_limit=math.pi/2-1e-4,
            phase_trainable=True, phase_projection_weight_decay=cfg["weight_decay"],
            phase_offset_weight_decay=0, phase_projection_dtype="FP32; outside BF16 autocast",
            phase_tau=1.0, phase_tau_units="squared radians", phase_tau_trainable=False,
            ltp_amplitude=1.0, ltd_amplitude=1.0, zero_lag_value=0.0,
            phase_difference_wrap=False,
            alias_control="Bounded phases ensure all pairwise differences lie strictly inside (-pi,pi), including FP32 tanh saturation",
            phase_gradient="ordinary first-order gradient; sign derivative zero; Gaussian envelope derivative -2*delta*L/tau; no surrogate",
            unit_normalization="none; sin/cos already represent unit phasors; fused kernel evaluates sign(sin(delta)) directly",
            common_carrier="cancels exactly; no clock or oscillator state",
            qk_normalization="none; existing real spatial RoPE retained",
            block_equation="u=RMSNorm(h+attention(h)); next=RMSNorm(u+bilinear_FFN(u))",
            block_normalization=dict(eps=1e-5, affine=False, accumulation_dtype="float32", count=2),
            recurrent_memory=False, eligibility_traces=False, interpolation=False,
            carry_matrix="current G for diagnostics only; ignored on the next step",
            within_segment_gradient="all 8 recurrent blocks; activation recomputation only, no no-grad or detach within a segment",
            segment_boundary="existing aligned harness: detach carry after every optimizer step; same sample retained for 16 segments",
            implementation="lt/unit_phase_stdp.py:unit_phase_gaussian_write; same unit-phase Triton operator as the benchmark",
            comparison_note="Compared with the old fixed-phase exponential run, both phase dependence and window shape change; not a single-variable ablation")
        kernel = Path(__file__).with_name("unit_phase_stdp.py").read_bytes()
        metadata["kernel_sha256"] = hashlib.sha256(kernel).hexdigest()
        (out / "unit_phase_stdp_snapshot.py").write_bytes(kernel)
    elif args.variant in ("phase_current_only", "phase_exp_current_only"):
        metadata.update(
            write_equation="G_ij=mean_tokens(V_i*RoPE(K)_j*sin(phiV_i-phiK_j))",
            read_equation="RoPE(Q) @ G.T (column convention: Gq)",
            phase_shape="independent K/V [heads, head_dim] per layer",
            phase_parameterization="phi=(pi/2)*tanh(theta_raw)",
            phase_initialization="independent uniform angles [-pi/4, pi/4]",
            phase_trainable=True, phase_weight_decay=0,
            phase_shared_over="tokens and recurrent iterations",
            common_carrier="cancels exactly; no clock or oscillator state",
            window="sine only; no exponential envelope or harmonics",
            qk_normalization="none; existing real spatial RoPE retained",
            block_equation="u=RMSNorm(h+attention(h)); next=RMSNorm(u+bilinear_FFN(u))",
            block_normalization=dict(eps=1e-5, affine=False, accumulation_dtype="float32", count=2),
            recurrent_memory=False, eligibility_traces=False, interpolation=False,
            carry_matrix="current G for diagnostics only; ignored on the next step",
            parameter_note="Original initialization and unused trace parameters retained; theta parameters excluded from decay by existing optimizer rule.")
        if args.variant == "phase_exp_current_only":
            metadata.update(
                write_equation="G=(V.T @ RoPE(K)/T) * L(phiV_i-phiK_j)",
                window="L(delta)=sign(delta)*exp(-abs(delta)/tau); exact, no Fourier approximation",
                phase_tau=1.0, phase_tau_units="radians; omega*tau_time",
                phase_tau_trainable=False, ltp_amplitude=1.0, ltd_amplitude=1.0,
                zero_lag_value=0.0, phase_difference_wrap=False,
                phase_gradient="ordinary autodiff; sign derivative zero; envelope derivative away from zero; no surrogate",
                distinction="Uses a phase-difference kernel on the real KV product, not the imaginary part of a single complex outer product.")
    elif args.variant == "current_only_urm_norm":
        metadata.update(
            read_equation="RoPE(Q) @ (V.T @ RoPE(K) / T).T; raw Q/K",
            qk_normalization="none",
            block_equation="u=RMSNorm(h+attention(h)); next=RMSNorm(u+bilinear_FFN(u))",
            block_normalization=dict(eps=1e-5, affine=False, accumulation_dtype="float32", count=2),
            phi_replaced=True, recurrent_memory=False, eligibility_traces=False,
            reference="UbiquantAI/URM c14e55f5f9227873617015cf60a239126b55adcd models/layers.py:rms_norm",
            unchanged="Original B-only projections, bilinear FFN, input injection, initialization, data, depth and optimizer.")
    elif args.variant == "current_only":
        normalized = bool(cfg.get("kv_qk_l2norm", False))
        metadata.update(
            read_equation="RoPE(Q_r) @ (V_r.T @ RoPE(K_r) / T).T",
            qk_normalization=("Q and K independently: x / (L2_norm(x) + eps), per token/head, FP32 before RoPE"
                              if normalized else "none; raw Q/K"),
            normalization_eps=cfg.get("eps", 1e-4) if normalized else None,
            unchanged="V/output projections, token mean reduction, optimizer/weight decay, data, seed and recurrent depth",
            recurrent_memory=False, eligibility_traces=False, interpolation=False,
            current_read_coefficient=1.0,
            carry_matrix="Current KV product for interface compatibility/diagnostics; never read by subsequent blocks.",
            parameter_note="Original initialization retained, including unused trace coefficients; no read gates.",
            primary_criterion="Training loss/accuracy and whether sustained deterioration returns; EMA test performance is secondary.")
    elif args.variant == "complex_current_only":
        metadata.update(
            complex_keys="RoPE(e_K_previous) + i RoPE(K_current)",
            complex_values="e_V_previous + i V_current",
            current_operator="G = Im(V_complex.T @ conj(K_complex)) / T = (V.T @ RoPE(e_K_previous) - e_V_previous.T @ RoPE(K)) / T",
            read_equation="RoPE(Q) @ G.T",
            recurrent_memory=False,eligibility_traces=True,interpolation=False,
            only_equation_change="M_new = M_old + G becomes M_new = G",
            carry_matrix="Current complex KV operator G; ignored by subsequent matrix updates. K/V eligibility traces are preserved and updated normally.",
            primary_criterion="Training loss/accuracy and whether sustained deterioration returns; EMA test performance is secondary.")
    elif args.variant == "current_plus_stdp":
        metadata.update(
            current_term="B = V.T @ RoPE(K) / T",
            difference_term="G = (V.T @ RoPE(e_K_previous) - e_V_previous.T @ RoPE(K)) / T",
            read_equation="RoPE(Q) @ (B + G).T",
            term_coefficients=dict(current=1.0,difference=1.0),
            recurrent_memory=False,eligibility_traces=True,interpolation=False,
            past_past_term=False,
            carry_matrix="Current B + G only; ignored by subsequent matrix updates. Original K/V eligibility traces remain.",
            checkpoint_note=f"Checkpoint interval {cfg['save_every_steps']} steps; keep_last={cfg['keep_last']}. These retention settings only control disk usage; all per-step training metrics remain recorded.",
            primary_criterion="Training loss/accuracy and whether sustained deterioration returns; EMA test performance is secondary.")
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2))
    (out / "research_runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out / "variant_snapshot.py").write_bytes(Path(__file__).with_name("kv_stability.py").read_bytes())
    started = time.monotonic()
    original_batch = t.train_batch
    original_check = t._check_finite_gradients
    current = {}

    def check(base, loss, device, ws):
        original_check(base, loss, device, ws)
        if current.get("measure"):
            grads = {}
            for name, p in base.named_parameters():
                if p.grad is not None:
                    grads[name] = scalar(p.grad.float().norm())
            current["gradients"] = grads

    def batch(model, base, state, *pos, **kw):
        current.clear()
        # Sample a whole 16-segment episode rather than always sampling segment 1.
        current["measure"] = state.step % (args.diagnostic_every * 4) < cfg["loops"]
        before = time.monotonic()
        result = original_batch(model, base, state, *pos, **kw)
        record = dict(step=state.step, elapsed=time.monotonic()-started,
                      seconds=time.monotonic()-before,
                      segment=int(state.carry.steps.max()), **result)
        append(out / "train.jsonl", record)
        if current["measure"]:
            append(out / "diagnostics.jsonl", dict(
                step=state.step, segment=record["segment"],
                gradients=current.get("gradients"), **diagnostics(base, state.carry)))
        return result

    t._check_finite_gradients = check
    t.train_batch = batch
    if args.variant == "phase_unit_gaussian_current_only":
        print("[RESEARCH] Actual variant: state-dependent per-token/channel K/V phases; "
              "current G only (no accumulated memory or eligibility traces); "
              "sign(sin(delta))*exp(-delta^2), exact tie=0; no phase/activity unit norm; "
              "two FP32 post-residual RMSNorms. Generic KV configuration fields below "
              "are inherited and do not describe this variant's memory equation.", flush=True)
    t.main(cfg)
    latest = t.find_latest_checkpoint(str(out))
    actual_step = int(Path(latest).stem.removeprefix("step_")) if latest else None
    status = "completed" if actual_step == args.steps else "stopped"
    (out / f"{status}.json").write_text(json.dumps(dict(
        requested_steps=args.steps, actual_step=actual_step,
        elapsed=time.monotonic()-started)))


if __name__ == "__main__":
    main()
