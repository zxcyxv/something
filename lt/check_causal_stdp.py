"""Audit the proposed timing rule against independent pair sums and its DTFT.

This is an architecture/correctness check, not a Sudoku performance experiment.
Run: python -m lt.check_causal_stdp
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from .causal_stdp import CausalLT, PairSTDP


def audit_pairs(writer):
    lags = torch.arange(-128, 129)
    batch, heads, channels = len(lags), writer.heads, writer.channels
    history = None
    observed = torch.zeros(batch, heads, 2, 2, dtype=torch.float64)
    first_event = None
    for k in range(132):
        x = torch.zeros(batch, 2, heads, channels, dtype=torch.float64)
        pre_k = (-lags).clamp_min(0)
        post_k = lags.clamp_min(0)
        x[pre_k == k, 0, :, 0] = .7
        x[post_k == k, 1, :, 0] = .4
        signal, history = writer.write(x, history)
        if k == 0:
            first_event = signal.abs().max().item()
        if k >= 129:
            assert torch.count_nonzero(signal) == 0, "old pairs must not be counted again"
        observed += signal
    expected = .28*writer.timing_window(lags).T
    error = (observed[:, :, 1, 0]-expected).abs().max().item()
    reverse_error = (observed[:, :, 0, 1]-.28*writer.timing_window(-lags).T).abs().max().item()
    assert error < 2e-14 and reverse_error < 2e-14
    assert first_event == 0
    assert (observed[lags > 0, :, 1, 0] > 0).all()
    assert (observed[lags < 0, :, 1, 0] < 0).all()
    assert (observed[lags == 0] == 0).all()
    return dict(lags=[-128, 128], heads=heads, checks=batch*heads,
                maximum_pair_error=error, maximum_reverse_error=reverse_error,
                simultaneous_write=0, repeated_write_after_last_event=0)


def audit_stream(writer):
    length, batch, cells = 19, 2, 3
    x = torch.rand(length, batch, cells, writer.heads, writer.channels, dtype=torch.float64)
    x = x/(1+x.sum(-1, keepdim=True))
    eta, gain = .07, 1.3
    distance = torch.tensor([[1, .8, .6], [.8, 1, .8], [.6, .8, 1]], dtype=torch.float64)
    memory = torch.zeros(batch, writer.heads, cells, cells, dtype=torch.float64)
    history = None
    for activity in x:
        signal, history = writer.write(activity, history)
        memory = (1-eta)*memory+eta*gain*distance*signal
        assert (history >= 0).all() and (history.sum(-1) <= 1+1e-12).all()
        assert signal.abs().max() <= 1+1e-12
    # Each pair is included once, at its later event, then only outer W decays.
    direct = torch.zeros_like(memory)
    for post_time in range(length):
        for pre_time in range(length):
            kernel = writer.timing_window([post_time-pre_time])[:, 0]
            activity_product = torch.einsum("bthc,bnhc->bhtn", x[post_time], x[pre_time])
            age = length-1-max(post_time, pre_time)
            direct += eta*(1-eta)**age*gain*distance*kernel[None, :, None, None]*activity_product
    error = (memory-direct).abs().max().item()
    assert error < 2e-14
    return dict(blocks=length, direct_all_pair_and_outer_leak_error=error,
                maximum_memory_magnitude=memory.abs().max().item())


def audit_spectrum(writer):
    omega = torch.linspace(0, torch.pi, 65, dtype=torch.float64)
    lags = torch.arange(-8192, 8193, dtype=torch.float64)
    direct = writer.timing_window(lags).to(torch.complex128) @ torch.exp(-1j*lags[:, None]*omega)
    analytic = writer.timing_spectrum(omega)
    error = (direct-analytic).abs().max().item()
    assert error < 3e-12
    # The pure odd special case has a negative imaginary spectrum in (0,pi).
    balanced = PairSTDP(1, 12, channels=4).double()
    spectrum = balanced.timing_spectrum(omega[1:-1])
    assert spectrum.real.abs().max() < 1e-14 and (spectrum.imag < 0).all()
    return dict(frequency_points=len(omega), explicit_DTFT_error=error,
                balanced_phase_radians=-torch.pi/2)


def audit_model():
    config = dict(batch_size=2, seq_len=9, grid=3, vocab_size=11,
                  num_puzzle_identifiers=1, puzzle_emb_ndim=0, hidden_size=32,
                  num_heads=2, loops=16, blocks_per_seg=4, num_layers=1,
                  mlp_expansion=1, legacy_gauge=False, use_trace=False,
                  block_order="pre", amp=False, forward_dtype="float64",
                  activity_channels=4, timing_tau_init=(2., 8.))
    model = CausalLT(config).double()
    batch = dict(inputs=torch.randint(0, 11, (2, 9)), puzzle_identifiers=torch.zeros(2, dtype=torch.long))
    with torch.no_grad():
        carry, _ = model(model.initial_carry(batch), batch)
        twice, output_two = model(carry, batch)
        whole = copy.deepcopy(model)
        whole.config.blocks_per_seg = 8
        once, output_one = whole(whole.initial_carry(batch), batch)
        torch.testing.assert_close(output_two["logits"], output_one["logits"], rtol=0, atol=0)
        torch.testing.assert_close(twice.trace, once.trace, rtol=0, atol=0)
        torch.testing.assert_close(twice.coupling, once.coupling, rtol=0, atol=0)
        contaminated = copy.deepcopy(carry)
        contaminated.halted[0] = True
        contaminated.current_hidden[0] = 1e6
        contaminated.trace[0] = 1e6
        contaminated.coupling[0] = 1e6
        reset, reset_output = model(contaminated, batch)
        clean, clean_output = model(model.initial_carry(batch), batch)
        torch.testing.assert_close(reset_output["logits"][0], clean_output["logits"][0], rtol=0, atol=0)
        torch.testing.assert_close(reset.trace[0], clean.trace[0], rtol=0, atol=0)
        torch.testing.assert_close(reset.coupling[0], clean.coupling[0], rtol=0, atol=0)
        torch.testing.assert_close(reset_output["logits"][1], output_two["logits"][1], rtol=0, atol=0)
        q = torch.randn(2, 9, 32, dtype=torch.float64)
        activities = model.inner.layers[0].temporal_write.activities(q)
        assert (activities >= 0).all() and (activities.sum(-1) < 1).all()
    _, out = model(model.initial_carry(batch), batch)
    out["logits"].square().mean().backward()
    gradients = {}
    for name, p in model.named_parameters():
        if "temporal_write" in name or name.endswith(("eta_raw", "lam_raw", "gain_raw")):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            gradients[name] = p.grad.norm().item()
            assert gradients[name] > 0, name
    # Loading into a fresh instance preserves parameters and mid-puzzle behavior.
    restored = CausalLT(config).double()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        ca, oa = model(copy.deepcopy(carry), batch)
        cb, ob = restored(copy.deepcopy(carry), batch)
        torch.testing.assert_close(oa["logits"], ob["logits"], rtol=0, atol=0)
        torch.testing.assert_close(ca.trace, cb.trace, rtol=0, atol=0)
        torch.testing.assert_close(ca.coupling, cb.coupling, rtol=0, atol=0)
    return dict(segment_carry_parity=True, per_puzzle_reset_parity=True,
                state_dict_and_carry_parity=True, gradient_norms=gradients)


def audit_cuda():
    from . import train_causal_stdp as entry
    from . import train
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda was requested but CUDA is unavailable")
    cfg = dict(entry.CFG, batch_size=2, seq_len=81, num_puzzle_identifiers=1,
               puzzle_emb_ndim=0, loops=2)
    model = CausalLT(cfg).cuda()
    batch = dict(inputs=torch.randint(0, 11, (2, 81), device="cuda"),
                 puzzle_identifiers=torch.zeros(2, dtype=torch.long, device="cuda"))
    carry, out = model(model.initial_carry(batch), batch)
    out["logits"].square().mean().backward()
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(carry.trace).all()
    assert torch.isfinite(carry.coupling).all()
    assert carry.trace.dtype == torch.float32 and carry.coupling.dtype == torch.float32
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    entry.configure_harness(cfg)
    for name, p in model.named_parameters():
        if "timing_tau_raw" in name or "timing_amplitude_logits" in name:
            assert train._is_no_decay(name, p)
        if "activity_weight" in name:
            assert not train._is_no_decay(name, p)
    return dict(device=torch.cuda.get_device_name(), hidden=832, heads=8, batch=2,
                blocks=8, amp=True, trace_dtype=str(carry.trace.dtype),
                memory_dtype=str(carry.coupling.dtype), all_parameter_gradients_finite=True,
                parameter_count=sum(p.numel() for p in model.parameters()),
                history_elements_per_puzzle=carry.trace[0].numel(),
                optimizer_decay_groups_checked=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also audit a full-size AMP forward/backward")
    args = parser.parse_args()
    torch.manual_seed(240926)
    torch.set_num_threads(2)
    writer = PairSTDP(3, 12, channels=4).double()
    with torch.no_grad():
        writer.timing_tau_raw[:, 1].add_(.21)
        writer.timing_amplitude_logits.normal_(0, .35)
        result = dict(pairs=audit_pairs(writer), stream=audit_stream(writer),
                      spectrum=audit_spectrum(writer))
    result["model"] = audit_model()
    if args.cuda:
        result["cuda"] = audit_cuda()
    result["scope"] = "Constructed-rule and implementation validation only; no Sudoku training or accuracy claim."
    root = Path("runs/causal_stdp_draft")
    root.mkdir(parents=True, exist_ok=True)
    (root/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
