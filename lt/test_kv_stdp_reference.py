"""Audit KV-STDP against an explicit sum over event pairs, not trace recursion.

Indices: r/s = recurrent block times, p = token, i/j = V/K channels.
For r > s, the actual heterogeneous decay convention gives
  + (1-lambda_j) lambda_j**(r-s-1) v[r,p,i] k[s,p,j]
  - (1-lambda_i) lambda_i**(r-s-1) v[s,p,i] k[r,p,j].
Spatial rotation acts only on the key channel. Query only reads the result.
"""
from dataclasses import replace
import unittest

import torch

from . import train as t
from .kv_stability import HistoricalKeyTraceInner


def config(**changes):
    values = dict(t.CFG, hidden_size=16, num_heads=2, grid=2, seq_len=4,
                  batch_size=2, num_puzzle_identifiers=1, puzzle_emb_ndim=0,
                  amp=False, activation_checkpoint=False, blocks_per_seg=2,
                  nograd_blocks=0, loops=4, forward_dtype="float64")
    values.update(changes)
    return t.LTConfig.from_dict(values)


def rotate_reference(x, theta, grid):
    """Independent real 2x2 rotation, with explicit position/channel loops."""
    positions = []
    for p in range(x.shape[-2]):
        channels = []
        for j in range(x.shape[-1] // 2):
            angle = theta[:, j, 0] * (p // grid) + theta[:, j, 1] * (p % grid)
            a, b = x[..., p, 2*j], x[..., p, 2*j+1]
            channels.extend((a * angle.cos() - b * angle.sin(),
                             a * angle.sin() + b * angle.cos()))
        positions.append(torch.stack(channels, dim=-1))
    return torch.stack(positions, dim=-2)


def pair_sum_reference(q, keys, values, lam, theta, grid, reduction="mean"):
    """Full history sum, directly forming every ordered pair once.

    No calls to implementation helpers and no eligibility/memory recurrence.
    """
    rotated = [rotate_reference(k, theta, grid) for k in keys]
    total = q.new_zeros(q.shape[0], q.shape[1], values[0].shape[-1], keys[0].shape[-1])
    for r in range(len(keys)):
        for s in range(r):
            weight = (1-lam) * lam.pow(r-s-1)
            for p in range(q.shape[-2]):
                vp, vs = values[r][:, :, p], values[s][:, :, p]
                kp, ks = rotated[r][:, :, p], rotated[s][:, :, p]
                total = total + vp.unsqueeze(-1) * (weight * ks).unsqueeze(-2)
                total = total - (weight * vs).unsqueeze(-1) * kp.unsqueeze(-2)
    if reduction == "mean":
        total = total / q.shape[-2]
    qr = rotate_reference(q, theta, grid)
    read = (qr.unsqueeze(-2) * total.unsqueeze(-3)).sum(-1)
    return read, total


def trace_reference(history, lam):
    return sum((1-lam[None, :, None, :]) * lam[None, :, None, :].pow(len(history)-1-s) * x
               for s, x in enumerate(history))


class KVSTDPReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(83)

    def activities(self, count=6):
        return [tuple(torch.randn(2, 2, 4, 8, dtype=torch.float64, requires_grad=True)
                      for _ in range(3)) for _ in range(count)]

    def test_every_prefix_matches_explicit_event_pairs_and_read(self):
        for mode in ("head", "pair"):
            for reduction in ("mean", "sum"):
                with self.subTest(mode=mode, reduction=reduction):
                    inner = t.KVSTDPInner(config(trace_decay_mode=mode,
                                                kv_write_reduction=reduction)).double()
                    layer = inner.layers[0]
                    with torch.no_grad():
                        layer.trace_lam_raw.copy_(torch.linspace(-2, 1, layer.trace_lam_raw.numel())
                                                 .reshape_as(layer.trace_lam_raw))
                    lam = layer.trace_decay_channels
                    keys, values = [], []
                    memory = ek = ev = None
                    for q, k, v in self.activities():
                        keys.append(k); values.append(v)
                        read, memory, ek, ev = inner.memory_step(layer, q, k, v, memory, ek, ev)
                        expected_read, expected_memory = pair_sum_reference(
                            q, keys, values, lam, layer.theta, 2, reduction)
                        for actual, expected in ((read, expected_read), (memory, expected_memory),
                                                 (ek, trace_reference(keys, lam)),
                                                 (ev, trace_reference(values, lam))):
                            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    def test_full_history_gradients_match_event_pair_reference(self):
        inner = t.KVSTDPInner(config()).double()
        layer = inner.layers[0]
        with torch.no_grad():
            layer.trace_lam_raw.normal_()
        activities = self.activities()
        memory = ek = ev = None
        for q, k, v in activities:
            read, memory, ek, ev = inner.memory_step(layer, q, k, v, memory, ek, ev)
        expected_read, expected_memory = pair_sum_reference(
            activities[-1][0], [x[1] for x in activities], [x[2] for x in activities],
            layer.trace_decay_channels, layer.theta, 2)
        targets = (activities[-1][0],) + tuple(x for triple in activities for x in triple[1:])
        targets += (layer.trace_lam_raw, layer.theta)
        actual_grads = torch.autograd.grad(read.square().sum()+memory.square().sum(), targets)
        expected_grads = torch.autograd.grad(expected_read.square().sum()+expected_memory.square().sum(), targets)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    def test_isolated_pulses_have_correct_sign_lag_and_no_same_time_pair(self):
        inner = t.KVSTDPInner(config(trace_decay_mode="head", kv_write_reduction="sum")).double()
        layer = inner.layers[0]
        with torch.no_grad():
            layer.theta.zero_()
        for delay in (0, 1, 2, 5):
            for pre_first in (False, True):
                memory = ek = ev = None
                for r in range(delay+1):
                    q, k, v = [torch.zeros(2, 2, 4, 8, dtype=torch.float64) for _ in range(3)]
                    if r == (0 if pre_first else delay):
                        k[0, 1, 2, 3] = 2
                    if r == (delay if pre_first else 0):
                        v[0, 1, 2, 5] = 3
                    _, memory, ek, ev = inner.memory_step(layer, q, k, v, memory, ek, ev)
                expected = torch.zeros_like(memory)
                if delay:
                    lam = layer.trace_decay[1]
                    expected[0, 1, 5, 3] = (1 if pre_first else -1)*6*(1-lam)*lam**(delay-1)
                torch.testing.assert_close(memory, expected, rtol=2e-12, atol=2e-12)

    def test_distinct_tokens_do_not_create_a_write_pair(self):
        inner = t.KVSTDPInner(config()).double()
        z = torch.zeros(2, 2, 4, 8, dtype=torch.float64)
        k, v = z.clone(), z.clone()
        k[0, 0, 0, 0] = 1
        v[0, 0, 1, 1] = 1
        _, memory, ek, ev = inner.memory_step(inner.layers[0], z, k, z)
        _, memory, _, _ = inner.memory_step(inner.layers[0], z, z, v, memory, ek, ev)
        torch.testing.assert_close(memory, torch.zeros_like(memory), rtol=0, atol=0)

    def test_mid_history_lane_reset_drops_only_that_lanes_old_pairs(self):
        inner = t.KVSTDPInner(config()).double()
        layer = inner.layers[0]
        history = self.activities()
        memory = ek = ev = None
        for r, (q, k, v) in enumerate(history):
            fresh = torch.tensor([True, False]) if r == 3 else None
            read, memory, ek, ev = inner.memory_step(layer, q, k, v, memory, ek, ev, fresh)
            for b in range(2):
                start = 3 if b == 0 and r >= 3 else 0
                keys = [x[1][b:b+1] for x in history[start:r+1]]
                values = [x[2][b:b+1] for x in history[start:r+1]]
                yr, mr = pair_sum_reference(q[b:b+1], keys, values, layer.trace_decay_channels,
                                           layer.theta, 2)
                torch.testing.assert_close(memory[b:b+1], mr, rtol=2e-12, atol=2e-12)
                torch.testing.assert_close(read[b:b+1], yr, rtol=2e-12, atol=2e-12)

    def test_segment_boundaries_preserve_state_and_detach_history(self):
        model = t.LT(vars(config())).double()
        batch = dict(inputs=torch.randint(1, 11, (2, 4)), labels=torch.randint(1, 11, (2, 4)),
                     puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        carry = model.initial_carry(batch)
        for _ in range(3):
            carry, out = model(carry, batch)
        model.config.blocks_per_seg = 6
        longer, other = model(model.initial_carry(batch), batch)
        for name in ("current_hidden", "coupling", "key_trace", "value_trace"):
            torch.testing.assert_close(getattr(carry, name), getattr(longer, name), rtol=0, atol=0)
            self.assertFalse(getattr(carry, name).requires_grad)
        torch.testing.assert_close(out["logits"], other["logits"], rtol=0, atol=0)

    def test_harness_resets_memory_traces_and_input_together(self):
        model = t.LT(vars(config())).double()
        batch = dict(inputs=torch.randint(1, 11, (2, 4)), labels=torch.randint(1, 11, (2, 4)),
                     puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        carry, _ = model(model.initial_carry(batch), batch)
        carry = replace(carry, halted=torch.tensor([True, False]))
        new = {k: v.flip(0) for k, v in batch.items()}
        mixed, mixed_out = model(carry, new)
        restarted, restart_out = model(model.initial_carry(new), new)
        continued, continue_out = model(replace(carry, halted=torch.zeros(2, dtype=torch.bool)), batch)
        for name in ("current_hidden", "coupling", "key_trace", "value_trace"):
            torch.testing.assert_close(getattr(mixed, name)[0], getattr(restarted, name)[0], rtol=0, atol=0)
            torch.testing.assert_close(getattr(mixed, name)[1], getattr(continued, name)[1], rtol=0, atol=0)
        torch.testing.assert_close(mixed_out["logits"][0], restart_out["logits"][0], rtol=0, atol=0)
        torch.testing.assert_close(mixed_out["logits"][1], continue_out["logits"][1], rtol=0, atol=0)
        self.assertTrue(torch.equal(mixed.current_data["inputs"][0], new["inputs"][0]))
        self.assertTrue(torch.equal(mixed.current_data["inputs"][1], batch["inputs"][1]))

    def test_checkpoint_recompute_preserves_forward_and_parameter_gradients(self):
        plain = t.LT(vars(config(blocks_per_seg=8))).double()
        checked = t.LT(vars(config(blocks_per_seg=8, activation_checkpoint=True))).double()
        checked.load_state_dict(plain.state_dict())
        batch = dict(inputs=torch.randint(1, 11, (2, 4)), labels=torch.randint(1, 11, (2, 4)),
                     puzzle_identifiers=torch.zeros(2, dtype=torch.int32))
        for model in (plain, checked):
            carry, _ = model(model.initial_carry(batch), batch)
            _, output = model(carry, batch)
            output["logits"].square().mean().backward()
        for (name, p), (other_name, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_historical_key_coordinates_preserve_initialization_and_fixed_theta_operator(self):
        torch.manual_seed(31)
        original = t.KVSTDPInner(config()).double()
        torch.manual_seed(31)
        historical = HistoricalKeyTraceInner(config()).double()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, historical.state_dict()[name], rtol=0, atol=0)
        activities = self.activities()
        outputs = []
        for inner in (original, historical):
            memory = ek = ev = None
            for q, k, v in activities:
                read, memory, ek, ev = inner.memory_step(inner.layers[0], q, k, v, memory, ek, ev)
            outputs.append((read, memory, ek, ev))
        for i in (0, 1, 3):
            torch.testing.assert_close(outputs[0][i], outputs[1][i], rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(original.apply_rope(outputs[0][2]), outputs[1][2], rtol=2e-12, atol=2e-12)
        gradients = []
        for inner, output in zip((original, historical), outputs):
            targets = (activities[-1][0],) + tuple(x for row in activities for x in row[1:])
            targets += (inner.layers[0].trace_lam_raw, inner.layers[0].theta)
            gradients.append(torch.autograd.grad(output[0].square().sum()+output[1].square().sum(), targets))
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-11)

    def test_historical_key_trace_matches_event_pairs_when_theta_changes(self):
        inner = HistoricalKeyTraceInner(config()).double()
        layer = inner.layers[0]
        memory = ek = ev = None
        keys, values = [], []
        lam = layer.trace_decay_channels.detach()
        with torch.no_grad():
            for q, k, v in self.activities():
                layer.theta.normal_()
                keys.append(rotate_reference(k, layer.theta, 2))
                values.append(v)
                read, memory, ek, ev = inner.memory_step(layer, q, k, v, memory, ek, ev)
                expected = torch.zeros_like(memory)
                for r in range(len(keys)):
                    for s in range(r):
                        weight = (1-lam)*lam.pow(r-s-1)
                        for p in range(4):
                            expected += values[r][:, :, p, :, None]*(weight*keys[s][:, :, p])[:, :, None, :]/4
                            expected -= (weight*values[s][:, :, p])[:, :, :, None]*keys[r][:, :, p, None, :]/4
                qr = rotate_reference(q, layer.theta, 2)
                wanted_read = (qr.unsqueeze(-2)*expected.unsqueeze(-3)).sum(-1)
                torch.testing.assert_close(memory, expected, rtol=2e-12, atol=2e-12)
                torch.testing.assert_close(read, wanted_read, rtol=2e-12, atol=2e-12)

    def test_historical_key_reset_ignores_old_coordinate_state(self):
        inner = HistoricalKeyTraceInner(config()).double()
        q, k, v = self.activities(1)[0]
        m = torch.randn(2, 2, 8, 8, dtype=torch.float64)
        ek, ev = torch.randn_like(k), torch.randn_like(v)
        reset = inner.memory_step(inner.layers[0], q, k, v, m, ek, ev, torch.tensor([True, False]))
        fresh = inner.memory_step(inner.layers[0], q, k, v)
        old = inner.memory_step(inner.layers[0], q, k, v, m, ek, ev)
        for x, y, z in zip(reset, fresh, old):
            torch.testing.assert_close(x[0], y[0], rtol=0, atol=0)
            torch.testing.assert_close(x[1], z[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
