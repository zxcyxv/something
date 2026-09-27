"""CPU regressions for replacing the v1.7 QR address with its effective matrix."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from . import train as t
from . import convert_v171 as conversion


def small_config(preset="v1.7", **overrides):
    cfg = dict(t.CFG)
    cfg.update(t.PRESETS[preset])
    cfg.update(hidden_size=16, num_heads=2, grid=3, seq_len=9, batch_size=2,
               num_puzzle_identifiers=1, puzzle_emb_ndim=16, vocab_size=11,
               num_layers=1, blocks_per_seg=2, loops=4, amp=False,
               forward_dtype="float32", compile=False)
    cfg.update(overrides)
    return cfg


def batch():
    labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [5, 6, 7, 8, 9, 10, 2, 3, 4]], dtype=torch.long)
    inputs = labels.clone()
    inputs[:, ::2] = 1
    return dict(inputs=inputs, labels=labels,
                puzzle_identifiers=torch.zeros(2, dtype=torch.int32))


def effective_linear_state(state):
    """Independent reference conversion; intentionally does not use the converter."""
    result = {}
    for name, value in state.items():
        if name.endswith(".wc_raw"):
            q, _ = torch.linalg.qr(value.transpose(-1, -2))
            result[name.removesuffix("_raw")] = q.transpose(-1, -2).contiguous()
        else:
            result[name] = value.clone()
    return result


class V171Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def converted_pair(self):
        old = t.LT(small_config(address_projection="qr")).eval()
        with torch.no_grad():
            # An initialized zero down projection would miss the MLP path.
            old.inner.layers[0].b_down.weight.normal_(mean=0, std=0.03)
            old.inner.puzzle_emb.weights.normal_(mean=0, std=0.02)
        new = t.LT(small_config("v1.71")).eval()
        new.load_state_dict(effective_linear_state(old.state_dict()), strict=True)
        return old, new

    def assert_carry_close(self, a, b):
        for name in ("current_hidden", "coupling", "trace"):
            av, bv = getattr(a, name), getattr(b, name)
            if av is None:
                self.assertIsNone(bv)
            else:
                torch.testing.assert_close(av, bv, atol=4e-6, rtol=4e-6)
        torch.testing.assert_close(a.steps, b.steps, atol=0, rtol=0)
        torch.testing.assert_close(a.halted, b.halted, atol=0, rtol=0)

    @torch.no_grad()
    def test_converted_logits_hidden_coupling_and_trace_across_segments(self):
        old, new = self.converted_pair()
        self.assertEqual(new.config.address_projection, "linear")
        self.assertEqual(new.config.block_order, "post")
        self.assertTrue(new.config.use_trace)
        data = batch()
        a, b = old.initial_carry(data), new.initial_carry(data)
        # Cover memory accumulation and the first automatic fresh-puzzle reset.
        for _ in range(6):
            a, ao = old(a, data)
            b, bo = new(b, data)
            torch.testing.assert_close(ao["logits"], bo["logits"], atol=4e-6, rtol=4e-6)
            self.assert_carry_close(a, b)
        self.assertGreater(float(a.coupling.abs().max()), 0)
        self.assertGreater(float(a.trace.abs().max()), 0)

    @torch.no_grad()
    def test_existing_presets_default_to_unchanged_qr_path(self):
        for preset in ("v1", "v1.1", "v2", "v1.7"):
            with self.subTest(preset=preset):
                cfg = small_config(preset)
                cfg.pop("address_projection", None)
                torch.manual_seed(23)
                implicit = t.LT(cfg).eval()
                torch.manual_seed(23)
                explicit = t.LT(dict(cfg, address_projection="qr")).eval()
                self.assertEqual(implicit.config.address_projection, "qr")
                self.assertIn("inner.layers.0.wc_raw", implicit.state_dict())
                self.assertNotIn("inner.layers.0.wc", implicit.state_dict())
                for key, value in implicit.state_dict().items():
                    torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
                data = batch()
                a, b = implicit.initial_carry(data), explicit.initial_carry(data)
                for _ in range(2):
                    a, ao = implicit(a, data)
                    b, bo = explicit(b, data)
                    torch.testing.assert_close(ao["logits"], bo["logits"], atol=0, rtol=0)
                    self.assert_carry_close(a, b)

    def test_linear_forward_backward_and_actual_update_do_not_call_qr(self):
        _, new = self.converted_pair()
        new.train()
        data = batch()
        address = new.inner.layers[0].wc
        before = address.detach().clone()
        optimizer = t.AdamATan2(new.parameters(), lr=1e-4, weight_decay=0)
        with mock.patch.object(torch.linalg, "qr", side_effect=AssertionError("linear path called QR")):
            carry = new.initial_carry(data)
            for _ in range(2):
                carry, outputs = new(carry, data)
            loss = t.stablemax_cross_entropy(outputs["logits"], data["labels"]).mean()
            loss.backward()
            self.assertIsNotNone(address.grad)
            self.assertTrue(bool(torch.isfinite(address.grad).all()))
            self.assertGreater(float(address.grad.norm()), 0)
            optimizer.step()
        self.assertTrue(bool(torch.isfinite(address).all()))
        self.assertGreater(float((address.detach() - before).norm()), 0)

    @torch.no_grad()
    def test_linear_state_save_reload_and_wrong_checkpoint_rejection(self):
        old, new = self.converted_pair()
        cfg = small_config("v1.71")
        stream = io.BytesIO()
        torch.save(dict(cfg=cfg, model_state_dict=new.state_dict()), stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        restored = t.LT(checkpoint["cfg"]).eval()
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        data = batch()
        a, b = new.initial_carry(data), restored.initial_carry(data)
        for _ in range(3):
            a, ao = new(a, data)
            b, bo = restored(b, data)
            torch.testing.assert_close(ao["logits"], bo["logits"], atol=0, rtol=0)
            self.assert_carry_close(a, b)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(old.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            old.load_state_dict(new.state_dict(), strict=True)
        # Exercise the actual resume loader, including historical configs that
        # omit the new field and a mislabeled old state with linear metadata.
        old_cfg = small_config()
        old_cfg.pop("address_projection", None)
        for saved_cfg in (old_cfg, dict(old_cfg, address_projection="linear")):
            invalid = io.BytesIO()
            torch.save(dict(cfg=saved_cfg, model_state_dict=old.state_dict()), invalid)
            invalid.seek(0)
            with self.assertRaisesRegex(ValueError, "address_projection"):
                t.load_checkpoint(invalid, restored, [], "cpu", load_optimizer=False)

    def test_conversion_helper_preserves_effective_projections_and_other_tensors(self):
        state = {
            "module._orig_mod.model.inner.layers.0.wc_raw": torch.randn(2, 8, 16),
            "module._orig_mod.model.inner.layers.1.wc_raw": torch.randn(2, 8, 16),
            "module._orig_mod.model.inner.init_hidden": torch.randn(16),
        }
        original = {key: value.clone() for key, value in state.items()}
        converted = conversion.convert_state(state, device="cpu")
        expected = effective_linear_state({key.replace("_orig_mod.", ""): value
                                           for key, value in state.items()})
        self.assertEqual(set(converted), set(expected))
        for key, value in converted.items():
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        for key, value in state.items():
            torch.testing.assert_close(value, original[key], atol=0, rtol=0)

    def test_checkpoint_converts_raw_evaluation_and_shadow_independently(self):
        old, _ = self.converted_pair()
        raw = {"model." + key: value.clone() for key, value in old.state_dict().items()}
        evaluation = {key: value.clone() for key, value in raw.items()}
        parameter_keys = {"model." + key for key, _ in old.named_parameters()}
        shadow = {key: value.clone() for key, value in raw.items() if key in parameter_keys}
        projection = "model.inner.layers.0.wc_raw"
        evaluation[projection] += 0.15 * torch.randn_like(evaluation[projection])
        shadow[projection] += 0.25 * torch.randn_like(shadow[projection])
        evaluation["model.inner.layers.0.beta"] += 0.1
        shadow["model.inner.layers.0.beta"] -= 0.2
        sections = dict(raw_model_state_dict=raw, model_state_dict=evaluation, ema_shadow=shadow)
        checkpoint = dict(cfg=small_config(preset="v1.7"), step=380000,
                          optimizer_states=[{"old_optimizer": True}], iter_id=42, batch_in_iter=99,
                          rng_state=torch.random.get_rng_state(), recurrent_carry={"old": True},
                          **sections)
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source_v17.pt", Path(directory) / "v171.pt"
            torch.save(checkpoint, source)
            source_before = source.read_bytes()
            conversion.convert_checkpoint(source, destination, device="cpu")
            converted = torch.load(destination, map_location="cpu", weights_only=True)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(converted["cfg"]["address_projection"], "linear")
            self.assertEqual(converted["step"], 380000)
            self.assertEqual(converted["iter_id"], 0)
            self.assertEqual(converted["batch_in_iter"], 0)
            for key in ("optimizer_states", "rng_state", "recurrent_carry"):
                self.assertNotIn(key, converted)
            for section, original in sections.items():
                expected = effective_linear_state(original)
                self.assertEqual(set(converted[section]), set(expected))
                for key, value in expected.items():
                    torch.testing.assert_close(converted[section][key], value, atol=0, rtol=0)
            linear_key = projection.removesuffix("_raw")
            self.assertFalse(torch.equal(converted["raw_model_state_dict"][linear_key],
                                         converted["model_state_dict"][linear_key]))
            self.assertFalse(torch.equal(converted["model_state_dict"][linear_key],
                                         converted["ema_shadow"][linear_key]))
            restored = t.ACTLossHead(t.LT(converted["cfg"]), "stablemax_cross_entropy").eval()
            restored.load_state_dict(converted["raw_model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
