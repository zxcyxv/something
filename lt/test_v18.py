"""CPU regressions for v1.8 selective plasticity and the harness options added with it."""
import tempfile
import unittest

import torch

from . import train as t


def small_config(**overrides):
    cfg = dict(t.DEFAULT_CFG)
    cfg.update(hidden_size=16, num_heads=2, puzzle_emb_ndim=16, global_batch_size=2, batch_size=2,
               seq_len=81, num_puzzle_identifiers=1, blocks_per_seg=2, loops=3, amp=False,
               amp_dtype="float32", activation_checkpoint=False, compile=False, epochs=4, eval_interval=2,
               lr_warmup_steps=0, lr=1e-3, puzzle_emb_lr=1e-3, ema_rate=0.9, num_aug=10,
               run_selftests=False, dataloader_workers=0, data_fingerprint="fp")
    cfg.update(overrides)
    return cfg


V171 = dict(plastic_select=False, lr_rewarm_start=None, lr_min_ratio=1.0, late_sup_prob=0.0)


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(inputs=torch.randint(1, 11, (2, 81), generator=g), labels=torch.randint(2, 11, (2, 81), generator=g),
                puzzle_identifiers=torch.zeros(2, dtype=torch.int32))


def build(cfg, seed=0):
    torch.manual_seed(seed)
    model = t.ACTLossHead(t.LT(cfg), q_weight=cfg["q_weight"])
    opts, lrs = t.create_optimizers(model, cfg, 1)
    ema = t.EMAHelper(cfg["ema_rate"])
    ema.register(model)
    return model, opts, lrs, ema


def train_steps(model, ts, opts, lrs, ema, cfg, n, seed0=0):
    for j in range(n):
        t.train_batch(model, model, ts, batch(seed0 + j), cfg, opts, lrs, 16, 0, 1, torch.device("cpu"))
        ema.update(model)
        ts.batch_in_iter += 1


def run_segments(lt, n):
    b = batch(99)
    carry, outs = lt.initial_carry(b), []
    with torch.no_grad():
        for _ in range(n):
            carry, out = lt(carry, b)
            outs.append(out["logits"])
    return carry, torch.stack(outs)


class V18Test(unittest.TestCase):
    def test_presets_keep_old_models_unselective(self):
        for name, preset in t.PRESETS.items():
            self.assertEqual(preset["plastic_select"], name == "v1.8")
        self.assertTrue(t.DEFAULT_CFG["plastic_select"])          # Kaggle CFG 는 실행마다 바꾸는 설정이라 검사하지 않는다
        self.assertEqual(set(t.CFG) - {"data_npz"}, set(t.DEFAULT_CFG) - {"data_npz"})

    def test_config_rejects_fixed_gain_or_lambda_with_selection(self):
        for key in ("stdp_gain_fixed", "stdp_lam_fixed"):
            with self.assertRaises(ValueError):
                t.LT(small_config(**{key: 0.5}))

    def test_legacy_v171_checkpoint_resumes_with_neutral_new_keys(self):
        cfg = small_config(**V171)
        model, opts, lrs, ema = build(cfg)
        ts = t.TrainState()
        train_steps(model, ts, opts, lrs, ema, cfg, 2)
        with tempfile.TemporaryDirectory() as d:
            path = t.save_training_checkpoint(d, ts, model, opts, ema, cfg, 0, 1, torch.device("cpu"))
            ck = torch.load(path, weights_only=False)
            for key in t._LEGACY_DEFAULTS:            # a checkpoint written before these keys existed
                ck["cfg"].pop(key, None)
            torch.save(ck, path)
            m2, o2, l2, e2 = build(cfg, seed=1)
            rs = t.load_training_checkpoint(path, m2, o2, e2, cfg, 0, 1, torch.device("cpu"))
            self.assertEqual(rs.step, 2)
            with self.assertRaisesRegex(ValueError, "model_id"):
                c18 = small_config()
                m3, o3, _, e3 = build(c18)
                t.load_training_checkpoint(path, m3, o3, e3, c18, 0, 1, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                t.load_training_checkpoint(path, m2, o2, e2, dict(cfg, lr_rewarm_start=0), 0, 1, torch.device("cpu"))

    def test_init_from_v171_converts_raw_and_ema_and_keeps_cursor(self):
        cfg = small_config(**V171)
        model, opts, lrs, ema = build(cfg)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith(("eta_raw", "lam_raw", "gain_raw")):
                    p.add_(torch.randn_like(p) * 0.5)
        ts = t.TrainState()
        train_steps(model, ts, opts, lrs, ema, cfg, 3)
        with tempfile.TemporaryDirectory() as d:
            path = t.save_training_checkpoint(d, ts, model, opts, ema, cfg, 0, 1, torch.device("cpu"))
            c18 = small_config()
            new, _, _, nema = build(c18, seed=3)
            rs = t.init_from_checkpoint(path, new, nema, c18, torch.device("cpu"))
            self.assertEqual((rs.step, rs.iter_id, rs.batch_in_iter, rs.carry), (3, ts.iter_id, ts.batch_in_iter, None))
            self.assertEqual(c18["init_from_model_id"], t.MODEL_ID)
            for weights in ("raw", "ema"):
                with t._EMASwap(model, ema if weights == "ema" else None), t._EMASwap(new, nema if weights == "ema" else None):
                    ca, la = run_segments(model.model, 3)
                    cb, lb = run_segments(new.model, 3)
                torch.testing.assert_close(lb, la, rtol=1e-4, atol=2e-5)
                torch.testing.assert_close(cb.coupling, ca.coupling, rtol=1e-4, atol=2e-5)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                t.init_from_checkpoint(path, new, nema, dict(c18, data_fingerprint="other"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "select_g_max"):
                t.init_from_checkpoint(path, new, nema, dict(c18, select_g_max=1e-3), torch.device("cpu"))

    def test_activation_checkpoint_gradients_match(self):
        grads = []
        for ckpt in (False, True):
            cfg = small_config(activation_checkpoint=ckpt)
            model, _, _, _ = build(cfg)
            with torch.no_grad():
                model.model.inner.layers[0].sel_w.normal_(0, 0.2)
            b = batch()
            _, loss, _, _, _ = model(carry=model.initial_carry(b), batch=b, return_keys=set())
            loss.backward()
            grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        self.assertEqual(set(grads[0]), set(grads[1]))
        for n in grads[0]:
            torch.testing.assert_close(grads[0][n], grads[1][n], rtol=2e-5, atol=3e-6, msg=n)

    def test_nograd_schedule(self):
        cfg = small_config(nograd_every=10000, nograd_start=100000, nograd_max=16)
        self.assertEqual([t.nograd_at(s, cfg) for s in (0, 99999, 100000, 109999, 110000, 139999, 140000, 10**6)],
                         [0, 0, 1, 1, 2, 4, 5, 16])
        self.assertEqual(t.nograd_at(10**6, small_config(nograd_every=0)), 0)
        fixed = small_config(nograd_fixed=8, nograd_every=10000, nograd_start=100000)
        self.assertEqual([t.nograd_at(s, fixed) for s in (0, 5, 10**6)], [8, 8, 8])

    def test_nograd_resume_across_depth_change_is_exact(self):
        cfg = small_config(nograd_every=2, nograd_start=1, nograd_max=3)
        model, opts, lrs, ema = build(cfg)
        ts = t.TrainState()
        train_steps(model, ts, opts, lrs, ema, cfg, 3)
        self.assertEqual(model.model.config.nograd_blocks, t.nograd_at(2, cfg))
        with tempfile.TemporaryDirectory() as d:
            path = t.save_training_checkpoint(d, ts, model, opts, ema, cfg, 0, 1, torch.device("cpu"))
            train_steps(model, ts, opts, lrs, ema, cfg, 2, seed0=3)          # 스텝 3 에서 깊이가 1 → 2 로 바뀐다
            m2, o2, l2, e2 = build(cfg, seed=5)
            rs = t.load_training_checkpoint(path, m2, o2, e2, cfg, 0, 1, torch.device("cpu"))
            train_steps(m2, rs, o2, l2, e2, cfg, 2, seed0=3)
            self.assertEqual((t.nograd_at(2, cfg), t.nograd_at(3, cfg)), (1, 2))
            self.assertEqual(model.model.config.nograd_blocks, m2.model.config.nograd_blocks)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, m2.state_dict()[name]), name)
            for name in ("current_hidden", "coupling", "trace", "steps", "halted"):
                self.assertTrue(torch.equal(getattr(ts.carry, name), getattr(rs.carry, name)), name)

    def test_lr_schedule_legacy_path_unchanged(self):
        cfg = small_config(lr_rewarm_start=None, lr_min_ratio=0.5, lr_warmup_steps=10)
        for s in (0, 5, 10, 50, 100):
            self.assertEqual(t.lr_at(s, 1.0, cfg, 100),
                             t.cosine_schedule_with_warmup_lr_lambda(s, base_lr=1.0, num_warmup_steps=10,
                                                                     num_training_steps=100, min_ratio=0.5))
        # 재개 재가열: 100k 에서 1e-5, 10k 동안 선형으로 1e-4, 이후 유지
        cfg = small_config(lr_rewarm_start=100000, lr_rewarm_steps=10000, lr_rewarm_from_ratio=0.1,
                           lr_min_ratio=1.0, lr_warmup_steps=2000)
        self.assertEqual([round(t.lr_at(s, 1e-4, cfg, 390625), 12) for s in (99999, 100000, 105000, 110000, 150000)],
                         [1e-4, 1e-5, 5.5e-5, 1e-4, 1e-4])


if __name__ == "__main__":
    unittest.main()
