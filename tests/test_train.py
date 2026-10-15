"""Harness: resuming from a checkpoint reproduces the uninterrupted run exactly (CPU, tiny model, real data)."""
import contextlib
import io
import tempfile
import unittest

import torch

from lt import train


def tiny_run(out, steps):
    cfg = dict(out_dir=out, max_steps=steps, hidden_size=16, num_heads=2, puzzle_emb_ndim=16, blocks_per_seg=2,
               loops=3, global_batch_size=4, test_size=8, epochs=2, eval_interval=1, lr_warmup_steps=2, compile=False,
               dataloader_workers=0, save_every_steps=0, log_every=1000, amp=False, kernel='torch')
    with contextlib.redirect_stdout(io.StringIO()):
        return train.main(cfg)


class Resume(unittest.TestCase):
    def test_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tiny_run(a, 7)
            tiny_run(b, 4)                       # stops at step 4 with a checkpoint (mid-puzzle: carry is saved)
            tiny_run(b, 7)                       # resumes from it
            ca = torch.load(f'{a}/step_7.pt', weights_only=False)
            cb = torch.load(f'{b}/step_7.pt', weights_only=False)
            for key in ('raw_model', 'model', 'ema'):
                for name, value in ca[key].items():
                    self.assertTrue(torch.equal(value, cb[key][name]), f'{key}.{name}')
            for name in ('hidden', 'clock', 'key_trace', 'value_trace'):
                self.assertTrue(torch.equal(ca['rank_states'][0]['carry'][name], cb['rank_states'][0]['carry'][name]))


if __name__ == '__main__':
    unittest.main()
