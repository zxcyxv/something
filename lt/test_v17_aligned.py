import unittest
import torch
from pathlib import Path
from . import train as t
from .audit_training_harness import load_old
from .research_v17_aligned import install_historical_initialization


class V17AlignedTests(unittest.TestCase):
    def test_exact_initialization_outputs_and_gradients(self):
        torch.set_num_threads(2)
        old=load_old(Path('docs/research/2026-10-03/reference/train_v17.py'))
        cfg=dict(t.CFG,**old.CFG);cfg.update(t.PRESETS['v1.7'])
        cfg.update(hidden_size=16,num_heads=2,grid=3,seq_len=9,batch_size=2,
            puzzle_emb_ndim=16,num_puzzle_identifiers=1,amp=False,
            activation_checkpoint=False,num_layers=1,blocks_per_seg=2)
        original_inner,original_id=t.LT_Inner,t.model_id_of
        try:
            torch.manual_seed(21);reference=old.LT(cfg)
            rng=torch.random.get_rng_state()
            install_historical_initialization(old)
            torch.manual_seed(21);current=t.LT(cfg)
            torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
            for key,value in reference.state_dict().items():
                torch.testing.assert_close(current.state_dict()[key],value,rtol=0,atol=0)
            batch=dict(inputs=torch.randint(0,11,(2,9)),puzzle_identifiers=torch.zeros(2,dtype=torch.long))
            a=reference.initial_carry(batch);b=current.initial_carry(batch)
            for step in range(1,17):
                a,ya=reference(a,batch);b,yb=current(b,batch)
                torch.testing.assert_close(ya['logits'],yb['logits'],rtol=0,atol=0)
                torch.testing.assert_close(a.current_hidden,b.current_hidden,rtol=0,atol=0)
                self.assertEqual(bool(b.halted.all()),step==16)
            ya['logits'].square().mean().backward();yb['logits'].square().mean().backward()
            params=dict(current.named_parameters())
            for name,pa in reference.named_parameters():
                pb=params[name]
                if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
        finally:t.LT_Inner,t.model_id_of=original_inner,original_id


if __name__=='__main__':unittest.main()
