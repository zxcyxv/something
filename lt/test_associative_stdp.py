"""Check time conventions, frozen reads, and paired experimental controls."""
import unittest

import numpy as np
import torch

from .simulate_associative_stdp import (AssociativeMemory,episodes,evaluate,
    mechanism_checks,numpy_step,outer,reconstruction_loss)


class AssociativeSTDPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_trace_matches_explicit_all_past_pairs(self):
        rng=np.random.default_rng(9)
        ks,vs=rng.normal(size=(2,12,7))
        ek=ev=np.zeros(7)
        lp,lm,ap,am=.8,.3,1.4,.6
        for r,(k,v) in enumerate(zip(ks,vs)):
            actual,ek,ev=numpy_step(k,v,ek,ev,'general_stdp',lp,lm,ap,am)
            expected=np.zeros((7,7))
            for s in range(r):
                expected += ap*(1-lp)*lp**(r-s-1)*outer(v,ks[s])
                expected -= am*(1-lm)*lm**(r-s-1)*outer(vs[s],k)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_pulse_sign_static_cancellation_skew_and_retention(self):
        checks=mechanism_checks()
        for value in checks.values():
            if isinstance(value,float):self.assertLess(value,1e-12)

    def test_general_window_initially_equals_project_rule(self):
        balanced=AssociativeMemory('balanced_stdp',dim=8,hidden=8)
        general=AssociativeMemory('general_stdp',dim=8,hidden=8)
        general.load_state_dict(balanced.state_dict(),strict=False)
        data=episodes(torch.Generator().manual_seed(51),4,3,8)
        p1,m1=balanced(*data[:3]);p2,m2=general(*data[:3])
        torch.testing.assert_close(p1,p2,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(m1,m2,rtol=1e-5,atol=1e-6)
        reconstruction_loss(p2[-1],data[3],data[2]).backward()
        for name in ('q','k','v'):
            grad=getattr(general,name).weight.grad
            self.assertTrue(grad.isfinite().all());self.assertGreater(float(grad.norm()),0)
        self.assertTrue(general.balance_logit.grad.isfinite())
        self.assertTrue(general.trace_logits.grad.isfinite().all())

    def test_frozen_recall_does_not_write_or_use_recall_target(self):
        model=AssociativeMemory('balanced_stdp',dim=8,hidden=8)
        bank,cue,observed,target=episodes(torch.Generator().manual_seed(8),4,3,8)
        memory=model.store(bank);before=memory.clone()
        with torch.no_grad():
            p,after=model.recall(memory,cue,observed)
            other,_=model.recall(memory,cue,observed)
        torch.testing.assert_close(before,memory,rtol=0,atol=0)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        torch.testing.assert_close(p,other,rtol=0,atol=0)
        self.assertEqual(evaluate(model,(bank,cue,observed,target))['read_memory_change_rms'],0)

    def test_parallel_storage_is_invariant_to_pattern_order(self):
        for rule in ('hebb','balanced_stdp','general_stdp'):
            model=AssociativeMemory(rule,dim=8,hidden=8)
            bank,_,_,_=episodes(torch.Generator().manual_seed(18),4,3,8)
            torch.testing.assert_close(model.store(bank),model.store(bank[:,[2,0,1]]),rtol=1e-5,atol=1e-6)

    def test_latent_skew_constraint_maps_through_separate_kv(self):
        bank=episodes(torch.Generator().manual_seed(91),4,3,8)[0]
        for rule in ('hebb','balanced_stdp','general_stdp'):
            model=AssociativeMemory(rule,dim=8,hidden=8)
            result=model.storage_diagnostics(bank)
            self.assertLess(result['latent_mapping_max_error'],2e-6)
            if rule=='balanced_stdp':self.assertLess(result['latent_symmetric_rms'],1e-7)


if __name__=='__main__':unittest.main()
