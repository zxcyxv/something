"""Meaningful regressions: architecture parity, symmetry, checkpointing, resume."""
import copy
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from . import new3 as m
from . import train_new3 as t


def small_config(directory):
    cfg=dict(t.DEFAULT_CFG,hidden_size=16,num_heads=2,head_dim=8,
             num_heads_t=2,head_dim_t=8,num_experts=4,num_active_experts=4,
             expert_intermediate=8,shared_intermediate=16,global_batch_size=4,
             eval_batch_size=4,blocks_per_seg=2,loops=4,amp=False,ckpt_blocks=False,
             epochs=8,eval_interval=2,num_aug=10,dataloader_workers=0,
             lr=0.001,lr_warmup_steps=0,ema_rate=0.9,gradient_log_every=0,
             out_dir=str(directory),test_size=8,milestone_extrap_n=8)
    t.validate_config(cfg)
    return cfg


def sequential_config(cfg):
    return dict(cfg,expert_schedule='sequential_then_routed',num_experts=8,
                num_active_experts=8,blocks_per_seg=9)


def boards(n=12):
    r,c=np.indices((9,9));base=(r*3+r//3+c)%9
    labels=np.stack([(base+i)%9+1 for i in range(n)]).astype(np.uint8)
    rng=np.random.default_rng(19)
    return np.where(rng.random(labels.shape)<0.3,labels,0).astype(np.uint8),labels


def batch(n=4):
    x,y=boards(n)
    return dict(inputs=torch.from_numpy(x.reshape(n,81).astype('int64')+1),
                labels=torch.from_numpy(y.reshape(n,81).astype('int64')+1),
                puzzle_identifiers=torch.zeros(n,dtype=torch.int32))


def gradient(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])


class New3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cfg=small_config(self.tmp.name)

    @unittest.skipUnless((t.ROOT/'새코드3.txt').exists(),'Uploaded reference source is optional')
    def test_uploaded_model_forward_and_backward_parity(self):
        src=types.ModuleType('uploaded_new3_reference');sys.modules[src.__name__]=src
        exec(compile((t.ROOT/'새코드3.txt').read_text(),'새코드3.txt','exec'),src.__dict__)
        old=src.LTv5(self.cfg).train();new=m.LTv5(self.cfg).train()
        new.load_state_dict(old.state_dict())
        data=batch();co,cn=old.initial_carry(data),new.initial_carry(data)
        for _ in range(5):
            co,oo=old(co,data);cn,on=new(cn,data)
            torch.testing.assert_close(oo['logits'],on['logits'],atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(co.coupling,cn.coupling,atol=2e-6,rtol=2e-6)
        lo=src.stablemax_cross_entropy(oo['logits'],data['labels']).mean()
        ln=m.stablemax_cross_entropy(on['logits'],data['labels']).mean()
        lo.backward();ln.backward()
        torch.testing.assert_close(gradient(old),gradient(new),atol=2e-6,rtol=2e-5)

    def test_digit_equivariance_and_training_gradient(self):
        self._assert_digit_equivariance(self.cfg)

    def test_sequential_digit_equivariance_and_training_gradient(self):
        self._assert_digit_equivariance(sequential_config(self.cfg))

    def _assert_digit_equivariance(self,cfg):
        a=m.ACTLossHead(m.LTv5(cfg)).train();b=copy.deepcopy(a)
        data=batch();perms=torch.stack([torch.cat((torch.arange(2),torch.randperm(9)+2)) for _ in range(4)])
        permuted={k:perms.gather(1,v) if k in ('inputs','labels') else v for k,v in data.items()}
        _,la,_,oa,_=a(return_keys={'logits'},carry=a.initial_carry(data),batch=data)
        _,lb,_,ob,_=b(return_keys={'logits'},carry=b.initial_carry(permuted),batch=permuted)
        expected=oa['logits'].gather(2,perms.argsort(1)[:,None,:].expand(4,81,11))
        torch.testing.assert_close(ob['logits'],expected,atol=3e-6,rtol=3e-6)
        la.backward();lb.backward()
        torch.testing.assert_close(gradient(a),gradient(b),atol=3e-6,rtol=3e-5)

    def test_checkpoint_backward_counts_routing_once(self):
        a=m.ACTLossHead(m.LTv5(self.cfg)).train();b=copy.deepcopy(a)
        b.model.config.ckpt_blocks=True
        for net in (a,b):
            data=batch();_,loss,_,_,_=net(return_keys=set(),carry=net.initial_carry(data),batch=data)
            loss.backward()
            counts=net.model.inner.layers[0].moe.router.acc_counts
            self.assertEqual(counts.sum().item(),4*11*81*self.cfg['blocks_per_seg'])
        torch.testing.assert_close(gradient(a),gradient(b),atol=1e-6,rtol=1e-5)

    def test_inactive_expert_cannot_win_after_bias(self):
        router=m.MoERouter(16,4)
        with torch.no_grad():router.gate.weight.zero_();router.bias[:2].fill_(-1)
        _,idx,_=router(torch.zeros(4,16),n_active=2)
        self.assertTrue((idx<2).all())

    def test_expert_dispatch_topk_matches_reference(self):
        expert=m.MoEExperts(4,16,8)
        x=torch.randn(12,16,requires_grad=True)
        idx=torch.stack((torch.arange(12)%4,(torch.arange(12)+1)%4),1)
        vals=torch.softmax(torch.randn(12,2),-1)
        actual=expert(x,idx,vals)
        expected=torch.zeros_like(x)
        for e in range(4):
            token,slot=(idx==e).nonzero(as_tuple=True)
            g,u=torch.nn.functional.linear(x[token],expert.gate_up[e]).chunk(2,-1)
            y=torch.nn.functional.linear(torch.nn.functional.silu(g)*u,expert.down[e])
            expected[token]+=y*vals[token,slot,None]
        torch.testing.assert_close(actual,expected)

    def test_mid_puzzle_resume_is_exact(self):
        self._assert_mid_puzzle_resume(self.cfg)

    def test_sequential_mid_puzzle_resume_is_exact(self):
        self._assert_mid_puzzle_resume(sequential_config(self.cfg))

    def _assert_mid_puzzle_resume(self,cfg):
        x,y=boards()
        a,oa,ea=t.make_training_objects(cfg,torch.device('cpu'));sa=t.TrainState()
        batches=list(t.make_loader(x,y,cfg,sa));steps_per_iter=cfg['eval_interval']*len(x)//4
        for _,data in batches[:3]:
            t.train_step(a,oa,ea,sa,data,cfg,24);t.advance_cursor(sa,steps_per_iter)
        path=t.save_checkpoint(cfg['out_dir'],sa,a,oa,ea,cfg)
        saved=torch.load(path,weights_only=False)
        self.assertFalse(all(torch.equal(saved['raw_model_state_dict'][k],saved['model_state_dict'][k])
                             for k in saved['raw_model_state_dict']))
        for _,data in batches[3:8]:
            t.train_step(a,oa,ea,sa,data,cfg,24);t.advance_cursor(sa,steps_per_iter)
        b,ob,eb=t.make_training_objects(cfg,torch.device('cpu'))
        sb=t.load_checkpoint(path,b,ob,eb,cfg,torch.device('cpu'))
        resumed=iter(t.make_loader(x,y,cfg,sb))
        for original in batches[3:8]:
            iteration,data=next(resumed)
            self.assertEqual(iteration,original[0])
            for key in data:self.assertTrue(torch.equal(data[key],original[1][key]))
            t.train_step(b,ob,eb,sb,data,cfg,24);t.advance_cursor(sb,steps_per_iter)
        self.assertEqual((sa.step,sa.iter_id,sa.batch_in_iter),(sb.step,sb.iter_id,sb.batch_in_iter))
        for name,value in a.state_dict().items():
            self.assertTrue(torch.equal(value,b.state_dict()[name]),name)
        for name,value in ea.shadow.items():self.assertTrue(torch.equal(value,eb.shadow[name]),name)
        self.assertTrue(torch.equal(sa.carry.hidden,sb.carry.hidden))
        self.assertTrue(torch.equal(sa.carry.coupling,sb.carry.coupling))
        # Reject changes that would silently alter the resumed experiment.
        bad=dict(cfg,num_aug=9)
        with self.assertRaisesRegex(ValueError,'num_aug'):
            t.load_checkpoint(path,b,ob,eb,bad,torch.device('cpu'))

    def test_sequential_order_all_tokens_and_state_carry(self):
        cfg=dict(sequential_config(self.cfg),loops=2)
        model=m.LTv5(cfg).eval();layer=model.inner.layers[0]
        frames=[]
        def record(_,args,output):
            frames.append(dict(index=args[4],x=args[0].clone(),
                memory=None if args[2] is None else args[2].clone(),
                h_out=output[0].clone(),memory_out=output[1].clone()))
        hook=layer.register_forward_hook(record)
        self.addCleanup(hook.remove)
        data=batch();carry=model.initial_carry(data)
        with torch.no_grad(), mock.patch.object(layer.moe.experts,'forward_single',
                wraps=layer.moe.experts.forward_single) as fixed, \
                mock.patch.object(layer.moe.router,'forward',wraps=layer.moe.router.forward) as router:
            first,out=model(carry,data)
            self.assertFalse(first.halted.any())
            self.assertEqual(len(out['router_logits']),1)
            second,out=model(first,data)
            self.assertTrue(second.halted.all())
            self.assertEqual(len(out['router_logits']),1)
            self.assertEqual(fixed.call_count,16)
            self.assertEqual(router.call_count,2)
            self.assertEqual([c.args[1] for c in fixed.call_args_list],list(range(8))*2)
            for c in fixed.call_args_list:
                self.assertEqual(c.args[0].shape,(4*11*81,cfg['hidden_size']))
            for c in router.call_args_list:
                self.assertEqual(c.args[0].shape,(4*11*81,cfg['hidden_size']))
        self.assertEqual([f['index'] for f in frames],(list(range(8))+[None])*2)
        self.assertIsNone(frames[0]['memory'])
        for start in (0,9):
            for i in range(start+1,start+9):
                torch.testing.assert_close(frames[i]['x'],frames[i-1]['h_out'],rtol=0,atol=0)
                torch.testing.assert_close(frames[i]['memory'],frames[i-1]['memory_out'],rtol=0,atol=0)
        torch.testing.assert_close(frames[9]['x'],first.hidden+model.inner.embed(data['inputs']),rtol=0,atol=0)
        torch.testing.assert_close(frames[9]['memory'],first.coupling,rtol=0,atol=0)
        torch.testing.assert_close(second.hidden,frames[-1]['h_out'],rtol=0,atol=0)

    def test_sequential_task_gradients_and_checkpoint_recomputation(self):
        cfg=sequential_config(self.cfg)
        # Exclude auxiliary losses so a nonzero router gradient must come from the answer loss.
        a=m.ACTLossHead(m.LTv5(cfg),q_weight=0,moe_aux_weight=0).train();b=copy.deepcopy(a)
        b.model.config.ckpt_blocks=True
        losses=[]
        for net in (a,b):
            data=batch();_,loss,_,_,_=net(return_keys=set(),carry=net.initial_carry(data),batch=data)
            loss.backward();losses.append(loss.detach())
            moe=net.model.inner.layers[0].moe
            self.assertEqual(moe.router.acc_counts.sum().item(),4*11*81)
            for p in (moe.experts.gate_up,moe.experts.down):
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertTrue((p.grad.reshape(8,-1).norm(dim=1)>0).all())
            self.assertGreater(moe.router.gate.weight.grad.norm().item(),0)
            self.assertGreater(net.model.inner.embed_common.grad.norm().item(),0)
        torch.testing.assert_close(losses[0],losses[1],atol=0,rtol=0)
        torch.testing.assert_close(gradient(a),gradient(b),atol=1e-6,rtol=1e-5)

    def test_schedule_validation_and_legacy_checkpoint_default(self):
        shipped=json.loads((t.ROOT/'configs/new3_sudoku_sequential_experts.json').read_text())
        t.validate_config(dict(t.DEFAULT_CFG,**shipped))
        self.assertEqual((shipped['num_experts'],shipped['blocks_per_seg'],shipped['loops']),(8,9,1))
        cfg=sequential_config(self.cfg)
        for changes in (dict(expert_schedule='unknown'),dict(blocks_per_seg=8),
                        dict(num_active_experts=7),dict(top_k=2)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                t.validate_config(dict(cfg,**changes))
        a,oa,ea=t.make_training_objects(self.cfg,torch.device('cpu'))
        path=t.save_checkpoint(self.cfg['out_dir'],t.TrainState(),a,oa,ea,self.cfg)
        saved=torch.load(path,weights_only=False)
        saved['cfg'].pop('expert_schedule')
        torch.save(saved,path)
        t.load_checkpoint(path,a,oa,ea,self.cfg,torch.device('cpu'))
        different=dict(self.cfg,expert_schedule='sequential_then_routed',blocks_per_seg=5)
        for training in (True,False):
            with self.subTest(training=training),self.assertRaisesRegex(ValueError,'expert_schedule'):
                t.load_checkpoint(path,a,oa,ea,different,torch.device('cpu'),training=training)

    def test_evaluation_restores_weights_modes_and_horizon(self):
        cfg=dict(self.cfg,loops=1)
        model,opt,ema=t.make_training_objects(cfg,torch.device('cpu'))
        state=t.TrainState();t.train_step(model,opt,ema,state,batch(),cfg,20)
        before={k:v.clone() for k,v in model.state_dict().items()}
        x,y=boards(5)
        result=t.evaluate(model,ema,x,y,cfg,1,6)
        self.assertEqual([r['n'] for r in result['rows']],[5]*6)
        self.assertTrue(model.training);self.assertEqual(model.model.config.loops,1)
        first,last=result['rows'][0],result['rows'][-1]
        self.assertEqual(result['train_segments'],1)
        self.assertEqual(result['error_reduction'],(last['exact']-first['exact'])/(5-first['exact']))
        for key,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
        self.assertTrue(t.valid_sudoku(batch()['labels']).all())

    def test_single_segment_defaults_and_next_batch_reset(self):
        shipped=json.loads((t.ROOT/'configs/new3_sudoku.json').read_text())
        for cfg in (t.DEFAULT_CFG,shipped):
            self.assertEqual((cfg['loops'],cfg['blocks_per_seg']),(1,8))
        model=m.LTv5(dict(self.cfg,loops=1)).eval()
        first=batch();second={k:v.roll(1,0) for k,v in first.items()}
        with torch.no_grad():
            carry,_=model(model.initial_carry(first),first)
            self.assertTrue(carry.halted.all())
            carry.hidden.fill_(100);carry.coupling.fill_(100)
            reused,output=model(carry,second)
            fresh,expected=model(model.initial_carry(second),second)
        self.assertTrue((reused.steps==1).all())
        self.assertTrue(reused.halted.all())
        for key in second:self.assertTrue(torch.equal(reused.data[key],second[key]))
        torch.testing.assert_close(output['logits'],expected['logits'],rtol=0,atol=0)
        torch.testing.assert_close(reused.coupling,fresh.coupling,rtol=0,atol=0)

    def test_batch_one_and_color_embedding_are_rejected(self):
        with self.assertRaises(ValueError):t.validate_config(dict(self.cfg,global_batch_size=1))
        with self.assertRaises(ValueError):t.validate_config(dict(self.cfg,puzzle_emb_ndim=27))


if __name__=='__main__':unittest.main()
