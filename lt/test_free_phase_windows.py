"""Research invariants: locality, freely reversing timing, exact KV placement,
full first-order derivatives, and the required two residual normalizations.
"""
import unittest

import torch

from .experiment_free_phase_windows import model_class
from .kv_stability import ExponentialPhaseCurrentReadInner
from .test_kv_stability import config


class FreePhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(409)

    def test_original_initialization_and_fixed_window_control(self):
        for window in ('exponential', 'fourier', 'biexponential', 'tanhsech'):
            torch.manual_seed(14)
            fixed = model_class(window, False)(config()).double()
            torch.manual_seed(14)
            free = model_class(window, True)(config()).double()
            for name, value in fixed.state_dict().items():
                torch.testing.assert_close(free.state_dict()[name], value, rtol=0, atol=0)
            q, k, v = [torch.randn(2,2,9,8,dtype=torch.float64) for _ in range(3)]
            a = fixed.memory_step(fixed.layers[0],q,k,v)
            b = free.memory_step(free.layers[0],q,k,v)
            for expected, actual in zip(a[:2],b[:2]):
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        torch.manual_seed(14)
        original = ExponentialPhaseCurrentReadInner(config())
        torch.manual_seed(14)
        free = model_class('exponential', True)(config())
        for name, value in original.state_dict().items():
            torch.testing.assert_close(free.state_dict()[name],value,rtol=0,atol=0)

    def test_token_local_phase_can_reverse_order_with_fixed_weights(self):
        inner = model_class()(config())
        layer = inner.layers[0]
        with torch.no_grad():
            layer.theta_k_raw.zero_(); layer.theta_v_raw.zero_()
            layer.phase_local.weight.zero_()
            layer.phase_local.weight[inner.dh, 0] = 1
        k, v = torch.zeros(2,2,9,8), torch.zeros(2,2,9,8)
        k[:,:,0,0] = -1
        first = inner.phases(layer,k,v)
        k[:,:,0,0] = 1
        second = inner.phases(layer,k,v)
        self.assertTrue((first[1][:,:,0,0]-first[0][:,:,0,0] < 0).all())
        self.assertTrue((second[1][:,:,0,0]-second[0][:,:,0,0] > 0).all())
        for a,b in zip(first,second):
            torch.testing.assert_close(a[:,:,1:],b[:,:,1:],rtol=0,atol=0)

    def test_dense_window_placement_and_all_parameter_gradients(self):
        for window in ('exponential', 'fourier', 'biexponential', 'tanhsech'):
            inner = model_class(window)(config()).double()
            layer = inner.layers[0]
            with torch.no_grad():
                layer.phase_local.weight.normal_(0,.05)
            q,k,v = [torch.randn(2,2,9,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
            qr,kr = (inner.apply_rope(x,layer) for x in (q,k))
            pk,pv = inner.phases(layer,kr,v)
            dense = (v[..., :, None]*kr[..., None, :]*inner.window(pv[..., :, None]-pk[..., None, :])).mean(-3)
            expected = qr @ dense.transpose(-1,-2)
            actual,current,ek,ev = inner.memory_step(layer,q,k,v,torch.full_like(dense,float('nan')))
            torch.testing.assert_close(current,dense,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            self.assertIsNone(ek); self.assertIsNone(ev)
            cot = torch.randn_like(actual)
            variables = (q,k,v,layer.theta_k_raw,layer.theta_v_raw,layer.phase_local.weight)
            a = torch.autograd.grad((actual*cot).sum(),variables)
            b = torch.autograd.grad((expected*cot).sum(),variables)
            for x,y in zip(a,b):
                torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)

    def test_actual_forward_has_two_correct_residual_norms(self):
        for window in ('exponential', 'fourier', 'biexponential', 'tanhsech'):
            inner = model_class(window)(config())
            layer = inner.layers[0]
            h = torch.randn(2,9,16)
            inj = torch.randn_like(h)
            x = h + inner.embed_scale*inj
            heads = lambda z:z.reshape(2,9,2,8).transpose(1,2)
            q,k,v = [heads(p(x)) for p in (layer.q_proj,layer.k_proj,layer.v_proj)]
            qr,kr = (inner.apply_rope(z,layer) for z in (q,k))
            pk,pv = inner.phases(layer,kr,v)
            g = (v[..., :, None]*kr[..., None, :]*inner.window(pv[..., :, None]-pk[..., None, :])).mean(-3)
            read = (qr@g.transpose(-1,-2)).transpose(1,2).reshape_as(h)
            norm = lambda z:(z.float()*torch.rsqrt(z.float().square().mean(-1,keepdim=True)+1e-5)).to(z.dtype)
            u = norm(x+layer.out_proj(read))
            gate,value = layer.b_gate_up(u).chunk(2,-1)
            expected = norm(u+layer.b_down(.5*gate*value))
            actual = inner.block(layer,h,inj,None,None,None,None)[0]
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            for dtype in (torch.float32,torch.bfloat16):
                z = torch.randn_like(h).to(dtype)
                torch.testing.assert_close(inner.phi(z),norm(z),rtol=0,atol=0)

    def test_diagonal_head_reverses_order_and_preserves_locality(self):
        inner = model_class(generator='diagonal')(config()).double()
        layer = inner.layers[0]
        with torch.no_grad():
            layer.phase_local_gain[:,inner.dh:] = 1
            layer.theta_k_raw.zero_(); layer.theta_v_raw.zero_()
        k = torch.randn(2,2,9,8,dtype=torch.float64)
        v = torch.ones_like(k)
        pk,pv = inner.phases(layer,k,v)
        v2 = v.clone(); v2[:,:,0] = -1
        pk2,pv2 = inner.phases(layer,k,v2)
        self.assertTrue((pv[:,:,0,0]-pk[:,:,0,0] > 0).all())
        self.assertTrue((pv2[:,:,0,0]-pk2[:,:,0,0] < 0).all())
        torch.testing.assert_close(pv[:,:,1:],pv2[:,:,1:],rtol=0,atol=0)
        q = torch.randn_like(k)
        actual = inner.memory_step(layer,q,k,v2)[0]
        qr,kr = (inner.apply_rope(z,layer) for z in (q,k))
        pk,pv = inner.phases(layer,kr,v2)
        g = (v2[..., :, None]*kr[..., None, :]*inner.window(pv[..., :, None]-pk[..., None, :])).mean(-3)
        expected = qr@g.transpose(-1,-2)
        a = torch.autograd.grad(actual.square().mean(),layer.phase_local_gain)[0]
        b = torch.autograd.grad(expected.square().mean(),layer.phase_local_gain)[0]
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


if __name__ == '__main__':
    unittest.main()
