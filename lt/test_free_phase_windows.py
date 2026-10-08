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

    def test_l2_qk_token_sum_write(self):
        for window in ('tanhsech', 'hebbian'):
            torch.manual_seed(14)
            base = model_class(window, generator='diagonal', window_scale=2.)(config()).double()
            torch.manual_seed(14)
            inner = model_class(window, generator='diagonal', window_scale=2., qk_l2=True, write_sum=True)(config()).double()
            layer = inner.layers[0]
            with torch.no_grad():
                layer.phase_local_gain.normal_(0, .3)
                base.layers[0].phase_local_gain.copy_(layer.phase_local_gain)
            q, k, v = [torch.randn(2,2,9,8,dtype=torch.float64)*3 for _ in range(3)]
            actual, current = inner.memory_step(layer,q,k,v)[:2]
            unit = lambda z: z/(z.norm(dim=-1,keepdim=True)+inner.config.eps)
            qr,kr = (inner.apply_rope(unit(z),layer) for z in (q,k))
            if window == 'hebbian':
                g = (v[..., :, None]*kr[..., None, :]).sum(-3)
            else:
                pk,pv = inner.phases(layer,kr,v)
                g = (v[..., :, None]*kr[..., None, :]*inner.window(pv[..., :, None]-pk[..., None, :])).sum(-3)
            torch.testing.assert_close(current,g,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(actual,qr@g.transpose(-1,-2),rtol=1e-12,atol=1e-12)
            # Scale-free in Q/K up to the norm+eps denominator; same parameters as the flag-off model.
            torch.testing.assert_close(inner.memory_step(layer,5*q,.2*k,v)[0],actual,rtol=0,atol=1e-3)
            for name, value in base.state_dict().items():
                torch.testing.assert_close(inner.state_dict()[name], value, rtol=0, atol=0)

    def test_complex_phase_window_matches_direct_pairs(self):
        from .experiment_free_phase_windows import periodized_tanhsech
        import math
        b, scale = periodized_tanhsech(2.0, 4)
        # Coefficients equal the direct periodization of tanh*sech on a grid.
        d = torch.linspace(1e-6, 2 * math.pi - 1e-6, 100001, dtype=torch.float64)
        W = sum((lambda x: torch.tanh(x) / torch.cosh(x))((d + 2 * math.pi * m) / 2.0) for m in range(-60, 61))
        for r in range(1, 5):
            br = torch.trapezoid(W * torch.sin(r * d), d) / math.pi
            torch.testing.assert_close(b[r - 1], br, rtol=1e-6, atol=1e-8)
        torch.manual_seed(5)
        base = model_class('hebbian', generator='diagonal', qk_l2=True, write_sum=True)(config()).double()
        torch.manual_seed(5)
        inner = model_class('complex', modes=3, qk_l2=True, write_sum=True, tau_phi=1.5)(config()).double()
        for name, value in base.state_dict().items():
            if 'window' not in name and 'phase_local' not in name:
                torch.testing.assert_close(inner.state_dict()[name], value, rtol=0, atol=0)
        layer = inner.layers[0]
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]
        read, G = inner.memory_step(layer, q, k, v)[:2]
        unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
        qr, kr = (inner.apply_rope(unit(z), layer) for z in (q, k))
        P = inner.dh // 2
        kc, vc, qc = (z.reshape(*z.shape[:-1], P, 2) for z in (kr, v, qr))
        phiK, phiV = torch.atan2(kc[..., 1], kc[..., 0]), torch.atan2(vc[..., 1], vc[..., 0])
        magK, magV = kc.norm(dim=-1), vc.norm(dim=-1)
        delta = phiV[..., :, None] - phiK[..., None, :]
        c = inner.window_coefficients.double()
        Wd = sum(c[r - 1] * torch.sin(r * delta) for r in range(1, 4))
        G_ref = (magV[..., :, None] * magK[..., None, :] * Wd).sum(-3)
        # Unit phasors use |z| + 1e-6, so higher harmonics carry a ~1e-5 relative bias.
        torch.testing.assert_close(G, G_ref, rtol=1e-4, atol=1e-7)
        read_ref = torch.stack((qc[..., 0] @ G_ref.transpose(-1, -2), qc[..., 1] @ G_ref.transpose(-1, -2)), -1).reshape_as(read)
        torch.testing.assert_close(read, read_ref, rtol=1e-4, atol=1e-7)
        # Q/K scale-free (up to norm+eps); peak of the window is 1.
        torch.testing.assert_close(inner.memory_step(layer, 5 * q, .2 * k, v)[0], read, rtol=0, atol=1e-3)
        x = torch.linspace(0, math.pi, 20001, dtype=torch.float64)
        self.assertAlmostEqual(float(sum(c[r - 1] * torch.sin(r * x) for r in range(1, 4)).max()), 1.0, places=6)
        # Full first-order derivatives through the complex write and read.
        def f(q, k, v):
            return inner.memory_step(layer, q, k, v)[0]
        self.assertTrue(torch.autograd.gradcheck(f, tuple(z.clone().requires_grad_(True) for z in (q, k, v)), eps=1e-6, atol=1e-6))

    def test_pairangle_phase_window_matches_direct_pairs(self):
        import math
        torch.manual_seed(5)
        base = model_class('hebbian', generator='diagonal', qk_l2=True, write_sum=True)(config()).double()
        torch.manual_seed(5)
        inner = model_class('pairangle', modes=3, qk_l2=True, write_sum=True, tau_phi=1.5)(config()).double()
        for name, value in base.state_dict().items():
            if 'window' not in name and 'phase_local' not in name:
                torch.testing.assert_close(inner.state_dict()[name], value, rtol=0, atol=0)
        layer = inner.layers[0]
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]
        read, G = inner.memory_step(layer, q, k, v)[:2]
        unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
        qr, kr = (inner.apply_rope(unit(z), layer) for z in (q, k))
        P = inner.dh // 2
        ang = lambda z: torch.atan2(z.reshape(*z.shape[:-1], P, 2)[..., 1], z.reshape(*z.shape[:-1], P, 2)[..., 0]).repeat_interleave(2, -1)
        phiK, phiV = ang(kr), ang(v)
        delta = phiV[..., :, None] - phiK[..., None, :]
        c = inner.window_coefficients.double()
        Wd = sum(c[r - 1] * torch.sin(r * delta) for r in range(1, 4))
        G_ref = (v[..., :, None] * kr[..., None, :] * Wd).sum(-3)
        torch.testing.assert_close(G, G_ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(read, qr @ G_ref.transpose(-1, -2), rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(inner.memory_step(layer, 5 * q, .2 * k, v)[0], read, rtol=0, atol=1e-3)
        x = torch.linspace(0, math.pi, 20001, dtype=torch.float64)
        self.assertAlmostEqual(float(sum(c[r - 1] * torch.sin(r * x) for r in range(1, 4)).max()), 1.0, places=6)
        def f(q, k, v):
            return inner.memory_step(layer, q, k, v)[0]
        self.assertTrue(torch.autograd.gradcheck(f, tuple(z.clone().requires_grad_(True) for z in (q, k, v)), eps=1e-6, atol=1e-6))

    def test_softangle_window_matches_direct_pairs_and_fades_weak_channels(self):
        import math
        torch.manual_seed(5)
        inner = model_class('softangle', modes=3, qk_l2=True, write_sum=True, tau_phi=1.5, phase_floor=0.5)(config()).double()
        torch.manual_seed(5)
        hard = model_class('pairangle', modes=3, qk_l2=True, write_sum=True, tau_phi=1.5)(config()).double()
        layer = inner.layers[0]
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]
        read, G = inner.memory_step(layer, q, k, v)[:2]
        unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
        qr, kr = (inner.apply_rope(unit(z), layer) for z in (q, k))
        P = inner.dh // 2
        pairs = lambda z: z.reshape(*z.shape[:-1], P, 2)
        ang = lambda z: torch.atan2(pairs(z)[..., 1], pairs(z)[..., 0]).repeat_interleave(2, -1)
        def rho(z):
            m2 = pairs(z).square().sum(-1)
            return (m2 / (m2 + 0.25 * m2.mean(-1, keepdim=True))).sqrt().repeat_interleave(2, -1)
        delta = ang(v)[..., :, None] - ang(kr)[..., None, :]
        coh = rho(v)[..., :, None] * rho(kr)[..., None, :]
        c = inner.window_coefficients.double()
        Wd = sum(c[r - 1] * coh ** r * torch.sin(r * delta) for r in range(1, 4))
        G_ref = (v[..., :, None] * kr[..., None, :] * Wd).sum(-3)
        torch.testing.assert_close(G, G_ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(read, qr @ G_ref.transpose(-1, -2), rtol=1e-9, atol=1e-9)
        # Coherence < 1 everywhere, so the soft write is strictly inside the hard one.
        G_hard = hard.memory_step(hard.layers[0], q, k, v)[1]
        self.assertLess(float(G.abs().sum()), float(G_hard.abs().sum()))
        # A channel at 1% of the RMS magnitude contributes ~0 window and a bounded gradient.
        k2 = k.clone(); k2[..., :2] *= 0.01
        k2.requires_grad_(True)
        G2 = inner.memory_step(layer, q, k2, v)[1]
        self.assertLess(float(G2[..., :, :2].abs().max()), 1e-3 * float(G2.abs().max()))
        grad = torch.autograd.grad(G2.square().sum(), k2)[0]
        self.assertTrue(torch.isfinite(grad).all())
        # The coherence gate is detached: gradients equal those of the reference with
        # rho held constant, and gradcheck holds for that function.
        def reference(q, k, v):
            qr, kr = (inner.apply_rope(unit(z), layer) for z in (q, k))
            delta = ang(v)[..., :, None] - ang(kr)[..., None, :]
            Wd = sum(c[r - 1] * coh.detach() ** r * torch.sin(r * delta) for r in range(1, 4))
            return qr @ (v[..., :, None] * kr[..., None, :] * Wd).sum(-3).transpose(-1, -2)
        ins = tuple(z.clone().requires_grad_(True) for z in (q, k, v))
        ref_ins = tuple(z.clone().requires_grad_(True) for z in (q, k, v))
        ga = torch.autograd.grad(inner.memory_step(layer, *ins)[0].square().sum(), ins)
        gb = torch.autograd.grad(reference(*ref_ins).square().sum(), ref_ins)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=1e-7, atol=1e-7)
        self.assertTrue(torch.autograd.gradcheck(reference, ref_ins, eps=1e-6, atol=1e-6))

    def test_value_normalisation_variants(self):
        import math
        torch.manual_seed(5)
        base = model_class('pairangle', modes=2, qk_l2=True, write_sum=True)(config()).double()
        for mode in ('unit', 'unit_gain'):
            torch.manual_seed(5)
            inner = model_class('pairangle', modes=2, qk_l2=True, write_sum=True, v_norm=mode)(config()).double()
            for name, value in base.state_dict().items():
                torch.testing.assert_close(inner.state_dict()[name], value, rtol=0, atol=0)
            layer = inner.layers[0]
            q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 3 for _ in range(3)]
            vn = v / (v.norm(dim=-1, keepdim=True) + inner.config.eps)
            if mode == 'unit_gain':
                self.assertAlmostEqual(float(torch.nn.functional.softplus(layer.v_gain_raw)[0]), math.sqrt(inner.dh), places=6)
                vn = vn * torch.nn.functional.softplus(layer.v_gain_raw)[None, :, None, None]
            read, G = inner.memory_step(layer, q, k, v)[:2]
            read_ref, G_ref = base.memory_step(base.layers[0], q, k, vn)[:2]
            torch.testing.assert_close(G, G_ref, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(read, read_ref, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(inner.memory_step(layer, q, k, 7 * v)[0], read, rtol=0, atol=1e-3)

    def test_relaxed_phase_window_and_tied_projection(self):
        torch.manual_seed(5)
        inner = model_class('relaxphase', modes=2, qk_l2=True, write_sum=True, tau_phi=2., v_norm='unit_gain',
                            tie_all=True, phase_kappa=1.5, phase_omega=0.25)(config()).double()
        layer = inner.layers[0]
        self.assertIs(layer.k_proj, layer.q_proj); self.assertIs(layer.v_proj, layer.q_proj)
        q, k, v = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]
        unit = lambda z: z / (z.norm(dim=-1, keepdim=True) + inner.config.eps)
        P = inner.dh // 2
        # Fresh puzzle: zero phase -> zero window, zero write; phase advances by kappa*|z_j| + omega.
        read0, G0, phi1, _ = inner.memory_step(layer, q, k, v)
        self.assertEqual(float(G0.abs().max()), 0.); self.assertEqual(float(read0.abs().max()), 0.)
        mag = unit(k).reshape(2, 2, 9, P, 2).norm(dim=-1)
        torch.testing.assert_close(phi1, 0.25 + 1.5 * mag, rtol=1e-12, atol=1e-12)
        # Second block: window on phi1 (history), direct pair sum.
        q2, k2, v2 = [torch.randn(2, 2, 9, 8, dtype=torch.float64) * 2 for _ in range(3)]
        read, G, phi2, _ = inner.memory_step(layer, q2, k2, v2, None, phi1, None, None)
        gain = torch.nn.functional.softplus(layer.v_gain_raw)[None, :, None, None]
        vn = unit(v2) * gain
        qr, kr = (inner.apply_rope(unit(z), layer) for z in (q2, k2))
        ph = phi1.repeat_interleave(2, -1)
        delta = ph[..., :, None] - ph[..., None, :]
        c = inner.window_coefficients.double()
        Wd = c[0] * torch.sin(delta) + c[1] * torch.sin(2 * delta)
        G_ref = (vn[..., :, None] * kr[..., None, :] * Wd).sum(-3)
        torch.testing.assert_close(G, G_ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(read, qr @ G_ref.transpose(-1, -2), rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(phi2, phi1 + 0.25 + 1.5 * unit(k2).reshape(2, 2, 9, P, 2).norm(dim=-1), rtol=1e-12, atol=1e-12)
        # fresh mask resets the phase of selected puzzles before use.
        fresh = torch.tensor([True, False])
        readf, Gf, phif, _ = inner.memory_step(layer, q2, k2, v2, None, phi1, None, fresh)
        self.assertEqual(float(Gf[0].abs().max()), 0.)
        torch.testing.assert_close(Gf[1], G[1], rtol=0, atol=0)
        torch.testing.assert_close(phif[0], phi1[0] * 0 + 0.25 + 1.5 * unit(k2).reshape(2, 2, 9, P, 2).norm(dim=-1)[0], rtol=1e-12, atol=1e-12)
        # Gradients through write, read and the phase advance.
        def f(q, k, v, phi):
            r, g, p, _ = inner.memory_step(layer, q, k, v, None, phi, None, None)
            return r, p
        self.assertTrue(torch.autograd.gradcheck(f, tuple(z.clone().requires_grad_(True) for z in (q2, k2, v2, phi1)), eps=1e-6, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
