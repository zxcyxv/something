"""C1 bipolar exponential-difference window with freely crossing delays.

L(d)=C sign(d) [exp(-|d|/slow)-exp(-|d|/fast)], 0<fast<slow.
The jump cancels, L(0)=0, and L'(0)=C(1/fast-1/slow). This is a
different learning window, not an approximation algorithm for discontinuous L.
"""
import math

import torch


def parameters(fast=.1, slow=1.):
    if not 0 < fast < slow:
        raise ValueError('Require 0 < fast < slow.')
    peak = math.log(slow / fast) / (1 / fast - 1 / slow)
    scale = 1 / (math.exp(-peak / slow) - math.exp(-peak / fast))
    return scale, peak


def window(delta, fast=.1, slow=1.):
    """Stable value, with the analytic derivative at zero made explicit.

    No surrogate: the mathematical window is differentiable at zero, although
    the sign-times-absolute-value expression obscures that removable singularity.
    """
    scale, _ = parameters(fast, slow)
    a = delta.abs()
    regular = delta.sign() * (-a / slow).exp() * (-torch.expm1(-a * (1 / fast - 1 / slow)))
    return scale * torch.where(delta == 0, (1 / fast - 1 / slow) * delta, regular)


def derivative(delta, fast=.1, slow=1.):
    scale, _ = parameters(fast, slow)
    return scale * ((-delta.abs() / fast).exp() / fast - (-delta.abs() / slow).exp() / slow)


def split_write(k, v, pk, pv, fast=.1, slow=1.):
    """Only O(TD) exponentials; exact sign regions still cost O(TD^2).

    Bounded phases avoid overflow in the inactive exponential branches.
    Ordinary autodiff includes all activity/phase paths. Exact ties get the
    continuous window's nonzero limiting derivative, not a straight-through rule.
    """
    scale, _ = parameters(fast, slow)
    vm_s, vp_s = v * (-pv / slow).exp(), v * (pv / slow).exp()
    kp_s, km_s = k * (pk / slow).exp(), k * (-pk / slow).exp()
    vm_f, vp_f = v * (-pv / fast).exp(), v * (pv / fast).exp()
    kp_f, km_f = k * (pk / fast).exp(), k * (-pk / fast).exp()
    delta = pv[..., :, None] - pk[..., None, :]
    plus = vm_s[..., :, None] * kp_s[..., None, :] - vm_f[..., :, None] * kp_f[..., None, :]
    minus = -vp_s[..., :, None] * km_s[..., None, :] + vp_f[..., :, None] * km_f[..., None, :]
    result = torch.where(delta >= 0, plus, minus)
    tie = (1 / fast - 1 / slow) * delta * v[..., :, None] * k[..., None, :]
    return scale * torch.where(delta == 0, tie, result).mean(-3)


def direct_write(k, v, pk, pv, fast=.1, slow=1.):
    delta = pv[..., :, None] - pk[..., None, :]
    return (v[..., :, None] * k[..., None, :] * window(delta, fast, slow)).mean(-3)


def verify():
    torch.set_num_threads(2)
    torch.manual_seed(372)
    shape = (2,2,7,8)
    inputs = [torch.randn(shape,dtype=torch.float64,requires_grad=True) for _ in range(4)]
    # Keep realistic phases and include exact ties in a separate case.
    for p in inputs[2:]:
        with torch.no_grad(): p.copy_(p.tanh()*1.57)
    results = []
    for ties in (False,True):
        if ties:
            with torch.no_grad(): inputs[3].copy_(inputs[2])
        actual,expected = split_write(*inputs),direct_write(*inputs)
        cot = torch.randn_like(actual)
        a = torch.autograd.grad((actual*cot).sum(),inputs)
        b = torch.autograd.grad((expected*cot).sum(),inputs)
        error = float((actual-expected).detach().abs().max())
        grad = max(float((x-y).abs().max()) for x,y in zip(a,b))
        assert error < 1e-12 and grad < 1e-11,(error,grad)
        results.append(dict(exact_ties=ties,output_max_abs_error=error,gradient_max_abs_error=grad))
    x = torch.tensor([-1e-7,0.,1e-7],dtype=torch.float64,requires_grad=True)
    g = torch.autograd.grad(window(x).sum(),x)[0]
    torch.testing.assert_close(g,derivative(x),rtol=1e-12,atol=1e-12)
    return dict(cases=results,zero_derivative=float(g[1]),scale=parameters()[0],peak=parameters()[1])


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
