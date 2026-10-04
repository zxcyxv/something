"""Numerical checks of channel-phase STDP; independent of the training harness.

Axes: a row is a token, a column is a neuron, and phase is event delay on a
common carrier. Recurrence only integrates the resulting writes. This module
does not modify or launch a model-training run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

PERIOD = 2 * math.pi


def principal(delta):
    return (np.asarray(delta) + math.pi) % PERIOD - math.pi


def line_window(delta, tau=1.0):
    """One isolated spike pair; simultaneous spikes have zero contribution."""
    delta = np.asarray(delta)
    return np.sign(delta) * np.exp(-np.abs(delta) / tau)


def periodic_window(delta, tau=1.0):
    """All pairs of two infinitely repeated, same-period spike trains.

    The expression uses expm1 to remain stable when tau is large. The exact
    jump at zero is kept; its chosen midpoint value is zero.
    """
    d = principal(delta)
    x = np.abs(d)
    return (np.sign(d) * np.exp(-x / tau)
            * (-np.expm1(-(PERIOD - 2 * x) / tau))
            / (-np.expm1(-PERIOD / tau)))


def coefficients(modes, tau=1.0, jitter=0.0):
    """Sine coefficients; jitter is the SD of the relative timing error."""
    n = np.arange(1, modes + 1, dtype=np.float64)
    return 2 * n / (math.pi * (n * n + tau ** -2)) * np.exp(-0.5 * (jitter * n) ** 2)


def spectral_window(delta, modes, tau=1.0, jitter=0.0):
    delta = np.asarray(delta)
    result = np.zeros_like(delta, dtype=np.float64)
    for n, b in enumerate(coefficients(modes, tau, jitter), 1):
        result += b * np.sin(n * delta)
    return result


def positive_power_window(delta, degree, tau=1.0):
    """Derivative of a positive power-series approximation to exp(-|lag|).

    With s=cos(lag/2)**2, the periodic even exponential is
    cosh(2*a*asin(sqrt(s)))/sinh(a*pi). Its coefficients satisfy
    c[m+1]/c[m]=(m*m+a*a)/((m+0.5)*(m+1)), hence are all positive.
    Every finite polynomial has the correct sign, and increases to the target
    pointwise away from zero. Resolution costs O(1/lag**2) degree.
    """
    delta = np.asarray(delta)
    s = np.cos(delta / 2) ** 2
    power = np.ones_like(s)
    derivative = np.zeros_like(s)
    c, a = 1.0, 1 / tau
    for m in range(1, degree + 1):
        c *= ((m - 1) ** 2 + a * a) / ((m - 0.5) * m)
        derivative += m * c * power
        power *= s
    return np.sin(delta) * derivative / (2 * a * np.sinh(a * math.pi))


def pulse_coefficients(degree, tau=1.0):
    """Exact finite Fourier spectrum from identical positive finite pulses.

    Each neuron fires a common waveform proportional to cos(t/2)**(2*degree).
    Its n-th normalized Fourier coefficient is binom(2d,d-n)/binom(2d,d).
    Cross-correlation squares this coefficient. The STDP-filtered result has
    exactly 'degree' harmonics, with no further hard Fourier truncation.
    """
    ratios, g = [], 1.0
    for n in range(1, degree + 1):
        g *= (degree - n + 1) / (degree + n)
        ratios.append(g)
    return coefficients(degree, tau) * np.asarray(ratios) ** 2


def pulse_window(delta, degree, tau=1.0):
    delta = np.asarray(delta)
    result = np.zeros_like(delta, dtype=np.float64)
    for n, b in enumerate(pulse_coefficients(degree, tau), 1):
        result += b * np.sin(n * delta)
    return result


def chord_window(delta, tau=1.0, resolution=0.01):
    """Phase-gradient alternative on the complex unit circle, not exact STDP.

    -tau*d/d_delta exp(-sqrt(|u_v-u_k|**2+resolution**2)/tau).
    It agrees with an exponential timing window at small resolved lags, uses
    only Re/Im(u_v*conj(u_k)), and makes no choice of an atan2 branch.
    """
    delta = np.asarray(delta)
    distance = np.sqrt(4 * np.sin(delta / 2) ** 2 + resolution ** 2)
    return np.sin(delta) / distance * np.exp(-distance / tau)


def certified_sign_bound(modes, tau=1.0, jitter=0.1):
    """Sufficient global sign guarantee, not just a sampled-grid check.

    The odd heat semigroup preserves L>=a/sinh(a*pi)*sin(delta) on (0,pi).
    |sin(n*delta)|<=n*|sin(delta)| bounds the omitted spectral tail relative
    to sin(delta). A decreasing Gaussian sum is bounded by its integral.
    """
    if jitter <= 0:
        raise ValueError('A finite positive jitter is required for this tail bound')
    a = 1 / tau
    lower = a / math.sinh(a * math.pi) * math.exp(-jitter ** 2 / 2)
    tail = math.sqrt(2 / math.pi) / jitter * math.erfc(jitter * modes / math.sqrt(2))
    return dict(modes=modes, tau=tau, jitter=jitter, positive_ratio_lower_bound=lower,
                omitted_tail_relative_to_sine_bound=tail, certified_positive=tail < lower)


def shape_preserving_metrics():
    x = np.linspace(0.0001, math.pi - 0.0001, 32768)
    records = []
    for tau in (0.25, 0.5, 1.0, 2.0):
        exact = periodic_window(x, tau)
        for family, function in [('positive_power_derivative', positive_power_window),
                                 ('positive_finite_pulse', pulse_window)]:
            last = np.zeros_like(x)
            for degree in (4, 8, 16, 32, 64, 128):
                y = function(x, degree, tau)
                assert y.min() > -2e-13
                row = dict(family=family, tau=tau, degree=degree,
                    wrong_sign_fraction=float(np.mean(y < -1e-8)),
                    peak_phase=float(x[np.argmax(y)]), peak_value=float(y.max()),
                    small_lag_values={str(d): float(function(d, degree, tau))
                                      for d in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)})
                if family == 'positive_power_derivative':
                    row['maximum_overshoot'] = float((y - exact).max())
                    row['minimum_increase_over_previous_degree'] = float((y - last).min())
                    assert (y - exact).max() < 2e-13
                    assert (y - last).min() > -2e-13
                records.append(row)
                last = y
    return records


def asymmetric_window(delta, ap=1.0, am=0.7, tp=1.0, tm=1.0):
    """Periodized asymmetric exponential, with midpoint convention at ties."""
    d = np.asarray(delta) % PERIOD
    pos = ap * np.exp(-d / tp) / (-np.expm1(-PERIOD / tp))
    neg = am * np.exp(-(PERIOD - d) / tm) / (-np.expm1(-PERIOD / tm))
    value = pos - neg
    at_zero = np.isclose(d, 0, atol=1e-14, rtol=0)
    midpoint = 0.5 * (ap - am) + ap / np.expm1(PERIOD / tp) - am / np.expm1(PERIOD / tm)
    return np.where(at_zero, midpoint, value)


def asymmetric_coefficients(modes, ap=1.0, am=0.7, tp=1.0, tm=1.0):
    n = np.arange(modes + 1, dtype=np.float64)
    return (ap / (1 / tp + 1j * n) - am / (1 / tm - 1j * n)) / PERIOD


def harmonic_write(v, k, modes=16, tau=1.0, jitter=0.04, amplitude_eps=0.0):
    """G[v,k] = mean_tokens sum_n b_n Im(V_n K_n^H).

    Inputs are complex [...,tokens,channels]. F_n(z)=|z| exp(i*n*arg(z))
    when eps=0. With eps>0, the smooth extension is
    F_n(z)=z**n/(|z|**2+eps**2)**((n-1)/2); its first mode remains exactly z.
    Each mode is reduced immediately, so no token*Dv*Dk intermediate is needed.
    """
    if not v.is_complex() or not k.is_complex():
        raise ValueError("v and k must contain complex activities")
    if amplitude_eps:
        rv = torch.sqrt(v.real.square() + v.imag.square() + amplitude_eps ** 2)
        rk = torch.sqrt(k.real.square() + k.imag.square() + amplitude_eps ** 2)
    else:
        # The finite epsilon branch should be used when differentiating at zero.
        rv = v.abs().clamp_min(torch.finfo(v.real.dtype).tiny)
        rk = k.abs().clamp_min(torch.finfo(k.real.dtype).tiny)
    uv, uk = v / rv, k / rk
    vm, km = v, k
    result = torch.zeros(*v.shape[:-2], v.shape[-1], k.shape[-1],
                         dtype=v.real.dtype, device=v.device)
    for b in coefficients(modes, tau, jitter):
        result = result + float(b) * (vm.imag.transpose(-1, -2) @ km.real
                                     - vm.real.transpose(-1, -2) @ km.imag)
        vm, km = vm * uv, km * uk
    return result / v.shape[-2]


def direct_write(v, k, tau=1.0):
    """Exact state-dependent phase reference, for small CPU inputs only."""
    pv, pk = np.angle(v), np.angle(k)
    weights = periodic_window(pv[..., :, :, None] - pk[..., :, None, :], tau)
    return (np.abs(v)[..., :, :, None] * np.abs(k)[..., :, None, :] * weights).mean(-3)


def filtered_pulse_write(v, k, phase_bins=1024, tau=1.0, jitter=0.04):
    """Independent phase-grid construction: filter pulse trains then correlate.

    Events are placed at integer grid positions. A normalized circular heat
    filter spreads each neuron with SD=jitter/sqrt(2). Applying the sampled
    STDP matrix to those activities must equal the jittered spectral write.
    """
    if v.ndim != 2 or k.ndim != 2:
        raise ValueError("This small reference expects tokens,channels")
    x = np.arange(phase_bins) * PERIOD / phase_bins
    freq = np.fft.fftfreq(phase_bins, d=1 / phase_bins)
    pulse = np.fft.ifft(np.exp(-0.25 * (jitter * freq) ** 2)).real
    # W[post,pre]; pulse sums to one, so no extra integration scaling remains.
    w = periodic_window(x[:, None] - x[None, :], tau)
    result = np.zeros((v.shape[1], k.shape[1]))
    for vt, kt in zip(v, k):
        vi = np.rint(np.angle(vt) * phase_bins / PERIOD).astype(int) % phase_bins
        ki = np.rint(np.angle(kt) * phase_bins / PERIOD).astype(int) % phase_bins
        pv = np.stack([np.roll(pulse, i) for i in vi]) * np.abs(vt)[:, None]
        pk = np.stack([np.roll(pulse, i) for i in ki]) * np.abs(kt)[:, None]
        result += pv @ w @ pk.T
    return result / len(v)


def kernel_metrics(tau=1.0):
    x = np.linspace(-math.pi, math.pi, 32769)
    exact = periodic_window(x, tau)
    records = []
    for jitter in (0.0, 0.04, 0.1):
        reference = exact if not jitter else spectral_window(x, 512, tau, jitter)
        for modes in (1, 4, 8, 16, 32, 64):
            y = spectral_window(x, modes, tau, jitter)
            outside = (np.abs(x) >= max(0.05, 3 * jitter)) & (np.abs(x) < math.pi - 0.02)
            positive = (x > 0.01) & (x < math.pi - 0.02)
            peak = np.argmax(np.where(x > 0, y, -np.inf))
            records.append(dict(tau=tau, jitter=jitter, modes=modes,
                rmse=float(np.sqrt(np.mean((y - reference) ** 2))),
                max_error_away_from_jump=float(np.abs(y[outside] - reference[outside]).max()),
                wrong_sign_fraction=float(np.mean(y[positive] < -1e-8)),
                peak_phase=float(x[peak]), peak_value=float(y[peak]),
                small_lag_values={str(d): float(spectral_window(d, modes, tau, jitter))
                                  for d in (0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)}))
    return records


def structural_checks(seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    # Independent definition: sum all time-separated pairs, rather than use
    # the closed-form window or its Fourier coefficients.
    d = np.r_[np.linspace(-math.pi, math.pi, 301), 0, 0.001, -0.001]
    for tau in (0.25, 1.0, 2.0, 10.0):
        pairs = sum(line_window(d + m * PERIOD, tau) for m in range(-120, 121))
        error = float(np.max(np.abs(pairs - periodic_window(d, tau))))
        assert error < 1e-12, (tau, error)
        results[f"all_pair_sum_error_tau_{tau}"] = error
    # Numerical integration over one cycle verifies the analytical spectrum.
    x = (np.arange(200000) + 0.5) * math.pi / 200000
    b = np.array([2 * np.mean(periodic_window(x) * np.sin(n * x)) for n in range(1, 65)])
    results['fourier_coefficient_error'] = float(np.abs(b - coefficients(64)).max())
    assert results['fourier_coefficient_error'] < 3e-8
    # Asymmetric amplitude and decay need complex coefficients, not Im-only.
    grid = (np.arange(30000) + 0.5) * PERIOD / 30000
    target = asymmetric_window(grid, 1.0, 0.7, 0.8, 1.3)
    estimated = np.array([np.mean(target * np.exp(-1j * n * grid)) for n in range(20)])
    predicted = asymmetric_coefficients(19, 1.0, 0.7, 0.8, 1.3)
    results['asymmetric_coefficient_error'] = float(np.abs(estimated - predicted).max())
    assert results['asymmetric_coefficient_error'] < 3e-8
    v = (0.2 + rng.random((7, 11))) * np.exp(1j * rng.uniform(-math.pi, math.pi, (7, 11)))
    k = (0.2 + rng.random((7, 9))) * np.exp(1j * rng.uniform(-math.pi, math.pi, (7, 9)))
    vt, kt = torch.tensor(v), torch.tensor(k)
    operator = harmonic_write(vt, kt, 32, jitter=0.1)
    # Evaluate the kernel entrywise independently of matrix feature products.
    phase_difference = np.angle(v)[:, :, None] - np.angle(k)[:, None, :]
    reference = (np.abs(v)[:, :, None] * np.abs(k)[:, None, :]
                 * spectral_window(phase_difference, 32, jitter=0.1)).mean(0)
    results['feature_vs_pair_kernel_error'] = float(np.max(np.abs(operator.numpy() - reference)))
    assert results['feature_vs_pair_kernel_error'] < 1e-13
    # Arbitrary rotation can even differ by token. Each token's common carrier
    # cancels before the token sum; it is not an oscillator state update.
    rotation = torch.tensor(np.exp(1j * rng.uniform(-math.pi, math.pi, (7, 1))))
    rotated = harmonic_write(vt * rotation, kt * rotation, 32, jitter=0.1)
    results['common_carrier_invariance_error'] = float((rotated - operator).abs().max())
    assert results['common_carrier_invariance_error'] < 1e-13
    reversal = harmonic_write(kt, vt, 32, jitter=0.1)
    results['role_reversal_error'] = float((operator + reversal.T).abs().max())
    assert results['role_reversal_error'] < 1e-13
    # Finite timing precision: compare its independent Gaussian-jitter meaning.
    relative_jitter = rng.normal(0, 0.1, 300000)
    points = np.array([0.02, 0.1, 0.3, 1.0, 2.8])
    monte_carlo = np.array([periodic_window(p + relative_jitter).mean() for p in points])
    smooth = spectral_window(points, 512, jitter=0.1)
    results['gaussian_jitter_monte_carlo_max_error'] = float(np.max(np.abs(monte_carlo - smooth)))
    assert results['gaussian_jitter_monte_carlo_max_error'] < 0.007
    # Compare full filtered pulse trains with the closed-form Fourier features.
    bins = 1024
    vp = np.rint(np.angle(v[:2, :4]) * bins / PERIOD) * PERIOD / bins
    kp = np.rint(np.angle(k[:2, :5]) * bins / PERIOD) * PERIOD / bins
    vg = np.abs(v[:2, :4]) * np.exp(1j * vp)
    kg = np.abs(k[:2, :5]) * np.exp(1j * kp)
    filtered = filtered_pulse_write(vg, kg, bins)
    featured = harmonic_write(torch.tensor(vg), torch.tensor(kg), 256, jitter=0.04).numpy()
    results['filtered_pulse_vs_features_max_error'] = float(np.max(np.abs(filtered - featured)))
    # A discrete quadrature of a discontinuity converges less quickly than the
    # analytic spectrum; the grid here is a reference, not a training proposal.
    assert results['filtered_pulse_vs_features_max_error'] < 2e-4
    # Trainability check: verify derivatives at generic activities and finite
    # gradients at a silent neuron using the disclosed smooth amplitude rule.
    av = vt[:2, :3].clone().requires_grad_()
    ak = kt[:2, :3].clone().requires_grad_()
    results['gradcheck_nonzero_activities'] = bool(torch.autograd.gradcheck(
        lambda a, b: harmonic_write(a, b, 8, jitter=0.1, amplitude_eps=1e-6),
        (av, ak), eps=1e-6, atol=2e-5, rtol=2e-4))
    av0 = torch.zeros(2, 3, dtype=torch.complex128, requires_grad=True)
    g0 = harmonic_write(av0, ak, 16, jitter=0.1, amplitude_eps=1e-6)
    g0.sum().backward()
    results['silent_neuron_gradient_finite'] = bool(torch.isfinite(av0.grad).all())
    assert results['silent_neuron_gradient_finite']
    # One token: a sine outer product has rank <=2. Multiple harmonics add
    # channel-kernel rank without inventing additional neuron carrier speeds.
    phase_v = rng.uniform(-math.pi, math.pi, (1, 32))
    phase_k = rng.uniform(-math.pi, math.pi, (1, 32))
    for modes in (1, 4, 8, 16):
        g = harmonic_write(torch.tensor(np.exp(1j * phase_v)),
                           torch.tensor(np.exp(1j * phase_k)), modes, jitter=0.04).numpy()
        results[f'one_token_rank_modes_{modes}'] = int(np.linalg.matrix_rank(g, tol=1e-10))
    tied = harmonic_write(vt, vt, 16, jitter=0.1)
    results['tied_kv_skew_symmetry_error'] = float((tied + tied.T).abs().max())
    assert results['tied_kv_skew_symmetry_error'] < 1e-13
    # A finite positive pulse is an independent construction, rather than a
    # Gaussian spectrum with an arbitrary truncation level. Verify its actual
    # temporal convolution against the finite analytical sine coefficients.
    bins, degree = 8192, 32
    x = np.arange(bins) * PERIOD / bins
    p = np.cos(x / 2) ** (2 * degree)
    p = p / p.sum()
    expected = np.fft.ifft(np.fft.fft(periodic_window(x)) * np.abs(np.fft.fft(p)) ** 2).real
    predicted = pulse_window(x, degree)
    results['finite_pulse_convolution_error'] = float(np.abs(expected - predicted).max())
    assert results['finite_pulse_convolution_error'] < 3e-6
    # Verify the positive-series derivative independently against finite
    # differences of its even polynomial, including its normalization.
    a, degree = 1.0, 16
    pts = np.array([0.1, 0.3, 1.0, 2.7])
    def even_polynomial(x):
        s, c = np.cos(x / 2) ** 2, 1.0
        total, power = np.ones_like(x), np.ones_like(x)
        for m in range(1, degree + 1):
            c *= ((m - 1) ** 2 + a * a) / ((m - 0.5) * m)
            power *= s
            total += c * power
        return total / np.sinh(a * math.pi)
    dx = 1e-6
    derivative = -(even_polynomial(pts + dx) - even_polynomial(pts - dx)) / (2 * dx * a)
    results['positive_series_derivative_error'] = float(np.abs(derivative - positive_power_window(pts, degree)).max())
    assert results['positive_series_derivative_error'] < 1e-8
    # A sinusoid-only activity is a decisive independent countercheck: even
    # with the exact exponential STDP filter, only the first harmonic survives.
    grid = np.arange(4096) * PERIOD / 4096
    wfft = np.fft.fft(periodic_window(grid))
    phase_k, modulation = 0.3, 0.8
    pk = (1 + modulation * np.cos(grid - phase_k)) / len(grid)
    numerical, exact = [], []
    for phase_v in (-2.0, -0.2, 0.6, 1.3, 2.2):
        pv = (1 + modulation * np.cos(grid - phase_v)) / len(grid)
        numerical.append(pv @ np.fft.ifft(wfft * np.fft.fft(pk)).real)
        exact.append(coefficients(1)[0] * modulation ** 2 / 4 * np.sin(phase_v - phase_k))
    results['sinusoidal_activity_exact_stdp_error'] = float(np.max(np.abs(np.asarray(numerical) - exact)))
    assert results['sinusoidal_activity_exact_stdp_error'] < 5e-8
    # The geometric candidate is also a derivative of an even kernel. It is
    # explicitly an alternative timing metric, not the exact periodized window.
    resolution, dx = 0.01, 1e-6
    def chord_potential(delta):
        return np.exp(-np.sqrt(4 * np.sin(delta / 2) ** 2 + resolution ** 2))
    numeric = -(chord_potential(pts + dx) - chord_potential(pts - dx)) / (2 * dx)
    results['chord_phase_gradient_error'] = float(np.abs(numeric - chord_window(pts)).max())
    assert results['chord_phase_gradient_error'] < 1e-8
    return results


def consistency_demo():
    """Mechanism only: FFN-like phase schedules are supplied, not learned.

    One useful relation keeps its sign; one nuisance relation alternates sign
    with the same amplitude. Unequal lags and magnitudes show the limitations.
    A Hebbian amplitude product cannot distinguish these supplied schedules.
    """
    records = []
    for rho in (1.0, 0.99, 0.95):
        stable = balanced = unequal = amplitude_bias = 0.0
        history = []
        for step in range(1, 129):
            sign = 1 if step % 2 else -1
            stable = rho * stable + float(periodic_window(0.2))
            balanced = rho * balanced + float(periodic_window(sign * 0.2))
            unequal = rho * unequal + float(periodic_window(0.1 if sign > 0 else -0.5))
            amplitude_bias = rho * amplitude_bias + (1.2 if sign > 0 else 0.8) * float(periodic_window(sign * 0.2))
            history.append([step, stable, balanced, unequal, amplitude_bias])
        records.append(dict(rho=rho, stable=stable, equal_alternation=balanced,
            unequal_lag_alternation=unequal, unequal_amplitude_alternation=amplitude_bias,
            ratio_equal_to_stable=abs(balanced / stable), history=history))
    assert abs(records[0]['equal_alternation']) < 1e-12
    return records


def precision_checks():
    """Quantify floating-point limits separately from the analytical proof."""
    x = np.linspace(1e-7, math.pi - 1e-7, 20001)
    v = torch.tensor(np.exp(1j * x)[None], dtype=torch.complex64)
    k = torch.ones(1, 1, dtype=torch.complex64)
    rotation = torch.tensor(np.exp(1.2j), dtype=torch.complex64)
    y = harmonic_write(v, k, 32, jitter=0.1).numpy()[:, 0]
    rotated = harmonic_write(v * rotation, k * rotation, 32, jitter=0.1).numpy()[:, 0]
    ref = spectral_window(x, 32, jitter=0.1)
    wrong = np.flatnonzero(rotated < 0)
    resolved = (x > 1e-5) & (x < math.pi - 1e-5)
    assert not np.any(rotated[resolved] < 0)
    return dict(points=len(x), float32_max_error=float(np.abs(y - ref).max()),
        common_rotation_max_error=float(np.abs(y - rotated).max()),
        unrotated_wrong_sign_count=int(np.sum(y < 0)),
        rotated_wrong_sign_count=len(wrong),
        rotated_wrong_sign_points=[dict(phase=float(x[i]), distance_to_pi=float(math.pi-x[i]),
                                       value=float(rotated[i])) for i in wrong],
        resolved_wrong_sign_count=int(np.sum(rotated[resolved] < 0)),
        note='The analytical certificate concerns exact arithmetic; distinguish rounding at ties from phase-cycle ambiguity.')


def monitor_training(root):
    result = []
    for name in ('urm_swiglu_loops16_20261004', 'v17_aligned_6000_20261004'):
        p = root / name
        rows = [json.loads(line) for line in (p / 'train.jsonl').read_text().splitlines()]
        windows = []
        for lo, hi in ((2000, 4000), (4000, 6000), (5000, 6000)):
            w = [r for r in rows if lo < r['step'] <= hi]
            terminal = [r for r in w if r.get('_count_raw', 0) > 0]
            windows.append(dict(start_exclusive=lo, end_inclusive=hi, all_segment_count=len(w),
                terminal_segment_count=len(terminal),
                all_segment_loss=float(np.mean([r['lm_loss'] for r in w])),
                terminal_loss=float(np.mean([r['lm_loss'] for r in terminal])),
                terminal_cell_accuracy=float(np.mean([r['accuracy'] for r in terminal])),
                terminal_exact_accuracy=float(np.mean([r['exact_accuracy'] for r in terminal]))))
        result.append(dict(run=name, last_step=rows[-1]['step'], windows=windows,
            terminal_selector='_count_raw > 0; verified fixed loops=16 with ACT disabled'))
    return result


def plots(out, metrics, consistency, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    x = np.linspace(-math.pi, math.pi, 10001)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for a in ax:
        a.plot(x, periodic_window(x), color='black', lw=1.5, label='Exact periodic STDP')
        a.axhline(0, color='gray', lw=0.5)
        a.set_xlabel('Relative channel phase (rad)')
        a.set_ylabel('Write window')
    ax[0].plot(x, np.sin(x), '--', label='Single sine (gain=1)')
    ax[0].plot(x, spectral_window(x, 16), label='16 harmonics, raw')
    ax[0].plot(x, spectral_window(x, 32, jitter=0.04), label='32 harmonics, jitter SD=0.04')
    ax[0].set_title('Shape and periodic wrap')
    ax[0].legend(fontsize=9)
    ax[1].plot(x, spectral_window(x, 16), label='16 harmonics, raw')
    ax[1].plot(x, spectral_window(x, 32, jitter=0.04), label='32 harmonics, jitter SD=0.04')
    ax[1].plot(x, spectral_window(x, 64, jitter=0.04), label='64 harmonics, jitter SD=0.04')
    ax[1].set_xlim(-0.3, 0.3)
    ax[1].set_title('The unavoidable resolution region around zero')
    ax[1].legend(fontsize=9)
    fig.savefig(out / 'window_shapes.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    positive = np.linspace(0, math.pi, 10001)
    for a in ax:
        a.plot(positive, periodic_window(positive), 'k', lw=1.5, label='Exact periodic STDP')
        a.set_xlabel('Positive relative channel phase (rad)')
        a.set_ylabel('Write strength')
        a.set_xlim(0, 1.5)
    for degree in (8, 32, 128):
        ax[0].plot(positive, positive_power_window(positive, degree), label=f'Positive series, degree={degree}')
        ax[1].plot(positive, pulse_window(positive, degree), label=f'Positive pulse, degree={degree}')
    ax[0].set_title('Derivative of a positive memory kernel')
    ax[1].set_title('Finite positive activities filtered by STDP')
    ax[0].legend(fontsize=9)
    ax[1].legend(fontsize=9)
    fig.savefig(out / 'sign_preserving_windows.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(positive, periodic_window(positive), 'k', label='Exact periodized exponential')
    for resolution in (0.01, 0.05, 0.1):
        ax.plot(positive, chord_window(positive, resolution=resolution), label=f'Chord-gradient, resolution={resolution}')
    ax.set_xlim(0, 1.5)
    ax.set_xlabel('Positive relative phase (rad)')
    ax.set_ylabel('Write strength')
    ax.set_title('A smooth geometric alternative; changes the lag metric')
    ax.legend(fontsize=9)
    fig.savefig(out / 'geometric_alternative.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for jitter in (0.0, 0.04, 0.1):
        subset = [m for m in metrics if m['tau'] == 1 and m['jitter'] == jitter]
        ax[0].loglog([m['modes'] for m in subset], [m['rmse'] for m in subset], 'o-', label=f'Jitter SD={jitter}')
    ax[0].set_xlabel('Number of harmonics')
    ax[0].set_ylabel('RMSE against exact or jittered window')
    ax[0].set_title('Accuracy costs bandwidth')
    ax[0].legend(fontsize=9)
    for row in consistency:
        history = np.asarray(row['history'])
        ax[1].plot(history[:, 0], history[:, 2] / np.maximum(history[:, 1], 1e-20), label=f'Equal sign alternation, rho={row["rho"]}')
    ax[1].set_xlabel('Recurrent integration step')
    ax[1].set_ylabel('Nuisance / persistent relation')
    ax[1].set_title('Cancellation depends on weighted balance')
    ax[1].legend(fontsize=9)
    fig.savefig(out / 'resolution_and_consistency.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for row in training:
        name = 'URM' if row['run'].startswith('urm_') else 'v1.7 aligned'
        w = row['windows'][:2]
        ax[0].plot([3000, 5000], [r['all_segment_loss'] for r in w], 'o-', label=f'{name}: all segments')
        ax[0].plot([3000, 5000], [r['terminal_loss'] for r in w], 'o--', label=f'{name}: segment 16')
        ax[1].plot([3000, 5000], [100 * r['terminal_cell_accuracy'] for r in w], 'o-', label=name)
    ax[0].set_xlabel('Center of 2,000-step window')
    ax[0].set_ylabel('Mean train loss')
    ax[0].set_title('Matched aggregation; different architectures')
    ax[0].legend(fontsize=9)
    ax[1].set_xlabel('Center of 2,000-step window')
    ax[1].set_ylabel('Terminal-segment cell accuracy (%)')
    ax[1].legend(fontsize=9)
    fig.savefig(out / 'training_monitor.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/phase_stdp_math_20261004')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = structural_checks()
    metrics = sum((kernel_metrics(tau) for tau in (0.25, 0.5, 1.0, 2.0)), [])
    consistency = consistency_demo()
    shape_preserving = shape_preserving_metrics()
    certificates = [certified_sign_bound(n, tau, jitter) for tau, jitter in
                    ((0.25, 0.04), (1.0, 0.04), (1.0, 0.1), (2.0, 0.04))
                    for n in (16, 26, 32, 64, 72, 128)]
    assert certified_sign_bound(32, 1.0, 0.1)['certified_positive']
    assert certified_sign_bound(72, 1.0, 0.04)['certified_positive']
    precision = precision_checks()
    training = monitor_training(Path('runs'))
    report = dict(protocol='Channel-phase kernel mathematics; no task-performance or universal-superiority claim.',
                  zero_lag_convention='midpoint=0; opposite one-sided limits',
                  phase_convention='phase stores event delay; physical e^(+i*omega*t) angle has opposite event-time sign',
                  jitter_convention='SD of relative lag; each independent neuron pulse has SD=jitter/sqrt(2)',
                  checks=checks, kernel_metrics=metrics, shape_preserving_metrics=shape_preserving,
                  sign_certificates=certificates, precision_checks=precision,
                  consistency=consistency, training=training,
                  sources=[
                      'https://arxiv.org/html/2603.15569v1',
                      'https://arxiv.org/html/2609.24797v1',
                      'https://proceedings.neurips.cc/paper_files/paper/2000/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf',
                      'https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2010.00032/full',
                      'https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004878',
                      'https://www.gatsby.ucl.ac.uk/~dayan/papers/lkdp05.pdf',
                      'https://www.gatsby.ucl.ac.uk/~dayan/papers/ld2006.pdf'])
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    plots(out, metrics, consistency, training)
    print(json.dumps(dict(checks=checks, selected_metrics=[m for m in metrics if m['tau'] == 1 and
        (m['modes'], m['jitter']) in ((1, 0.0), (16, 0.0), (32, 0.04), (64, 0.04))],
        consistency=[{k: v for k, v in row.items() if k != 'history'} for row in consistency]), indent=2))


if __name__ == '__main__':
    main()
