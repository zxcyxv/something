"""Independent algebra checks for recurrent-axis pair STDP.

These are checks of stated identities/counterexamples, not training experiments
or floating-point stability interventions.  The analytical statements apply to
prescribed K/V histories unless an explicit model-level check says otherwise.
"""
import argparse
import json
from pathlib import Path

import sympy as sp
import torch


def outer(v, k):
    return v[..., :, None] * k[..., None, :]


def pair_memory(keys, values, lam):
    """Direct temporal pair sum. Shapes [time, channel], scalar common decay."""
    return sum((1-lam)*lam**(r-s-1) *
               (outer(values[r], keys[s])-outer(values[s], keys[r]))
               for r in range(len(keys)) for s in range(r))


def trace_writes(keys, values, lam_k, lam_v=None, ek=None, ev=None):
    lam_v = lam_k if lam_v is None else lam_v
    ek = torch.zeros_like(keys[0]) if ek is None else ek.clone()
    ev = torch.zeros_like(values[0]) if ev is None else ev.clone()
    writes = []
    for k, v in zip(keys, values):
        writes.append(outer(v, ek)-outer(ev, k))
        ek, ev = lam_k*ek+(1-lam_k)*k, lam_v*ev+(1-lam_v)*v
    return torch.stack(writes), ek, ev


def error(actual, expected):
    return float((actual-expected).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7301)
    torch.set_default_dtype(torch.float64)
    results = {'scope': 'Algebraic identities and explicit counterexamples; no training conclusions.',
               'checks': {}}
    records = results['checks']

    # Temporal matrix is skew, but heteroassociative channel memory need not be.
    n, dk, dv, lam = 9, 4, 3, 0.3
    k, v = torch.randn(n, dk), torch.randn(n, dv)
    a = torch.zeros(n, n)
    for r in range(n):
        for s in range(r):
            a[r, s] = (1-lam)*lam**(r-s-1)
            a[s, r] = -a[r, s]
    direct = pair_memory(k, v, lam)
    recursive = trace_writes(k, v, lam)[0].sum(0)
    records['temporal_operator'] = {
        'V_T_A_K_error': error(v.T @ a @ k, direct),
        'trace_vs_pair_error': error(recursive, direct),
        'A_skew_error': error(a, -a.T),
        'finite_window_A_times_constant_norm': float((a @ torch.ones(n)).norm()),
        'note': 'A is skew in time indices. V.T @ A @ K is not generally skew in channel indices. A @ 1 is a finite-window boundary term, not identically zero.'}

    # Full complex Hebbian accumulation and its STDP imaginary component.
    p = torch.tril(a, diagonal=-1)
    complex_k = p @ k+1j*k
    complex_v = p @ v+1j*v
    complex_memory = complex_v.T @ complex_k.conj()
    complex_temporal = p.T @ p+torch.eye(n)+1j*(p-p.T)
    n_small = 3
    ps = torch.tensor([[0.,0.,0.],[.9,0.,0.],[.09,.9,0.]])
    full_gram = ps.T @ ps+torch.eye(n_small)+1j*(ps-ps.T)
    without_E = torch.eye(n_small)+1j*(ps-ps.T)
    records['complex_extension'] = {
        'full_temporal_operator_error':error(complex_memory,v.to(torch.complex128).T @ complex_temporal @ k.to(torch.complex128)),
        'imaginary_STDP_error':error(complex_memory.imag,direct),
        'full_Gram_min_eigenvalue':float(torch.linalg.eigvalsh(full_gram).min()),
        'B_plus_iG_without_E_min_eigenvalue':float(torch.linalg.eigvalsh(without_E).min()),
        'formula':'sum C = V.T @ [P.T@P+I+i*(P-P.T)] @ K; sum G = V.T @ (P-P.T) @ K',
        'note':'P is the strict-past trace filter. Imaginary selection gives STDP exactly. Dropping E changes full complex Hebbian accumulation; the self-associative Gram example can lose positive semidefiniteness. PSD is not required or guaranteed for the actual independent-K/V signed model.'}

    # Identical endpoints and static content do not determine STDP memory.
    e1, e2, z = torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.zeros(2)
    cw = torch.stack((e1, e2, -e1, -e2, e1))
    ccw = cw.flip(0)
    mcw, mccw = pair_memory(cw, cw, 0.), pair_memory(ccw, ccw, 0.)
    records['path_dependence'] = {
        'same_first_and_last': bool(torch.equal(cw[[0, -1]], ccw[[0, -1]])),
        'same_sum_current_KV': bool(torch.equal(cw.T @ cw, ccw.T @ ccw)),
        'clockwise_memory': mcw.tolist(), 'counterclockwise_memory': mccw.tolist(),
        'reversal_error': error(mcw, -mccw),
        'note': 'At lambda=0 these are nearest-neighbour writes. Order, not just final state or occupancy, determines the memory.'}

    # A pure-STDP memory can store an ordinary association with timed activity.
    key, value = torch.tensor([2., -1.]), torch.tensor([3., 4., -2.])
    kp = torch.stack((key, torch.zeros_like(key)))
    vp = torch.stack((torch.zeros_like(value), value))
    mem = pair_memory(kp, vp, lam)
    q = key/key.square().sum()/(1-lam)
    simultaneous = pair_memory(torch.stack((key, key)),
                               torch.stack((value, value)), lam)
    records['association_capacity'] = {
        'timed_memory_error': error(mem, (1-lam)*outer(value, key)),
        'real_query_retrieval_error': error(mem @ q, value),
        'same_time_constant_activity_memory_norm': float(simultaneous.norm()),
        'note': 'Q can be a real cue with no trace. Pure STDP can store KV; synchronous stationary activation and separated pre/post events are different encodings.'}

    # G-only is an ordinary signed associative read of two temporal populations.
    n, d = 7, 4
    k, v, ek, ev, q = [torch.randn(n, d) for _ in range(5)]
    g = v.T @ ek-ev.T @ k
    extended_keys = torch.cat((ek, k), dim=0)
    extended_values = torch.cat((v, -ev), dim=0)
    records['G_as_associative_read'] = {
        'extended_attention_error': error(q @ g.T, (q @ extended_keys.T) @ extended_values),
        'formula': 'keys=[past_K; current_K], values=[current_V; -past_V], read=Q @ keys.T @ values',
        'note': 'This uses real Q with the standard associative read. The temporal association list, rather than the matrix-read algebra, changes from current KV.'}

    # A pure temporal read can emulate a static KV lookup via two-phase coding.
    key, value, query = torch.randn(4), torch.randn(3), torch.randn(4)
    amplitude = 1/(1+lam)
    keys = torch.stack((key, torch.zeros_like(key)))
    values = torch.stack((torch.zeros_like(value), value))
    # First phase is K-active. Equilibrated trace just before it follows V-active.
    ek0, ev0 = lam/(1+lam)*key, 1/(1+lam)*value
    phase_writes = trace_writes(keys, values, lam, ek=ek0, ev=ev0)[0]
    phase_queries = torch.stack((-query, query))
    phase_reads = torch.einsum('rij,rj->ri', phase_writes, phase_queries)
    target = amplitude*value*(key @ query)
    records['two_phase_G_only_lookup'] = {
        'both_phases_lookup_error': error(phase_reads, target.expand_as(phase_reads)),
        'cycle_sum_write_norm': float(phase_writes.sum(0).norm()),
        'note': 'Constructive capacity example, not the learned network trajectory. Alternate K-only and V-only activity; alternate Q sign. G changes sign while its Q read stays fixed. This disproves an impossibility argument based solely on zero cycle-average writing.'}

    # Independent projections of a common hidden vector need not yield skew M.
    # h=[k;v]; K picks the first half and V the second half.
    h = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
    kx, vx = h[:, :2], h[:, 2:]
    projected = pair_memory(kx, vx, 0.)
    latent_skew = pair_memory(h, h, 0.)
    pk = torch.cat((torch.eye(2), torch.zeros(2, 2)), dim=1)
    pv = torch.cat((torch.zeros(2, 2), torch.eye(2)), dim=1)
    records['projection_counterexample'] = {
        'latent_skew_error': error(latent_skew, -latent_skew.T),
        'projected_memory': projected.tolist(),
        'projected_has_nonzero_diagonal': bool(projected.diagonal().abs().sum() > 0),
        'projected_factorization_error': error(projected, pv @ latent_skew @ pk.T),
        'note': 'Even with K and V both linear in the same h, the channel M need not be skew. Inferring imaginary eigenvalues for the actual model from STDP alone is invalid.'}

    # Shared hidden activity does not force each actual multihead M to be skew.
    # For H>=3, K_h reads hidden group h and V_h reads group h+1 (cyclic).
    # The selected off-diagonal blocks of one skew hidden matrix are independent.
    nheads, width = 4, 3
    target_memories = torch.randn(nheads,width,width)
    latent = torch.zeros(nheads*width,nheads*width)
    for head in range(nheads):
        pre = slice(head*width,(head+1)*width)
        nxt = (head+1)%nheads
        post = slice(nxt*width,(nxt+1)*width)
        latent[post,pre] = target_memories[head]
        latent[pre,post] = -target_memories[head].T
    actual_memories = torch.stack([latent[((head+1)%nheads)*width:((head+1)%nheads+1)*width,
                                          head*width:(head+1)*width] for head in range(nheads)])
    # Every skew latent matrix can be written by a finite signed trace path.
    # e: 0 -> basis(j) -> c*basis(i) -> 0, one wedge per nonzero entry.
    trace_path = [torch.zeros(nheads*width)]
    for i in range(nheads*width):
        for j in range(i):
            if latent[i,j] == 0:
                continue
            ej, ai = torch.zeros(nheads*width), torch.zeros(nheads*width)
            ej[j] = 1.
            ai[i] = latent[i,j]*(1-lam)
            trace_path.extend((ej,ai,torch.zeros(nheads*width)))
    trace_path = torch.stack(trace_path)
    activities = (trace_path[1:]-lam*trace_path[:-1])/(1-lam)
    written = trace_writes(activities,activities,lam)[0].sum(0)
    records['multihead_capacity_construction'] = {
        'arbitrary_head_memory_error':error(actual_memories,target_memories),
        'latent_skew_error':error(latent,-latent.T),
        'finite_activity_history_write_error':error(written,latent),
        'note':'For at least three heads, cyclic block-selection K/V projections can represent arbitrary independent head matrices through a shared skew hidden-history matrix. The actual H=8 model is not subject to a general per-head skew/capacity obstruction. This is an operator capacity construction, not a proof that the learned recurrence reaches the required histories.'}

    # Closed-form period-2 writes with different channel time constants.
    km, vm, kd, vd = [torch.randn(5) for _ in range(4)]
    lk, lv = torch.linspace(.08, .14, 5), torch.linspace(.09, .13, 5)
    ak, av = (1-lk)/(1+lk), (1-lv)/(1+lv)
    ks, vs = torch.stack((km+kd, km-kd)), torch.stack((vm+vd, vm-vd))
    gs = trace_writes(ks, vs, lk, lv, ek=km-ak*kd, ev=vm-av*vd)[0]
    gmean = outer(av*vd, kd)-outer(vd, ak*kd)
    galt = outer((1+av)*vd, km)-outer(vm, (1+ak)*kd)
    ashared = (1-lam)/(1+lam)
    gshared = trace_writes(ks, vs, lam, ek=km-ashared*kd, ev=vm-ashared*vd)[0]
    records['period_two'] = {
        'heterogeneous_mean_formula_error': error(gs.mean(0), gmean),
        'heterogeneous_alternating_formula_error': error((gs[0]-gs[1])/2, galt),
        'heterogeneous_mean_norm': float(gmean.norm()),
        'shared_decay_cycle_mean_norm': float(gshared.mean(0).norm()),
        'shared_decay_instantaneous_write_norm': float(gshared[0].norm()),
        'note': 'Common decay cancels cycle-average writing but not the alternating write. Heterogeneous decays can produce secular accumulation on an exactly periodic prescribed trajectory.'}

    # Shift by a constant input contributes only a filtered endpoint term.
    n, d = 11, 4
    k, v = torch.randn(n, d), torch.randn(n, d)
    ki, vi = torch.randn(d), torch.randn(d)
    full = pair_memory(k+ki, v+vi, lam)
    dynamic = pair_memory(k, v, lam)
    coeff = torch.tensor([lam**(n-1-r)-lam**r for r in range(n)])
    endpoint = outer((coeff[:, None]*v).sum(0), ki)-outer(vi, (coeff[:, None]*k).sum(0))
    records['static_input_boundary'] = {
        'endpoint_identity_error': error(full-dynamic, endpoint),
        'endpoint_coefficient_sum': float(coeff.sum()),
        'formula': 'D_R(x)=sum_{r=1..R}(lambda**(R-r)-lambda**(r-1))*x_r; M(k+a,v+b)-M(k,v)=D_R(v)*a.T-b*D_R(k).T',
        'note': 'Requires common fixed decay, zero initial traces and fixed input offset. It is not a claim that static task information cannot be encoded in changing hidden activity.'}

    # For common decay, current/past writes are exactly adjacent differences
    # of the filtered trajectory. Integrating them is a path integral.
    k, v = torch.randn(13, 4), torch.randn(13, 3)
    ek_hist, ev_hist = [torch.zeros(4)], [torch.zeros(3)]
    for kr, vr in zip(k, v):
        ek_hist.append(lam*ek_hist[-1]+(1-lam)*kr)
        ev_hist.append(lam*ev_hist[-1]+(1-lam)*vr)
    ek_hist, ev_hist = torch.stack(ek_hist), torch.stack(ev_hist)
    direct_g = trace_writes(k, v, lam)[0]
    trace_area_g = (outer(ev_hist[1:],ek_hist[:-1])-outer(ev_hist[:-1],ek_hist[1:]))/(1-lam)
    endpoint_product = outer(ev_hist[-1],ek_hist[-1])/(1-lam)
    moving_key_path = (outer(ev_hist[1:]+ev_hist[:-1],ek_hist[1:]-ek_hist[:-1])).sum(0)/(1-lam)
    records['filtered_path_integral'] = {
        'adjacent_filtered_path_error': error(direct_g, trace_area_g),
        'endpoint_minus_key_motion_error': error(direct_g.sum(0),endpoint_product-moving_key_path),
        'formula': 'G_r=(eV_r*eK_{r-1}.T-eV_{r-1}*eK_r.T)/(1-lambda); M_R=(eV_R*eK_R.T-sum_r[(eV_r+eV_{r-1})*(eK_r-eK_{r-1}).T])/(1-lambda)',
        'note': 'Zero initial traces, fixed common decay. The second term is a path-dependent correction from key motion; M is not generally a rescaled current or filtered KV product.'}

    # A bounded exactly periodic full state needs zero plasticity per period.
    cycles = {'period_two': torch.stack((e1,e2)),
              'period_three': torch.stack((e1,e2,-e1))}
    cyc_rows = {}
    for name, cycle in cycles.items():
        p = len(cycle)
        dec = .1
        weights = torch.tensor([(1-dec)*dec**(p-r-1)/(1-dec**p) for r in range(p)])
        trace0 = (weights[:,None]*cycle).sum(0)
        gs, ek_end, ev_end = trace_writes(cycle, cycle, dec, ek=trace0, ev=trace0)
        cyc_rows[name] = {'net_write_norm':float(gs.sum(0).norm()),
                          'instantaneous_write_norm':float(gs.norm()),
                          'trace_periodicity_error':error(ek_end,trace0)}
    records['periodic_full_state_condition'] = {
        'examples':cyc_rows,
        'condition':'M_{r+P}=M_r requires sum_{s=r+1..r+P} G_s=0. Otherwise an exactly repeated prescribed activity cycle causes secular memory drift.',
        'note':'Necessary condition, not a theorem that dynamics is attracted to period two. The original per-channel decay can give nonzero mean writing even for period two.'}

    # Fixed-address STDP read is a value change, without any complex Query.
    value_before, value_after = torch.randn(3),torch.randn(3)
    address = torch.randn(4)
    q_real = address/address.square().sum()
    delta_write = outer(value_after,address)-outer(value_before,address)
    records['fixed_address_increment_read'] = {
        'value_increment_read_error':error(delta_write @ q_real,value_after-value_before),
        'note':'For lambda=0 and unchanged K, G with a current real Q retrieves the change in V. This fits a residual update interpretation and does not require a Query trace.'}

    # The unnormalised cross-spectrum formula retains both amplitude and phase.
    w = .71
    ka, va = torch.randn(4, dtype=torch.complex128), torch.randn(3, dtype=torch.complex128)
    lk, lv = torch.tensor([.1, .2, .3, .4]), torch.tensor([.15, .25, .35])
    hc_k = (1-lk)*torch.exp(torch.tensor(-1j*w))/(1-lk*torch.exp(torch.tensor(-1j*w)))
    hc_v = (1-lv)*torch.exp(torch.tensor(-1j*w))/(1-lv*torch.exp(torch.tensor(-1j*w)))
    # Average at equally spaced phases eliminates the 2*omega phase contribution exactly.
    phases = torch.arange(16)*2*torch.pi/16
    ks = (torch.exp(1j*phases[:, None])*ka).real
    vs = (torch.exp(1j*phases[:, None])*va).real
    eks = (torch.exp(1j*phases[:, None])*hc_k*ka).real
    evs = (torch.exp(1j*phases[:, None])*hc_v*va).real
    actual = (outer(vs, eks)-outer(evs, ks)).mean(0)
    predicted = .5*(outer(va, ka.conj())*(hc_k.conj()[None, :]-hc_v[:, None])).real
    records['harmonic_mean'] = {'formula_error': error(actual, predicted),
        'formula': 'mean G_ij=0.5*Re[V_i*conj(K_j)*(conj(H_j)-H_i)], H_j=(1-lambda_j)*exp(-iw)/(1-lambda_j*exp(-iw))'}

    # With a common lambda, learning measures phase-lagged temporal correlation.
    l, c, s = sp.symbols('l c s', real=True)
    symbolic = sp.simplify((1-l)*(c-sp.I*s)/(1-l*(c-sp.I*s)))
    imag = sp.im(sp.together(symbolic)).subs(s**2, 1-c**2)
    expected_imag = -(1-l)*s/(1-2*l*c+l*l)
    records['symbolic_filter'] = {
        'imaginary_filter_identity': bool(sp.simplify(imag-expected_imag) == 0),
        'odd_window_DTFT': '-2i*(1-lambda)*sin(w)/(1-2*lambda*cos(w)+lambda**2)',
        'null_frequencies': ['0 (constant activity)', 'pi (period-2 activity)'],
        'note': 'Zero spectral weight concerns long-run accumulated association at that frequency, not zero instantaneous G on every period-2 block.'}

    # A truncation boundary changes what is held fixed when differentiating.
    # At a fixed activity, full-prefix STDP is identically zero for all a,b.
    # If incoming traces were generated with a0=b0=1 and then detached,
    # their response to a current change of projection scales is absent.
    prefix, chunk = 32, 8
    a_scale = torch.tensor(1., requires_grad=True)
    b_scale = torch.tensor(1., requires_grad=True)
    key = torch.tensor([1., -2.])
    value = torch.tensor([3., 1.])
    kk = (a_scale*key)[None].expand(prefix+chunk, -1)
    vv = (b_scale*value)[None].expand(prefix+chunk, -1)
    writes_full, _, _ = trace_writes(kk, vv, lam)
    probe = outer(value, key)/outer(value, key).square().sum()
    loss_full = (writes_full.sum(0)*probe).sum()
    grad_full = torch.autograd.grad(loss_full, (a_scale, b_scale), retain_graph=True)
    c0 = 1-lam**prefix
    fixed_ek, fixed_ev = c0*key, c0*value
    writes_cut, _, _ = trace_writes(kk[:chunk], vv[:chunk], lam,
                                   ek=fixed_ek, ev=fixed_ev)
    loss_cut = (writes_cut.sum(0)*probe).sum()
    grad_cut = torch.autograd.grad(loss_cut, (a_scale, b_scale))
    coefficient = c0*(1-lam**chunk)/(1-lam)
    expected = torch.tensor([-coefficient, coefficient])
    # A real optimizer-induced change of projections yields the same term.
    a_after, b_after = 0.99, 1.01
    jumped, _, _ = trace_writes((a_after*key)[None].expand(chunk, -1),
                                (b_after*value)[None].expand(chunk, -1), lam,
                                ek=fixed_ek, ev=fixed_ev)
    expected_jump = coefficient*(b_after-a_after)*outer(value, key)
    records['detached_trace_boundary'] = {
        'full_forward_memory_norm': float(writes_full.detach().sum(0).norm()),
        'cut_forward_memory_norm': float(writes_cut.detach().sum(0).norm()),
        'full_gradient_of_probe': [float(x) for x in grad_full],
        'cut_gradient_of_probe': [float(x) for x in grad_cut],
        'cut_gradient_formula_error': error(torch.stack(grad_cut), expected),
        'projection_jump_write_formula_error': error(jumped.sum(0), expected_jump),
        'formula': 'At incoming traces c0*k0,c0*v0 and constant within-segment k=a*k0,v=b*v0: DeltaM=c0*(1-lambda**B)/(1-lambda)*(b-a)*v0*k0.T.',
        'note': 'This is a derivative of the fixed-incoming-state continuation objective, not an autograd bug. An optimizer-induced change is itself a temporal event seen by STDP, although a frozen-parameter evaluation has no such event.'}

    # The initialized production block gives an exact, identifiable first write.
    from . import train as t
    cfg = t.LTConfig.from_dict(dict(t.CFG, hidden_size=16, num_heads=2, grid=2,
                  seq_len=4, batch_size=2, num_puzzle_identifiers=1,
                  puzzle_emb_ndim=0, amp=False, activation_checkpoint=False,
                  forward_dtype='float64', trace_decay_mode='head',
                  kv_qk_l2norm=False, kv_qk_rmsnorm=False))
    inner = t.KVSTDPInner(cfg).double()
    layer = inner.layers[0]
    inj = torch.randn(2, 4, 16)
    h0 = inner.init_hidden[None, None, :].expand_as(inj)
    def heads(x):
        return x.reshape(2, 4, 2, 8).transpose(1, 2)
    with torch.no_grad():
        first = inner.block(layer, h0, inj, None, None, None, None)
        second = inner.block(layer, first[0], inj, *first[1:], None)
        iv = heads(layer.v_proj(inner.embed_scale*inj))
        ik = inner.apply_rope(heads(layer.k_proj(inner.embed_scale*inj)), layer)
        hv = heads(layer.v_proj(h0))
        hk = inner.apply_rope(heads(layer.k_proj(h0)), layer)
        decay = layer.trace_decay[None, :, None, None]
        expected_first_write = (1-decay)*(iv.transpose(-1,-2)@hk-hv.transpose(-1,-2)@ik)/4
        # Removing the noncollinear initial reference, with zero initial bilinear
        # output, yields a collinear input-driven trajectory and no global read.
        state = torch.zeros_like(h0), None, None, None
        for _ in range(12):
            state = inner.block(layer, state[0], inj, *state[1:], None)
    records['initialized_production_block'] = {
        'first_block_memory_norm': float(first[1].norm()),
        'second_block_formula_error': error(second[1], expected_first_write),
        'zero_reference_memory_after_12_blocks_norm': float(state[1].norm()),
        'formula': 'G2=(1-lambda)*mean_p[(WV*I_p)*(R_p*WK*h0).T-(WV*h0)*(R_p*WK*I_p).T]',
        'note': 'Uses zero b_down at initialization and radial Phi. Actual h0 is nonzero, so this is not a dead-network bug. First cross-token communication is driven by the transition from the common initial reference toward the input.'}

    # Reading an accumulated synapse and reading its increment are different
    # even before the nonlinear neuron update: the Query itself also changes.
    count, dv, dk = 11, 3, 4
    increments = torch.randn(count,dv,dk)
    memories = increments.cumsum(0)
    queries = torch.randn(count,dk)
    reads = torch.einsum('tvk,tk->tv',memories,queries)
    previous_m = torch.cat((torch.zeros_like(memories[:1]),memories[:-1]))
    previous_q = torch.cat((torch.zeros_like(queries[:1]),queries[:-1]))
    previous_read = torch.cat((torch.zeros_like(reads[:1]),reads[:-1]))
    increment_read = torch.einsum('tvk,tk->tv',increments,queries)
    moving_query = torch.einsum('tvk,tk->tv',previous_m,queries-previous_q)
    future_queries = queries.flip(0).cumsum(0).flip(0)
    swapped = torch.einsum('tvk,tk->v',increments,future_queries)
    fixed_q = torch.randn(dk)
    fixed_reads = memories@fixed_q
    multiplicities = torch.arange(count,0,-1)
    records['accumulated_read_and_query_motion'] = {
        'product_difference_error': error(reads-previous_read,increment_read+moving_query),
        'sum_order_error': error(reads.sum(0),swapped),
        'constant_query_triangular_weight_error': error(fixed_reads.sum(0),
              (multiplicities[:,None]*(increments@fixed_q)).sum(0)),
        'nonzero_query_motion_term_norm': float(moving_query.norm()),
        'formula': 'M_r q_r-M_previous q_previous = G_r q_r+M_previous(q_r-q_previous). Sum_r M_r q_r = Sum_u G_u Sum_{r>=u} q_r.',
        'note': 'The sum of read drives is exact. It is not the final hidden state: input, bilinear FFN and Phi also act. Two integrations are not by themselves an STDP implementation error.'}

    center, amplitude = torch.randn(dv,dk),torch.randn(dv,dk)
    query_mean, query_delta = torch.randn(dk),torch.randn(dk)
    signs = torch.tensor([-1.,1.])
    cycle_q = query_mean+signs[:,None]*query_delta
    cycle_m = center+signs[:,None,None]*amplitude/2
    cycle_g = signs[:,None,None]*amplitude
    cycle_y = torch.einsum('tvk,tk->tv',cycle_m,cycle_q)
    cycle_gy = torch.einsum('tvk,tk->tv',cycle_g,cycle_q)
    alpha = .37
    mixed = cycle_g+alpha*(center-signs[:,None,None]*amplitude/2)
    mixed_y = torch.einsum('tvk,tk->tv',mixed,cycle_q)
    records['period_two_read_rectification'] = {
        'memory_mean_read_error': error(cycle_y.mean(0),center@query_mean+amplitude@query_delta/2),
        'memory_alternating_read_error': error((cycle_y[1]-cycle_y[0])/2,
                                              center@query_delta+amplitude@query_mean/2),
        'current_mean_read_error': error(cycle_gy.mean(0),amplitude@query_delta),
        'mixed_mean_read_error': error(mixed_y.mean(0),alpha*(center@query_mean)+(1-alpha/2)*(amplitude@query_delta)),
        'net_write_norm': float(cycle_g.sum(0).norm()),
        'nonzero_mean_current_read_norm': float(cycle_gy.mean(0).norm()),
        'formula': 'For G_r=s_r D, M_r=C+s_r D/2, q_r=q_mean+s_r q_delta: mean(Mq)=C q_mean+D q_delta/2; mean(Gq)=D q_delta.',
        'note': 'Equilibrated zero-mean period-two writes; C depends on the earlier path. Query oscillations can rectify a zero-mean write into a nonzero mean drive. This is not proof of a useful or harmful cycle.'}

    for name, row in records.items():
        for key, val in row.items():
            if key.endswith('_error'):
                assert val < 2e-11, (name, key, val)
    assert records['symbolic_filter']['imaginary_filter_identity']
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
