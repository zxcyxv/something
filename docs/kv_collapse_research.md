# KV-STDP collapse investigation — 2026-10-03

Status at publication, 2026-10-03: all reported long training runs have stopped
or completed. The original M, current B-only, G-only, historical-key-trace,
and normalization-free v1.7 runs completed6000. B+G and quarter-read completed
the requested3000-step observation. The v1.7 W-only and B/M interpolation runs
were stopped at2385 and2875 respectively. Normalized KV and pure G/M
interpolation have not been trained. No long training run is currently active.

Read the [Korean research summary and evidence index](research/2026-10-03/README.md)
for the current conclusions, consistent notation, comparison tables, and
archived scalar logs. The remaining document preserves the chronological
investigation; statements about a next experiment refer to that stage of work.
The primary outcome is training loss/accuracy and deterioration after early
improvement. Test accuracy is secondary by user request. The cause of the
original training reversal remains unconfirmed; frozen G-only computation is
substantially stronger than its online training metrics suggest.

## Updated comparison requested by the user

The primary experiment now removes the eight no-grad prefix blocks. Use
`configs/kv_stdp_ng0_research.json`, eight gradient-bearing blocks per optimizer
step, and compare accuracy at the same optimizer steps against the original
v1.7 log. The earlier `baseline` run used 8+8 blocks and was stopped with its
checkpoint preserved when the comparison protocol was corrected. It must not
be reported as the requested no-grad-free baseline.

The user also clarified that persistent lack of improvement matters, even if
the temporary decline near 73% eventually recovers. Structural fixes remain
experimental until the corrected baseline and same-step comparison are done.

Removing the no-grad prefix is a **depth-matching control, not a proposed
stability fix**. The smaller recurrent computation budget can reduce accuracy.
An accuracy difference between 8+8 and 0+8 blocks is not evidence of instability
or of a failed fix. The primary comparison is KV versus v1.7 at 0+8 blocks,
16 segments, and matching optimizer steps.

The original v1.7 log and trainer were recovered from LinearTuring commit
`f5793d22344e601c6e3a25d12c519e82ad2b7632`, paths
`2026-09-09/train_v17.log` and `2026-09-09/analysis/train_v17.py`.
The NPZ is byte-identical (SHA256
`b209d23895b87a64942a5e525af7a0c826a41cfff45cbe13b90041aaeaf52a3e`).
Forty generated batches across dataset iterations0,1,3 match exactly.
The manifest, data parity record, and source files are under the experiment's
`reference/` directory.

| v1.7 optimizer step | Training cell accuracy | EMA test cell accuracy |
|---:|---:|---:|
| 2000 | 77.90% | — |
| 4000 | 78.67% | — |
| 6000 | 81.66% | — |
| 1953 | — | 48.05% |
| 3906 | — | 62.65% |
| 5859 | — | 66.80% |

The comparison script aligns training accuracy to the recorded `halt step`
(for example the v1.7 step1000 line reports accuracy from step992). Loss is
compared at the actual logged step; testing is compared at evaluation steps.
Raw training and EMA testing must not be mixed.

## Completed baseline and next controlled experiments

All runs in this section use 8 gradient-bearing blocks per segment, 16 segments,
and no no-grad prefix. The 8+8 run is retained only as supplementary evidence.

| Optimizer step | KV baseline | Historical v1.7 | Historical R1B8 bilinear |
|---:|---:|---:|---:|
| 2000 | 58.90% | 77.90% | 61.87% |
| 4000 | 59.54% | 78.67% | 74.41% |
| 6000 | 60.51% | 81.66% | 80.84% |

These are terminal training-batch cell accuracies at the listed optimizer steps.
KV's terminal accuracy averaged over the last 500 steps is 61.99%, versus
59.63% in steps1501–2000 and 54.75% in steps2501–3000. It declines and partially
recovers, but substantially underperforms the two references at6000. The EMA
test result at5859 is52.12% (0/2048 exact), versus v1.7's66.80% (141 exact) and
R1B8's66.13% (48 exact). Final KV evaluation at6000 is51.23% (0 exact).

R1B8 remains an essential **mechanism reference**: it learned using signed
instantaneous attention and a bilinear layer, without persistent KV memory.
Its historical trainer, initialization and augmentation pipeline differ; its
block count is matched, but identical training batches have not been established.
The successful log, config, model source and harness patch are archived with
hashes in `reference/r1b8_manifest.json`. `r1b8_comparison.png` includes both
historical references and all new runs without no-grad blocks.

Next, as explicitly requested, import the exact recovered v1.7 trainer and
replace `_unit(x,y)` with `(x,y)`. This removes Q/K address normalization from
both the instantaneous read kernel and the trace write kernel. It preserves
value cosine agreement, QR projection, trace dynamics, Phi, token sum and the
training equations. The runner records every step and saves original checkpoints
plus separate carry/RNG sidecars. A normalized KV run was planned next, but is
now deferred under the user's latest instruction.

For KV normalization, normalize current K **before updating its eligibility
trace**, and normalize current Q for reading. Re-normalizing the entire K trace
would change the exponential temporal weights and would test another rule.
The additive memory and signed pairwise temporal difference are retained.

## Question and supplied observations

The October 2 model initially reduced training loss, then loss rose and did not
recover. The user observed the same failure with plain KV outer-product writes.
Changing memory retention from 1 to 0.95 prevented substantial initial learning.
These are user observations, not independently reproduced results in this study.
The primary reproduction retains signed temporal writes and additive memory.

## Reproduction protocol

- Source: `lt/train.py` at commit `70a8dba`.
- Original top-level CFG: d832, H8, batch128, 16 segments, each with 8 no-grad
  blocks and 8 grad blocks; BF16 projections, FP32 memory/trace calculations.
- Same repository NPZ, seed0, augmentation1000, AdamATan2, lr1e-4 with 2000-step
  warmup, weight decay1, EMA0.999. Stop at optimizer step6000.
- Operational overrides: local data/output paths, no time deadline, checkpoint
  every500 steps, per-step JSON metrics, diagnostic sampling. Compilation stays on.
- Original selftest fails bitwise resume equality for `q_proj.weight` on CPU:
  maximum difference7.45e-9, both tensors finite. It is not a nonfinite-gradient
  failure. This is recorded; baseline equations are not changed to pass it.
- Runner: `python -m lt.research_kv_collapse --out runs/kv_collapse_20261003/baseline`.
- Each run stores the trainer snapshot/hash, actual config, per-step metrics,
  diagnostic measurements, and full resumable checkpoints.

Baseline diagnostic sampling records segment1 every64 steps. Subsequent runs
sample complete episodes to avoid confounding depth with training progress.

## Successful reference model

Repository: `zxcyxv/LinearTuring`, inspected commit `1591470`.
Actual successful runs are `R1B8_bilin_ok` and resumed `R1B8_bilin_r2`.
The earlier `R1B8_bilin` ran with its bilinear flag inactive and is not the control.

Authoritative files:

- `sudoku_runs/2026-08-26/code/lt.py` and root `model1.py`.
- `sudoku_runs/2026-08-26/R1B8_bilin_ok.log`.
- `sudoku_runs/2026-08-27/checkpoints/R1B8_bilin_r2_config.yaml`.
- `sudoku_runs/2026-08-26/REPORT.md`, sections7.2 and7.5.

Reference log: step400 loss1.9683/accuracy0.4694; step2000
loss1.1671/accuracy0.6187; step4000 loss0.7475/accuracy0.7441;
step6000 loss0.5960/accuracy0.8084. These are individual logged training batches,
not moving averages, and the reference has 8 grad blocks without the new model's
additional 8 no-grad blocks. They are historical results, not a paired rerun.

Important structural differences:

| Property | Successful R1B8 | Current KV-STDP |
|---|---|---|
| Address amplitudes | Unit L2 norm; signed bounded similarities | Raw independent Q/K projections |
| Connection state | Recomputed from current hidden every block | Persistent channel matrix M |
| Value transport | Shared W and transpose W | Independent V and output projections |
| FFN input | Previous block's dissipated hidden | Hidden plus current M read, before dissipation |
| Dissipation | Strang pre/post Phi; learned gamma | One post-FFN Phi; fixed gamma=1/d |
| Position | Learned phase and distance decay | Learned 2D rotation, no distance decay |

The table above describes the earlier R1B8 code. v1.7 already uses a post-FFN
fixed Phi with gamma=1/d, like KV-STDP. Therefore the earlier R1B8 ordering
difference is not a valid explanation by itself for a gap against v1.7.
Normalized addresses, memory representation/update, and the instantaneous
attention path remain architectural differences to investigate.

## Hypotheses to distinguish

1. **Growing effective feedback gain:** unnormalized K/V writes and Q read
   produce amplitude-dependent feedback; the quadratic FFN receives amplified
   read outputs. A bounded hidden after Phi alone does not bound M or guarantee
   bounded derivatives before Phi.
2. **Loss of a stable recurrent regime:** a learned change could move the coupled
   (h,M,eK,eV) system from a fixed point to oscillations or strong sensitivity.
   Temporal writes vanish at a fixed point but can accumulate on an orbit.
3. **Stale state under parameter updates:** M is carried across optimizer steps
   while Q/K/V change; old Hebbian control recomputes its connectivity each block.
4. **Projection precision:** BF16 rounds activity before FP32 subtraction;
   quantify this rather than assuming that cancellation itself causes collapse.

The first frozen checkpoint probe (step500, fixed test puzzles) shows original
memory becoming nearly constant by block16 and remaining so through block256.
Its maximum singular value is approximately30 in that small test batch.
FP32 K/V projections produce virtually the same terminal loss and dynamics at
this checkpoint. This contradicts simple unconditional linear growth of M and
does not yet settle what happens at the collapse transition.

Frozen-weight interventions change inference distributions; their loss alone is
not a test of whether a modification can train successfully. Every proposed fix
must be trained from the same initialization/data protocol to step6000.

## Block-level evidence in the completed KV baseline

A frozen step3000 checkpoint on four saved training puzzles, reset to fresh
state, reaches an approximate two-cycle: at block32, one-step hidden RMS change
is0.7297 while two-step change is0.00338. Prediction flips are28.40% between
successive blocks, versus0.62% between blocks two apart. FFN input RMS is4.67
and FFN delta RMS14.74 on the even phase. The memory norm is approximately
stationary, despite a nonzero alternating memory increment. This is evidence of
a coupled oscillation in this small probe, not unbounded growth or proof that
oscillation causes the training deficit. With the same checkpoint, a read-operator
bound reduces flips to9.57%, and an extra pre-FFN Phi removes most flips but also
reduces accuracy. Stabilizing a frozen trajectory is not by itself a training fix.

The address-normalization hypothesis comes from a concrete difference with
R1B8: scaling an entire raw Q/K/V activity history by c scales M by c² and its
read by c³. With unit Q/K activities and raw V, the corresponding read scales
approximately by c (up to epsilon). This removes one amplitude feedback path;
it does not bound accumulated M uniformly over arbitrary trajectories. Both
implementations still require an empirical training test.

## v1.7 address-normalization ablation: completed results

The first2000 optimizer steps of `v17_no_address_norm` already challenge the
claim that missing Q/K normalization alone explains the KV deficit. Training
cell accuracy at2000 is90.68%, with8/128 exact. The matched first EMA evaluation
at1953 is57.79%, with3/2048 exact, compared with48.05% and0 exact in the original
v1.7 log. These early results do not establish a lasting improvement; the later
matched evaluation below changes that interpretation.

A frozen raw step1500 v1.7 checkpoint on the same first16 test puzzles has
pre-FFN RMS309.5 and FFN delta RMS8473 at block32, yet only1.23% of predictions
change from the previous block. Thus a large pre-Phi amplitude by itself is
not a sufficient explanation for the KV two-cycle or its training deficit.

The matched second EMA evaluation at3906 is64.89%, with62/2048 exact, compared
with original v1.7's62.65% and26 exact. Training accuracy nevertheless declines:
the last512-step terminal mean is89.50% at2000 and approximately83.83% around5000.
Early optimization speed, later training behavior and EMA generalization must
therefore be assessed separately.

| EMA test optimizer step | Original v1.7 cell accuracy | Exact /2048 | Without address norm cell accuracy | Exact /2048 |
|---:|---:|---:|---:|---:|
| 1953 | 48.05% | 0 | 57.79% | 3 |
| 3906 | 62.65% | 26 | 64.89% | 62 |
| 5859 | 66.80% | 141 | 66.26% | 83 |

At6000, the ablation's training cell accuracy is82.70% (6/128 exact) and its
EMA test accuracy is66.48% (88/2048 exact). The original log has training81.66%
at6000 but no test evaluation at that exact step, so5859 is the final directly
matched test comparison. At5859, removing normalization is0.54 percentage points
lower in cell accuracy and58 puzzles lower in exact matches. It speeds early
training but does not demonstrate a final generalization advantage at this
horizon. In contrast, the KV baseline's matched test result is52.12%,0 exact.
Missing Q/K normalization alone therefore does not account for that large deficit.

That run stopped at6000; normalized KV training was deferred. The
activity-normalization implementation has passing invariant
tests, including the temporal pair-sum identity and finite gradients at zero,
but its training effect remains unknown.

The step6000 KV probe was repeated with K/V projections computed in FP32
instead of BF16. On the same16 test puzzles at block32, cell accuracy stays
55.40% and consecutive prediction flips stay47.92%; hidden change RMS is1.2068
versus1.2067. This intervention does not remove the observed two-cycle. It does
not test whether an entire FP32 training run would follow another trajectory.

The prepared (not launched) KV normalization control retains the original **token mean**, as well as
raw V and additive M. Only Q/K activity normalization changes. Unit addresses
also reduce initial read gain, so any result must acknowledge that consequence;
do not silently replace the mean by a sum and call it a normalization-only test.

### The two memory organizations are not temporally equivalent

For one fixed set of activities, `(Q K.T) V = Q (K.T V)` by associativity.
After accumulation over recurrent time, even the simplified rules differ:

```
token-link state:  (sum_s Q_s K_s.T) V_t
KV state:         Q_t (sum_s K_s.T V_s)
```

The first transports current values along previously formed links; the second
retrieves past values using current queries. The actual v1.7 also has value
cosine agreement, a moving average of token links, and an instantaneous read
path; the equations above illustrate the temporal distinction rather than
claiming to reproduce those full update rules. Replacing current-value message
passing with past-value retrieval is a further mechanism hypothesis, not yet a
causal result. A dynamic token-link matrix is compatible with a broad fast-weight
interpretation, while its N-by-N storage differs from the sequence-length
complexity of standard KV linear attention. Relevant primary reference:
[Schlag, Irie & Schmidhuber, ICML2021](https://proceedings.mlr.press/v139/schlag21a.html).

### Token time versus recurrent depth

The ordinary linear-attention recurrence accumulates over token index i within
a layer or recurrent block r: `S[r,i] = S[r,i-1] + k[r,i] v[r,i].T`, starting
with `S[r,0] = 0`. Repeating attention on updated hidden states does not require
retaining the preceding block's S. Carrying KV over r is an additional memory
design, rather than a consequence of the token-wise linear-attention identity.
See [Katharopoulos et al., ICML2020](https://proceedings.mlr.press/v119/katharopoulos20a.html).

In v1.7 the actual read is `[(1-lambda) A(h_r) + lambda W_r] V(h_r)`.
Even a hypothetical W-only read would still transport current V; its
instantaneous/remembered-link interpolation is a separate architectural choice
from storing past V in channel-space memory. This distinction is now the main
mechanistic question to reconsider; it has not yet been isolated by retraining.

## Artifacts

`runs/kv_collapse_20261003/` contains experiment configs, metrics, checkpoints,
frozen-state probes, summary JSON and `comparison.png`. This directory is ignored
by git under the existing repository policy; the runner/probe/analysis source and
this report are versionable. Full checkpoints contain optimizer, carry and RNG
states to allow later examination or exact resume.

## Requested control: v1.7 without normalization, W-only read

Run directory: `runs/kv_collapse_20261003/v17_no_norm_memory_read`.
Command:

```
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m lt.research_v17_normalization \
  --read-mode memory --out runs/kv_collapse_20261003/v17_no_norm_memory_read
```

This starts from the same seed0 initialization as `v17_no_address_norm`, with
the same input stream, optimizer, 8 gradient blocks,16 segments, and6000-step
limit. It changes exactly one additional model equation:

```
a_eff = (1 - lam) * a + lam * w    # previous run
a_eff = w                        # new run; coefficient of W is exactly1
```

The current value read remains `W_r V(h_r)`, including the tied transpose value
projection. Address normalization stays disabled for instantaneous and trace
addresses. The W target, value cosine agreement, trace, EMA update of W, Phi,
and all parameter initialization are preserved. Unused interpolation parameters
remain allocated to preserve the initialization draw order; they receive no
read-path gradients. The effective source and exact one-line diff are saved
alongside the run.

Two invariant tests pass: identical parameter initialization, and identical W /
trace updates for the same input with both an existing memory and a fresh lane.
The W-only output is also checked exactly against the explicit `W V` read,
with gradients through its value/write path and none through the removed
interpolation/instantaneous-read parameters.

The user shortened this control to approximately2000 steps, requesting a switch
to KV interpolation if it clearly underperformed. At2000 the last256-step
terminal-batch mean accuracy is56.64%, versus90.05% for the mixed read. The exact
step2000 batch is58.19% versus90.68%. EMA testing at1953 is52.51%,0/2048 exact,
versus57.79%,3 exact. It was stopped with a saved step2385 checkpoint; the
comparison uses its intact step2000 checkpoint and metrics, not an assumed
2000 shutdown. `stopped.json` records this distinction.

## Requested KV interpolation control

Run: `kv_interpolated_read_ng0`; variant: `interpolated_read` in
`lt/kv_stability.py`. The user's final preference is **one scalar per head**,
not one coefficient per feature. There are eight additional trainable scalars,
sigmoid-initialized to0.25, allocated after all original initialization draws.

Precise scope: its instantaneous branch is **real current KV**; only the
accumulated branch uses the original complex trace/current STDP product.
It must not be described as interpolating the current complex operator with
its accumulated history. The subsequently completed `current_only` run removed
that accumulated branch and all traces, yielding a real-current-KV control.
The user had intended to test complex-current-KV first; both controls are useful,
but their order and interpretation must be recorded accurately.

Using the code's matrix orientation, with spatially rotated Q and K:

```
B_r = mean_tokens(V_r.T @ RoPE(K_r))          # current KV; not stored
M_r = M_(r-1) + STDP_write_r                 # unchanged original update
read_r = (1-lambda_h) * RoPE(Q_r) @ B_r.T
         + lambda_h * RoPE(Q_r) @ M_r.T
```

In the expression for B, `mean_tokens` means divide the token contraction by
T=81. Both read paths use the original token mean. Q/K remain unnormalized;
V, eligibility traces, additive memory, no-grad0,8 blocks,16 segments, data,
seed, and optimizer remain unchanged. The new scalars use the existing
no-weight-decay rule for `lam_raw` parameters.

Ten KV invariant tests pass, including preserved original initialization,
bit-identical writes/traces for fixed activities, equivalence of the fresh KV
read to an independently computed token-space `(Q K.T) V / T`, scalar-per-head
shape, interpolation endpoints and nonzero finite gate gradients. This validates
the intended intervention, not its training outcome.

The user stopped this run at2875, with the final checkpoint and all metrics
preserved. It did not complete6000 and cannot establish later stability.
Recent256-step means over terminal training batches were:

| Window ending | Training loss | Cell accuracy |
|---:|---:|---:|
| 2000 | 0.598897 | 81.60% |
| 2304 | 0.477663 | 87.45% |
| 2560 | 0.489192 | 86.17% |
| 2864 | 0.455709 | 88.50% |

There was a temporary dip, followed by improvement before interruption; no
persistent reversal had been established within the observed interval.

## Requested current-only attention control

Run: `kv_current_only_ng0`; variant: `current_only` in `lt/kv_stability.py`.
This is an independent seed0 restart with the same data,8 gradient blocks,
16 segments, raw Q/K, optimizer, and6000-step limit as the preceding KV runs.

```
B_r = V_r.T @ RoPE(K_r) / T
read_r = RoPE(Q_r) @ B_r.T = (RoPE(Q_r) @ RoPE(K_r).T) @ V_r / T
```

There is no recurrent matrix accumulation, eligibility trace update, temporal
subtraction, or interpolation. The coefficient of the current read is1,
not the previous mixed path's initial0.75. Hidden-state recurrence, input
injection, independent Q/K/V projections, learned spatial rotation, output
projection, bilinear layer, and Phi retain the original implementation.

The inherited carry structure contains the current B solely for compatibility
and diagnostics; the next block ignores it completely. The unused trace
coefficients remain allocated to preserve all original random initialization
draws and receive no gradients. The legacy startup banner prints the original
KV-STDP configuration; the variant snapshot and protocol describe the actual
current-only computation.

Twelve invariant tests pass. The two new checks verify exact initialization
parity, equivalence to independent token-space attention in both outputs and
Q/K/V gradients, no dependence on past matrices or traces (including NaN-filled
states), no trace coefficient gradients, and valid projection gradients through
the full recurrent model. The config parity check permits only output path,
model identifier, and research variant to differ from the original KV control.

The requested primary comparison is the training trajectory, particularly any
sustained rise in loss and decline in accuracy after early improvement.
`training_stability.png` plots matched256-step terminal-batch means; EMA testing
remains recorded by the shared trainer but is not the decision criterion.

Completed6000. The final terminal batch has loss0.370395, accuracy91.29%; the
last256-step terminal means are loss0.334031 and accuracy92.75%. The corresponding
means at2992 were0.320600 and94.24%. Thus this run retains high training accuracy
but shows a modest late decline of1.49 percentage points; it does not establish
perfectly monotonic training. All6000 steps are present, saved model weights are
finite, and saved K/V traces are absent as intended. At2000 its current training
inputs, labels and puzzle identifiers match the original and interpolated KV
runs exactly. `completed_run_checks.json` and `paired_batch_check_2000.json`
record these checks.

## Complex K/V retained, current attention only

Run: `kv_complex_current_only_ng0`; variant: `complex_current_only`.
The original code defines the complex activities using past eligibility traces
and current projections, rather than an independent real/imaginary projection:

```
Kc_r = RoPE(eK_(r-1)) + i RoPE(K_r)
Vc_r = eV_(r-1) + i V_r
G_r = Im(Vc_r.T @ conj(Kc_r)) / T
    = (V_r.T @ RoPE(eK_(r-1)) - eV_(r-1).T @ RoPE(K_r)) / T
read_r = RoPE(Q_r) @ G_r.T
```

The only change from the original KV-STDP equations is
`M_r = M_(r-1) + G_r` becoming `M_r = G_r`. Both eligibility traces, their learned
decay parameters, the original update order and fresh-lane reset, raw Q/K/V,
RoPE, hidden recurrence, optimizer, initialization, data and0+8 depth remain
unchanged. There is no interpolation coefficient. The returned carry matrix is
the current G only; its previous value is ignored. Unlike the real current-only
control, this model retains a history of K/V activity through the traces.

Fourteen invariant tests pass. New checks compare this real-arithmetic
implementation against an actual complex-tensor outer product, verify exact
initialization and trace-update parity, establish independence from old M even
when filled with NaNs, and verify finite nonzero gradients through K/V histories
and the learned trace decay. Configuration parity permits only the run path,
model identifier, and variant name to differ from the original control.

## Implementation audit against temporal pair-STDP (October 3)

The audit treats channel values as neuron activities, recurrent blocks as time,
and tokens as independent samples whose writes are averaged. Its purpose is to
check the implementation of the STDP rule, not to assume that STDP is invalid.
The original trainer is byte-identical to the baseline run snapshots. No model
equation, checkpoint, or training process was changed during this audit.

`lt/test_kv_stdp_reference.py` independently sums all ordered event pairs, with
explicit position/channel rotation rather than the production trace recursion.
For r>s, the actual heterogeneous window adds

```
+ (1-lambda_j) * lambda_j**(r-s-1) * v[r,p,i] * rotated_k[s,p,j]
- (1-lambda_i) * lambda_i**(r-s-1) * v[s,p,i] * rotated_k[r,p,j]
```

The sum matches every prefix of M and its Q read in FP64, including gradients
of Q, K/V histories, lambda and theta. Isolated pre/post pulses verify sign,
lag and L(0)=0; separate tokens do not accidentally form write pairs. Mixed-lane
reset, input retention, segment continuity, boundary detach and checkpoint
recomputation checks also pass. The eight new checks plus the sixteen existing
variant checks pass (24 total). No sign, transpose, duplicate-pair accumulation,
state-reset or missing-gradient error was identified.

`lt/audit_kv_stdp_runtime.py` uses the original step6000 raw weights and four
saved training puzzles, forcing continuation of their saved carry for one
eight-block segment. This is a local execution comparison, not another training
run. BF16 eager with/without activation checkpointing gives identical outputs
and parameter gradients. Compiled BF16 versus eager BF16 has gradient relative
L2 difference 1.06%, cosine 0.999944, and no missing/nonfinite/zero parameter
gradients. BF16 versus FP32 gradient difference is 1.07%. On identical captured
activity inputs, FP32 outer-product subtraction versus FP64 has relative L2
error at most 3.43e-7. TF32 is disabled. These checks do not establish equality
of complete training trajectories, but provide no evidence for a dropped
gradient or catastrophic subtraction error in this checkpoint.

Two distinctions from a simplified fixed scalar-window derivation remain:

1. Per-channel decay makes the exact window L_ij(dt), rather than a shared odd
   L(dt). Positive lags use the presynaptic channel's decay and negative lags
   the postsynaptic channel's decay. This is a valid heterogeneous STDP rule;
   calling it one shared antisymmetric window is inaccurate. At step6000 the
   decays range from0.09587 to0.10479 (time constants0.4265–0.4433 blocks).
   A constant-activity startup probe accumulates a boundary term of0.276% of
   the same-input current-KV norm; shared head decay removes it up to numerical
   error. This difference has not been established as the training failure.
2. K traces are stored before RoPE and rotated using the current theta when
   written. At fixed theta this exactly equals tracing rotated K. When theta
   changes between optimizer steps, it differs from using the historical
   rotated activity at the time of each event. A two-event synthetic example
   demonstrates the difference. Reconstructing the actual final optimizer
   theta update at step6000 gives a local write difference of0.0453% and a read
   difference of0.0238%. This is not a replay of all historical parameter
   updates and is not evidence that this small local difference causes collapse.
   If historical rotated activity is intended, the consistent implementation
   stores the rotated key trace and does not rotate that trace again at write
   time; that requires explicit checkpoint/trace-coordinate compatibility.

The same-input norm of G is0.439–0.946 times that of current KV in this saved
continuation. Therefore the claim that subtraction simply destroys the signal
through numerical cancellation is not supported here. Existing frozen probes
still show an approximately two-block cycle in the coupled hidden/M dynamics;
that observation does not identify a code bug or prove a cause of the accuracy
decline. The cause remains unconfirmed.

Reproduce the audit with:

```
python -m unittest lt.test_kv_stdp_reference lt.test_kv_stability -v
python -m lt.audit_kv_stdp_runtime runs/kv_collapse_20261003/baseline_ng0/step_6000.pt --out runs/kv_collapse_20261003/audit/runtime_baseline6000.json
python -m lt.audit_kv_temporal_basis runs/kv_collapse_20261003/baseline_ng0/step_6000.pt --out runs/kv_collapse_20261003/audit/temporal_basis_baseline6000.json
```

Machine-readable results and the test transcript are in
`runs/kv_collapse_20261003/audit/`; the compact record is `summary.json`.

## One-hour follow-up: coordinate, feedback and credit-assignment controls

The follow-up keeps the STDP rule as the object to implement and investigate.
It does not infer that STDP is invalid from poor training. Its new controls are
isolated in `lt/kv_stability.py`; the original trainer remains unchanged.

Two same-initialization, no-grad0 training controls were started:

- `kv_historical_key_trace_ng0`, target6000: store each K after its event-time
  RoPE. G, additive M, Q read, V trace, timing coefficients and parameter count
  remain the same. A new independent pair-sum check varies theta between events
  and verifies the historical-coordinate interpretation. At fixed theta, the
  uninterrupted forward and backward match the original coordinate form.
- `kv_read_gain_quarter_ng0`, target3000: keep the original raw-key traces and
  every STDP write, multiplying only the memory read by0.25. No current KV term,
  interpolation, memory decay or new parameter is introduced. This is a gain
  diagnostic, not a claimed theoretically required coefficient.

The data, seed, batch size, depth, optimizer and learning-rate schedule match
`baseline_ng0`. The shorter3000-step target does not alter learning rate because
`lr_min_ratio=1` after the same2000-step warmup. Only the final checkpoint is
retained automatically. Configuration comparisons and28 reference/variant
tests passed. Current/final measurements are in `audit/training_comparison.json`
and `.csv`, with a plot in `audit/training_comparison.png`. Accuracy comparisons
use terminal training segments within the same256-step windows, not EMA tests.

### A small forward discrepancy can change a truncation-boundary gradient

`lt/probe_kv_trace_boundary_gradient.py` uses the same physical incoming state
at fixed step6000 weights. With fresh state, original and rotated-trace logits
and theta gradients agree to FP32 error. With detached incoming state, logits
still agree (relative L2 error2.08e-7), but the theta gradient differs by52.24%
(cosine0.90350). Every other parameter-gradient difference is below1.5e-6
relative L2 in this two-puzzle probe.

This difference follows the coordinate-change chain rule:

```
rotated_trace = R(theta) raw_trace
original_theta_gradient - historical_theta_gradient
    = (d loss / d rotated_trace) * d[R(theta) raw_trace]/d theta
```

The measured gradient difference matches that boundary term with relative
residual1.95e-6. Holding raw trace constant and holding event-time rotated trace
constant define different truncated derivatives. Thus the earlier0.024% local
forward effect alone was insufficient to dismiss a training effect. A failed
training control would still not make the two derivatives identical.

### Frozen feedback intervention: compare the supervised phase

`lt/probe_kv_feedback.py` starts nine interventions from each of two consecutive
states after the same128-block warm-up on eight saved training puzzles. It
continues for48 blocks at frozen original step6000 weights. The original
trajectory alternates strongly; halving/quartering its read reduces that
alternation while the memory RMS remains similar. Stopping memory writes alone
is phase-dependent and does not generally remove the cycle.

Crucially, training observes every eighth block, which is always even. Averaging
odd and even phases would misleadingly favor removing the cycle:

| Continuation from block128 | Even-block accuracy | Even-block loss | Mean one-block hidden change | Mean two-block hidden change |
|---|---:|---:|---:|---:|
| Original |56.81%|0.95678|1.22799|0.00402|
| Read gain0.25 |55.59%|1.16573|0.00183|0.00219|

Accuracy/loss use the last eight even observations; hidden-change columns use
the last16 observations. The quarter-read intervention suppresses the cycle
but does not improve the matched supervised-phase loss or accuracy. This is
why actual training, not suppression of the frozen cycle, is the criterion.
All branches and both parities are in `audit/feedback_baseline6000.json`.

### Long gradients predominantly pass through M in the STDP model

`lt/probe_kv_credit_assignment.py` holds weights fixed, runs128 blocks, and
differentiates the same final loss with different detach boundaries. Forward
logits are exactly identical across modes. A one-puzzle pilot was repeated on
four common puzzles for original STDP and the successful current-KV control.

| Four-puzzle probe: cosine to full128-block gradient | STDP | Current KV |
|---|---:|---:|
| Detach all states every8 |0.53383|0.16261|
| Detach hidden only; retain memory/trace graph |0.99915|0.16261|
| Detach memory/traces only; retain hidden graph |0.53451|1.00000|

The missing long dependency in this STDP checkpoint predominantly passes
through M/traces. This suggests a controlled study of memory credit assignment.
It does not establish a bug or causation: the successful current-KV model also
has a large full-versus-truncated gradient difference, and the real trainer
supervises and updates parameters at every segment, unlike this frozen
final-loss-only diagnostic. Retaining a graph across in-place optimizer updates
would not be a correct fix; a training comparison must specify its optimizer
schedule and amount of computation explicitly.

Full gradient records are in `audit/credit_assignment6000_batch4.json`. Research
time, live/completed runs and the final audit are tracked in
`runs/kv_collapse_20261003/free_research_1h.json`.

### Read gain is equivalent to STDP write amplitude here

For zero-initialized additive memory and a constant c, define N=c*M:

```
M_new = M_old + G
N_new = N_old + c*G
Q @ N_new.T = c * (Q @ M_new.T)
```

Thus the quarter-read control is also a control of STDP window amplitude /
plasticity learning rate, with coefficient0.25 instead of1. This equivalence
holds through the hidden recurrence, since the two reads generate the same
next activity, and through truncated differentiation because c is independent
of the learned parameters. An eight-block FP64 comparison verified hidden
states, rescaled memory, traces and all parameter gradients; results are in
`audit/read_write_gain_equivalence.json`. It introduces no instantaneous KV
term. The STDP derivation itself does not fix this amplitude to1.

### Completed follow-up and limits

Both runs reached their requested targets: historical-key6000 and quarter-read
3000. Training logs are contiguous, losses/weights/carry states are finite, and
each run retains only its final checkpoint. The last observed terminal segment
for the3000-step run is2992. Matched256-step training windows are:

| Control | Endpoint | Loss | Cell accuracy | Original at same endpoint |
|---|---:|---:|---:|---:|
| Historical rotated-key trace |6000|0.79979|63.99%|0.84316 /61.84%|
| STDP read/write gain0.25 |2992|1.04397|56.80%|1.04130 /54.84%|

The historical-coordinate correction gives a modest final difference in this
single-seed comparison, but did not remove the intermediate reversal. The
quarter-gain run peaked at59.90% on the rolling window ending2384 and fell to
56.80%; its final loss is essentially unchanged from the original. Neither
experiment establishes the cause or a solution of the training decline.

The user challenged this experimental prioritization. The historical-coordinate
control tests a concrete interpretation mismatch, but its causal connection to
collapse was unproven. The quarter-gain control was generic stability tuning:
reducing a frozen cycle had not been shown to improve the supervised phase.
Further work should first identify a required derivation/implementation
mismatch, rather than treating any stabilizing intervention as a repair.

An optional final frozen probe of the quarter-gain checkpoint failed during
short-window reporting because its first row has no two-step predecessor. No
result from that probe is used above. The aggregation helper was corrected and
checked against existing records; the model experiment was not rerun. Training
was unaffected. See `audit/training_completion_checks.json` for completion
checks and `free_research_1h.json` for the final research record.

## Additional hour: derive the computation before proposing another repair

Started 2026-10-03 10:23:37 UTC at the user's request. This follow-up prioritizes
theoretical correspondence. It leaves `lt/train.py` and every saved checkpoint
unchanged. It does not start another long training run. A short, isolated
16-update diagnostic on copies of existing checkpoints is described separately.
All formulas below use plain text: `r,s` are recurrent block times, `p` is a
token/sample, and `i,j` are V/K channels. Channels, not tokens, are the neurons
in the plasticity analogy. Column-vector notation `M q` equals the code's
row-vector expression `q @ M.T`.

### What the pair-STDP derivation does establish

At fixed parameters, with one common decay per head and zero initial traces:

```
w(d) = (1-lambda) * lambda**(d-1), d >= 1
G_r = mean_p[v_rp eK_(r-1),p.T - eV_(r-1),p k_rp.T]
M_R = sum_{r<=R} G_r
    = mean_p sum_{s<r<=R} w(r-s) * [v_rp k_sp.T - v_sp k_rp.T]
```

K includes the fixed spatial rotation here. There is no same-time pair, and
each temporal pair is counted once. This is the rate/bilinear extension of a
balanced all-pairs STDP window. General pair STDP need not have equal positive
and negative amplitudes or time constants; equality is a modeling choice, not
a requirement of the term STDP. The standard two-trace construction is given
in [Neuronal Dynamics, section 19.2.2](https://neuronaldynamics.epfl.ch/online/Ch19.S2.html).
For reference, independent K/V programming and a separate Q read are already
present in [Schlag et al., equations 4, 10 and 11](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf).
These sources justify the respective plasticity and associative-read forms;
their combination does not prove a task-specific recurrent solver works.

Let P be the strict-past trace matrix on time indices, P_rs=w(r-s) for r>s.
With K and V stacked along time, each token contributes:

```
M = V.T @ A @ K,       A = P - P.T
```

A is skew in time indices. The actual M is NOT necessarily skew in channel
indices: K and V are different learned projections. An explicit common-hidden
example produces `M=diag(1,0)`, although the underlying hidden-space temporal
outer-product difference is skew. Therefore an argument that the actual M has
only imaginary eigenvalues, or that its read must be orthogonal to h, is invalid.

There is a stronger constructive capacity check for multiple heads. With three
or more heads, let K_head select hidden group head and V_head select the next
group cyclically. Arbitrary independent head matrices are then different
off-diagonal blocks of one skew hidden-history matrix. Every such skew matrix
can be produced by a finite signed eligibility-trace path. Thus sharing one
hidden state does not impose a general per-head skew-matrix capacity obstruction
on the actual eight-head parameterization. This is an existence construction,
not a claim that the trained recurrence reaches those paths.

The full complex construction has a precise but different temporal operator:

```
Kc = P @ K + i*K
Vc = P @ V + i*V
sum C = Vc.T @ conj(Kc)
      = V.T @ [P.T@P + I + i*(P-P.T)] @ K
sum Im(C) = V.T @ (P-P.T) @ K
```

Thus selecting Im gives this STDP rule exactly. Keeping the full complex
Hebbian product also keeps the trace-trace and simultaneous products. Deleting
E to retain B+iG defines another operator; it is not the same complex Gram
matrix. A three-time self-associative counterexample even loses positive
semidefiniteness after deleting E. This is a distinction between operators,
not a PSD requirement for the actual signed, independent-K/V architecture.

### The real Query read is valid; what it retrieves changes

G-only reading has an exact ordinary associative form:

```
extended_keys   = [past_K; current_K]
extended_values = [current_V; -past_V]
Q @ G.T = Q @ extended_keys.T @ extended_values
```

It retrieves the two directed temporal associations. No Query trace or complex
Query is mathematically necessary. A two-event construction with K active first
and V active second writes an ordinary KV association using pure STDP; a real
Q retrieves it exactly. Conversely, identical synchronous K,V activity at all
times gives zero write for a common window. No scalar gain can turn these two
different statistics into the same rule on every input history.

There are useful counterexamples to overly strong failure arguments:

- With unchanged K and lambda=0, `G q = V_now - V_before` for
  `q=K/||K||^2`. The read is a value increment, appropriate for a residual
  update interpretation. Current Query is sufficient in this case.
- Alternating K-only and V-only phases gives opposite G on alternate blocks.
  Alternating the Query sign makes both reads return the same KV lookup. Zero
  cycle-average writing therefore does NOT imply that G-only computation fails.
- At a fixed recurrent state, G tends to zero, but previously accumulated M
  need not. A G-only network can also retain answers in local hidden attractors
  or maintain changing hidden activity while its output decisions settle.

### Accumulated M is a path integral, not generally a current KV estimate

For common fixed lambda, the updated traces satisfy an exact identity:

```
G_r = [eV_r eK_(r-1).T - eV_(r-1) eK_r.T] / (1-lambda)

M_R = [eV_R eK_R.T
       - sum_r (eV_r+eV_(r-1)) (eK_r-eK_(r-1)).T] / (1-lambda)
```

The second line assumes zero initial traces. It separates a final filtered
outer product from a path-dependent term due to key changes. Different paths
with the same initial/final states and the same sum of current KV products can
produce opposite M. This is a property of the intended time-order statistic,
not a subtraction-precision error.

For slowly changing activities, the strict-past normalized EMA gives
`G approximately [V_dot K.T - V K_dot.T]/(1-lambda)` in block-time units.
This differential interpretation requires slow variation; it must not be
applied to the original model's near-period-two motion. The connection between
balanced timing windows and differential Hebbian learning is established under
explicit rate/correlation approximations in
[Xie and Seung, sections 3-4](https://www.cs.cmu.edu/Groups/NIPS/NIPS99/99papers-pub-on-web/Named/XieSeung.pdf).
Their sign-dependent result for their specified neural dynamics does not imply
that flipping the sign of our G would fix this independent-Q/K/V model.

For constant offsets `K_r=a+k_r`, `V_r=b+v_r`, the contribution involving the
offset is exactly an endpoint term:

```
D_R(x) = sum_{r=1..R} [lambda**(R-r)-lambda**(r-1)] * x_r
M(a+k,b+v)-M(k,v) = D_R(v) a.T - b D_R(k).T
```

This identity was checked on the production model's captured trajectories. It
does not say that a static puzzle cannot be represented: the changing hidden
trajectory can encode it. The initial production block provides a concrete
example. Because b_down is initially zero, its first read is zero; its second
write is exactly:

```
G2 = (1-lambda) * mean_p[
       (WV I_p) (R_p WK h0).T - (WV h0) (R_p WK I_p).T]
```

I is the scaled constant input injection. The first global communication
therefore records movement from the common initial reference h0 toward the
input. The real implementation has nonzero h0; the conditional zero-h0,
zero-b_down collinearity counterexample is not a bug in the actual initialization.

A further exact constraint follows directly from additive M:

```
M_(r+P)=M_r  requires  sum_{s=r+1..r+P} G_s=0.
```

An exactly repeated activity cycle with nonzero net STDP writing keeps moving
M, so it cannot be an exactly periodic full hidden/trace/M state. A common
balanced window gives zero net writing on an equilibrated two-state cycle,
but generic directed cycles of length three or more can have nonzero net write.
This is a necessary condition, not a proof that the network is attracted to a
two-cycle or that its two-cycle causes poor accuracy. Related STDP models can
support useful oscillatory memories; see
[Susman, Brenner and Barak](https://arxiv.org/html/1808.00756). Their particular
self-associative dynamics does not establish properties of our independent
Q/K/V model.

### Per-channel windows: a real difference, not the demonstrated solution

The actual learned pair-wise lambda produces L_ij(r-s), not one common odd
window. Positive lags depend on lambda_j, negative lags on lambda_i. For an
exact prescribed two-state cycle, write `K=Kmean+sign*Kdelta` and similarly V.
Let `a_j=(1-lambda_j)/(1+lambda_j)`. Its equilibrated cycle-average write is:

```
mean G_ij = (a_i-a_j) * Vdelta_i * Kdelta_j
```

Mean over tokens is implicit. Equal decay removes this term; different decays
can accumulate it on every cycle. The common-window frequency response is
`-2i*(1-lambda)*sin(w)/(1-2*lambda*cos(w)+lambda^2)`, so both zero frequency and
the period-two frequency pi have zero cycle-average weight. This statement
does not make instantaneous period-two G zero.

At the original step6000 checkpoint, four common saved puzzles and a frozen
128-block FP32 trajectory gave 99.990% of centered K/V tail energy in the
two-state component. The predicted heterogeneous mean-write RMS was 0.000464,
against measured 0.000509; the alternating-write formula had relative error
0.000524. Replaying the same activities with one mean lambda per head changed
final M by 11.74% relative to the common-window M norm. This effect can accumulate
despite a small spread of time constants.

However, actually replacing each pair decay by its head mean and rerunning all
128 identical saved puzzles from fresh state gave 56.55% cell accuracy, versus
56.53% unchanged, and zero exact matches in both. Enforcing a common odd window
did not restore computation at this checkpoint. No head-decay retraining was
launched based on this result. Heterogeneous pair STDP remains a valid broader
rule; it is not itself proof of an implementation error.

### Slow-weight updates are temporal events, but the simple causal claim fails

If hidden activity stays fixed while K/V projections change between segments,
STDP observes changing neural activities. For a simple constant-activity
example, incoming detached traces are `c0*k0,c0*v0`, while the current segment
uses `K=a*k0,V=b*v0`. Over B blocks:

```
DeltaM = c0*(1-lambda**B)/(1-lambda) * (b-a) * v0*k0.T
```

At a=b=1 the forward write is zero, but the fixed-incoming-trace derivative is
nonzero. If the entire constant-history trajectory is differentiated with one
shared projection, its write and that derivative are both zero. This is a
difference between objectives at a truncation boundary, not broken autograd.
An actual optimizer-induced change also gives the forward event in this formula.

The final AdamATan2 K/V updates were inverted from saved moments. Holding hidden,
input and current spatial rotation fixed verified the resulting write exactly.
The integrated eight-block stationary-activity event was 0.77% of current-KV
norm for accumulated M and 1.29% for G-only. These are single-update diagnostics,
not a reconstruction of all previous training updates or proof of a cause.

The stronger observation is the difference between the saved online-training
carry and a fresh fixed-parameter episode. All three step6000 checkpoints have
byte-identical saved inputs/labels/identifiers (128 puzzles). The same final raw
classifier is used in both columns, with the original BF16/FP32 precision:

| Model | Saved online carry: cell / exact | Fresh fixed-weight128 blocks: cell / exact | Fresh loss |
|---|---:|---:|---:|
| Accumulated STDP M |60.54% /0%|56.51% /0%|0.99633|
| G-only |81.33% /10.94%|90.10% /53.91%|0.35107|
| Current KV-only |91.28% /22.66%|93.13% /25.78%|0.35264|

The fixed G-only model's exact-match rate increases from 7.03% at block32 to
53.91% at block128. It can perform sustained recurrent computation. These are
saved training puzzles, not a generalization evaluation, and the fixed replay
is not the same metric as the earlier rolling training table.

To test whether ongoing updates necessarily disrupt that computation, each
checkpoint was copied in memory and the same saved batch was trained for one
fresh 16-segment episode using the original optimizer/harness. No checkpoint
was written. At the end of that episode:

| Model | Online16 updates: cell / exact | Fresh replay at resulting fixed weights: cell / exact |
|---|---:|---:|
| Accumulated M |66.68% /0%|63.09% /0%|
| G-only |97.49% /85.16%|96.86% /85.16%|
| Current KV |99.95% /99.22%|98.66% /89.84%|

This already-seen-batch fitting diagnostic is not a new training benchmark.
It contradicts the simple local claim that within-puzzle updates necessarily
destroy G-only computation. It does not rule out a harmful historical effect
earlier in training. The optimizer-event derivation and the training/frozen
trajectory difference must not be reported as an established explanation of
the original reversal.

### Reproduction and current conclusion

One experimental distinction remains essential. The existing
`kv_interpolated_read_ng0` used current real B with accumulated M; the implemented
`kv_current_plus_stdp_ng0` used B+G without accumulation. Neither is a pure
G/M interpolation. For additive memory, the previously requested latter control
would be:

```
M_r = M_(r-1) + G_r
read_operator = (1-alpha)*G_r + alpha*M_r
              = G_r + alpha*M_(r-1)
```

Alpha can be one scalar per head. This preserves the current STDP read while
changing the contribution from earlier writes. The quarter-read experiment was
instead `0.25*(G_r+M_(r-1))`; it changed both contributions together. Its result
does not establish the outcome of G/M interpolation. The artifact audit found
no completed pure G/M interpolation run. No implementation or training of that
additional control was undertaken in this theoretical hour; it is a clearly
specified missing comparison, not a proposed correction required by STDP.

Analytical checks are in `lt/analyze_kv_stdp_theory.py`; independent prescribed
activity histories, constructive retrieval examples, symbolic filter algebra
and the actual initialized production block are included. Every asserted
identity passed in FP64, typically at 1e-14 or better. These checks verify the
stated mathematical claims, not successful model training.

The trajectory-content, frozen-episode, recorded-projection-update and isolated
online-episode diagnostics are respectively in:

```
lt/probe_kv_temporal_content.py
lt/probe_kv_frozen_episode.py
lt/probe_kv_projection_update.py
lt/probe_kv_online_episode.py
```

Results and the research manifest are under
`runs/kv_collapse_20261003/theory_1h/`. The common-window intervention is recorded
in `shared_window_frozen.json`; it changed only an in-memory checkpoint copy.

The derivation supports the STDP write and real Query read. It exposes the
specific history statistic and the additional constraints introduced by an
accumulating M. It does not establish a required current-KV addition, Query
trace, complex Query, sign reversal, or memory forgetting correction. The
cause of the original training reversal remains unconfirmed. The evidence
also requires separating the poorly progressing accumulated-M dynamics from
the G-only model, which already computes many exact solutions at frozen weights.
