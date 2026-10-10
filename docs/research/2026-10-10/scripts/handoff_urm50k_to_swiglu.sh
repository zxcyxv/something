#!/bin/bash
# Stop URM 1-layer right after its step-50000 checkpoint, then start alpha=1 + SwiGLU boundary.
set -u
cd /workspace/something
URM=runs/urm_swiglu_1layer_loops16_20261010
PID=3153
OUT=runs/pairangle_unrotated_dca1_swiglu_tau2_r2_qkl2_sum_20261010
log() { echo "[handoff $(date -u +%H:%M:%S)] $*"; }

until [ -f "$URM/step_50000.pt" ]; do
  kill -0 $PID 2>/dev/null || { log "URM exited before 50k; not launching"; exit 1; }
  sleep 15
done
log "step_50000.pt saved; SIGTERM to URM ($PID)"
kill -TERM $PID
while kill -0 $PID 2>/dev/null; do sleep 5; done
log "URM stopped: $(ls $URM/*.pt | tail -2 | tr '\n' ' ')"

source /venv/main/bin/activate
mkdir -p "$OUT"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m lt.experiment_free_phase_windows \
  --window pairangle --modes 2 --tau-phi 2 --qk-l2 --write-sum --phase-frame unrotated \
  --dc-hebbian --dc-alpha-init 1 --boundary-ffn swiglu --steps 0 --save-every 5000 \
  --out "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
log "launched SwiGLU run pid $! -> $OUT"
