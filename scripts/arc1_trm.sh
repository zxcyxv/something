#!/bin/bash
# ARC-AGI-1 in the TRM recipe (global batch 768, configs/arc1_trm_loops16.json, ACT 16) on one multi-GPU instance.
# Run from the repository root:
#   bash scripts/arc1_trm.sh setup          GPUs, torch build vs GPU arch, NCCL, disk (>= 30 GB), matplotlib, quick tests
#   PRICE=8.0 bash scripts/arc1_trm.sh bench
#                                           speed of each precision (fp32 / tf32 / bf16 bands) at the fastest
#                                           grad_accum x act_ckpt that fits, BENCH_STEPS steps each, no checkpoints
#                                           -> bench/<label>/bench.json and a BENCH_JSON line (paste it on the other machine)
#   bash scripts/arc1_trm.sh decide A B     cheaper per optimizer step: bench.json paths, BENCH_JSON payloads or
#                                           label:s_per_step:price
#   PREC=fp32 bash scripts/arc1_trm.sh start
#                                           launch (or resume) runs/$RUN in the background with this machine's bench
#                                           setting for PREC, or SETTING="ga=2 ckpt=false prec=fp32"
#   bash scripts/arc1_trm.sh status | stop | curves | archive
# Environment: PRICE ($/h of the whole instance), RUN (arc1_trm_l16), CONFIG (configs/arc1_trm_loops16.json),
#   NGPU (all GPUs), PRECS ("fp32 tf32 bf16"), PREC (fp32), SETTING, BENCH_STEPS (40), MAX_HOURS, MAX_STEPS,
#   EXTRA (more KEY=VALUE overrides for lt.train), REF (TRM reference curves for `curves`).
# Disk: a checkpoint of this recipe is ~6 GB (the 960,960 x 512 puzzle-embedding table, raw + EMA); the run keeps 2.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate
[ -f bench/env.sh ] && . bench/env.sh                      # NCCL_P2P_DISABLE=1 when setup found P2P broken

RUN=${RUN:-arc1_trm_l16}
CONFIG=${CONFIG:-configs/arc1_trm_loops16.json}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PRECS=${PRECS:-fp32}
PREC=${PREC:-fp32}
BENCH_STEPS=${BENCH_STEPS:-40}
REF=${REF:-docs/evidence/trm_arc1/reference.json}
GBS=$(python -c "import json; print(json.load(open('$CONFIG'))['global_batch_size'])")

gpu_label() { echo "${NGPU}x$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/ /_/g')"; }
gpu_mem_gib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%.1f", $1 / 1024}'; }
prec_args() {
  case $1 in
    fp32) echo "matmul_precision=highest band_bf16=false" ;;
    tf32) echo "matmul_precision=high band_bf16=false" ;;
    bf16) echo "matmul_precision=highest band_bf16=true" ;;
    *) echo "unknown precision $1" >&2; exit 1 ;;
  esac
}
# Peak GiB per GPU, measured on an RTX 4090 (loops 1, 1000 versions): 3.8 + slope x lanes, plus the carry of the
# other micro-batches (grad_accum > 1, ~0.036 GiB per lane; bf16 halves the traces in it).
predict_mem() {  # prec ckpt lanes ga
  awk -v p="$1" -v c="$2" -v n="$3" -v g="$4" 'BEGIN {
    s = (c == "true") ? 0.387 : 0.509; k = 0.036
    if (p == "bf16") { s = (c == "true") ? BF16_CKPT : BF16_NOCKPT; k = 0.022 }
    printf "%.1f", 3.8 + s * n + k * n * (g - 1) }' BF16_CKPT=0.21 BF16_NOCKPT=0.33
}
worker_pids() {  # the lt.train python processes of runs/$RUN (not torchrun, not the wrapper shell)
  ps -eo pid=,args= | awk -v o="--out runs/$RUN " 'index($0, o) && $2 ~ /python[0-9.]*$/ && index($0, "-m lt.train") && !index($0, "torchrun") {print $1}'
}
launch() {  # out_dir, then KEY=VALUE overrides; foreground
  local out=$1; shift
  torchrun --standalone --nproc_per_node="$NGPU" -m lt.train --config "$CONFIG" --out "$out" --set "$@"
}

setup() {
  echo "== repo $(git rev-parse --short HEAD 2>/dev/null || echo '?'), $(git status --porcelain 2>/dev/null | wc -l) uncommitted paths"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  python - <<'EOF'
import torch
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPUs', torch.cuda.device_count(), 'archs', torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    cc = torch.cuda.get_device_capability(i)
    if f'sm_{cc[0]}{cc[1]}' not in torch.cuda.get_arch_list():
        raise SystemExit(f'GPU {i} is sm_{cc[0]}{cc[1]}, not in this torch build: '
                         'uv pip install torch --index-url https://download.pytorch.org/whl/cu128 (CLAUDE.md section 12)')
    a = torch.randn(4096, 4096, device=f'cuda:{i}', dtype=torch.bfloat16)
    f = torch.compile(lambda x: (x @ x).float().sum())
    assert torch.isfinite(f(a)).item()
print('bf16 matmul + torch.compile ok on every GPU')
EOF
  if ! timeout 120 torchrun --standalone --nproc_per_node="$NGPU" scripts/nccl_check.py; then
    echo "NCCL check failed or hung; retrying with NCCL_P2P_DISABLE=1 (this model's all-reduce is ~15 MB/step, P2P is not needed)"
    NCCL_P2P_DISABLE=1 timeout 120 torchrun --standalone --nproc_per_node="$NGPU" scripts/nccl_check.py
    mkdir -p bench && echo 'export NCCL_P2P_DISABLE=1' > bench/env.sh
  fi
  local free; free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
  echo "disk: ${free} GB free"
  [ "$free" -ge 30 ] || { echo "need >= 30 GB free (2 kept checkpoints of ~6 GB + one being written)"; exit 1; }
  python -c "import matplotlib" 2>/dev/null || uv pip install matplotlib || pip install matplotlib
  test -f data/arc1.npz
  python -m unittest tests.test_bands tests.test_model 2>&1 | tail -3
}

bench() {
  local label; label=${LABEL:-$(gpu_label)}
  local dir=bench/$label cap; cap=$(gpu_mem_gib)
  mkdir -p "$dir"
  echo "== bench $label: ${cap} GiB per GPU, global batch $GBS, $BENCH_STEPS steps per setting"
  for prec in $PRECS; do
    for cand in "1 false" "2 false" "1 true" "4 false" "2 true" "4 true"; do
      set -- $cand
      local ga=$1 ck=$2 lanes=$((GBS / NGPU / $1)) name=${prec}_ga$1_ckpt$2 pred
      pred=$(predict_mem "$prec" "$ck" "$lanes" "$ga")
      if awk -v p="$pred" -v c="$cap" 'BEGIN { exit !(p > 0.92 * c) }'; then
        echo "  $name: predicted ${pred} GiB > 92% of ${cap}, skipped"; continue
      fi
      [ -d "$dir/$name" ] && mv "$dir/$name" "$dir/$name.old.$(date +%s)"
      mkdir -p "$dir/$name"
      echo "  $name: $lanes lanes per GPU, predicted ${pred} GiB ..."
      if timeout "${BENCH_TIMEOUT:-1200}" bash -c "$(declare -f launch); NGPU=$NGPU CONFIG=$CONFIG launch $dir/$name \
          grad_accum=$ga act_ckpt=$ck carry_offload=false $(prec_args "$prec") max_steps=$BENCH_STEPS \
          save_final=false save_every_steps=0 save_on_eval=false test_size=64" > "$dir/$name/stdout.log" 2>&1; then
        break                                            # fastest setting that fits for this precision
      fi
      echo "    failed: $(grep -m1 -oE 'OutOfMemoryError|CUDA out of memory|Error[^\n]{0,80}' "$dir/$name/stdout.log" || echo timeout)"
    done
  done
  python scripts/arc1_trm_bench.py collect "$dir" --label "$label" --gpu-mem "$cap" ${PRICE:+--price $PRICE}
}

start() {
  local ga ck pr
  if [ -n "${SETTING:-}" ]; then
    eval "$(echo "$SETTING" | tr ' ' '\n' | sed -n 's/^ga=/ga=/p; s/^ckpt=/ck=/p; s/^prec=/pr=/p')"
  else
    local bj="bench/$(gpu_label)/bench.json"
    [ -f "$bj" ] || { echo "no $bj: run bench first or set SETTING"; exit 1; }
    read -r ga ck pr < <(python -c "import json; b = json.load(open('$bj'))['best']['$PREC']['setting']; print(b['grad_accum'], str(b['act_ckpt']).lower(), b['precision'])")
  fi
  [ -z "$(worker_pids)" ] || { echo "runs/$RUN is already running: $(worker_pids)"; exit 1; }
  local limits=""
  [ -n "${MAX_HOURS:-}" ] && limits="$limits max_hours=$MAX_HOURS"
  [ -n "${MAX_STEPS:-}" ] && limits="$limits max_steps=$MAX_STEPS"
  mkdir -p "runs/$RUN"
  echo "== $(date -u +%FT%TZ) start runs/$RUN: $CONFIG, $NGPU GPUs, grad_accum $ga act_ckpt $ck precision $pr$limits ${EXTRA:-}" | tee -a "runs/$RUN/stdout.log"
  setsid nohup bash -c "$(declare -f launch); NGPU=$NGPU CONFIG=$CONFIG launch runs/$RUN grad_accum=$ga act_ckpt=$ck \
      carry_offload=false $(prec_args "$pr") $limits ${EXTRA:-}; echo EXITCODE \$?" >> "runs/$RUN/stdout.log" 2>&1 < /dev/null &
  sleep 20
  echo "workers: $(worker_pids | tr '\n' ' ')  (log: runs/$RUN/stdout.log)"
}

status() {
  echo "workers: $(worker_pids | tr '\n' ' ')"
  grep -E '^\[(LT|EVAL)\]|^EXITCODE|Error' "runs/$RUN/stdout.log" | tail -n 8 || true
  python - "runs/$RUN" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / 'train.jsonl')][-200:] if (d / 'train.jsonl').exists() else []
cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else {}
if len(rows) > 1 and cfg:
    sps = (rows[-1]['elapsed'] - rows[0]['elapsed']) / (len(rows) - 1)
    pairs = json.load(open(cfg["data_npz"].replace(".npz", ".json")))["train_pairs"]
    total = cfg['epochs'] // cfg['eval_interval'] * (cfg['eval_interval'] * pairs // cfg['global_batch_size'])
    total = min(total, cfg['max_steps'] or total)
    left = total - rows[-1]['step']
    print(f"step {rows[-1]['step']:,} / {total:,}: {sps:.3f} s/step, {3600 / sps:,.0f} steps/h, ETA {left * sps / 3600:.1f} h "
          f"(peak {max(r.get('mem_gb', 0) for r in rows):.1f} GiB)")
EOF
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  df -h . | tail -1
}

stop() {
  local p; p=$(worker_pids)
  [ -n "$p" ] || { echo "runs/$RUN is not running"; exit 0; }
  kill -TERM $p                                           # every rank stops together and saves a checkpoint
  echo "SIGTERM sent to $p; the run saves step_N.pt and exits (tail -f runs/$RUN/stdout.log)"
}

curves() {
  local refs=""
  [ -f "$REF" ] && refs="--ref $REF"
  python -m lt.analysis.curves "runs/$RUN" $refs --out "runs/$RUN/curves" --x step epoch samples hours
}

archive() {
  local e=docs/evidence/run_$RUN
  mkdir -p "$e"
  cp "runs/$RUN/config.json" "runs/$RUN/stdout.log" "$e/"
  [ -f "runs/$RUN/eval.jsonl" ] && cp "runs/$RUN/eval.jsonl" "$e/"
  gzip -c "runs/$RUN/train.jsonl" > "$e/train.jsonl.gz"
  [ -d "runs/$RUN/curves" ] && cp -r "runs/$RUN/curves" "$e/"
  du -sh "$e"
  echo "archived to $e (checkpoints not included); commit and push it before releasing the instance"
}

cmd=${1:-}; shift || true
case $cmd in
  setup|bench|start|status|stop|curves|archive) $cmd ;;
  decide) python scripts/arc1_trm_bench.py decide "$@" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
