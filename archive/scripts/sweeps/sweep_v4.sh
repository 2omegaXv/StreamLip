#!/bin/bash
# StreamLip V4 training sweep
# Usage: bash scripts/sweep_v4.sh
#
# Edit the two lines below, then run.

EPOCHS=30
LR=3e-4
BS=256       # must match --batch_size in train_v4.py

# ── derived (don't edit below) ────────────────────────────────────────────────
cd "$(dirname "$0")/.."
source .venv/bin/activate

run_name="v4_ctc_aux_lr${LR}_ep${EPOCHS}"
out_dir="runs/v4/${run_name}"

steps_per_epoch=$(python -c "
import csv
n = sum(1 for r in csv.DictReader(open('data/processed/manifest.csv')) if r['split']=='pretrain')
print(max(1, n * 9 // 10 // ${BS}))
" 2>/dev/null || echo 400)
final_step=$(( steps_per_epoch * EPOCHS ))
final_ckpt="${out_dir}/step_$(printf '%06d' ${final_step}).pt"

if [ -f "${final_ckpt}" ]; then
  echo "SKIP (already done): ${run_name}"
  exit 0
fi

echo "=========================================="
echo "START: lr=${LR}  epochs=${EPOCHS}  run=${run_name}"
echo "steps/epoch=${steps_per_epoch}  total=${final_step}"
echo "=========================================="

python scripts/train_v4.py \
  --lr          "${LR}"     \
  --max_epochs  "${EPOCHS}" \
  --batch_size  "${BS}"     \
  --num_workers 12           \
  --warmup_epochs 2.0        \
  --run_name    "${run_name}" \
  --output_dir  "${out_dir}"
&& echo "DONE: ${run_name}" \
|| echo "FAILED: ${run_name}"
