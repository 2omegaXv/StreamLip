#!/bin/bash
# Parameter sweep: lr x alpha, epochs fixed at 30
# Usage: bash scripts/sweep_v2.sh

cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS=30
lr=1e-3

run_name="v2_lower_lr${lr}_ep${EPOCHS}"
out_dir="runs/v2/${run_name}"

steps_per_epoch=$(python -c "
import csv
n = sum(1 for r in csv.DictReader(open('data/processed/manifest.csv')) if r['split']=='pretrain')
print(max(1, n * 9 // 10 // 256))
" 2>/dev/null || echo 443)
final_step=$(( steps_per_epoch * EPOCHS ))
final_ckpt="${out_dir}/step_$(printf '%06d' ${final_step}).pt"

if [ -f "${final_ckpt}" ]; then
  echo "SKIP (already done): ${run_name}"
  continue
fi

echo "=========================================="
echo "START: lr=${lr}  epochs=${EPOCHS}  run=${run_name}"
echo "=========================================="

python scripts/train_v2.py \
  --lr          "${lr}" \
  --max_epochs  "${EPOCHS}" \
  --run_name    "${run_name}" \
  --output_dir  "${out_dir}" \
  --num_workers 8 \
&& echo "DONE: ${run_name}" \
|| echo "FAILED: ${run_name}"
