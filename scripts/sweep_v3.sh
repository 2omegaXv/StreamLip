#!/bin/bash
# StreamLip V3 training sweep
# Usage: bash scripts/sweep_v3.sh

cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS=30
lr=3e-4

run_name="v3_tanh0.1_causal_lr${lr}_ep${EPOCHS}"
out_dir="runs/v3/${run_name}"

steps_per_epoch=$(python -c "
import csv
n = sum(1 for r in csv.DictReader(open('data/processed/manifest.csv')) if r['split']=='pretrain')
print(max(1, n * 9 // 10 // 512))
" 2>/dev/null || echo 221)
final_step=$(( steps_per_epoch * EPOCHS ))
final_ckpt="${out_dir}/step_$(printf '%06d' ${final_step}).pt"

if [ -f "${final_ckpt}" ]; then
  echo "SKIP (already done): ${run_name}"
  exit 0
fi

echo "=========================================="
echo "START: lr=${lr}  epochs=${EPOCHS}  run=${run_name}"
echo "=========================================="

python scripts/train_v3.py \
  --lr          "${lr}" \
  --max_epochs  "${EPOCHS}" \
  --run_name    "${run_name}" \
  --output_dir  "${out_dir}" \
  --num_workers 8 \
&& echo "DONE: ${run_name}" \
|| echo "FAILED: ${run_name}"
