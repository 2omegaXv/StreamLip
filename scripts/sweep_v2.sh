#!/bin/bash
# Parameter sweep: lr x alpha, epochs fixed at 30
# Usage: bash scripts/sweep_v2.sh

cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS=25
lr=3e-2
alpha=1.0

run_name="v2_subword_lr${lr}_alpha${alpha}_ep${EPOCHS}"
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
echo "START: lr=${lr}  alpha=${alpha}  epochs=${EPOCHS}  run=${run_name}"
echo "=========================================="

python scripts/train_v2.py \
  --lr          "${lr}" \
  --alpha       "${alpha}" \
  --max_epochs  "${EPOCHS}" \
  --run_name    "${run_name}" \
  --output_dir  "${out_dir}" \
  --num_workers 12 \
&& echo "DONE: ${run_name}" \
|| echo "FAILED: ${run_name}"
