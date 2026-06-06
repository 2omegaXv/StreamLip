#!/bin/bash
# StreamLip Offline training sweep
# Edit the two lines below, then run.

EPOCHS=150
LR=6e-4
BS=256

cd "$(dirname "$0")/.."
source .venv/bin/activate

run_name="offline_lora16_glr100_lr${LR}_ep${EPOCHS}"
out_dir="runs/offline/${run_name}"

steps_per_epoch=$(python -c "
import csv, json
from pathlib import Path
root = Path('data/processed')
cache = root / '_pre_only_pretrain.txt'
if cache.exists():
    n = len(cache.read_text().split())
else:
    n = sum(1 for r in csv.DictReader(open(root/'manifest.csv')) if r['split']=='pretrain')
print(max(1, int(n * 0.9) // ${BS}))
" 2>/dev/null || echo 6000)
final_step=$(( steps_per_epoch * EPOCHS ))
final_ckpt="${out_dir}/step_$(printf '%06d' ${final_step}).pt"

if [ -f "${final_ckpt}" ]; then
  echo "SKIP (already done): ${run_name}"; exit 0
fi

echo "=========================================="
echo "START: lr=${LR}  epochs=${EPOCHS}  bs=${BS}  run=${run_name}"
echo "steps/epoch=${steps_per_epoch}  total=${final_step}"
echo "=========================================="

python scripts/train_offline.py \
  --lr          "${LR}"     \
  --max_epochs  "${EPOCHS}" \
  --batch_size  "${BS}"     \
  --num_workers 8          \
  --run_name    "${run_name}" \
  --output_dir  "${out_dir}" \
  --warmup_epochs 5.0
&& echo "DONE: ${run_name}" \
|| echo "FAILED: ${run_name}"
