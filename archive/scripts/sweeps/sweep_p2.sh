#!/bin/bash
# StreamLip V4 Phase 2 — joint training sweep
# Runs two ablations back-to-back:
#   1. with_text: FM conditioned on vis + h_lm + speaker
#   2. no_text:   FM conditioned on vis + zeros + speaker  (visual-only baseline)
#
# Edit the lines below, then run.

EPOCHS=30
LR=3e-4
BS=512        # FM head is memory-heavy; reduce if OOM

# ── derived (don't edit below) ────────────────────────────────────────────────
cd "$(dirname "$0")/.."
source .venv/bin/activate

steps_per_epoch=$(python -c "
import csv
n = sum(1 for r in csv.DictReader(open('data/processed/manifest.csv')) if r['split']=='pretrain')
print(max(1, n * 9 // 10 // ${BS}))
" 2>/dev/null || echo 3200)
final_step=$(( steps_per_epoch * EPOCHS ))

echo "steps/epoch=${steps_per_epoch}  total=${final_step}"
echo ""

for variant in with_text no_text; do

  run_name="v4_p2_${variant}_lr${LR}_ep${EPOCHS}"
  out_dir="runs/v4/${run_name}"
  final_ckpt="${out_dir}/step_$(printf '%06d' ${final_step}).pt"

  if [ -f "${final_ckpt}" ]; then
    echo "SKIP (already done): ${run_name}"
    continue
  fi

  echo "=========================================="
  echo "START: variant=${variant}  lr=${LR}  epochs=${EPOCHS}"
  echo "=========================================="

  extra_args=""
  [ "${variant}" = "no_text" ] && extra_args="--no_text_cond"

  python scripts/train_v4.py \
    --lr          "${LR}"     \
    --max_epochs  "${EPOCHS}" \
    --batch_size  "${BS}"     \
    --num_workers 16           \
    --load_latent             \
    --load_face               \
    --eval_every  500         \
    --save_every  1000        \
    --run_name    "${run_name}" \
    --output_dir  "${out_dir}" \
    ${extra_args} \
  && echo "DONE: ${run_name}" \
  || echo "FAILED: ${run_name}"

  echo ""

done
