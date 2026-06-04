#!/bin/bash
# LM fine-tune 超参扫描：lr × epochs 网格（OLMo-1B，小写 LRS3）
# lr ∈ {1e-4, 3e-4, 1e-3}，epochs ∈ {2, 3}

GPU=${1:-0}
BATCH=64

declare -a LRS=("1e-5")
declare -a EPOCHS=("6" "7" "8")

echo "===== OLMo-1B LM finetune sweep (lr × epochs) ====="
printf "%-8s %-8s %-12s %-10s\n" "lr" "epochs" "best_ppl" "best_ep"
echo "--------------------------------------------"

for LR in "${LRS[@]}"; do
    for EP in "${EPOCHS[@]}"; do
        TAG="lr${LR}_ep${EP}"
        OUT="pretrained/olmo-1b-lrs3-${TAG}"
        LOG="logs/finetune_olmo_sweep_${TAG}.log"

        .venv/bin/python scripts/finetune_lm.py \
            --model_path pretrained/olmo-1b \
            --lr "$LR" --epochs "$EP" --batch "$BATCH" \
            --output "$OUT" --gpu "$GPU" \
            > "$LOG" 2>&1

        .venv/bin/python -c "
import re
log = open('$LOG').read()
epochs = re.findall(r'Epoch (\d+)/\d+ \| .*val_ppl=([\d.]+)', log)
if epochs:
    best = min(epochs, key=lambda x: float(x[1]))
    print(f'lr={\"$LR\":<8} ep={\"$EP\":<6} best_ppl={best[1]:<10} @ epoch {best[0]}')
else:
    print(f'lr={\"$LR\":<8} ep={\"$EP\":<6} FAILED')
" 2>/dev/null
    done
done

echo ""
echo "模型保存在 pretrained/olmo-1b-lrs3-lr*_ep*/"
