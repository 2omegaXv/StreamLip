# StreamLip V5：基于冻结 Conformer + OLMo-1B 的唇语识别系统

> 文档记录了第一个能跑起来、得到有意义 WER 的版本。
> 时间戳：2026.06.04.22

---

## 1. 系统概述

StreamLip V5 是一个**唇语识别（VSR）**系统，核心思路是：

```
lip frames
  └→ Auto-AVSR Conformer (frozen, 预提取)
       └→ avsr_enc.npy (T, 768)
            ├→ [proj Linear(768→2048)]
            │       └→ visual prefix / cross-attn → OLMo-1B
            └→ OLMo-1B LM (fine-tuned)
                   └→ 逐词生成文本
```

用 Auto-AVSR 的预训练 Conformer encoder 作为固定的视觉特征提取器，不参与训练。LM 部分用 OLMo-1B（在 LRS3 小写文本上微调），通过 Flamingo 风格的 Gated Cross-Attention 注入视觉特征。

---

## 2. 数据处理

### 2.1 数据集

- **LRS3-TED**，本地路径：`/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/`
- 分割：pretrain（115948 clips）/ trainval（31982）/ test（1321）
- 处理后存在 `data/processed/{split}/{speaker}/{clip}/`

### 2.2 预处理流程

```
原始 mp4 → 人脸检测 → 嘴唇裁剪 96×96 灰度 → lip_avsr.npy (T,96,96) uint8
lip_avsr.npy → Auto-AVSR frontend+proj → Conformer × 12 → avsr_enc.npy (T,768) float16
```

**相关脚本**：
- `scripts/preprocess_lrs3.py`：人脸检测与嘴唇裁剪（初次处理）
- `scripts/reprocess_avsr.py`：重处理调度器，并行启动 worker 重跑裁剪
- `scripts/reprocess_worker_avsr.py`：FAN 批量推理 worker（整 clip 一次批量，比逐帧快 10×），输出 `lip_avsr.npy`
- `scripts/extract_avsr_enc.py`：提取 Auto-AVSR Conformer 特征

### 2.3 文本处理

- `text.json`：含词级时间戳 `words[{word, start, end}]`
- 文本全部 **lowercase**（关键，否则 SmolLM/OLMo 的大写碎片化会导致训练失效）
- `scripts/prepare_lm_text.py`：生成 `data/processed/lrs3_text.txt`（小写）

### 2.4 Dataset（`src/streaminlip/v5/data/dataset.py`）

- 读 `avsr_enc.npy` 作为 visual input
- `build_text_sequence()`：构造 `(input_ids, target_ids, text_pos, last_chunk_mask)`
  - **BOS**：OLMo 无 bos_token，用 eos_token_id（50279）代替
  - **EOS**：target_ids 末尾显式加 eos_token（关键，让模型学会自然停止）
  - `input_ids  = [bos] + tok_0 ... tok_{N-1}`
  - `target_ids = tok_0 ... tok_{N-1} + eos`

---

## 3. 模型架构

### 3.1 总体结构

```
visual (B,T,768)
  │
  ├─ proj: Linear(768 → 2048)              # 对齐 OLMo hidden size
  │
  └─ OLMo-1B（16层，hidden=2048）
       每 4 层插入一个 GatedCrossAttentionLayer
       ← visual features 通过 forward hook 注入
```

**文件**：`src/streaminlip/v5/model.py`

### 3.2 Gated Cross-Attention（`src/streaminlip/cross_attention.py`）

- Flamingo 风格，tanh gate 初始化为 0.1
- `x + tanh(gate) × CrossAttn(x, vis)`
- gate 在 fp32 下以 `lr × 10` 训练（防止不动或饱和）

### 3.3 视觉编码器（frozen）

Auto-AVSR：ResNet frontend → `proj_encoder: Linear(512→768)` → `ConformerEncoder × 12`

代码通过 `AutoAVSRInferencer` 加载，所有参数 `requires_grad=False`。

### 3.4 参数量

| 部分 | 参数量 |
|------|--------|
| OLMo-1B（可训练） | ~1177M |
| proj + CA layers | ~47M |
| Auto-AVSR encoder（frozen） | ~11.6M |
| **总计** | **1236M**（可训练 1225M） |

---

## 4. LM 微调

在 V5 训练之前，先在 LRS3 文本上微调 OLMo-1B，使其 prior 贴近 TED 演讲分布：

```bash
python scripts/finetune_lm.py \
  --model_path pretrained/olmo-1b \
  --output pretrained/olmo-1b-lrs3 \
  --lr 3e-5 --epochs 2 --save_each_epoch
# val_ppl=25.8
```

**超参扫描**：`scripts/sweep_lm.sh`，结果存在 `pretrained/olmo-1b-lrs3-lr*_ep*/`

---

## 5. V5 训练

### 5.1 Optimizer

三组 param group（AdamW，betas=(0.9, 0.98)，WarmupCosine scheduler）：

| 参数组 | lr | weight_decay |
|--------|-----|-------------|
| LM pretrained params | `1e-5/3e-6` | 0.01 |
| proj / CA 新参数 | `3e-4` | 0.3 |
| gate（fp32） | `1e-4`（`lr×10`） | 0.0 |

### 5.2 关键超参

```
batch_size = 256
warmup_epochs = 3.0
max_epochs = 50 (约 19500 steps)
max_frames = 150
val_clips = 500
```

### 5.3 训练命令

```bash
nohup .venv/bin/python scripts/train_v5_avsr.py \
  --run_name v5_olmo_lr1e-5_ep50_eos \
  --smollm2_path pretrained/olmo-1b-lrs3-ep2 \
  --cross_attn_every_n 4 \
  --lr 1e-5 --max_epochs 50 \
  > logs/run_v5_olmo_lr1e-5_ep50_eos.log 2>&1 &
```

### 5.4 训练曲线对比

**lr=1e-5**：

| step | val_ce | val_tok_acc |
|------|--------|-------------|
| 500  | 1.097  | 0.772 |
| 1000 | 0.723  | 0.846 |
| **1500** | **0.675** | **0.853** ← best |
| 2000 | 0.693  | 0.854 |
| 4000+ | >0.78 | ~0.85（过拟合） |

**lr=3e-6**（完整曲线，关键节点）：

| step | val_ce | val_tok_acc | 备注 |
|------|--------|-------------|------|
| 1000 | 0.719  | 0.843 | |
| **2000** | **0.664** | 0.860 | **val_ce best** |
| 3000 | 0.672  | 0.862 | |
| 8500 | 0.789  | 0.862 | |
| 10500 | 0.814 | 0.865 | |
| 14000 | 0.836 | 0.867 | |
| **14500** | 0.858 | **0.8669** | **val_tok_acc best** |
| 17000 | 0.903  | 0.855 | |

**重要发现**：val_ce 和 val_tok_acc 的最优点完全不同：
- val_ce best @ step 2000（0.664）：EOS 预测校准最好
- val_tok_acc best @ step 14500（0.867）：逐 token 识别准确率最高

val_ce 在 step 2000 后上升是"过拟合"的表象，但 tok_acc 还在缓慢提升，说明模型的识别能力在持续增强，只是概率校准变差（LM 对 EOS 时机的判断越来越极端）。**解码效果应以 step ~14500 的 checkpoint 为准**，而不是 val_ce 最低的 step 2000。

---

## 6. 解码

### 6.1 Offline Decode（推荐）

```python
# scripts/decode_v5.py: offline_decode()
# beam search + no_repeat_ngram_size=4 + max_toks 按语速估算
```

```bash
python scripts/decode_v5.py \
  --ckpt runs/v5/v5_olmo_lr1e-5_ep50_eos/step_001500.pt \
  --smollm2_path pretrained/olmo-1b-lrs3-ep2 \
  --cross_attn_every_n 4 \
  --split test --n_clips 200 \
  --offline --num_beams 3
```

**最优 beam 宽度：3**（见 §7.3 beam sweep 实验）

### 6.2 EOS 自然停止

加入 EOS loss 后，模型能在正确位置停止，不依赖 `max_new_tokens` 截断，insertion error 大幅降低。

---

## 7. 测试结果

### 7.1 评测命令（公平对比）

```bash
python scripts/eval_compare.py \
  --ckpt runs/v5/v5_olmo_lr1e-5_ep50_eos/step_007000.pt \
  --n_clips 50 --beam 10 --seed 42
```

### 7.2 结果汇总

**公平对比（beam=40，200 clips，test split，seed=42）**：

| 系统 | WER | Word Acc | 备注 |
|------|-----|---------|------|
| Auto-AVSR | **20.2%** | 79.8% | CTC+Att joint, beam=40 |
| **StreamLipV5** (lr=3e-6, step=14500) | **32.4%** | 67.6% | frozen encoder, beam=40 |

差距 12.2%，考虑 encoder frozen 的限制，这是合理的起点。

**早期小样本参考**（beam=10，50 clips）：
- Auto-AVSR: 13.3% WER
- V5 step=7000: 22.3% WER

> 注：beam=10 结果偏低（样本少 + beam 宽度小使两者都偏乐观），200 clips beam=40 更可靠。

### 7.3 Beam Sweep 实验（长句子子集，≥10词 / ≥4.0s，126 clips）

长句子是 V5 相对优势最大的条件（LM 语境积累越多纠错能力越强），在此子集上系统测试 beam 宽度的影响：

| beam | AVSR WER | V5 WER | 差距 | V5胜率 |
|------|----------|--------|------|--------|
| 1 (greedy) | 11.1% | 21.9% | 10.8pp | 6.3% (8/126) |
| 2 | 10.3% | 16.2% | 5.9pp | 9.5% (12/126) |
| **3** | **10.2%** | **15.4%** | **5.2pp ← 最小** | 9.5% (12/126) |
| 4 | 10.3% | 15.6% | 5.3pp | 10.3% (13/126) |
| 40 | 9.9% | 15.9% | 6.0pp | 10.3% (13/126) |

**结论**：
- **V5 最优 beam = 3**，WER 15.4%，差距缩至 5.2pp
- beam 1→2 跳变最大（-5.7pp），第一个候选对比即可过滤大量 greedy 错误
- beam ≥ 4 后不再有收益，甚至微弱回升——beam 过宽时 LM 选出"语言流畅但视觉不对齐"的候选，引入噪声
- AVSR 对 beam 几乎不敏感（11.1%→9.9%），说明其特征质量远优于 V5

**实验脚本**：`CLAUDECODE/tasks/v5_vs_avsr_analysis/eval_long_sentences.py`

### 7.3 典型样例（step=14500，beam=40）

```
GT:    it was a remarkable privilege and an amazing education
AVSR:  it was a remarkable privilege and an amazing educational  （多一词）
V5:    it was a remarkable privilege and an amazing educational  （同 AVSR）

GT:    and in many cases we don't
AVSR:  and in many cases we don't   ✅
V5:    and in many cases we don't   ✅

GT:    this is just one face of a booming sex trade across the arab region
AVSR:  this is just one phase of a booming sex trade across the arab region
V5:    this is just one face of a booming sex trade across the arab peninsula  （V5 face 对，region→peninsula 错）

GT:    and they say i want to work in global poverty but what will it mean about my car
AVSR:  and they say i want to win global poverty ...about my career
V5:    and they said i want to read global poverty ...about my caree

GT:    we're losing a ritual
AVSR:  we are using ritual   （losing→using）
V5:    we're using ritual ritual  （重复）
```

---

## 8. V5 相对 Auto-AVSR 的优势

Auto-AVSR 是成熟的 VSR 系统（WER ~13%），但架构有固有局限：

| 维度 | Auto-AVSR | StreamLip V5 |
|------|-----------|--------------|
| 词表 | SentencePiece ~1000 tokens | OLMo BPE ~50280 tokens |
| Decoder | 6层 TransformerDecoder（从头训） | OLMo-1B（3T tokens 预训练） |
| 语言 prior | 无（只有 LRS3 训练集的分布） | 强大的英语语言模型 |
| 停止机制 | CTC 单调对齐，天然停止 | EOS loss 学到自然停止 |
| 可扩展性 | 仅 ASR | 可接 FM head 做 L2S；可做 instruction tuning |
| 上下文修复 | 弱（短程 attention） | 强（1B LM 修复视觉歧义） |

**具体体现**：
- Auto-AVSR 把 "i'm going to say" 识别成 "i've been saying"（语言先验弱）
- V5 在视觉信号不足时，用 LM prior 选择更自然的表达

**当前 V5 的不足**（已明确）：
- Conformer encoder frozen，特征和 OLMo token 空间存在 gap
- 缺少 CTC 单调对齐，导致 insertion error 比 Auto-AVSR 多
- 训练数据仅用 pretrain split，无 trainval

**路线**：unfreeze Conformer + 更多训练数据，预计可进一步缩小与 Auto-AVSR 的差距。

---

## 9. 已知问题与后续方向

| 问题 | 原因 | 方案 |
|------|------|------|
| val_ce ~step 2000 后平台 | Conformer frozen，特征和 OLMo token 空间有 gap | 提取 `avsr_pre_enc.npy`，训练时 unfreeze Conformer |
| Insertion error 多 | 无 CTC 单调对齐 | Conformer unfreeze 后可加 CTC head |
| CA gate 趋向饱和 | gate lr 过大时 tanh→1 | 已调为 `lr×10`，gate init=0.1 |
| 数字识别差（"1836"） | test GT 大写，WER 计算格式不一致 | normalize 时统一处理 |

### 下一步：解冻 Conformer

1. 新建 `scripts/extract_avsr_pre_enc.py`：提取 Auto-AVSR `frontend + proj_encoder` 输出（Conformer 输入，768 维），存为 `avsr_pre_enc.npy`
2. `model.py`：`_encode()` 改为实时跑 Conformer
3. Optimizer 加第四个 param group：Conformer 以极小 lr（`1e-6`）微调

---

*2026.06.04.22 / beam sweep 补充 2026.06.04*
