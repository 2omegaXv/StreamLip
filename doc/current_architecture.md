# StreamLip 当前代码架构说明

更新时间：2026-05-31

本文档描述当前 worktree 中的最新实际代码状态。重点结论是：当前最新主线已经不是一个在线端到端的 `video -> text -> audio` 系统，而是一个以预提取 Auto-AVSR 与 SmolLM2 特征为条件的独立 FM 声码器头训练路线。

## 1. 当前最新主线：FM-AVSR

当前最活跃、最新的训练路线由以下文件组成：

| 功能 | 文件 |
| --- | --- |
| 训练 FM head | `scripts/train_fm_avsr.py` |
| FM-AVSR 数据集 | `src/streaminlip/fm_avsr_dataset.py` |
| 生成音频评估 | `scripts/eval_fm_avsr.py` |
| Auto-AVSR 封装 | `src/streaminlip/auto_avsr.py` |
| Auto-AVSR 特征提取 | `scripts/extract_avsr_enc.py` |
| SmolLM2 hidden 预提取 | `scripts/extract_smollm2_h.py` |
| speaker embedding 预提取 | `scripts/extract_speaker_emb.py` |
| DiT/OT-CFM 基类 | `src/streaminlip/v2/fm_head.py` |

### 1.1 是否是 AVSR 生成 text 然后再转 audio？

准确说：训练时不是在线 `AVSR -> text -> audio`。

当前流程是离线预处理后再训练 FM：

```text
lip.npy
  -> Auto-AVSR
  -> avsr_enc.npy      (T_v, 768)   视觉连续特征
  -> avsr_text.txt     文本转录，来自 Auto-AVSR CTC greedy decode

avsr_text.txt
  -> SmolLM2
  -> smollm2_h.npy     (L, 960)     token-level LM hidden

face.npz
  -> SpeakerEncoder
  -> speaker_emb.npy   (256,)

audio.wav
  -> Mimi encoder/transformer/downsample
  -> latent.npz        (T_a, 512)   量化前连续 latent

训练:
  avsr_enc.npy + smollm2_h.npy + speaker_emb.npy
  -> FMHeadAVSR
  -> 预测 Mimi latent 的 OT-CFM velocity
```

所以文本路径确实来自 Auto-AVSR，但在 FM 训练中已经被固化为 `smollm2_h.npy`。`train_fm_avsr.py` 不再加载 Auto-AVSR 或 SmolLM2，也不在线解码文本。

推理评估 `eval_fm_avsr.py` 同样读取已经存在的 `avsr_enc.npy`、`smollm2_h.npy` 和 `speaker_emb.npy`。严格的端到端视频输入推理还需要先运行特征提取脚本，或者把这些步骤串成一个 pipeline。

### 1.2 FMHeadAVSR 条件

`scripts/train_fm_avsr.py` 定义了一个继承 `src/streaminlip/v2/fm_head.py` 基类结构的 `FMHeadAVSR`：

```text
condition = concat[
  v_down: (B, T_a, 768),
  h_down: (B, T_a, 960),
  spk:    (B, T_a, 256)
] -> Linear(1984, 512)
```

FM head 本体是 6 层 DiT block，隐维 512，训练目标是 OT-CFM：

```text
x0 ~ N(0, I)
x1 = target Mimi latent
xt = (1 - t) * x0 + t * x1
target velocity = x1 - x0
loss = MSE(v_theta(xt, condition, t), x1 - x0)
```

`--no_text_cond` 消融会把 `h_down` 置零，用于对比“有文本条件”和“无文本条件”。

## 2. 数据处理现状

### 2.1 原始处理目录

当前代码默认数据目录是：

```text
data/processed/
```

manifest：

```text
data/processed/manifest.csv
```

FM-AVSR 有效 clip 缓存：

```text
data/processed/_fm_avsr_pretrain.txt
```

之前核对过，`extract_smollm2_h.py` 日志显示：

```text
Done: 115947  Skip: 0  Err: 0
```

说明 `smollm2_h.npy` 已经完成预提取。

### 2.2 每个 clip 的主要文件

当前 FM-AVSR 主线需要：

| 文件 | shape | 产生方式 | 用途 |
| --- | --- | --- | --- |
| `lip.npy` | `(T, 96, 96, 3)` | `scripts/preprocess_lrs3.py` | Auto-AVSR 输入 |
| `face.npz` | JPEG bytes + offsets | `scripts/preprocess_lrs3.py` | speaker embedding 提取 |
| `audio.wav` | 24 kHz mono | `scripts/preprocess_lrs3.py` | Mimi latent 提取与 GT 保存 |
| `latent.npz` | `(T_a, 512)` | Mimi encoder path | FM target |
| `avsr_enc.npy` | `(T_v, 768)` | `scripts/extract_avsr_enc.py` | 视觉条件 |
| `avsr_text.txt` | text | `scripts/extract_avsr_enc.py` | SmolLM2 输入 |
| `smollm2_h.npy` | `(L, 960)` | `scripts/extract_smollm2_h.py` | 文本/LM 条件 |
| `speaker_emb.npy` | `(256,)` | `scripts/extract_speaker_emb.py` | 说话人条件 |

`latent_norm_stats.npz` 若存在，会被 `FMAVSRDataset` 用于按维度标准化 Mimi latent；评估时 `eval_fm_avsr.py` 会调用 `denormalize_latent` 还原再送入 Mimi decoder。

### 2.3 预处理脚本顺序

推荐顺序：

```bash
# 基础 LRS3 预处理，生成 lip/face/audio/latent/text.json 等
uv run python scripts/preprocess_lrs3.py

# speaker embedding
uv run python scripts/extract_speaker_emb.py --split pretrain

# Auto-AVSR visual feature + CTC text
uv run python scripts/extract_avsr_enc.py --split pretrain

# SmolLM2 hidden，依赖 avsr_text.txt
uv run python scripts/extract_smollm2_h.py --split pretrain --batch_size 256
```

实际运行时可能已经完成大部分步骤。不要盲目 `--force` 或 `--overwrite`，否则会重算大量 NFS 小文件与 GPU 任务。

## 3. 当前可用训练方式

### 3.1 FM-AVSR with text

这是当前最新主线的“有文本条件”版本：

```bash
uv run python scripts/train_fm_avsr.py \
  --run_name fm_avsr_with_text \
  --batch_size 128 \
  --max_epochs 30 \
  --num_workers 8
```

已有日志显示 `fm_avsr_with_text` 已经训练到 `step_026580.pt` 并完成。

### 3.2 FM-AVSR no text

这是核心消融：

```bash
uv run python scripts/train_fm_avsr.py \
  --no_text_cond \
  --run_name fm_avsr_no_text \
  --batch_size 128 \
  --max_epochs 30 \
  --num_workers 8
```

注意：之前失败的 `fm_avsr_no_text` 日志显示 DataLoader worker 被 kill。那次运行使用了默认 `batch_size=1024`，很可能过大。当前建议显式传 `--batch_size 128`。

### 3.3 FM-AVSR 生成音频

```bash
uv run python scripts/eval_fm_avsr.py \
  --ckpt runs/fm_avsr/fm_avsr_with_text/step_026580.pt \
  --n 20 \
  --output_dir eval_out/with_text \
  --save_gt
```

no-text 版本：

```bash
uv run python scripts/eval_fm_avsr.py \
  --ckpt runs/fm_avsr/fm_avsr_no_text/step_026580.pt \
  --no_text_cond \
  --n 20 \
  --output_dir eval_out/no_text \
  --save_gt
```

评估脚本会：

```text
FM inference -> denormalize latent -> Mimi upsample -> Mimi decoder_transformer -> Mimi decoder -> wav
```

### 3.4 eval/推理时到底发生了什么？

`scripts/eval_fm_avsr.py` 的行为更像“离线条件生成 demo”，不是从原始视频直接一键跑完整系统。它假设每个 clip 的以下文件已经存在：

```text
clip/
  avsr_enc.npy
  smollm2_h.npy
  speaker_emb.npy
  latent.npz
  avsr_text.txt
  audio.wav
```

单个 clip 的 eval 行为如下：

```text
                              ┌────────────────────────────────────┐
                              │ data/processed/.../{clip}/          │
                              └────────────────────────────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        │                                       │                                       │
        ▼                                       ▼                                       ▼
  avsr_enc.npy                            smollm2_h.npy                          speaker_emb.npy
  (T_v, 768)                              (L, 960)                               (256,)
        │                                       │                                       │
        │                                       │                                       │
        │ 当前代码: enc[::2][:T_a]              │ token hidden 重采样到 T_a              │ expand 到每个 latent frame
        │ 建议: T_v -> T_a 通用重采样           │ idx[j] = j * L // T_a                  │
        ▼                                       ▼                                       ▼
  v_down                                  h_down                                  spk_expanded
  (1, T_a, 768)                           (1, T_a, 960)                           (1, T_a, 256)
        │                                       │                                       │
        └─────────────────────────────── concat over channel ───────────────────────────┘
                                                │
                                                ▼
                                  condition: (1, T_a, 1984)
                                                │
                                                ▼
                                  Linear(1984 -> 512)
                                                │
                                                ▼
                                  cond: (1, T_a, 512)
                                                │
                                                ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ FMHeadAVSR.forward_inference(nfe=args.nfe)          │
                    │                                                     │
                    │ x_0 ~ N(0, I), shape (1, T_a, 512)                  │
                    │ for step in 0..nfe-1:                               │
                    │   t = step / nfe                                    │
                    │   velocity = DiT(x, cond, t)                        │
                    │   x = x + (1/nfe) * velocity                        │
                    │                                                     │
                    │ output: pred_latent_norm (1, T_a, 512)              │
                    └─────────────────────────────────────────────────────┘
                                                │
                                                ▼
                                  denormalize_latent(...)
                                                │
                                                ▼
                                  pred_latent: (T_a, 512)
                                                │
                                                ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ Mimi decode path                                    │
                    │                                                     │
                    │ pred_latent.T -> (1, 512, T_a)                      │
                    │ mimi.upsample -> (1, 512, ~2*T_a)                   │
                    │ mimi.decoder_transformer                            │
                    │ mimi.decoder -> waveform @ 24 kHz                   │
                    └─────────────────────────────────────────────────────┘
                                                │
                                                ▼
                                  eval_out/.../{i:04d}_pred.wav
```

如果加 `--save_gt`，脚本还会走一条 GT latent 解码路径：

```text
latent.npz["latent"][:T_a]
  -> Mimi decode
  -> eval_out/.../{i:04d}_gt.wav
```

这个 `gt.wav` 不是原始 `audio.wav` 的直接复制，而是把存盘的 Mimi latent 再解码得到的重建音频。它用于判断 Mimi latent/decoder 本身的重建上限，以及和 FM 预测音频做同路径对比。

#### with-text 和 no-text 的 eval 差异

with-text：

```text
smollm2_h.npy -> h_down -> FM condition
```

no-text：

```text
h_down = zeros(1, T_a, 960)
```

其余输入相同：`avsr_enc.npy` 和 `speaker_emb.npy` 仍然会被使用。因此 no-text 不是“无条件音频生成”，而是“视觉 + 说话人条件，去掉文本/LM hidden 条件”。

#### eval 不做的事情

`eval_fm_avsr.py` 不会做以下事情：

- 不读取原始 `lip.npy` 做 Auto-AVSR 推理。
- 不在线生成 `avsr_text.txt`。
- 不在线运行 SmolLM2。
- 不做 Whisper/UTMOS/SECS 等指标计算。
- 不做真正 streaming chunk-by-chunk 输出。

因此，如果只有原始视频或 `lip.npy`，需要先补齐：

```text
lip.npy -> extract_avsr_enc.py -> avsr_enc.npy + avsr_text.txt
avsr_text.txt -> extract_smollm2_h.py -> smollm2_h.npy
face.npz -> extract_speaker_emb.py -> speaker_emb.npy
```

然后才能用 `eval_fm_avsr.py` 生成音频。

#### 输出文件含义

| 文件 | 含义 |
| --- | --- |
| `{i:04d}_pred.wav` | FMHeadAVSR 从噪声采样得到的预测 latent，经 Mimi 解码后的音频 |
| `{i:04d}_gt.wav` | 真实 Mimi latent 经同一个 Mimi decode path 重建出的参考音频 |

如果想和原始音频比较，还需要额外查看 clip 目录下的 `audio.wav`。

### 3.5 旧路线：v2/v3/v4/offline

这些路线仍在代码库中，但不是当前最新 FM-AVSR 主线。

| 路线 | 入口 | 目标 | 当前定位 |
| --- | --- | --- | --- |
| offline | `scripts/train_offline.py` | 全视频特征 + Gemma/LM cross-attn 做 transcript next-token | 历史离线文本实验 |
| v2 | `scripts/train_v2.py` | 早期 PoE / SmolLM2 / FM 联合设计 | 历史路线 |
| v3 | `scripts/train_v3.py` | Flamingo-style gated cross-attn，Phase 1 text path | 历史文本路线 |
| v4 phase1 | `scripts/train_v4.py` | 在线 AV-HuBERT + SmolLM2 cross-attn 文本路径 | 历史/候选文本路线 |
| v4 phase2 | `scripts/train_v4_phase2.py` | 加载 phase1，冻结后训 FM head | 旧的端到端式候选路线 |
| ctc only | `scripts/train_ctc_only.py` | 诊断 AV-HuBERT 特征是否可做 CTC 唇读 | 诊断实验 |

如果当前目标是复现实验结果、完成文本条件消融，优先使用 `train_fm_avsr.py`，不要混用 v4 phase2；v4 phase2 的输入、checkpoint 和模型结构都不同。

## 4. 当前代码库存在的问题

### 4.1 视觉条件时间对齐可能有严重 bug

当前 `train_fm_avsr.py` 和 `eval_fm_avsr.py` 使用：

```python
v_down = enc[:, ::2, :][:, :T_a, :]
```

这是假设 `avsr_enc.npy` 是 25 Hz、而 Mimi latent 是 12.5 Hz，所以需要 `::2`。

但实际抽样发现，很多 `avsr_enc.npy` 的长度已经接近 `latent.npz` 的 `T_a`，即比例接近 `1.0`，不是稳定 `2.0`。这意味着当前代码可能把视觉条件砍掉一半，然后补零到 `T_a`。后果：

- with-text 训练可能主要依赖 SmolLM2 hidden，掩盖视觉问题。
- no-text 消融会被严重低估，因为视觉条件被错误降采样。
- 训练和评估都会受影响，因为两边用了同样的 `::2`。

建议改为通用重采样：

```text
resample enc length T_v -> T_a
idx[j] = j * T_v // T_a
v_down = enc[:, idx, :]
```

而不是硬编码 `::2`。

### 4.2 `extract_avsr_enc.py` 注释与实际 shape 不一致

脚本顶部注释说：

```text
T' ≈ T/4
```

但 `AutoAVSRInferencer.encode_batch` 注释和实测样本显示输出长度经常接近输入/latent 长度，不应继续按固定 `T/4` 或 `T/2` 推断。文档、训练代码和提取脚本需要统一。

### 4.3 默认 batch size 过大

`train_fm_avsr.py` 默认：

```python
--batch_size 1024
```

但 no-text 训练曾因 DataLoader worker 被 kill。考虑到每 batch 里有可变长 `enc/latent/h_lm` padding，默认值不安全。建议：

- 默认改为 128，或
- 在命令中始终显式传 `--batch_size 128`。

### 4.4 DataLoader 内存与 shared memory 风险

代码已经设置：

```python
torch.multiprocessing.set_sharing_strategy("file_system")
TMPDIR=.tmp
pin_memory=False
```

但仍可能遇到 worker kill、系统 semaphore 或 NFS 临时文件堆积。出现类似问题时：

- 降低 `batch_size`
- 降低 `num_workers`
- 关闭 `persistent_workers`
- 清理 `.tmp/`
- 必要时清理系统 semaphore

### 4.5 当前不是严格流式系统

项目最初目标是 streaming lip-to-speech，但 FM-AVSR 当前主线是预提取特征、离线训练 FM head。它验证的是：

```text
Auto-AVSR 文本先验 / SmolLM2 hidden 是否帮助 Mimi latent 生成
```

它还没有完成：

- 在线视频输入即时提取 Auto-AVSR 特征。
- 在线 Auto-AVSR 解码并驱动 SmolLM2。
- chunk-by-chunk FM 采样与 overlap-add。
- 端到端延迟测量。

因此论文/报告中应把它描述为“当前实验主线/退路路线”，不要声称已经是完整 streaming 系统。

### 4.6 `third_party` 管理混乱

当前：

- `third_party/auto_avsr/` 是新增未跟踪目录，但 `src/streaminlip/auto_avsr.py` 实际依赖它。
- `third_party/av_hubert/` 是已有 submodule/嵌套仓库，并且内部有未提交改动。

建议二选一统一策略：

1. 将 `third_party/auto_avsr/` 正式作为 submodule/外部依赖记录；或
2. 只保留必要 wrapper，写清下载与放置路径，将 `third_party/auto_avsr/` ignore。

当前为了可复现，暂时不要 ignore `third_party/auto_avsr/`。

### 4.7 模型定义重复

`FMHeadAVSR` 在 `train_fm_avsr.py` 和 `eval_fm_avsr.py` 各定义了一遍。两者结构必须完全一致才能加载 checkpoint。建议后续抽到：

```text
src/streaminlip/fm_avsr_model.py
```

这样训练和评估共享同一个类。

## 5. 建议下一步

优先级从高到低：

1. 围绕真实 FM sampling endpoint 优化，而不是继续只看 deterministic recon。
2. 用 `lambda_sample_recon > 0` 的配置做小规模/全量验证；这个 loss 现在必须走可微 `FMHead.sample()`。
3. 评估 `sample_recon_nfe=4/8/10` 与 eval `nfe=4/10/50` 的 mismatch。
4. 将 `FMHeadAVSR` 抽到共享模块，避免训练/评估结构漂移。
5. 重跑 fixed-latents full training 的评估，并和单样本 overfit 结果对齐。
6. 重跑 with-text/no-text 消融。
7. 决定 `third_party/auto_avsr` 是提交、submodule，还是外部安装依赖。

## 5.1 2026-05-31 关键实验发现

### Mimi latent 数据已经重新固定到 12.5 Hz

之前 GT 音频在 eval 中像 2x speed，根因是大量 `latent.npz` 仍是旧的 25 Hz 表示，而当前 Mimi decode 路径期望 12.5 Hz latent。现在处理逻辑是：

```text
audio.wav -> Mimi encode/downsample -> latent.npz (T_a, 512), 约 12.5 Hz
```

`FMAVSRDataset.validate_latent_frame_rate()` 会拒绝疑似 25 Hz latent，避免再用 `lat[::2]` 这种错误修补。全量 latent 已重新提取并重算：

```text
data/processed/latent_norm_stats.npz
```

训练读入 latent 后会按维度标准化，eval 前再 denormalize 后送 Mimi decoder。

### cross-attention 不是当前主要瓶颈

`src/streaminlip/v2/fm_head.py` 现在支持可选 `use_cross_attn`。结构是：

```text
x -> self-attn(x)
  -> cross-attn(query=x, key/value=condition tokens)
  -> FFN
```

condition tokens 会加 sinusoidal position embedding，否则 attention 对 K/V token 顺序本身不敏感。对应实验配置：

```text
configs/fm_avsr_overfit_12p5hz_crossattn.yaml
```

单样本 overfit 结果：

```text
cross-attn FM nfe50:
  audio corr   0.9466
  SI-SDR       9.36 dB
  latent MSE   0.1832

cross-attn deterministic recon:
  audio corr   0.9982
  SI-SDR       24.55 dB
```

解释：cross-attention 证明模型能读取条件并直接重建 latent，但没有解决从随机噪声沿 velocity field 积分到正确 endpoint 的问题。因此 deterministic recon 只能作为诊断，不应作为主要优化目标。

### 之前的 sample recon loss 实际没有梯度

旧训练代码中：

```python
pred_sample = fm.forward_inference(...)
loss_sample_recon = mse(pred_sample, lat_gt)
```

但 `forward_inference()` 带 `@torch.no_grad()`，所以旧的 `lambda_sample_recon` 没有真正通过 Euler sampling path 回传梯度。当前已修复为：

```text
FMHead.sample(...)             可微 Euler solver，用于训练 endpoint loss
FMHead.forward_inference(...)  no-grad wrapper，用于 eval/inference
```

训练脚本现在用：

```python
pred_sample = fm.sample(v_down, h_down, spk, nfe=args.sample_recon_nfe)
loss_sample_recon = mse(pred_sample, lat_gt)
```

对应测试：

```text
tests/test_fm_head_temporal_condition.py
  test_sample_can_backpropagate_for_sample_recon_loss
```

### 当前最有效的单样本方向：sample endpoint loss

对应配置：

```text
configs/fm_avsr_overfit_12p5hz_sample_endpoint.yaml
```

设置：

```yaml
lambda_recon: 0.0
lambda_sample_recon: 1.0
sample_recon_nfe: 4
```

单样本 overfit 5000 step 结果：

```text
train loss:
  step 1:    fm 1.9701 | sample 1.9654 | total 3.9354
  step 1000: fm 0.3484 | sample 0.2442 | total 0.5926
  step 3000: fm 0.1190 | sample 0.0980 | total 0.2170
  step 5000: fm 0.1119 | sample 0.0896 | total 0.2015

eval nfe=4:
  audio corr   0.9726
  SI-SDR       12.43 dB
  latent MSE   0.0918
  latent corr  0.9511

eval nfe=50:
  audio corr   0.9688
  SI-SDR       11.84 dB
  latent MSE   0.0922
  latent corr  0.9507
```

对比：

```text
cross-attn nfe50:
  corr 0.9466 | SI-SDR 9.36 dB | latent MSE 0.1832

旧 sample_recon nfe50:
  corr 0.9662 | SI-SDR 11.47 dB

旧 recon_aux nfe50:
  corr 0.9660 | SI-SDR 11.45 dB
```

结论：当前最明确的改进不是加强 condition 注入，而是让训练目标直接约束真实 sampling endpoint。后续优先测试 `sample_recon_nfe=8/10`、不同 `lambda_sample_recon`，并在全量训练中验证。

## 6. 当前推荐命令

with-text 已有 checkpoint 时，先评估：

```bash
uv run python scripts/eval_fm_avsr.py \
  --ckpt runs/fm_avsr/fm_avsr_with_text/step_026580.pt \
  --n 20 \
  --output_dir eval_out/with_text \
  --save_gt
```

修复视觉重采样后，重跑 no-text：

```bash
uv run python scripts/train_fm_avsr.py \
  --no_text_cond \
  --run_name fm_avsr_no_text \
  --batch_size 128 \
  --max_epochs 30 \
  --num_workers 8
```

如果只是 smoke test：

```bash
uv run python scripts/train_fm_avsr.py \
  --debug \
  --no_wandb \
  --batch_size 4 \
  --num_workers 0
```

## 7. 文件地图

```text
scripts/
  train_fm_avsr.py          当前 FM-AVSR 主训练
  eval_fm_avsr.py           当前 FM-AVSR 生成音频
  extract_avsr_enc.py       Auto-AVSR enc/text 预提取
  extract_smollm2_h.py      SmolLM2 hidden 预提取
  extract_speaker_emb.py    speaker embedding 预提取
  train_v4.py               旧/候选 StreamLip V4 phase1
  train_v4_phase2.py        旧/候选 StreamLip V4 FM phase2
  train_v3.py               旧 V3 text path
  train_offline.py          旧 offline text path
  train_ctc_only.py         CTC 诊断

src/streaminlip/
  fm_avsr_dataset.py        当前 FM-AVSR dataset
  auto_avsr.py              Auto-AVSR wrapper
  v2/fm_head.py             DiT OT-CFM 基类
  v2/speaker_encoder.py     speaker embedding model
  v3/, v4/, offline/        历史/候选路线
```
