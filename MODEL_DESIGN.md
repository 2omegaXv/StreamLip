# StreamLip 模型架构设计规范

> 本文档是模型实现的**唯一参考**。所有模块的 I/O 形状、预训练模型用法、训练目标均以此为准。
> 理论推导见 `theory.md`，数据格式依赖见 `DATA_DESIGN.md`。

---

## 1. 系统概述

StreamLip 是一个**因果流式唇语语音合成**系统。与原始设计的核心区别在于：视觉路径和语言路径**完全解耦**，通过 Product of Experts 在 logit 空间融合，音频路径绕过离散 token 决策直接条件化在连续表示上。

```
                    ┌─────────────────────────────────┐
                    │   1. Visual Encoder              │
lip (B,T,3,96,96) ──→  AV-HuBERT + Conformer Adapter ├──→ ṽ_t (B,T,960)
                    │   chunk-bidirectional            │     │
                    └──────────────────────────────────┘     │
                                                             │
                    ┌───────────────┐                        │
                    │ 2. Visual     │                        │
             ṽ_t ──→   Head        ├──→ s_vis (B,T,V)       │
                    │ Linear(960,V) │        │               │
                    └───────────────┘        │               │
                                            PoE              │
                    ┌───────────────────┐   +α·s             │
text tokens ───────→  3. LM Backbone   ├──→ s_LM (B,T,V)    │
x̂_{1:t-1}          │   SmolLM2-360M    │        │           │
(or GT x*)          │   text-only       ├──→ h̃_t^LM (B,T,960)
                    └───────────────────┘   │               │
                                            ▼               ▼
                                     argmax → x̂_t    ┌─────────────────┐
                                     (WER/LM 自回归) │  5. FM Head      │
                                                      │  DiT + OT-CFM   │◀── id̂ (B,256)
                                                      └────────┬────────┘
                                    ┌──────────────────┐       │             ▲
video ─────────────────────────────→ 4. Speaker Encoder│──→ id̂              │
                                    │ face crop → 256d │       │    ┌────────┴────────┐
                                    └──────────────────┘       │    │ Speaker Encoder │
                                                               ▼
                                                        latent (B,T_a,512)
                                                               │
                                                        Mimi decoder
                                                               │
                                                        audio (B,1,T_s)

L_total = L_fm + λ·L_CE(s_vis + α·s_LM, x*)    λ=0.005, α=1.0（初始值）
```

**chunk 大小：C = 6 帧（240ms @ 25fps）**

### 1.1 与原架构的核心差异

| 维度 | 原架构 | 新架构 |
|------|--------|--------|
| SmolLM2 输入 | 视觉特征 vis_feat（inputs_embeds 模式） | 文本 token IDs（input_ids 模式）|
| 文本预测方式 | backbone hidden → Text Head → logits | s_vis + α·s_LM（Product of Experts）|
| FM 条件 | sg(hidden)[:, ::2, :] | sg(ṽ_t) ∥ sg(h̃_t^LM) ∥ id̂ |
| 说话人身份 | 无 | Speaker Encoder → id̂ |
| 梯度隔离 | FM 不更新 backbone | FM 不更新 Visual Encoder 和 LM；CE 驱动两者 |
| 因果结构 | v_t → x_t → a_t（倒置）| x_t → h_t → {v_t, a_t}（正确）|

---

## 2. 关键硬件常数

继承自 `DATA_DESIGN.md`，此处仅列与模型直接相关的部分：

| 常数 | 值 | 来源 |
|------|----|------|
| 视频帧率 | 25 fps | LRS3 固定 |
| chunk 帧数 | C = 6 | 240ms，整除 |
| 前瞻帧数 | Δ = 5 | chunk 内后 5 帧 |
| Mimi 输出帧率 | 12.5 Hz | `downsample` 后 |
| latent 维度 | 512 | Mimi `hidden_size` |
| 视频/latent 比 | 2 : 1 | 25fps / 12.5Hz |
| T_a | T // 2 | latent 帧数 |
| SmolLM2 hidden | 960 | `config.hidden_size` |
| SmolLM2 vocab | 49152 | `config.vocab_size` (= V) |
| Speaker embedding dim | 256 | ECAPA-TDNN 输出 |

---

## 3. 预训练模型验证（实测 I/O）

### 3.1 Mimi（`pretrained/mimi/`）

```python
# 完整编码路径（preprocess_lrs3.py 中使用）
wav: (1, 1, T_s)                              # 24kHz mono
  → mimi.encoder(wav)                         # (1, 512, T_s/960)  @ 25 Hz
  → mimi.encoder_transformer(...).last_hidden_state  # (1, T'/T_s/960, 512)  @ 25 Hz
  → mimi.downsample(...)                      # (1, 512, T_a)      @ 12.5 Hz
  → .transpose(1,2).squeeze(0).half()         # (T_a, 512)  ← 存储为 latent.npz
```

```python
# 推理解码路径（FM Head 预测完成后）
pred_latent: (B, T_a, 512)
  → mimi.upsample(pred_latent.transpose(1,2)) # (B, 512, 2·T_a) @ 25 Hz
  → mimi.decoder(...)                         # (B, 1, T_samples) @ 24kHz
```

**关键**：latent 是 `downsample` 后、VQ **前** 的连续特征。推理时跳过 `quantizer`，直接 `upsample → decoder`。

| 属性 | 类型 | 用途 |
|------|------|------|
| `encoder` | MimiEncoder (CNN) | 25 Hz 特征提取 |
| `encoder_transformer` | MimiTransformerModel | 上下文建模 |
| `downsample` | MimiConv1d | 25→12.5 Hz |
| `upsample` | MimiConvTranspose1d | 12.5→25 Hz（推理用）|
| `decoder` | MimiDecoder (CNN) | 25 Hz → waveform（推理用）|
| `quantizer` | MimiSplitResidualVQ | **训练/推理均不使用** |

### 3.2 SmolLM2-360M（`pretrained/smollm2-360m/`）

| 参数 | 值 |
|------|----|
| 架构 | LlamaForCausalLM（32层 decoder-only）|
| hidden_size | 960 |
| intermediate_size | 2560 |
| num_attention_heads | 15 |
| num_key_value_heads | 5（GQA）|
| vocab_size | 49152 |
| max_position_embeddings | 8192 |
| dtype | bfloat16 |

**使用方式（与原架构不同）**：传入 `input_ids=(B, T)`，**使用** `embed_tokens` 层和 `lm_head`；`lm_head` 输出作为 LM logits `s_LM`，最后一层 hidden state 作为 `h̃_t^LM`。

```python
# LM Backbone 前向
outputs = smollm2.model(input_ids=text_token_ids, attention_mask=mask)
h_lm = outputs.last_hidden_state   # (B, T, 960)  ← h̃_t^LM
s_lm = smollm2.lm_head(h_lm)       # (B, T, 49152) ← s_LM
```

### 3.3 AV-HuBERT Large（`pretrained/av-hubert/model.pt`）

| 参数 | 值（来自官方论文/代码）|
|------|------|
| encoder_embed_dim | 1024 |
| encoder_layers | 24 |
| encoder_attention_heads | 16 |
| 输入 | (B·T, 1, 96, 96) 灰度，**改为 3 通道** |
| 输出 | (B, T, 1024) per-frame features |
| 加载方式 | `torch.load(..., weights_only=False)`（fairseq format）|

**通道修改策略**：将第一层 Conv2d 的 `in_channels: 1→3`，权重用 `mean` 复制（保持预训练特征响应）。

---

## 4. 模块详细设计

### 4.1 Visual Encoder

**职责**：将唇部视频帧编码为连续视觉表示 ṽ_t，并投影到词表空间得到视觉 logits s_vis。

```
输入: lip  (B, T, 3, 96, 96)  float32, ImageNet 归一化
          │
          ▼
  reshape → (B·T, 3, 96, 96)
          │
          ▼
  AV-HuBERT (frozen)           # 修改 in_ch: 1→3
          │
          ▼
  (B·T, 1024)  reshape → (B, T, 1024)
          │
          ▼
  Conformer Adapter (2-4层)    # chunk-bidir attention
  chunk_size=C, lookahead=Δ
          │
          ▼
  (B, T, 1024)
          │
          ▼
  Linear(1024, 960)
          │
          ▼
  ṽ_t: (B, T, 960)             ← 连续视觉特征，送往 FM 条件
          │
          ▼
  Visual Head: Linear(960, V)  # V=49152，复用 SmolLM2 lm_head.weight（weight tying）
          │
          ▼
  s_vis: (B, T, 49152)         ← 视觉 logits，log p(x_t | v_t)
```

**Chunk 注意力 Mask**（Conformer Adapter 中使用）：

```python
def make_chunk_causal_mask(T: int, C: int) -> torch.BoolTensor:
    """True = 可见。shape (T, T)"""
    chunk_idx = torch.arange(T) // C
    return chunk_idx.unsqueeze(1) >= chunk_idx.unsqueeze(0)
```

### 4.2 LM Backbone（SmolLM2-360M）

**职责**：纯文本语言模型，估计语言先验 p(x_t | x_{1:t-1})，输出 LM hidden state 和 LM logits。

**关键**：LM 全程只看文本 token，不接触视觉特征（保持语言先验纯粹性）。

```
输入: text_ids  (B, T)           # 训练时为 GT x*（teacher forcing）
                                  # 推理时为 x̂_{1:t-1}（自回归）
         │
         ▼
  smollm2.model(input_ids=text_ids,
                attention_mask=padding_mask)
         │
         ▼
  last_hidden_state  (B, T, 960)
         ├──────────────────────────────→ h̃_t^LM (B, T, 960)  ← FM 条件
         │
         ▼
  smollm2.lm_head(...)           # Linear(960, 49152)
         │
         ▼
  s_LM: (B, T, 49152)            ← LM logits，log p(x_t | x_{1:t-1})
```

- **训练**：`input_ids = x*` (GT, 右移 1 步)，teacher forcing
- **推理**：自回归，每 chunk 提交 x̂_t 后追加到 `input_ids`；用 kv-cache 做增量推理
- **初始化**：从 `pretrained/smollm2-360m/` 加载，**包含** `embed_tokens` 和 `lm_head`
- **微调**：可选 LoRA (rank=16) 或全参数微调

### 4.3 Speaker Encoder

**职责**：从视频第一个 chunk（前 C=6 帧）的人脸外观估计说话人身份向量 id̂，供 FM Head 条件化使用。每段序列只运行一次，结果复用。推理时无需参考音频。

**数据来源**：预处理阶段已生成 `face.npz`（256×256 RGB uint8），与 `lip.npy` 同目录。格式为 JPEG 压缩单文件：

```python
# 读取前 C 帧（scripts/preprocess_worker.py 中已有 load_face_jpeg 工具函数）
f = np.load("face.npz")
data, offsets = f["data"], f["offsets"]
# 按帧随机读取：frame i = data[offsets[i]:offsets[i+1]] → cv2.imdecode
frames = [cv2.cvtColor(
              cv2.imdecode(np.frombuffer(data[offsets[i]:offsets[i+1]], np.uint8),
                           cv2.IMREAD_COLOR),
              cv2.COLOR_BGR2RGB)
          for i in range(min(C, len(offsets)-1))]  # (C, 256, 256, 3)
face_chunk = np.stack(frames).mean(axis=0)           # (256, 256, 3) float32 均值
```

```
输入: face_chunk  (B, 3, 256, 256)      # 第一 chunk 的 C 帧人脸均值，float32 [0,1]
         │
         ▼
  face_encoder (ArcFace R50，frozen)     # 预训练人脸识别，输入 112×112 或 256×256
         │  内部 resize + 归一化
         ▼
  face_embed: (B, 512)                  # ArcFace 原始 L2-norm embedding
         │
         ▼
  Linear(512, 256)                      # 投影到 FM 条件维度
         │
         ▼
  id̂: (B, 256)                          ← 说话人身份向量
```

- **训练/推理一致**：均读 `face.npz` 前 C 帧，无音频依赖，分布完全一致
- **冻结**：Speaker Encoder 参数全程冻结，不参与梯度更新
- **无额外预处理**：`face.npz` 在现有预处理流程中已生成，无需修改 `preprocess_lrs3.py`

### 4.4 PoE Text Decoder

**职责**：以 Product of Experts 方式融合视觉 logits 和 LM logits，得到文本后验 logits，用于 CE 训练和自回归解码。

```python
# 训练时（有 GT x*）
posterior_logits = s_vis + alpha * s_lm   # (B, T, V)，Product of Experts

loss_text = F.cross_entropy(
    posterior_logits.reshape(-1, V)[valid],
    frame_labels.reshape(-1)[valid],
)

# 推理时（Greedy decode）
x_hat = (s_vis + alpha * s_lm).argmax(dim=-1)   # (B, T)
```

- `alpha`：LM 权重超参，初始值 1.0，在验证集上搜索 WER 最优值
- `valid`：padding mask，`True` 表示有效帧
- SIL token 不 ignore，让模型学习停顿
- 可选 label smoothing α_ls=0.1

### 4.5 FM Head（Flow Matching）

**职责**：以连续特征三元组 (ṽ_t, h̃_t^LM, id̂) 为条件，生成 Mimi latent 序列。**音频路径完全绕过离散 token 决策**。

```
条件构造（stop-gradient + 时间对齐 + 投影）:
  ṽ_t    (B, T, 960)  → .detach() → [:, ::2, :]  → (B, T_a, 960)   sg
  h̃_t^LM (B, T, 960)  → .detach() → [:, ::2, :]  → (B, T_a, 960)   sg
  id̂     (B, 256)     → unsqueeze(1).expand(T_a)  → (B, T_a, 256)

  concat → (B, T_a, 960+960+256=2176)
  Linear(2176, 512)
  → cond (B, T_a, 512)

训练时前向（OT-CFM）:
  x_1 = latent                    # (B, T_a, 512) 目标
  t   ~ Uniform(0, 1)
  x_0 ~ N(0, I)
  x_t = (1-t) x_0 + t x_1         # OT 直线插值
  t_emb = sinusoidal(t)           # (B, D_t)

  DiT (4-6层):
    输入: x_t    (B, T_a, 512)
    条件: cond   (B, T_a, 512)     # adaLN-Zero 注入
          t_emb (B, D_t)           # adaLN-Zero 注入
    输出: v_θ    (B, T_a, 512)     # 预测速度场

  L_fm = ||v_θ - (x_1 - x_0)||²   # OT-CFM loss
```

**DiT 单层结构**（adaLN-Zero）：

```
x: (B, T_a, 512)
         │
  LayerNorm(x)  ──adaLN(t_emb + pool(cond))──→ scale/shift/gate
         │
  Self-Attention（双向，FM 不需要因果）
         │
  gate_attn × x  + residual
         │
  LayerNorm  ──adaLN──→
         │
  FFN (MLP)
         │
  gate_ffn × x  + residual
```

**推理（Euler solver，NFE=10）**：

```python
x = torch.randn(B, T_a, 512)
dt = 1.0 / nfe
for step in range(nfe):
    t = torch.full((B,), step / nfe)
    v = fm_head(x, cond, t)
    x = x + dt * v
return x  # predicted latent (B, T_a, 512)
```

---

## 5. 完整 Forward Pass

```python
class StreamLip(nn.Module):
    def forward(self, lip, text_ids, latent, frame_labels, mask):
        # lip:          (B, T, 3, 96, 96)
        # text_ids:     (B, T)            GT token IDs（teacher forcing）
        # latent:       (B, T_a, 512)     T_a = T // 2
        # frame_labels: (B, T)            MFA 对齐的帧级 token labels
        # mask:         (B, T)            True = 有效帧

        # ── 路径一：文本后验 ───────────────────────────────────────────
        # 1. 视觉编码
        vis_feat, s_vis = self.visual_encoder(lip)   # (B,T,960), (B,T,V)

        # 2. 语言模型（纯文本，不看视觉）
        h_lm, s_lm = self.lm_backbone(text_ids, mask)  # (B,T,960), (B,T,V)

        # 3. PoE 后验 logits
        posterior = s_vis + self.alpha * s_lm           # (B, T, V)

        # 4. CE Loss
        valid = mask.reshape(-1)
        loss_text = F.cross_entropy(
            posterior.reshape(-1, self.vocab_size)[valid],
            frame_labels.reshape(-1)[valid],
        )

        # ── 路径二：音频生成 ───────────────────────────────────────────
        # 5. 说话人身份（序列内常量，可提前缓存）
        id_vec = self.speaker_encoder(lip)              # (B, 256)

        # 6. FM 条件构造（stop-gradient 隔离）
        v_down = vis_feat.detach()[:, ::2, :]           # (B, T_a, 960)
        h_down = h_lm.detach()[:, ::2, :]               # (B, T_a, 960)
        id_exp = id_vec.unsqueeze(1).expand(-1, v_down.size(1), -1)  # (B, T_a, 256)
        cond = self.cond_proj(
            torch.cat([v_down, h_down, id_exp], dim=-1)  # (B, T_a, 2176)
        )                                                # (B, T_a, 512)

        # 7. FM Loss
        loss_fm = self.fm_head(cond, latent)             # OT-CFM

        # ── 合并 Loss ──────────────────────────────────────────────────
        loss_total = loss_fm + self.lambda_text * loss_text

        return {
            "loss":       loss_total,
            "loss_fm":    loss_fm.detach(),
            "loss_text":  loss_text.detach(),
            "posterior":  posterior,        # (B,T,V) 可用于解码 x̂_t
        }
```

---

## 6. 训练配置

### 6.1 Loss 权重

```python
lambda_text = 0.005   # 来自 UniVoice 消融：FM task 更难
alpha       = 1.0     # PoE LM 权重，在验证集上搜索 WER 最优值（无需重训）
```

### 6.2 梯度流控制

| 模块 | CE Loss 梯度 | FM Loss 梯度 |
|------|-------------|-------------|
| AV-HuBERT | frozen | frozen |
| Conformer Adapter | ✓ | ✗（sg 隔离）|
| Visual Head | ✓ | ✗（sg 隔离）|
| LM Backbone | ✓ | ✗（sg 隔离）|
| Speaker Encoder | frozen | frozen |
| FM Head | ✗ | ✓ |
| cond_proj | ✗（输入已 sg） | ✓ |

### 6.3 超参初值

| 超参 | 值 | 来源 |
|------|----|------|
| batch_size | 16/GPU | FM 显存占用更大 |
| window_frames T | 150 | ~6s，dataset.py 默认 |
| optimizer | AdamW | β=(0.9, 0.95), wd=0.1 |
| lr（Conformer, Visual Head, LM）| 2e-4 | cosine decay |
| lr（FM head, cond_proj）| 2e-4 | 从头训练 |
| warmup_steps | 2000 | linear warmup |
| total_steps (phase 1) | 50k-100k | |
| gradient_clip | 1.0 | |
| mixed precision | bfloat16 | |
| nfe_train | 1 | OT-CFM 标准 |
| nfe_inference | 10 | Euler solver |
| cfg_scale (inference) | 2.5 | |

### 6.4 阶段训练策略

**Phase 1**（Week 3-4）：验证 PoE 文本路径的流式唇读 WER

```
AV-HuBERT:          frozen
Conformer Adapter:  trainable
Visual Head:        trainable
LM Backbone:        trainable（或 LoRA rank=16）
Speaker Encoder:    frozen（预训练）
FM Head:            不训练
Loss:               λ · L_CE(s_vis + α·s_LM, x*)
```

验收标准：LRS3-TED 验证集 WER ≤ 50%（Whisper-large-v3 评估）

**Phase 2**（Week 5-6）：接入 FM Head 端到端训练

```
AV-HuBERT:          frozen
Conformer Adapter:  frozen（或小 lr 微调）
Visual Head:        frozen（或小 lr 微调）
LM Backbone:        frozen（stop-gradient 保证 FM loss 不更新）
Speaker Encoder:    frozen
FM Head + cond_proj: trainable（从头训练）
Loss:               L_FM + λ · L_CE
```

---

## 7. 推理流水线（流式，per-chunk）

```python
state = {
    "lm_past_kv":    None,        # SmolLM2 kv-cache
    "lm_input_ids":  [BOS],       # 已提交文本 token 序列
    "last_audio":    zeros(240ms),# overlap-add 缓冲
}
id_vec = speaker_encoder(full_video)   # 一次性估计，(1, 256)

for chunk_frames in chunked(video_stream, C=6):
    # 1. 视觉编码（chunk-bidir attention 内完成）
    vis_feat, s_vis = visual_encoder(chunk_frames)  # (1, C, 960), (1, C, V)

    # 2. LM 推进（增量，仅用文本 kv-cache）
    h_lm, s_lm, state.lm_past_kv = lm_backbone_incremental(
        state.lm_input_ids[-C:],    # 当前 chunk 对应的历史 token（长度 C）
        state.lm_past_kv
    )                               # (1, C, 960), (1, C, V)

    # 3. Greedy decode（可选输出，用于 WER 监控）
    x_hat = (s_vis + alpha * s_lm).argmax(-1)       # (1, C)
    state.lm_input_ids.extend(x_hat[0].tolist())

    # 4. FM 条件构造
    T_a = C // 2  # = 3
    v_down = vis_feat[:, ::2, :].detach()            # (1, 3, 960)
    h_down = h_lm[:, ::2, :].detach()               # (1, 3, 960)
    id_exp = id_vec.unsqueeze(1).expand(-1, T_a, -1) # (1, 3, 256)
    cond = cond_proj(torch.cat([v_down, h_down, id_exp], -1))  # (1, 3, 512)

    # 5. FM 采样（NFE=10，约 20-30ms）
    pred_latent = euler_solve(fm_head, cond, T_a)    # (1, 3, 512)

    # 6. Mimi 解码
    up = mimi.upsample(pred_latent.transpose(1, 2))  # (1, 512, C)
    audio_chunk = mimi.decoder(up)                   # (1, 1, 5760) = 240ms

    # 7. Overlap-add（50ms 交叉淡化）
    yield overlap_add(audio_chunk, state.last_audio)
    state.last_audio = audio_chunk
```

**延迟组成（每 chunk 240ms）**：
- 前瞻等待：240ms（等 chunk 完整到达）
- Visual Encoder（含 Conformer）：~10ms
- LM 推进（GQA + kv-cache）：~5ms
- FM 采样（NFE=10）：~20-30ms
- Mimi 解码：~5ms
- **端到端延迟预估：~270ms**（接近 250ms 目标）

---

## 8. 文件结构（待实现）

```
src/streaminlip/
├── models/
│   ├── __init__.py
│   ├── visual_encoder.py    # AV-HuBERT + Conformer Adapter + Visual Head
│   ├── lm_backbone.py       # SmolLM2 wrapper（input_ids 模式）
│   ├── speaker_encoder.py   # face crop → id̂ (256d)
│   ├── fm_head.py           # DiT + OT-CFM（含 cond_proj）
│   └── streamlip.py         # 组装完整模型 + PoE 融合
├── training/
│   ├── __init__.py
│   ├── trainer.py           # 训练循环
│   └── losses.py            # OT-CFM loss
└── inference/
    ├── __init__.py
    └── streaming.py         # 流式推理 pipeline（per-chunk）
```

---

## 9. 待确认事项

| # | 问题 | 影响范围 | 优先级 |
|---|------|----------|--------|
| 1 | LM Backbone 是否使用 LoRA（rank=16）还是全参微调；LRS3 数据量下全参可能过拟合 | LM Backbone | 高 |
| 2 | ~~Speaker Encoder 方案~~ **已定**：ArcFace R50，取第一 chunk 人脸均值，推理无需音频 | - | ✓ |
| 3 | PoE alpha 初始值：训练时固定 alpha=1.0，还是也作为可学习参数（temperature scaling）| PoE | 中 |
| 4 | frame_labels 的粒度：MFA 帧级对齐（词/音素）vs chunk 级平均（每 chunk 一 token）；两种方式对 LM 先验质量影响不同 | 数据/训练 | 中 |
| 5 | Visual Head 权重是否与 SmolLM2 lm_head.weight tied（节省参数，但两者角色不同）| Visual Head | 低 |
| 6 | CFG（Classifier-Free Guidance）在推理时的 cfg_scale 消融（1.0/1.5/2.5）| FM Head | 低 |

---

## 10. 版本记录

| 日期 | 变更 |
|------|------|
| 2026-05-16 | 初始版本，基于 DATA_DESIGN.md + 实测预训练模型 I/O 完成 |
| 2026-05-21 | **架构重写**：基于 theory.md 因果推导。SmolLM2 改为纯文本输入；拆分 Visual Head；新增 Speaker Encoder；文本预测改用 Product of Experts；FM 条件化改为 (ṽ_t, h̃_t^LM, id̂) 三元组；梯度隔离规则更新 |
