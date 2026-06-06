# 精读笔记：流式 V2A 相关论文

---

## 1. LipVoicer (ICLR 2024) — L2S + 文本引导扩散基线

### 核心思路
训练时**不用文本**，推理时用文本做 classifier guidance。

### 架构
- **MelGen (扩散模型)**：基于 DiffWave 残差骨架的条件 DDPM
  - 输入条件：裁剪嘴唇视频 V_L + 随机选取的全脸图像 I_F（提供说话人信息）
  - 视频编码：3D Conv → ShuffleNet v2 → TCN → 嘴部嵌入 m ∈ R^{N×D_m}
  - 人脸嵌入：ResNet-18（去最后两层）→ f ∈ R^{D_f}，复制 N 次后与 m 拼接得 v ∈ R^{N×D}
  - 训练：标准 DDPM，用 classifier-free guidance（有/无条件 v）
- **唇读模型**：Ma et al. (2023) 预训练，LRS2 WER=14.6%，LRS3 WER=19.1%
- **ASR 分类器**：Audio-Visual Efficient Conformer（加了扩散时间步嵌入），在 LRS2/LRS3 微调
- **Vocoder**：DiffWave

### 推理公式（核心）
修正噪声 = CFG噪声 + ASR classifier guidance：
```
ε̂ = ε_mg(x_t, V_L, I_F, w1) - w2 · γ_t · √(1-ᾱ_t) · ∇_{x_t} log p(t_LR | x_t)
```
其中：
- `ε_mg = (1+w1)·ε_θ(x_t, V_L, I_F) - w1·ε_θ(x_t)` （CFG）
- `γ_t = ||ε_mg|| / (√(1-ᾱ_t) · ||∇ log p(t_LR|x_t)||)` （梯度归一化）
- `t_LR` 是唇读模型预测的文本

### 关键消融
| 配置 | WER (LRS3) |
|------|-----------|
| w2=0（无 ASR guidance） | **86.2%** |
| w2=1.5（最终配置） | **21.4%** |
| 使用 GT 文本替代唇读 | **5.4%** |

**结论：文本引导将 WER 从 86% 降到 21%，是可理解语音生成的决定性因素。**

### 结果 (LRS3)
| 指标 | LipVoicer | GT |
|------|-----------|-----|
| WER | 21.4% | 1.0% |
| STOI-Net | 0.92 | 0.93 |
| DNSMOS | 3.11 | 3.30 |
| MOS-Intelligibility | 3.44 | 4.38 |
| MOS-Naturalness | 3.52 | 4.45 |

### 对你项目的启示
1. **文本对 L2S 至关重要** —— 验证了你"联合解码输出文本"的 motivation
2. **外挂方式的局限**：需要单独的唇读模型 + 单独的 ASR + 梯度归一化，推理复杂且慢（DDPM 采样步数多）
3. **全局非自回归**：需要完整视频输入，无法流式

---

## 2. V2SFlow (ICASSP 2025) — L2S + Flow Matching

### 核心思路
将语音分解为 content/pitch/speaker 三个子空间，分别从视觉预测，再用 RFM 解码器重建。

### 架构
- **视觉编码器**：AV-HuBERT (Large)，冻结参数
- **三个属性编码器**（均为 Conformer）：
  - Content Encoder → 预测 HuBERT K-means 离散 token，CE loss + label smoothing (α=0.1)
  - Pitch Encoder → 预测 VQ-VAE pitch token，同上
  - Speaker Encoder → 时间维平均 → cosine similarity loss
- **RFM Speech Decoder**：8层 Transformer, hidden=512, 4 heads
  - 条件 c = concat(content_token, pitch_token, speaker_emb)
  - 用 adaLN-Zero 注入时间步 t
  - RFM 目标：`L = ||v(x_t, t|c; θ) - (x1 - x0)||²`
  - x_t = (1-t)x0 + t·x1，t ∈ [0,1]
  - CFG：训练时 10% drop all condition；推理用 Euler solver 30 步，γ=2
  - 输出：80-bin mel-spectrogram (hop=160, win=640, 16kHz)

### 训练细节
- 各编码器独立训练 50k steps，batch=144s/GPU × 8 GPU
- 解码器独立训练 400k steps，batch=384s/GPU × 1 GPU
- 时间步采样：logit-normal sampling
- Vocoder：HiFi-GAN

### 结果 (LRS3-TED)
| 方法 | UTMOS↑ | WER↓ | SECS↑ | MAE_F0↓ | MOS-Nat↑ |
|------|--------|------|-------|---------|----------|
| GT | 3.519 | 2.5 | - | - | 4.42 |
| V2SFlow-V | **3.780** | **28.5** | 0.664 | **0.251** | **4.60** |
| DiffV2S | 2.989 | 39.2 | 0.627 | 0.290 | 3.28 |

**V2SFlow UTMOS 和 MOS-Naturalness 超过 GT**（因为去除了背景噪声）。

### 关键消融
- 去掉 content token → WER 暴增至 106.4%（最关键属性）
- 去掉 pitch token → MAE_F0 上升但 WER 不变
- 去掉 speech decomposition → 全面下降
- RFM vs DDIM（30步）：RFM 在所有指标上优于 DDIM

### 对你项目的启示
1. **RFM + DiT 是当前 L2S 最优生成范式**，30 步就能得到高质量
2. **语音分解（content/pitch/speaker）是有效的** —— 你可以让 AR 头预测 content token，FM 头生成音频
3. **局限：全局生成，非流式；WER=28.5% 远不及有文本引导的 LipVoicer 21.4%**

---

## 3. Visatronic (Apple, 2024/2026) — Video+Text→Speech Decoder-Only

### 核心思路
提出 **VTTS 任务**（Video-Text-to-Speech）：给定说话视频+文本转录，用统一 decoder-only Transformer 生成语音。

### 架构
- **视频表示**：VQ-VAE (Kinetics-600预训练) → 每帧编码为 16×16 spatial grid → 离散token → embedding → 空间求和聚合为单向量 z_t^v
- **文本表示**：字符级 tokenizer → embedding
- **语音表示**：dMel 离散化 (4-bit, 80 mel-filterbank channels) → embedding → 线性层投射
- **说话人**：d-vector (512维) → 线性投射
- **统一 decoder**：单个 Transformer，用 cross-entropy loss L_CE 只在语音 token 上计算
  - 输入序列：[Speaker] [Text tokens] [Video tokens] [Speech tokens]（多种排列策略）
  - 位置编码：RoPE

### 关键发现：模态排列策略
| 策略 | WER (VoxCeleb2) | WER (LRS3 zero-shot) |
|------|----------------|---------------------|
| TV-CoTemporal | 14.1 | **17.9** |
| VT-Scaled | **12.2** | 28.9 |
| Video-Causal-Streaming | 14.5 | - |
| TTS (text only) | 14.7 | 27.8 |

- **TV-CoTemporal**：视频和语音共享 position ID（共时间轴），**跨域泛化最好**
- **VT-Scaled**：文本在前+视频时间缩放的 prefix，**域内性能最好**
- **Video-Causal-Streaming**：文本先、然后视频和语音交错（因果），**适合流式**

### 模态消融
| 配置 | WER |
|------|-----|
| VT-Scaled (full) | 12.2% |
| 去掉 Text | 74.5% |
| 去掉 Video | 46.4% |

**两种模态缺一不可，文本对内容更关键，视频对时序和情感更关键。**

### 训练细节
- 数据集：VoxCeleb2 (1.6k小时, 6k说话人)，Demucs 去噪 + Whisper-large-v2 伪标注
- 随机 span masking（对视频、文本、语音以概率 p 遮蔽）
- 训练 2M-3M iterations

### 对你项目的启示
1. **最接近你方案的前人工作**，但输入需要 GT 文本
2. **Video-Causal-Streaming 排列证明了 decoder-only 可以做因果/流式 VTTS**
3. **Co-temporal position ID 是跨域泛化的关键** —— 视频和语音共享位置 ID 比 global sequential 好
4. **dMel 离散化是可行的语音表示方案**（无需额外 vocoder 训练，只需 off-the-shelf vocoder）
5. **你的差异化：模型自己预测文本而非输入 GT 文本**

---

## 4. Moshi (Kyutai, 2024) — 全双工流式对话 + Inner Monologue

### 核心思路
端到端流式语音对话模型，**同时建模用户和系统两路音频流 + 系统侧的文本流（Inner Monologue）**。

### 架构
- **Helium**：7B 参数 text LLM (RoPE, GatedLinearUnits, SiLU, RMSNorm, 32层, 32头, context=4096)
- **Mimi**：流式 neural audio codec
  - SeaNet 编码器 + RVQ (Q=8, N_A=2048, 12.5Hz, 1.1kbps)
  - **Split RVQ**：第一层 VQ 蒸馏 WavLM 语义信息 + 7层 RVQ 声学信息，并行而非级联
  - 全因果（encoder + decoder 都是因果卷积），80ms 帧延迟
  - 纯对抗训练（去掉重建 loss，只用 feature loss + discriminator loss）→ 主观质量更好
- **RQ-Transformer**：
  - **Temporal Transformer**（大，=Helium 7B）：建模时间维度，每步输入 = K 个子序列 embedding 的和
  - **Depth Transformer**（小，6层1024维16头）：建模 codebook 维度，预测每个时间步的 K 个 token

### Inner Monologue（核心机制）
每个时间步 s 的联合序列 V_s 结构（自底向上）：
```
文本 token → Moshi 语义 token → Moshi 声学 tokens (×7, 延迟 τ=1) → 用户语义 token → 用户声学 tokens (×7)
```
- 文本 token 作为**每帧的前缀**，在 Depth Transformer 中**最先被预测**
- 训练时用 forced alignment 将文本与音频帧对齐，未对齐的帧填 PAD
- 文本延迟：预训练时 ±0.6 随机偏移，后续阶段固定为 0

### Inner Monologue 的效果
- **大幅提升语音质量和语言学正确性**
- 允许从 Moshi 模型**直接派生出 streaming ASR 和 streaming TTS**（通过控制文本-音频延迟）

### 训练阶段（4阶段）
1. **预训练**：无监督音频数据（Temporal Transformer 从 Helium 初始化），1M steps, batch=1.2M tokens
2. **后训练**：模拟多流（基于 diarization），100k steps
3. **Fisher 微调**：全双工对话数据，10k steps
4. **指令微调**：合成交互脚本，30k steps

### 延迟
- 理论：160ms（Mimi 80ms帧 + 80ms声学延迟）
- 实际：~200ms (L4 GPU)

### 关键超参
| 组件 | 参数量 | 关键设计 |
|------|--------|---------|
| Temporal Transformer | 7B | 32层, 4096维, 32头 |
| Depth Transformer | ~300M | 6层, 1024维, 16头, per-codebook 参数 |
| Mimi | ~100M | Split VQ + 7层RVQ, 纯对抗训练 |
| 音频帧率 | 12.5 Hz | 8 codebook × 12.5 = 100 tokens/s |
| 文本帧率 | ~3-4 tokens/s | 远低于音频 |

### 对你项目的启示
1. **Inner Monologue 是你"联合解码中文本作为语义先验"的最直接前人验证**
   - Moshi 证明：先预测文本 token 再预测音频 token，能显著提升质量
   - 你可以将此范式从 audio→audio 移植到 video→audio
2. **Split RVQ 分离语义和声学是实用的 codec 设计**
3. **Depth Transformer 处理 codebook 维度 + Temporal Transformer 处理时间维度** 是高效的层级生成方案
4. **200ms 延迟是可接受的** —— 与你的 200ms 前瞻窗口刚好匹配
5. **4阶段训练是工业级做法** —— 你的课程项目可以简化

---

## 5. UniVoice (2025) — ASR+TTS 统一框架 (AR + FM)

### 核心思路
在同一 LLM 中**用连续表示统一 ASR（自回归）和 TTS（Flow Matching）**。

### 架构
- **基座**：SmolLM2-360M（小型 LLM）
- **ASR 分支**：Whisper-large-v3-turbo encoder + adaptive avg pooling adapter → LLM → AR 文本解码
- **TTS 分支**：文本 token 序列 + 参考音频 + masked speech → LLM → Flow Matching 解码 mel
  - 用 in-context learning 替代 adaLN-zero（保持 LLM 原始结构）
  - 文本作为 prefix，speech features 在其后
  - Flow step t 通过正弦位置编码插入 text 和 speech 之间
- **Vocoder**：BigvGAN

### Dual Attention（核心技术）
解决 AR 和 FM 的注意力矛盾：
- **ASR**：标准 causal mask（因果）
- **TTS**：bidirectional attention mask（双向）—— 这是关键差异！

训练时根据任务切换 mask 类型。

### 统一训练目标
```
L_total = λ · L_LM(θ) + L^cfm_audio(θ)
```
- L_LM：AR 交叉熵 loss（ASR）
- L^cfm_audio：OT-CFM loss（TTS），在 masked speech 区域计算
- **λ = 0.005** 是最优（注意：ASR loss 权重极小！）

### λ 消融
| λ | TTS WER↓ | TTS UTMOS↑ | ASR WER-c↓ |
|---|----------|-----------|-----------|
| 0.01 | 4.66 | 3.69 | 4.21 |
| **0.005** | **4.06** | **3.72** | **3.01** |

**TTS（FM）任务更难，需要更高权重；ASR 更简单，小权重即可。**

### Attention Mask 消融
| Mask | TTS WER↓ | SIM↑ | UTMOS↑ |
|------|----------|------|--------|
| AR Mask | 9.85 | 0.49 | 2.23 |
| **Full (Bidirectional)** | **4.66** | **0.56** | **3.92** |

**双向 attention 对 TTS 质量至关重要** —— 但这意味着 TTS 不是流式的。

### 结果
| 模型 | 类型 | Params | WER↓ | UTMOS↑ | ASR WER-c↓ |
|------|------|--------|------|--------|-----------|
| UniVoice | 统一 | 0.4B | **4.06** | **3.72** | **3.0** |
| F5-TTS | TTS only | 0.3B | 2.54 | 3.84 | - |
| Whisper-turbo | ASR only | 0.8B | - | - | 1.9 |

统一模型存在轻微 trade-off，但参数效率极高。

### 对你项目的启示
1. **λ=0.005 是平衡 AR+FM 的关键** —— FM 需要更高权重
2. **双向 attention 对生成质量很重要**，但与你的流式需求冲突 → 你可能需要**有限窗口的双向 attention**（前瞻窗口内双向）
3. **连续表示优于离散** —— 直接在 mel 空间做 FM 比离散 codec token 更好
4. **text-prefix-guided speech infilling** 是实用的 TTS 范式，可适配到你的 V2A 场景

---

## 6. SoundReactor (Sony, 2025) — 帧级在线 V2A

### 核心思路
首个**帧级在线 V2A**：因果自回归 Transformer + Diffusion Head，逐帧生成音频，不访问未来视频帧。

### 架构
三组件：
1. **Video Token Modeling**
   - DINOv2-small (21M) 提取 grid (patch) features V̂_i ∈ R^{H'×W'×c}
   - 与前一帧做时间差分：concat(V̂_i, V̂_i - V̂_{i-1})
   - 2D conv 下采样 → flatten → 前置 learnable aggregation token → shallow Transformer aggregator → 输出 **v_i（单个 token/帧）**
   - **完全因果**：只用当前帧和前一帧
2. **Audio Token Modeling**
   - 自训 VAE（非 RVQ！）：48kHz stereo → 30Hz 连续潜变量
   - **每帧一个连续 token**（vs 离散 codec 需要多个 token/帧）
3. **Multimodal AR Transformer + Diffusion Head**
   - LLaMA-style decoder-only (RMSNorm, SwiGLU, RoPE)
   - 输入：交错的 [v_1, x_1, v_2, x_2, ...] 序列
   - 输出条件 z_i = Concat(z̄_{2i-1}, z̄_{2i})（前一音频位置和当前视频位置的 Transformer 输出）
   - **Diffusion Head**：每帧独立的小型扩散模型，从 z_i 生成 x_i

### 训练（2阶段）
1. **Stage 1: Diffusion pretraining** (EDM2 框架)
   ```
   L = E[λ(t)·e^{-u_θ(t)} · Σ||x_i^0 - D_θ(x_i^t, t, z_i)||² + u_θ(t)]
   ```
   - u_θ(t)：可学习的不确定性函数
2. **Stage 2: ECT fine-tuning**（Easy Consistency Tuning）
   - 将 DM 蒸馏为 CM，减少采样步数
   - Δt 渐进退火到 0

### 推理
- CFG：ω=3.0，随机替换 v_i 为 null embedding
- NFE=1~4 步即可（ECT 后）

### 延迟
| NFE | Token-level | Waveform-level |
|-----|-------------|----------------|
| 1 | 24.3ms | **26.3ms** |
| 4 | 28.6ms | **31.5ms** |

30FPS = 33.3ms/帧，ECT NFE≤4 都可实时。

### 结果 (OGameData250K)
- 在 FAD、MMD、IB-Score、DeSync 上全面超越 V-AURA
- 主观测试：AV-Sem=65.2, AV-Temp=64.3 (vs V-AURA 42.5, 50.5)

### 对你项目的启示
1. **AR + Diffusion Head 是在线 V2A 的可行范式**
2. **每帧一个连续 token（VAE）比多个离散 token（RVQ）更适合流式**
3. **DINOv2 grid features + temporal difference 是因果视觉编码的实用方案**
4. **ECT 蒸馏能将 NFE 降到 1-4 步，实现 <33ms/帧延迟**
5. **关键差异：SoundReactor 做通用音效（游戏），不做语音** → 不需要解决同形音/语境依赖问题
6. **你的项目 = SoundReactor 架构 + 语音特有的文本先验 + 前瞻窗口**

---

## 7. SLD-L2S (2026.02) — 当前 L2S SOTA

### 核心思路
在 **neural audio codec 的连续潜空间**直接做层级子空间潜在扩散。

### 架构
- **视觉前端**：AV-HuBERT Large（冻结），输出 1024 维特征
- **Subspace Decomposition Module (SDM)**：
  - 将 1024 维视觉特征分到 **8 个并行子空间**
  - 每个子空间：LayerNorm + 1D Conv → 独立处理
- **DiCB (Diffusion Convolution Block)**：12 层
  - **卷积注意力**（替代 self-attention）：depthwise conv (kernel 5×7) + Hadamard product
  - **卷积 FFN**：kernel=3
  - 条件注入：AdaLN-SOLA（共享 AdaLN + 低秩调整矩阵 β_α，rank=32）
- **Subspace Recomposition Module (SRM)**：
  - 每个子空间投影到 128 维 → concat → 1024 维 → ConvNeXt blocks (3层, 512 hidden)
- **Audio Codec**：X-Codec-hubert (16kHz)

### Reparameterized Flow Matching（关键创新）
不直接预测速度场 v_θ，而是预测**目标数据点 x_1**：
```
d_θ(x_t, c_s, t) = x_t + (1-t)·v_θ(x_t, c_s, t)     （Eq.4）
```
Reparameterized loss（加权 MSE）：
```
L_FM = E[ ||d_θ(x_t, c_s, t) - x_1||² / (1-t)² ]     （Eq.5）
```
**(1-t)² 加权使得接近目标时（t→1）梯度更大**

### 多目标训练
```
L = L_FM + λ_1 · L_SLM + λ_2 · L_sem     （Eq.8）
```
- **L_SLM**：将预测和目标分别解码为波形 → 过 SLM (WavLM) → L2 距离
- **L_sem**：将预测和目标的 codec latent → 过语义解码器 D_s（HuBERT-base 特征空间）→ L2 距离
- λ_1 = 1, λ_2 = 100

### 训练细节
- AdamW, lr=2e-4, cosine decay, batch=16, 150K iters, 单卡 H100, ~5天
- 说话人嵌入：GE2E, 256维
- 推理：Euler solver, **NFE=10**（极高效！）

### 结果 (LRS3-TED)
| 方法 | NFE | UTMOS↑ | SCOREQ↑ | WER↓ | SECS↑ |
|------|-----|--------|---------|------|-------|
| GT | - | 3.572 | 3.810 | - | 0.71 |
| LipVoicer | 400 | 2.454 | 2.660 | **20.62** | 0.588 |
| V2SFlow | 30 | 3.694 | 4.071 | 27.55 | **0.851** |
| **SLD-L2S** | **10** | **4.210** | **4.608** | 30.22 | 0.804 |

- **感知质量 (UTMOS/SCOREQ) 新 SOTA**，大幅超过所有前人
- WER 略高于 LipVoicer（因为不用 VSR 模型），但 MOS 自然度远超（4.17 vs 2.51）
- **只需 10 步推理**

### 消融
- 去掉 DiCB → 全面崩溃（DiT 替代效果差）
- 去掉 Reparameterized FM → UTMOS/SCOREQ 下降
- 去掉 L_sem → 质量/说话人相似度轻微下降
- 去掉 L_SLM → 声学指标微升但 D-BERT/SECS 下降（内容和身份保持退化）
- **8 子空间最优**（4/16/32 都更差）

### 对你项目的启示
1. **在 codec 潜空间做 FM 比在 mel 空间做更好** —— 你可以考虑用 audio codec
2. **Reparameterized FM（预测 x_1 而非 v）更稳定**，且允许加入辅助 loss
3. **DiCB（卷积替代 attention）对序列长度不敏感** —— 适合流式（不需要全局 attention）
4. **SLM loss + Semantic loss 的组合是提升内容正确性的有效方法**
5. **NFE=10 已足够高质量** —— 你的 audio chunking 每 200ms 跑一次 FM 是可行的

---

## 横向对比总结

| 维度 | LipVoicer | V2SFlow | Visatronic | Moshi | UniVoice | SoundReactor | SLD-L2S |
|------|-----------|---------|------------|-------|----------|--------------|---------|
| **生成范式** | DDPM | RFM | AR (离散) | AR (离散) | AR+FM | AR+DM | RFM |
| **流式** | ✗ | ✗ | ✓(Streaming变体) | ✓ | ✗ | ✓ | ✗ |
| **文本利用** | 推理时 CG | ✗ | 输入 GT 文本 | Inner Monologue | prefix text | ✗ | ✗ |
| **输出空间** | mel | mel | 离散 dMel | 离散 codec | mel (FM) | 连续 VAE latent | codec latent |
| **视觉编码** | 3DConv+TCN | AV-HuBERT | VQ-VAE | - | - | DINOv2 | AV-HuBERT |
| **L2S WER** | 21.4% | 28.5% | 12.2%(有GT文本) | - | - | - | 30.2% |
| **感知质量** | 低 | 高 | 中 | - | - | - | 最高 |
| **NFE/步数** | ~1000 | 30 | AR | AR | 32 | 1-4 | 10 |

---

## 你项目的技术路线建议（基于精读）

### 视觉编码器
- **AV-HuBERT**（V2SFlow, SLD-L2S 验证）或 **DINOv2**（SoundReactor 验证，更适合因果）
- 如需因果：DINOv2 + temporal difference（SoundReactor 方案）
- 前瞻窗口内可用小范围双向 attention

### 文本分支（AR 头）
- 借鉴 **Moshi Inner Monologue**：文本 token 作为每帧前缀，先于音频 token 预测
- 或借鉴 **Visatronic**：decoder-only 统一处理 video+text+speech token 序列

### 音频分支（FM 头）
- 借鉴 **SoundReactor**：每帧/每 chunk 独立的 diffusion/FM head
- 在 **audio codec latent 空间**生成（SLD-L2S 验证最优）
- Reparameterized FM + 辅助 semantic/SLM loss（SLD-L2S 方案）

### Loss 平衡
- **λ_AR ≈ 0.005**（UniVoice 验证）
- 或 stop-gradient 隔离（你和 Gemini 讨论的 VLA 策略）

### Audio Chunking
- 每 200ms（前瞻窗口）为一个 chunk
- FM head 一次生成整个 chunk 的 codec latent
- Euler solver NFE=10（SLD-L2S）或 ECT NFE=1-4（SoundReactor）
