# StreamLip 项目行动规划

> **项目名称**: StreamLip: Streaming Lip-to-Speech Synthesis via Joint Text-Audio Decoding with Flow Matching  
> **当前状态**: Proposed（Proposal 已完成，尚未开始实现）  
> **时间预算**: 9 周  
> **核心目标**: 在 LRS3-TED 上实现首个流式唇语语音合成系统，WER ≤ 40%，UTMOS ≥ 3.0，延迟 ≤ 250ms

---

## 阶段总览

| 阶段 | 周次 | 核心目标 | 退路方案 |
|------|------|----------|----------|
| 0. 环境与数据准备 | Week 1-2 | 跑通数据流水线与基础组件 | — |
| 1. AR Backbone + 文本头 | Week 3-4 | 流式唇读模型，验证文本先验可行性 | 若唇读完全不行，转纯视觉流式方案 |
| 2. FM 头联合训练 | Week 5-6 | 端到端流式 L2S，能出声 | 若联合解码失败，退回纯流式视觉→FM（退路A） |
| 3. 消融实验与调优 | Week 7-8 | 完整实验表格，验证各设计选择 | — |
| 4. 论文撰写与展示 | Week 9 | Final report + Poster | — |

---

## 阶段 0: 环境与数据准备 (Week 1-2)

### 0.1 数据获取与预处理

- [ ] **下载 LRS3-TED 数据集**
  - 包含 ~400h 视频，需申请 academic license
  - 若获取困难，备选 LRS2 (224h) 或 VoxCeleb2 (2.4kh, 无GT文本需Whisper伪标注)
- [ ] **视频预处理流水线**
  - 人脸检测 + 嘴唇裁剪 (96×96)，使用 RetinaFace / MediaPipe
  - 25fps 帧提取，灰度/RGB 标准化
  - 参考 AV-HuBERT 官方预处理脚本
- [ ] **音频预处理**
  - 16kHz 重采样
  - 通过预训练 audio codec (X-Codec 或 Mimi) 提取连续 latent 作为训练目标
  - 验证 codec 的重建质量 (PESQ, UTMOS)
- [ ] **文本对齐**
  - 使用 Montreal Forced Aligner (MFA) 获取 word/phoneme-level 时间戳
  - 将文本 token 与视频帧对齐（每帧分配 0~1 个 token，未对齐帧填 PAD）
  - 参考 Moshi 的文本-音频对齐方案

### 0.2 预训练模型准备

- [ ] **AV-HuBERT Large** (视觉编码器)
  - 下载预训练权重，验证在 LRS3 上的特征质量
  - 冻结参数，仅训练适配层
- [ ] **Audio Codec** (X-Codec-hubert 或 Mimi)
  - 下载/训练，验证重建质量
  - 确认 latent 维度、帧率 (如 Mimi: 12.5Hz, 1024维)
- [ ] **SmolLM2-360M** (可选，作为 AR backbone 初始化)
  - 下载权重，测试推理速度
  - 评估是否需要 LoRA 适配

### 0.3 实验基础设施

- [ ] **训练环境搭建**
  - GPU 配置确认 (目标: 1-2×H100 或等效)
  - PyTorch 2.x + FlashAttention + DeepSpeed/FSDP
  - WandB / TensorBoard 实验追踪
- [ ] **评估流水线**
  - WER: Whisper-large-v3 作为 ASR 评测模型
  - UTMOS: 预训练 MOS 预测模型
  - SECS: 预训练说话人编码器 (GE2E / ECAPA-TDNN)
  - LSE-C: SyncNet 唇音同步评估
  - 延迟: 计时工具

### 0.4 阶段 0 验收标准

- [x] LRS3 数据已预处理，DataLoader 可正常出批
- [x] AV-HuBERT 可提取视觉特征 (shape 验证)
- [x] Audio Codec 可编码/解码，重建质量达标
- [x] 文本强制对齐完成，对齐质量抽查通过
- [x] 评估流水线端到端跑通 (用 GT 音频验证)

---

## 阶段 1: AR Backbone + 文本头 (Week 3-4)

### 1.1 核心目标

训练一个**流式视觉条件语言模型** (Streaming Visual-Conditioned LM)，即"视觉唇读+语言模型先验"的联合体。这是整个系统的语义核心。

### 1.2 架构实现

- [ ] **Chunk-level Bidirectional Visual Encoder**
  - AV-HuBERT 输出 → Conformer Adapter (2-4层)
  - Chunk 内 (1+Δ=6帧) 双向 attention，chunk 间因果
  - 输出: 每帧 1024维 → 投影到 backbone 维度
- [ ] **Causal AR Backbone**
  - Decoder-only Transformer: Gemma-3-1B，LoRA rank=16 微调
avhubert    - 备选（待速度实验对比）：Qwen3-1.7B、SmolLM2-1.7B、Qwen3-0.6B
  - 输入: 交错的 [visual_feat, text_embedding] 序列
  - 因果 mask (只看过去 + 当前 chunk 的前瞻)
- [ ] **Text Head**
  - Linear projection → vocabulary logits
  - CE Loss with label smoothing (α=0.1)
  - 词表: BPE (SentencePiece, vocab=1000-4000) 或字符级

### 1.3 训练配置

```python
# 关键超参 (初始值)
batch_size = 32  # 每 GPU
max_seq_len = 600  # ~24s video at 25fps
lr = 2e-4  # AdamW, cosine decay
warmup_steps = 2000
total_steps = 50000-100000
lookahead_frames = 5  # 200ms
```

### 1.4 验证实验

- [ ] **流式唇读 WER 评估**
  - 目标: WER ≤ 50% on LRS3 test (比全局 VSR 19% 差是预期的)
  - 对比:
    - 无前瞻 (Δ=0) vs 有前瞻 (Δ=5)
    - 有/无 LM 初始化 (Gemma-3-1B LoRA vs random init)
- [ ] **文本质量分析**
  - 逐句 WER 分布，识别失败模式
  - 同形音错误 (buy/by, there/their) 是否被 LM 先验解决
- [ ] **关键决策点**: 如果 WER > 60% 且无改善趋势 → 考虑退路方案

### 1.5 阶段 1 验收标准

- [x] 流式唇读模型收敛，WER ≤ 50%
- [x] 前瞻窗口 vs 无前瞻有明显差距 (验证设计有效性)
- [x] LM 初始化 vs 随机初始化有差距 (验证语言先验价值)
- [x] 模型能实时推理 (单帧处理 < 40ms)

---

## 阶段 2: FM 头联合训练 (Week 5-6)

### 2.1 核心目标

在阶段 1 的流式唇读模型之上，接入 Flow Matching 头，实现端到端的流式语音生成。

### 2.2 FM Head 实现

- [ ] **轻量 DiT (4-6层)**
  - 输入: 噪声 latent x_t ∈ R^D (D = codec latent 维度)
  - 条件: backbone hidden state h_i (通过 adaLN-Zero 注入)
  - 时间步 t: 正弦编码 → adaLN
  - 输出: 预测速度场 v_θ 或 目标 x_1 (reparameterized FM)
- [ ] **Audio Chunking**
  - 每 200ms (5帧@25fps) 为一个 audio chunk
  - FM head 一次生成一个 chunk 的所有 codec latent
  - Euler solver, NFE=10 (参考 SLD-L2S)
- [ ] **Overlap-add 合成**
  - 相邻 chunk 重叠 50ms + linear crossfade
  - 验证边界平滑度

### 2.3 联合训练策略

- [ ] **Stop-gradient 实现**
  ```python
  h_i_detached = h_i.detach()  # FM 不更新 backbone
  loss_fm = flow_matching_loss(fm_head, h_i_detached, x_1, t)
  loss_ar = ce_loss(text_head(h_i), gt_text)
  loss_total = loss_fm + loss_ar  # λ=1 for AR (sole backbone driver)
  ```
- [ ] **Scheduled Corruption**
  - p_mask = 0.1: 以 10% 概率将 h_i 替换为零向量
  - p_noise = 0.05: 以 5% 概率添加高斯噪声
  - 使 FM 头对不完美文本先验鲁棒
- [ ] **CFG 训练**
  - 10% 概率 drop 所有条件 (h_i → null)
  - 推理时 CFG scale γ=2-3

### 2.4 训练配置

```python
# Phase 2 超参
lr_fm_head = 2e-4       # FM head 从头训练
lr_backbone = 0         # Backbone 冻结 (stop-grad)
batch_size = 16         # 每 GPU (FM 更占显存)
total_steps = 50000-100000
nfe_train = 1           # 训练时单步 (CFM 标准做法)
nfe_inference = 10      # 推理时 10 步
cfg_scale = 2.5
```

### 2.5 阶段 2 验收标准

- [x] 系统能从视频生成可听的语音 (主观判断)
- [x] WER ≤ 45% (有文本先验应优于纯视觉方法)
- [x] UTMOS ≥ 2.5 (基本自然度)
- [x] 单帧延迟 ≤ 50ms (不含前瞻等待时间)
- [x] 音频 chunk 边界无明显断裂

---

## 阶段 3: 消融实验与调优 (Week 7-8)

### 3.1 核心消融实验

| # | 实验 | 对比 | 验证目标 |
|---|------|------|----------|
| 1 | StreamLip (full) vs w/o text head | 去掉文本分支，纯视觉→FM | 文本先验的贡献 |
| 2 | StreamLip (full) vs w/o look-ahead | Δ=0 纯因果 | 前瞻窗口的贡献 |
| 3 | StreamLip (full) vs w/ GT text | 用 GT 文本隐状态替代预测 | 性能上界 |
| 4 | StreamLip (full) vs Chunked V2SFlow | V2SFlow 按 chunk 独立处理 | 公平流式 baseline |
| 5 | Stop-gradient vs 固定 λ=0.005 | UniVoice 式平衡 vs VLA 式隔离 | 最优训练策略 |
| 6 | NFE=4/10/20 | 不同推理步数 | 质量-延迟 trade-off |
| 7 | Chunk size 100/200/500ms | 不同生成粒度 | 最优 chunk 大小 |

### 3.2 Baseline 构造

- [ ] **Chunked V2SFlow**: 复现 V2SFlow 但每 chunk 独立处理 (不看未来)
- [ ] **SoundReactor-L2S**: SoundReactor 架构直接用于 LRS3 (无文本)
- [ ] **V2SFlow (offline)**: 完整视频输入的离线上界参考

### 3.3 超参调优

- [ ] CFG scale 搜索: γ ∈ {1.5, 2.0, 2.5, 3.0, 4.0}
- [ ] Look-ahead 帧数: Δ ∈ {0, 3, 5, 7, 10}
- [ ] Scheduled corruption 比例: p_mask ∈ {0.05, 0.1, 0.15, 0.2}
- [ ] FM head 深度: 2/4/6 层 DiT

### 3.4 分析维度

- [ ] **序列长度 vs 质量衰退**: 2s/5s/10s/20s 的 WER 曲线
- [ ] **错误类型分析**: 同形音错误 vs 语义错误 vs 完全乱码
- [ ] **延迟 profiling**: 各组件耗时 breakdown
- [ ] **失败案例可视化**: 文本预测错误如何传播到音频

### 3.5 阶段 3 验收标准

- [x] 完整消融表格 (7 组实验)
- [x] WER ≤ 40% (最终目标)
- [x] UTMOS ≥ 3.0 (最终目标)
- [x] 延迟 ≤ 250ms (最终目标)
- [x] 文本头消融证明 WER 有显著改善 (Δ ≥ 5%)

---

## 阶段 4: 论文撰写与展示 (Week 9)

### 4.1 Report 结构

1. **Abstract** (已有初稿)
2. **Introduction**: 动机 + 前人空白 + 三点贡献
3. **Related Work**: L2S 方法 / 流式生成 / 联合解码
4. **Method**: 问题定义 → 架构 → 训练策略 → 推理流水线
5. **Experiments**: 主表 + 消融 + 分析
6. **Discussion**: Trade-off 分析 + 局限性 + 未来方向
7. **Conclusion**

### 4.2 关键写作要点

- [ ] 核心叙事: "我们定义了流式 L2S 新任务，提出首个解决方案，并通过系统消融验证设计"
- [ ] 不需要 beat 离线 SOTA，但需要 convincing ablation
- [ ] 诚实讨论延迟-质量 trade-off
- [ ] Poster: 架构图 + 主表 + 1-2个关键消融 + demo 二维码

### 4.3 Demo 准备

- [ ] 录制 3-5 个视频的流式生成 demo (对比 GT)
- [ ] 延迟实时展示 (可选)

---

## 风险管理清单

| 风险 | 等级 | 触发条件 | 应对措施 |
|------|------|----------|----------|
| 流式唇读 WER 过高 | 高 | WER > 60% 且不收敛 | 增大前瞻窗口 / 换更大 LM / 退路 A |
| AR 错误累积 | 高 | 长序列 (>5s) 质量崩溃 | 增大 scheduled corruption / 定期重置文本历史 |
| FM 因果质量差 | 中高 | UTMOS < 2.5 | 增大 chunk size / 参考 StreamFlow causal noising |
| Chunk 边界断裂 | 中 | 主观听测不可接受 | 增大 overlap / 用 infilling 范式 |
| Loss 不收敛 | 中 | FM/AR 其一不下降 | 确认 stop-gradient 正确 / 调整 lr |
| 算力不足 | 中 | 训练过慢 | 减小模型 / 减少训练步数 / 简化为退路 A |
| 数据获取困难 | 低-中 | LRS3 申请被拒 | 转用 LRS2 或 VoxCeleb2 + Whisper 伪标注 |

---

## 退路方案详情

### 退路 A: 纯流式视觉→FM (无文本头)

**保留**: 首个流式 L2S + 前瞻窗口  
**放弃**: 联合解码  
**实现难度**: 低 (去掉文本分支即可)  
**预期效果**: WER ~35-50% (参考 V2SFlow 28.5% 但加流式约束会更差)  
**学术价值**: 仍有首篇流式 L2S 的贡献

### 退路 B: 非流式联合解码

**保留**: AR+FM 联合训练验证  
**放弃**: 流式生成  
**实现难度**: 中 (全局 attention，更易训练)  
**预期效果**: WER ~20-30% (接近 LipVoicer)  
**学术价值**: AR+FM 在 L2S 的首次验证

### 退路 C: 流式 + 外挂 VSR

**保留**: 流式 + 文本引导  
**放弃**: 端到端联合解码  
**实现难度**: 中 (需要额外 VSR 模型)  
**预期效果**: WER ~25-35%  
**学术价值**: 流式文本引导 L2S

---

## 关键里程碑检查点

| 时间点 | 检查项 | Go/No-Go 决策 |
|--------|--------|---------------|
| Week 2 末 | 数据流水线跑通? | 若数据无法获取 → 换数据集 |
| Week 4 末 | 流式唇读 WER < 55%? | 若 > 60% → 考虑退路 A 或增大前瞻 |
| Week 6 末 | 系统能生成可听语音? | 若完全不可听 → 切换退路方案 |
| Week 8 末 | WER < 40% 且 UTMOS > 3.0? | 若未达标 → Report 中诚实讨论 limitation |

---

## 参考资源索引

| 资源 | 用途 | 链接/位置 |
|------|------|-----------|
| LipVoicer 代码 | 文本引导参考 | github.com/yochaiye/LipVoicer |
| Moshi 代码 | Inner Monologue 参考 | github.com/kyutai-labs/moshi |
| UniVoice 代码 | AR+FM 联合训练参考 | github.com/gwh22/UniVoice |
| AlignDiT 代码 | 多模态 DiT 参考 | github.com/kaistmm/AlignDiT |
| Visatronic 代码 | Decoder-only VTTS 参考 | github.com/apple/visatronic-demo |
| AV-HuBERT | 视觉编码器 | github.com/facebookresearch/av_hubert |
| SLD-L2S 论文 | 当前 SOTA 方法 | arxiv.org/abs/2602.11477 |
| StreamFlow 论文 | 流式 FM 技术 | openreview.net/forum?id=1cURNMriee |
