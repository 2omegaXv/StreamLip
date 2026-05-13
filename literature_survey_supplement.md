# 文献调研补充：2025-2026 遗漏工作

> 本文档补充初次调研中遗漏的关键论文，重点关注：LLM 用于唇读解码、层级 V2S 表示、因果/流式 VSR。

---

## 一、L2S (Lip-to-Speech) 补充

### 1.1 From Faces to Voices (CVPR 2025, Highlight)

**Hierarchical Representations for High-quality Video-to-Speech**

- 作者：Ji-Hoon Kim, Jeongsoo Choi 等（KAIST，V2SFlow/SLD-L2S 同一团队）
- 论文：https://arxiv.org/abs/2503.16956
- 项目页：https://mm.kaist.ac.kr/projects/faces2voices/

**核心方法：**
- 提出层级视觉编码器，将视频逐步映射到声学空间：
  - Stage 1 (Content): 唇部运动 → 语言内容表示（对齐 HuBERT token）
  - Stage 2 (Timbre): 面部身份 → 说话人音色嵌入（对齐 speaker embedding）
  - Stage 3 (Prosody): 面部表情 → 韵律特征（对齐 pitch/energy）
- **Flow Matching 解码器**：Optimal Transport 条件流，从层级视觉表示生成 mel-spectrogram
- 结果：LRS3 上质量接近真实语音，显著超越 V2SFlow

**与 StreamLip 的关系：**
- 层级对齐策略可用于我们的视觉编码器——在 chunk-bidirectional 框架内分阶段提取 content/speaker/prosody
- 验证了 Flow Matching 在 L2S 中的优越性
- **局限：全局非自回归，非流式**

---

## 二、LLM-based VSR (唇读) 补充

### 2.1 VALLR (ICCV 2025)

**Visual ASR Language Model for Lip Reading**

- 作者：Marshall Thomas Edward Fish 等
- 论文：https://arxiv.org/abs/2503.21408

**核心方法：**
- 两阶段管道：
  1. Video Transformer + CTC head → 预测**音素序列**（紧凑、speaker-invariant）
  2. 音素序列 → fine-tuned LLaMA 3.2 → 重建完整句子
- 关键设计：用音素作为中间表示，而非直接从视频预测文字

**关键结果 (LRS3)：**

| 方法 | WER↓ | 标注数据量 |
|------|------|----------|
| VALLR (Llama 3.2-3B) | **18.7%** | 30h |
| 去掉音素中间表示 | 25.6% | 30h |
| Auto-AVSR | 19.1% | 1759h |

**核心发现：**
- 音素中间表示将 WER 从 25.6% 降到 18.7%（降低 27%）
- 仅用 30h 标注数据就超越了用 1759h 数据训练的 Auto-AVSR
- LLM 的语言推理能力弥补了视觉信息的不足

**对 StreamLip 的重大启示：**
- **文本头应先预测音素而非直接预测文字**
- 音素与唇动的映射更直接（viseme→phoneme 歧义远小于 viseme→word）
- 音素序列与帧率对齐更自然
- 音素隐状态作为 FM 头条件同样有效（音素与发音直接对应）

---

### 2.2 Leveraging LLMs in VSR (arXiv 2506.02012, May 2025)

**Model Scaling, Context-Aware Decoding, and Iterative Polishing**

- 作者：Zehua Liu 等（北邮 + 清华）
- 论文：https://arxiv.org/abs/2506.02012
- 代码：https://github.com/liu12366262626/VSR-LLM

**核心方法：**
- 架构：Visual Encoder (ResNet18 + 3D-CNN + Conformer) → Linear Connector → Qwen2.5 LLM (QLoRA)
- 三个贡献：
  1. **Scaling Law**：更大 LLM = 更好唇读（Transformer 49.3% → Qwen2.5-32B 42.5% CER）
  2. **Context-Aware Decoding (CAD)**：将前 30 秒的 ASR 文本作为 LLM prompt
  3. **Iterative Polishing (IP)**：多轮解码，将上一轮输出作为参考再次解码

**关键结果 (CNVSRC.Single, CER)：**

| 方法 | Valid | Test |
|------|-------|------|
| Transformer baseline | 50.53% | 49.32% |
| Qwen2.5-7B | 46.48% | 45.20% |
| + CAD | 44.58% | 43.20% |
| + IP | 45.80% | 44.75% |
| + CAID (CAD+IP) | 44.29% | **42.87%** |
| Qwen2.5-32B-Instruct | 43.23% | 41.86% |
| + CAID | 39.90% | **38.18%** |

**对 StreamLip 的启示：**
- **直接验证了"历史文本作为 LLM 上下文"能显著降低唇读错误率** —— 这正是我们 AR 文本头的工作原理
- CAD 对同形音和领域特定术语特别有效
- 论文脚注："不同类型的上下文文本效果差异不大，只要包含足够的主题指示词即可" → 我们的自预测文本即使有错误，只要保持主题一致性就有效
- **挑战**：即使 32B + 完美上下文，CER 仍有 38% → 纯唇读极其困难

---

### 2.3 Not Only Vision (ICCV 2025)

**Evolve Visual Speech Recognition via Peripheral Information**

- 会议：ICCV 2025
- 论文：https://openaccess.thecvf.com/content/ICCV2025/papers/Yuan_Not_Only_Vision_Evolve_Visual_Speech_Recognition_via_Peripheral_Information_ICCV_2025_paper.pdf

**核心方法：**
- 除唇部视频外，利用面部周边区域信息（表情、头部姿态、下巴运动等）
- Modality adapter：2× average downsampling + projection → LLM embedding space
- LLM 解码器 + LoRA 微调

**对 StreamLip 的启示：**
- 视觉编码器不应只看嘴唇 crop，面部周边信息（下巴、面颊肌肉）也有助于语音内容推断
- 可能需要扩大 crop 区域或使用全脸特征

---

### 2.4 Personalized Lip Reading (2024/2025)

**Adapting to Your Unique Lip Movements with Vision and Language**

- 论文：https://arxiv.org/abs/2409.00986

**核心方法：**
- LLaMA3-8B 作为 LLM 解码器
- 视觉编码器冻结，仅对 LLM 做 LoRA 适配（Q/K/V 层）
- 个性化：对特定说话人的唇形做适配

**对 StreamLip 的启示：**
- 说话人适配对唇读重要 → 你的模型应包含 speaker conditioning
- LoRA 微调 LLM 是高效的 VSR 适配策略

---

### 2.5 MMS-LLaMA (ACL Findings 2025)

**Efficient LLM-based Audio-Visual Speech Recognition**

- 论文：https://aclanthology.org/2025.findings-acl.1065.pdf

**核心方法：**
- AV Q-Former：跨模态注意力融合音频和视频特征
- Speech Rate Predictor：估计语速，动态分配 query 数量
- Query Allocation Strategy：根据语速给 LLM 提供不同数量的 token

**对 StreamLip 的启示：**
- **语速预测 + 动态 token 分配**是处理语音和视频帧率不匹配的优雅方案
- 你的 AR backbone 每帧输出一个 hidden state，但人类说话时可能某些帧没有新音素 → 可以引入类似的 rate predictor 决定何时跳过文本预测

---

## 三、因果/流式 VSR 补充

### 3.1 SwinLip (2025)

**Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer**

- 期刊：Neurocomputing, 2025
- 论文：https://www.sciencedirect.com/science/article/abs/pii/S0925231225009610

**核心方法：**
- 3D Spatio-Temporal Embedding Module（单层 3D CNN）编码时间特征
- 1D Convolutional Attention Module 在最后一层 Swin Transformer stage
- **关键设计：为流式 VSR/AVSR 开发因果模型**
  - 去除 self-attention 中的非因果成分
  - 去除 Batch Normalization（替换为 Layer Normalization 或 Group Normalization）
  - 使用因果卷积替代标准卷积

**对 StreamLip 的启示：**
- **最直接的流式视觉编码器设计参考**
- 因果化策略：
  1. 将 bidirectional self-attention 限制为 causal attention（或 chunk-level bidirectional）
  2. BN → LN/GN（BN 依赖 batch 统计量，不适合流式）
  3. 标准卷积 → 因果卷积（padding 只在左侧）

---

### 3.2 Towards Inclusive Communication (arXiv, March 2026)

**Unified Framework for Generating Spoken Language from Sign, Lip, and Audio**

- 论文：https://arxiv.org/abs/2508.20476

**核心方法：**
- 统一框架同时处理 SLT（手语翻译）、VSR、ASR、AVSR 四个任务
- 三个模态编码器 + Mapping Network → 统一语言表示 → LLM 解码器

**对 StreamLip 的启示：**
- 统一多任务框架的趋势 → 你的联合解码（唇读+语音合成）是这一方向的自然延伸
- Mapping Network 的设计可参考

---

## 四、综合更新：修订后的前人空白分析

| 维度 | 已有工作（含补充） | 仍未解决（StreamLip 的机会） |
|------|-----------------|---------------------------|
| **L2S 生成范式** | V2SFlow, SLD-L2S, From Faces to Voices (全部离线 FM) | **无人做过流式 L2S** |
| **文本/语言利用** | LipVoicer (外挂 CG); LLMVSR (上下文解码); VALLR (音素+LLM) | **无人让 L2S 模型内部联合输出音素/文本 + 音频** |
| **LLM for VSR** | VALLR, LLMVSR, Not Only Vision, Personalized LR (全部离线) | **无人在流式 L2S 中使用 LLM 语言先验** |
| **因果/流式 VSR** | SwinLip (因果编码器, 仅输出文本) | **无人将因果 VSR 与流式语音生成结合** |
| **中间表示** | VALLR: 音素中间表示大幅降低 WER | **无人用音素作为 L2S 中 FM 头的条件** |

---

## 五、对 StreamLip 架构的修订建议

基于补充调研，建议考虑以下修改：

### 修改 1：文本头改为音素头

VALLR 证明了 lip → phoneme → LLM → text 优于 lip → text。建议：

```
原方案：  Visual Features → AR Backbone → Text Token (word-level)
修订方案：Visual Features → AR Backbone → Phoneme Token → (可选) LM Layer → Word Token
```

好处：
- 音素与帧率对齐更自然（~10-15 phonemes/sec vs ~3 words/sec）
- viseme→phoneme 歧义远小于 viseme→word
- 音素隐状态与发音动作直接对应，作为 FM 头的条件更有效
- 训练数据需求更低（VALLR 仅用 30h 数据就达到 SOTA）

### 修改 2：视觉编码器扩大感受野

Not Only Vision (ICCV 2025) 证明面部周边信息有助于 VSR：

```
原方案：  仅嘴唇 crop → AV-HuBERT
修订方案：下半脸 crop (含下巴/面颊) → AV-HuBERT + 额外 facial expression 特征
```

### 修改 3：层级视觉对齐

From Faces to Voices (CVPR 2025) 证明层级对齐有效：

```
原方案：  Visual Features → 直接作为 AR backbone 输入
修订方案：Visual Features → Content Align → Timbre Align → Prosody Align → AR backbone
```

### 修改 4：语速感知 token 生成

MMS-LLaMA 的 Speech Rate Predictor 启发：

```
原方案：  每帧固定输出 1 个音素 token + 1 个音频 chunk
修订方案：Speech Rate Predictor 决定当前帧是否需要生成新音素，避免静默帧产生冗余 token
```

---

## 六、2026 年 AR+Diffusion 联合生成工作

以下工作虽非直接做 L2S，但其 AR+Diffusion/FM 联合架构为 StreamLip 的训练策略提供了最新的 2026 年支撑。

### 6.1 MAViD (arXiv 2512.03034, 2026.03)

**A Multimodal Framework for Audio-Visual Dialogue Understanding and Generation**

- 论文：https://arxiv.org/abs/2512.03034

**核心方法：**
- AR + Diffusion 联合网络：
  - AR 模块负责语义级别的文本/对话理解和规划
  - Diffusion 模块负责连续信号（音频+视频）的高保真生成
- 可生成 ~30 秒的同步音视频对话内容
- 联合训练策略：AR 模块的隐状态作为 Diffusion 模块的条件

**对 StreamLip 的启示：**
- **直接验证了 AR+Diffusion 联合架构在多模态生成中的有效性（2026 年最新）**
- 你的方案（AR 文本头 + FM 音频头）与 MAViD 的 AR+Diffusion 设计同构
- 证明 AR 隐状态作为扩散/流匹配条件是可行且有效的

---

### 6.2 Apollo (arXiv 2601.04151, 2026.01)

**Unified Multi-Task Audio-Video Joint Generation**

- 论文：https://arxiv.org/abs/2601.04151

**核心方法：**
- 26B 参数统一 DiT 架构
- **Omni-Full Attention**：跨模态全注意力，所有模态 token 互相 attend
- **Random Modality Masking**：训练时随机 mask 掉 1-2 个模态，强迫模型学习跨模态推断
- **多阶段课程训练**：
  - Stage 1: 单模态预训练
  - Stage 2: 双模态联合训练
  - Stage 3: 全模态微调
- Multimodal RoPE：为不同模态设计的旋转位置编码

**对 StreamLip 的启示：**
- **Random Modality Masking ≈ 你的 Scheduled Corruption**：两者思路完全一致——通过训练时随机遮蔽/破坏某个条件，防止模型过度依赖单一模态
- 多阶段课程训练策略可参考（你的两阶段方案是简化版）
- Multimodal RoPE 的设计可能对处理视频帧率与音频帧率不匹配有用

---

### 6.3 UniTalking (arXiv 2603.01418, 2026.03)

**A Unified Audio-Video Framework for Talking Portrait Generation**

- 论文：https://arxiv.org/abs/2603.01418

**核心方法：**
- 对称双流架构：音频流 + 视频流
- **Joint-Attention**：在 Multi-Modal Transformer block 中对拼接的 audio-visual token 做联合注意力
- 多任务学习：T2AV, TV2A, TI2AV, TR2AV 四个任务交替训练
- Flow Matching 用于连续信号（音频 mel + 视频 latent）的生成

**对 StreamLip 的启示：**
- **Joint-Attention 在 AV token 上的设计**可参考——你的 AR backbone 也需要同时处理视觉 token 和文本 token
- 多任务交替训练（不同输入→输出组合）有助于学习更鲁棒的跨模态表示
- UniTalking 做的是 Text→Audio+Video（生成方向），你做的是 Video→Audio（理解方向），互补

---

### 6.4 LTX-2 (arXiv 2601.03233, 2026.01)

**Efficient Joint Audio-Visual Foundation Model**

- 论文：https://arxiv.org/abs/2601.03233

**核心方法：**
- 基座：13B 预训练视频 DiT
- 音频流：轻量 3B 参数，通过以下机制连接：
  - **Bidirectional Cross-Attention**：音频流和视频流互相 attend
  - **1D Temporal RoPE**：音频流使用 1D 时间旋转位置编码
  - **Cross-modality AdaLN**：用一个模态的特征调制另一个模态的归一化参数
- 开源，是同类中最快的模型

**对 StreamLip 的启示：**
- **Cross-modality AdaLN**是一种轻量的跨模态条件注入方式——可以用来将文本隐状态注入 FM 头（替代简单的拼接或加法）
- Bidirectional Cross-Attention 的设计可参考，但需要改为 chunk-level 以适配流式
- 轻量音频流（3B vs 13B 视频流）的不对称设计合理——你的 FM head 也应远小于 AR backbone

---

### 6.5 这些工作的共同验证

| 设计选择 | MAViD | Apollo | UniTalking | LTX-2 | StreamLip |
|---------|-------|--------|-----------|-------|-----------|
| AR + 连续生成联合 | ✓ (AR+Diff) | ✓ (DiT) | ✓ (FM) | ✓ (DiT) | ✓ (AR+FM) |
| 隐状态作为生成条件 | ✓ | ✓ | ✓ | ✓ (AdaLN) | ✓ |
| 随机模态遮蔽/corruption | - | ✓ | - | - | ✓ |
| 多阶段课程训练 | - | ✓ | ✓ (多任务) | - | ✓ (2阶段) |
| 跨模态注意力 | ✓ | ✓ (Omni) | ✓ (Joint) | ✓ (Cross) | ✓ (chunk-bidir) |

**结论：StreamLip 的 AR+FM 联合架构设计与 2026 年多个大规模多模态生成工作的方向完全一致。** 区别在于：它们做生成（T→AV），你做理解+生成（V→Text+Audio）；它们是离线的，你是流式的。

---

## 七、新增参考文献

```bibtex
@inproceedings{faces2voices,
  title={From Faces to Voices: Learning Hierarchical Representations for High-quality Video-to-Speech},
  author={Kim, Ji-Hoon and Choi, Jeongsoo and Kim, Jaehun and Jung, Chaeyoung and Chung, Joon Son},
  booktitle={Proc. CVPR},
  year={2025}
}

@inproceedings{vallr,
  title={VALLR: Visual ASR Language Model for Lip Reading},
  author={Fish, Marshall Thomas Edward and others},
  booktitle={Proc. ICCV},
  year={2025}
}

@article{llmvsr,
  title={Leveraging Large Language Models in Visual Speech Recognition: Model Scaling, Context-Aware Decoding, and Iterative Polishing},
  author={Liu, Zehua and Li, Xiaolou and Guo, Li and Li, Lantian and Wang, Dong},
  journal={arXiv preprint arXiv:2506.02012},
  year={2025}
}

@inproceedings{notonlyvision,
  title={Not Only Vision: Evolve Visual Speech Recognition via Peripheral Information},
  author={Yuan, others},
  booktitle={Proc. ICCV},
  year={2025}
}

@article{swinlip,
  title={SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer},
  author={various},
  journal={Neurocomputing},
  year={2025}
}

@inproceedings{mmslamma,
  title={Efficient LLM-based Audio-Visual Speech Recognition},
  author={various},
  booktitle={Findings of ACL},
  year={2025}
}

@article{unitalking2026,
  title={UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation},
  author={various},
  journal={arXiv preprint arXiv:2603.01418},
  year={2026}
}

@article{ltx2,
  title={LTX-2: Efficient Joint Audio-Visual Foundation Model},
  author={various},
  journal={arXiv preprint arXiv:2601.03233},
  year={2026}
}

@article{apollo2026,
  title={Apollo: Unified Multi-Task Audio-Video Joint Generation},
  author={various},
  journal={arXiv preprint arXiv:2601.04151},
  year={2026}
}

@article{mavid,
  title={MAViD: A Multimodal Framework for Audio-Visual Dialogue Understanding and Generation},
  author={various},
  journal={arXiv preprint arXiv:2512.03034},
  year={2026}
}
```
