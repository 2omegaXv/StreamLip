# 流式 V2A 唇语语音合成 —— 文献调研报告

> 核心关注：前人在 V2A（唇部视频→语音）合成中做到了什么程度？特别关注：流式/实时生成、联合解码（同时输出音频+文本）、AR+FM 联合训练。

---

## 一、V2A (Lip-to-Speech) 核心论文

### 1.1 早期多任务学习范式

| 论文 | 年份/会议 | 核心方法 | 关键结论 |
|------|-----------|----------|----------|
| **Lip2Speech: Multi-task Learning** | ICASSP 2023 | 多任务学习同时预测音频+文本，文本分支提供语义监督 | 首次证明文本辅助监督能显著提升 wild 环境下语音内容的准确性 |
| **Towards Accurate L2S in-the-Wild** | 2024 (arXiv 2403.01087) | 利用预训练唇读模型推断文本，再将文本作为条件生成语音 | 在 LRW/LRS2/LRS3 上大幅超越前人方法 |

- **与你的项目的关系：** Lip2Speech (ICASSP 2023) 是**最早将"联合输出文本+音频"用于 L2S 的工作**，直接验证了你方案中"联合解码"思路的可行性。但它只用文本做辅助 loss，并未做到真正的 joint decoding（文本隐状态作为音频生成的 conditioning）。
- 代码：https://github.com/ms-dot-k/Lip-to-Speech-Synthesis-in-the-Wild
- 论文：https://arxiv.org/abs/2302.08841

---

### 1.2 文本引导扩散生成

| 论文 | 年份/会议 | 核心方法 | WER (LRS3) |
|------|-----------|----------|------------|
| **LipVoicer** | **ICLR 2024** | 扩散模型生成 mel + 唇读分类器引导 (Classifier Guidance) | 当时 SOTA |
| **LipSody** | 2026.02 (arXiv) | 在 LipVoicer 基础上加入韵律一致性（pitch/energy/emotion） | 韵律指标大幅提升 |

**LipVoicer** 是该方向里程碑式工作：
- 核心思路：先用 VSR 模型从唇部视频推断文本，再用文本作为 Classifier Guidance 引导扩散模型生成 mel-spectrogram
- 关键发现：**文本引导对于解决 viseme→phoneme 一对多歧义至关重要**（恰好对应你项目中 LLM 语义先验的 motivation）
- 局限：**非流式、非自回归**，需要完整视频输入
- 代码：https://github.com/yochaiye/LipVoicer
- 论文：https://openreview.net/forum?id=ZZCPSC5OgD

**LipSody** (2026) 发现 LipVoicer 等扩散模型的韵律一致性差，提出用说话人身份、语言内容、面部情感三种 cue 显式估计 pitch 和 energy。
- 论文：https://arxiv.org/abs/2602.01908

---

### 1.3 Flow Matching 范式

| 论文 | 年份/会议 | 核心方法 | 亮点 |
|------|-----------|----------|------|
| **V2SFlow** | **ICASSP 2025** | Rectified Flow Matching + 语音分解 (content/pitch/speaker) | MOS 甚至超过 GT（因为去除了背景噪声） |
| **SLD-L2S** | 2026.02 (arXiv) | 层级子空间潜在扩散 + reparameterized flow matching | LRS3 上感知质量新 SOTA；直接在 neural codec 潜空间生成 |
| **LipDiffuser** | WASPAA 2025 | MP-ADM 扩散 + MP-FiLM 视觉调制 | 语音质量和说话人相似度超越前人，WER 可竞争 |

**V2SFlow** 是**第一个将 Flow Matching 用于 L2S 的工作**：
- 将语音信号分解为 content、pitch、speaker 三个子空间，分别从视觉输入预测
- 用 Rectified Flow Matching + DiT 从噪声映射到目标 mel
- 局限：仍为**全局非自回归生成**，非流式
- 论文：https://arxiv.org/abs/2411.19486

**SLD-L2S** 是**目前 L2S 领域最新 SOTA** (2026.02)：
- 在 pre-trained neural audio codec (X-Codec) 的连续潜空间直接做扩散
- 引入层级子空间分解 + DiCB（Diffusion Convolution Block）
- 多目标训练：flow matching loss + semantic loss + SLM loss
- 论文：https://arxiv.org/abs/2602.11477

---

### 1.4 多模态对齐 / 视频配音

| 论文 | 年份/会议 | 核心方法 | 亮点 |
|------|-----------|----------|------|
| **AlignDiT** | **ACM MM 2025** | DiT 架构 + 多模态 (video/text/ref-audio) in-context learning | 无需外部 forced aligner；多模态 CFG 自适应平衡各模态 |
| **DiFlowDubber** | 2026.03 (arXiv) | Discrete Flow Matching + 两阶段 TTS→dubbing 迁移 | 首个端到端 discrete flow matching 视频配音框架 |
| **MAVFlow** | 2025.03 (arXiv) | CFM + 双模态 (audio+visual) 引导零样本 AV2AV 翻译 | 保持说话人特征的跨语言音视频翻译 |

**AlignDiT** 与你的项目最相关：
- 在 DiT 框架中**同时处理视频、文本和参考音频**三种模态
- 提出 multimodal classifier-free guidance：推理时自适应平衡各模态的贡献
- **隐式学习跨模态对齐**，不依赖外部时间对齐工具
- 代码：https://github.com/kaistmm/AlignDiT
- 论文：https://arxiv.org/abs/2504.20629

---

## 二、联合解码（同时输出音频+文本）

这是你项目的**核心创新方向**之一。以下是前人在"同时输出语音和文本"方面的工作：

### 2.1 Visatronic (Apple, 2024)

**Visatronic: A Multimodal Decoder-Only Model for Speech Synthesis**
- 提出 **VTTS 任务 (Video-Text-to-Speech)**：从说话视频+文本联合生成语音
- **统一的 decoder-only Transformer**，将视觉、文本、语音作为时间对齐的 token 流
- 用**自回归 loss 生成离散化的 mel-spectrogram**
- LRS3 上 WER=4.5%，VoxCeleb2 上 WER=12.2%
- **关键洞察：** 研究了异质采样率模态的同步机制，位置编码策略如何实现同步
- 论文：https://arxiv.org/abs/2411.17690
- 代码：https://github.com/apple/visatronic-demo

> **与你项目的关系：** Visatronic 是**最接近你方案的前人工作** —— 它也在一个 decoder-only 模型内同时处理视频+文本→语音。但它的输入需要 GT 文本，而你的方案是**模型自己预测文本**（联合解码），这是关键差异。

### 2.2 Moshi (Kyutai, 2024) —— 对话式全双工模型

**Moshi: A Speech-Text Foundation Model for Real-Time Dialogue**
- **全双工流式对话**：同时建模"系统说"和"用户说"两路音频流
- **Inner Monologue 机制：** 在生成音频 token 前先预测时间对齐的文本 token —— 这恰好对应你方案中"模型内部同时输出文本和音频"的思路
- 7B 参数 Temporal Transformer + 小型 Depth Transformer
- 使用 Mimi streaming codec，端到端延迟约 **200ms**
- 训练 4 阶段：预训练→多流后训练→全双工微调→指令微调
- 论文：https://arxiv.org/abs/2410.00037
- 代码：https://github.com/kyutai-labs/moshi

> **与你项目的关系：** Moshi 的 **Inner Monologue** 是你"联合解码中文本分支作为语义先验"最直接的前人验证。它证明了：让模型先预测文本 token 再预测音频 token，能显著提升生成语音的质量和事实性。但 Moshi 是 audio→audio 对话模型，不是 video→audio。

### 2.3 UniVoice (2025)

**UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS**
- **在同一 LLM 框架中统一 ASR（自回归）和 TTS（Flow Matching）**
- 核心技术：**Dual Attention Mechanism** —— ASR 用 causal mask，TTS 用不同的 mask
- 基座：SmolLM2-360M + Whisper encoder + BigvGAN vocoder
- 零样本 TTS：WER 降低 12%，UTMOS 提升 0.4（vs. 其他统一模型）
- 论文：https://arxiv.org/abs/2510.04593
- 代码：https://github.com/gwh22/UniVoice

> **与你项目的关系：** UniVoice **直接在一个模型内结合了 AR (离散文本) + FM (连续语音)** 的双分支训练，并设计了 dual attention 来解决两种训练范式的冲突。这与你方案中的 AR+FM 联合训练高度一致，是最重要的技术参考之一。

---

## 三、流式/实时生成

你项目的另一核心创新是**流式（streaming）V2A**。以下是前人在流式音频/视频生成方面的进展：

### 3.1 SoundReactor (Sony, 2025) —— 最相关的流式 V2A

**SoundReactor: Frame-level Online Video-to-Audio Generation**
- **首个提出"帧级在线 V2A 生成"任务的工作**
- 核心约束：**端到端因果性** —— 不允许访问未来视频帧
- 架构：decoder-only causal transformer + diffusion head
- 视觉：DINOv2 每帧聚合为一个 token（保持因果性）
- 音频：自训 VAE 将 48kHz 立体声编码为 30Hz 连续潜变量
- 延迟：**26.3ms/帧**（NFE=1）
- 局限：针对游戏音效等通用 V2A，**非唇语语音合成**
- 论文：https://arxiv.org/abs/2510.02110（提交 ICLR 2026 后撤回）
- 项目页：https://koichi-saito-sony.github.io/soundreactor/

> **与你项目的关系：** SoundReactor 的因果 AR + diffusion head 架构与你的方案非常相似，但它做的是通用音效而非语音。你的项目如果实现了，将是**首个将此范式应用于唇语语音合成的工作**。

### 3.2 StreamFlow (NeurIPS 2025) —— 流式 Flow Matching

**StreamFlow: Streaming Audio Generation from Discrete Tokens via Streaming Flow Matching**
- **因果化的 Flow Matching**：沿时间轴做 causal noising，每次预测多个时间步的 vector field
- Scale-DiT 架构提升鲁棒性
- 成功替换了 Moshi 的 Mimi 解码器
- 论文：https://openreview.net/forum?id=1cURNMriee

> **与你项目的关系：** 证明了 Flow Matching 可以被改造为流式生成范式，你的 FM 解码器可以参考其 causal noising 策略。

### 3.3 VSSFlow (Apple, 2025-2026) —— 统一 V2S + VisualTTS

**VSSFlow: Unifying Video-conditioned Sound and Speech Generation**
- 在 DiT 架构中统一 Video-to-Sound 和 Visual Text-to-Speech
- **解耦条件聚合：** cross-attention 处理语义条件，self-attention 处理时序条件
- 关键发现：**联合训练 sound + speech 两个任务反而互相提升**，无需复杂的训练策略
- 论文：https://arxiv.org/abs/2509.24773

> **与你项目的关系：** 证明了在统一框架中联合学习多个音频生成任务是可行的，且互相受益。

---

## 四、相关技术基础

### 4.1 视觉语音识别 (VSR) —— 为联合解码提供文本能力

| 模型 | 年份 | WER (LRS3) | 方法 |
|------|------|------------|------|
| AV-HuBERT | 2022 | ~26.9% | 自监督预训练 |
| Auto-AVSR | 2023 | ~19.1% | 端到端音视频 ASR |
| VALLR | ICCV 2025 | 18.7% (beam) | LLaMA 3.2 + phoneme |
| CAV2vec | ICLR 2025 | SOTA (带噪) | 多任务腐蚀预测 |

### 4.2 流式 TTS —— 延迟参考基准

| 系统 | 首包延迟 | 年份 | 特点 |
|------|----------|------|------|
| StreamMel | 10ms | 2025 | 极致低延迟 |
| SpeakStream | 30ms+15ms | 2025 | Mac M4 Pro 本地运行 |
| VoXtream | 102ms | 2025 | 增量式 phoneme transformer |
| Moshi | ~200ms | 2024 | 全双工对话，含理解 |

---

## 五、前人空白与你项目的定位

通过以上调研，**明确的学术空白** 如下：

| 维度 | 现有工作做到了什么 | 没做到什么（你的机会） |
|------|-------------------|----------------------|
| **生成范式** | L2S 全部为非自回归/全局生成（LipVoicer, V2SFlow, SLD-L2S 等） | **无人做过流式/自回归的 L2S** |
| **文本利用方式** | LipVoicer: 外挂 VSR→文本→classifier guidance；Lip2Speech: 文本做辅助 loss | **无人在 L2S 中做过真正的联合解码（模型同时 AR 输出文本 + FM 输出音频）** |
| **流式 V2A** | SoundReactor 做了通用流式 V2A（游戏音效）| **无人做过流式唇语→语音合成** |
| **前瞻窗口** | SoundReactor 为零前瞻（完全因果）；流式 TTS 有 chunk-based 方法 | **"短前瞻窗口 + 语义先验"解决 viseme 歧义的组合策略无人尝试过** |
| **AR+FM 联合训练** | UniVoice 在 ASR+TTS 上验证可行；Moshi 用 Inner Monologue 先文本后音频 | **无人在 V2A/L2S 场景下结合 AR+FM 联合训练** |

### 你项目的核心贡献可总结为：

1. **首个流式 L2S 系统**：将 SoundReactor 式的因果 AR 框架引入唇语语音合成
2. **前瞻窗口 + 语义先验的组合**：比 SoundReactor 的零前瞻更合理（语音比通用音效更需要上下文），比 LipVoicer 的全局生成更实时
3. **联合解码范式**：借鉴 Moshi 的 Inner Monologue 和 UniVoice 的 dual attention，在一个网络内同时输出文本和音频，文本隐状态直接作为 FM 头的 conditioning
4. **Audio Chunking**：类似流式 TTS 的 chunk-based 生成，一次输出 200ms 音频块

---

## 六、推荐重点精读论文（按优先级）

1. **LipVoicer** (ICLR 2024) —— L2S + 文本引导扩散的基线 [[paper]](https://openreview.net/forum?id=ZZCPSC5OgD) [[code]](https://github.com/yochaiye/LipVoicer)
2. **V2SFlow** (ICASSP 2025) —— L2S + Flow Matching 的基线 [[paper]](https://arxiv.org/abs/2411.19486)
3. **Visatronic** (Apple, 2024) —— 最接近你方案的 Video+Text→Speech decoder-only 模型 [[paper]](https://arxiv.org/abs/2411.17690) [[code]](https://github.com/apple/visatronic-demo)
4. **Moshi** (Kyutai, 2024) —— Inner Monologue = 你的"文本先验" [[paper]](https://arxiv.org/abs/2410.00037) [[code]](https://github.com/kyutai-labs/moshi)
5. **UniVoice** (2025) —— AR+FM 在同一模型中联合训练的技术方案 [[paper]](https://arxiv.org/abs/2510.04593) [[code]](https://github.com/gwh22/UniVoice)
6. **SoundReactor** (Sony, 2025) —— 在线因果 V2A 框架参考 [[paper]](https://arxiv.org/abs/2510.02110)
7. **SLD-L2S** (2026) —— 当前 L2S SOTA [[paper]](https://arxiv.org/abs/2602.11477)
8. **StreamFlow** (NeurIPS 2025) —— 流式 Flow Matching 技术参考 [[paper]](https://openreview.net/forum?id=1cURNMriee)
9. **AlignDiT** (ACM MM 2025) —— 多模态 DiT 对齐参考 [[paper]](https://arxiv.org/abs/2504.20629) [[code]](https://github.com/kaistmm/AlignDiT)
10. **VSSFlow** (Apple, 2025) —— 联合训练 sound+speech 的统一框架 [[paper]](https://arxiv.org/abs/2509.24773)
