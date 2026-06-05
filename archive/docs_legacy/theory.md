# StreamLip 概率建模理论推导

---

## 1. 因果结构

### 1.1 原始建模的问题

原始建模隐含了如下因果方向：

$$v_t \;\longrightarrow\; x_t \;\longrightarrow\; a_t$$

即视频帧 $v_t$ 驱动隐状态 $x_t$，再生成音频 $a_t$。这在因果上是颠倒的。真实的生成过程是：说话人先有语言意图 $x_t$，继而驱动发音器官做出三维运动 $h_t$，最终同时产生 2D 嘴唇外观 $v_t$ 和声学输出 $a_t$。

### 1.2 三层因果图

引入三类变量：

- $x_t$：**语言内容**（linguistic content），离散文本 token 或音素，沿时间轴构成自回归链
- $h_t$：**3D 发音运动**（articulatory motion），下颌、舌、唇等关节的三维轨迹，是 $v_t$ 和 $a_t$ 的共同物理原因
- $\text{id}$：**说话人身份**（speaker identity），捕获声道特征和面部外观，在一段话中恒定

正确的因果有向图为（以单个时间步 $t$ 为代表）：

```
x_{t-1} ───→ x_t ───→ x_{t+1}        AR chain（语言先验）
               │
               ↓
              h_t                      3D 发音运动（中间层，不可观测）
            ↙     ↘
          v_t       a_t                视觉观测 & 声学输出（均可观测）
          ↑           ↑
           ╲         ╱
              id                       说话人身份（序列内恒定）
```

**关键条件独立性**（由因果图直接读出）：

$$a_t \;\perp\; v_t \;\Big|\; h_t,\, \text{id} \qquad \text{给定发音运动和身份，音频与视频条件独立}$$

$$v_t \;\perp\; v_s \;\Big|\; h_{1:T},\, \text{id} \quad (s \neq t) \qquad \text{给定隐序列，各帧视频观测独立}$$

### 1.3 关于韵律

音高、节奏、重音等韵律属性不需要独立的潜变量。形式上，韵律 $p_t$ 被文本上下文积分掉：

$$p(a_t \mid h_t,\, \text{id}) = \int p(a_t \mid h_t,\, p_t,\, \text{id})\cdot p(p_t \mid x_{1:t})\, dp_t$$

由于 LM hidden state 编码了完整文本上下文 $x_{1:t}$（句法结构、语义重音、话语连贯性），以 LM hidden state 为条件的 FM Head 可隐式地学习 $p(p_t \mid x_{1:t})$ 所约束的韵律分布。这在 LRS3-TED 演讲风格（韵律高度规范）下尤其成立。

### 1.4 关于时序依赖

物理上 $v_{t+1}$ 和 $a_{t+1}$ 依赖于 $v_t$ 和 $a_t$。在模型中，这一依赖由各解码器隐式满足，无需显式建模为因果边：

- 音频时序连续性：Mimi decoder 的因果卷积天然引入时序记忆
- 视觉时序连续性：Conformer Adapter 的时序 attention 覆盖
- Chunk 边界：overlap-add crossfade 在波形层面平滑过渡

---

## 2. 生成模型

### 2.1 联合概率分解

将 $\text{id}$ 视为从视频中可估计的观测变量（取点估计 $\hat{\text{id}}$），联合分布分解为：

$$p(a_{1:T},\, v_{1:T},\, h_{1:T},\, x_{1:T} \mid \text{id}) =
\underbrace{p(x_1)\prod_{t=2}^{T} p(x_t \mid x_{1:t-1})}_{\text{AR prior（语言模型）}}
\cdot \prod_{t=1}^{T} \underbrace{p(h_t \mid x_t)}_{\text{发音运动}}
\cdot \prod_{t=1}^{T} \underbrace{p(v_t \mid h_t,\, \text{id})}_{\text{视觉外观}}
\cdot \prod_{t=1}^{T} \underbrace{p(a_t \mid h_t,\, \text{id})}_{\text{声学输出}}$$

### 2.2 对 $h_t$ 的边缘化

$h_t$ 不可直接观测，将其积分掉，得到**内容-身份联合条件**下的边缘分布：

$$p(a_t \mid x_t,\, \text{id}) \triangleq \int p(a_t \mid h_t,\, \text{id})\cdot p(h_t \mid x_t)\, dh_t$$

$$p(v_t \mid x_t,\, \text{id}) \triangleq \int p(v_t \mid h_t,\, \text{id})\cdot p(h_t \mid x_t)\, dh_t$$

**实践含义**：AV-HuBERT 在提取视觉特征 $\tilde{v}_t$ 时已隐式完成了对 $h_t$ 的边缘化；$\tilde{v}_t$ 是 $h_t$ 的充分统计量代理，直接用于后续推理和生成。

### 2.3 各分量的模型对应

| 分量 | 概率 | 模型 |
|------|------|------|
| AR prior | $p(x_t \mid x_{1:t-1})$ | SmolLM2（纯语言先验，不接触视觉）|
| 视觉似然 | $p(v_t \mid x_t)$ | AV-HuBERT（边缘化 $h_t$ 后经 Bayes 反用）|
| 音频生成 | $p(a_t \mid x_t,\, \text{id})$ | FM Head（$h_t$ 隐式，见 §7）|
| 身份估计 | $\hat{\text{id}} = \mathcal{E}(v_{1:T})$ | Speaker Encoder（GE2E / d-vector）|

---

## 3. 推理问题：给定视频，生成音频

### 3.1 目标分布

测试时仅观测视频 $v_{1:T}$，先点估计身份，再对内容序列积分：

$$p(a_{1:T} \mid v_{1:T}) \approx \sum_{x_{1:T}} \underbrace{p(a_{1:T} \mid x_{1:T},\, \hat{\text{id}})}_{\text{FM Head}} \cdot \underbrace{p(x_{1:T} \mid v_{1:T})}_{\text{内容后验}}$$

### 3.2 内容后验

由贝叶斯定理，在 $\text{id}$ 主要影响外观而非嘴唇运动模式的近似下：

$$p(x_{1:T} \mid v_{1:T}) \;\propto\; \underbrace{\prod_{t=1}^{T} p(v_t \mid x_t)}_{\text{视觉似然}} \cdot \underbrace{\prod_{t=1}^{T} p(x_t \mid x_{1:t-1})}_{\text{语言先验}}$$

---

## 4. 流式贝叶斯滤波

### 4.1 前向递推（Forward Algorithm）

定义前向概率 $\alpha_t(x_t) \triangleq p(x_t,\, v_{1:t})$，满足递推：

$$\boxed{\alpha_t(x_t) = p(v_t \mid x_t) \cdot \sum_{x_{t-1}} p(x_t \mid x_{t-1})\cdot \alpha_{t-1}(x_{t-1})}$$

这正是 **HMM 的 Forward Algorithm**，两步构成：

**预测步**（语言先验驱动）：
$$p(x_t \mid v_{1:t-1}) = \sum_{x_{t-1}} p(x_t \mid x_{t-1})\cdot p(x_{t-1} \mid v_{1:t-1})$$

**更新步**（视觉证据修正）：
$$p(x_t \mid v_{1:t}) \;\propto\; p(v_t \mid x_t)\cdot p(x_t \mid v_{1:t-1})$$

### 4.2 对连续 $x_t$ 的推广

若 $x_t \in \mathbb{R}^d$，求和改为积分：线性高斯系统退化为 Kalman Filter，一般情形需粒子滤波或变分推断。

---

## 5. 对数域：Product of Experts

### 5.1 分解公式

取对数，利用 $p(v_t \mid x_t) \propto p(x_t \mid v_t) / p(x_t)$：

$$\log p(x_t \mid v_{1:t}) = \underbrace{\log p(x_t \mid v_t)}_{\substack{\text{视觉模型得分} \\ (s_{\text{vis}})}} + \underbrace{\log p(x_t \mid x_{1:t-1})}_{\substack{\text{语言模型得分} \\ (s_{\text{LM}})}} - \underbrace{\log p(x_t)}_{\text{边际先验（可吸收）}} + \text{const}$$

### 5.2 实用形式（Shallow Fusion）

$$\log p(x_t \mid v_{1:t}) \approx s_{\text{vis}}(x_t) + \alpha \cdot s_{\text{LM}}(x_t)$$

其中 $\alpha > 0$ 为 LM 权重超参，等价于 ASR beam search 中的语言模型权重。

```python
vis_logits = visual_head(avhubert_feat)          # (B, T, V): log p(x_t | v_t)
lm_logits  = lm_head(lm_backbone(prev_tokens))   # (B, T, V): log p(x_t | x_{1:t-1})
posterior  = vis_logits + alpha * lm_logits      # (B, T, V): log p(x_t | v_{1:t})
```

---

## 6. 流式解码约束

### 6.1 文本必须 Greedy 解码

说话速率约 3–4 token/s，每 chunk（240ms）约对应 1 个文本 token。Beam search 需同时维护 $K$ 条假设，每条需独立运行 FM 采样，代价随 $K$ 线性增长，且无法确定何时"提交"某条 beam 而不引入额外延迟。因此流式场景只能采用 greedy 解码：

$$\hat{x}_t = \operatorname{arg\,max}_{x_t}\; p(x_t \mid v_{1:t}) = \operatorname{arg\,max}_{x_t}\;\bigl[s_{\text{vis}}(x_t) + \alpha \cdot s_{\text{LM}}(x_t)\bigr]$$

### 6.2 FM 不应直接条件化在 $\hat{x}_t$ 上

Greedy 解码在唇读场景下的 WER 可达 40–50%，即相当比例的 $\hat{x}_t$ 是错误的。若 FM Head 直接以 `embed`$(\hat{x}_t)$ 为条件，则：

- 文本错误直接传播为音频错误，且无任何缓冲
- Token embedding 是离散决策的 one-hot 代理，已经丢失了后验分布 $p(x_t \mid v_{1:t})$ 的全部不确定性信息

因此，**FM 的条件不应是离散 token，而应是产生该 token 的底层连续表示**。

### 6.3 充分统计量论证

设 FM 的条件为连续向量 $\mathbf{c}_t$。理想情况下，$\mathbf{c}_t$ 应是 $a_t$ 关于 $v_{1:t}$ 的**充分统计量**，即：

$$p(a_t \mid v_{1:t},\, \hat{\text{id}}) = p(a_t \mid \mathbf{c}_t,\, \hat{\text{id}})$$

以下两个连续表示均比 $\hat{x}_t$ 的 token embedding 更接近这一充分统计量：

| 表示 | 符号 | 编码内容 | 对文本错误的鲁棒性 |
|------|------|---------|-----------------|
| 视觉编码器输出 | $\tilde{v}_t$ | 当前帧的发音运动（$h_t$ 的代理） | **强**：独立于文本预测 |
| LM hidden state | $\tilde{h}_t^{\text{LM}}$ | 历史文本上下文、隐式韵律 | **中**：依赖历史，不依赖当前预测 |
| Argmax token embedding | $\phi(\hat{x}_t)$ | 仅编码当前决策的 token ID | **差**：文本错一词，FM 全错 |

$\tilde{v}_t$ 的关键性质：即使 $\hat{x}_t$ 预测错误，$\tilde{v}_t$ 仍正确地反映了嘴唇的实际运动；$\tilde{h}_t^{\text{LM}}$ 则由已提交的历史 token $\hat{x}_{1:t-1}$ 确定，与当前步骤的预测错误无关。

---

## 7. FM Head 的条件分布

### 7.1 条件的选取

综合 §6 的分析，FM Head 的条件为：

$$\mathbf{c}_t = \operatorname{proj}\!\left(\tilde{v}_t \;\|\; \tilde{h}_t^{\text{LM}} \;\|\; \hat{\text{id}}\right)$$

其中 $\|$ 表示拼接，$\operatorname{proj}$ 为线性投影至 FM 的 latent 维度。目标分布因此近似为：

$$\boxed{p(a_t \mid v_{1:t},\, \hat{\text{id}}) \;\approx\; p(a_t \mid \tilde{v}_t,\; \tilde{h}_t^{\text{LM}},\; \hat{\text{id}})}$$

### 7.2 与文本预测的关系

文本预测和音频生成**共享底层表示，但互不直接约束**：

$$\hat{x}_t = \operatorname{arg\,max}\bigl(s_{\text{vis}}(\tilde{v}_t) + \alpha\cdot s_{\text{LM}}(\tilde{h}_t^{\text{LM}})\bigr) \quad \text{（文本输出，用于 WER 评估和 LM 下一步）}$$

$$a_t \;\sim\; p(a_t \mid \tilde{v}_t,\; \tilde{h}_t^{\text{LM}},\; \hat{\text{id}}) \quad \text{（音频输出，从不接触 }\hat{x}_t\text{）}$$

两者均从同一组连续特征派生，因而是**软一致**（soft consistent）而非**硬绑定**（hard-committed）。当文本预测出错时，音频仍由正确的视觉特征和历史语言上下文引导，错误传播被局部化。

### 7.3 OT-CFM 目标

在 Optimal Transport Conditional Flow Matching 框架下，FM Head 学习速度场 $v_\theta$：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\, x_0,\, x_1}\left\| v_\theta\!\left(x_t,\; \operatorname{sg}(\mathbf{c}_t),\; t\right) - (x_1 - x_0) \right\|^2$$

其中 $x_t = (1-t)x_0 + t\,x_1$，$x_0 \sim \mathcal{N}(0,I)$，$x_1$ 为目标 Mimi latent，$\operatorname{sg}(\cdot)$ 为 stop-gradient，确保 FM loss 不回传到视觉编码器或 LM。

---

## 8. 完整架构

### 8.1 数据流

系统对每个 chunk 执行两条路径。身份向量 $\hat{\text{id}}$ 由 Speaker Encoder 对整段视频一次性估计，后续复用。

**路径一：文本解码**（输出 $\hat{x}_t$，用于 WER 评估与 LM 自回归推进）

```
v_t ──[AV-HuBERT + Conformer]──→ ṽ_t
                              └──[Visual Head]──→ s_vis
                                                    ↘
                                                 s_vis + α·s_LM ──→ argmax ──→ x̂_t
                                                    ↗                            │
x̂_{1:t-1} ──[SmolLM2]──→ h̃_t^LM                                               │
                       └──[LM Head]──→ s_LM          ←──────────────────────────┘
                                                           （回传，下一步使用）
```

**路径二：音频生成**（输出 $a_t$，从不接触离散决策 $\hat{x}_t$）

```
ṽ_t    ──────────────────────────┐
h̃_t^LM ──────────────────────────┼──[proj]──→ c_t ──[FM Head (DiT)]──→ latent_t ──[Mimi]──→ a_t
id̂    ──────────────────────────┘
```

两路共享 $\tilde{v}_t$ 和 $\tilde{h}_t^{\text{LM}}$，音频路径绕过离散决策，直接以连续表示为条件。

### 8.2 数据流文字说明

每个 chunk 的处理分五步顺序执行：

1. **视觉编码**：AV-HuBERT 对 chunk 内各帧提取特征，Conformer Adapter 在 chunk 内做双向 attention，输出连续视觉表示 $\tilde{v}_t \in \mathbb{R}^{960}$；Visual Head 将其线性投影到词表空间，得到视觉 logits $s_{\text{vis}}$。

2. **语言模型推进**：SmolLM2 以已提交的历史文本 $\hat{x}_{1:t-1}$ 为输入，输出 LM hidden state $\tilde{h}_t^{\text{LM}} \in \mathbb{R}^{960}$ 和 LM logits $s_{\text{LM}}$。LM 全程只看文本，不接触视觉特征，保持语言先验的纯粹性。

3. **文本后验与 Greedy 解码**：将视觉 logits 与 LM logits 按 Product of Experts 相加得到后验 logits，取 argmax 得 $\hat{x}_t$。$\hat{x}_t$ 用于：① 对外输出文本（WER 评估）；② 回传 SmolLM2 作为下一步的自回归输入。

4. **FM 条件构造**：将 $\tilde{v}_t$、$\tilde{h}_t^{\text{LM}}$、$\hat{\text{id}}$ 拼接后线性投影，得到 FM 条件向量 $\mathbf{c}_t \in \mathbb{R}^{512}$。此处施加 stop-gradient，FM loss 不回传到视觉编码器和 LM。

5. **音频生成**：FM Head（轻量 DiT）以 $\mathbf{c}_t$ 为条件，通过 10 步 Euler solver 将高斯噪声流变为 Mimi latent，再经 Mimi decoder 解码为 240ms 波形。相邻 chunk 经 overlap-add crossfade 拼接。

### 8.3 各模块职责划分

| 模块 | 输入 | 输出 | 优化目标 |
|------|------|------|---------|
| AV-HuBERT + Conformer | $v_{1:T}$ | $\tilde{v}_t$，$s_{\text{vis}}$ | 视觉似然 $\log p(x_t \mid v_t)$ |
| LM Backbone（SmolLM2）| $\hat{x}_{1:t-1}$ | $\tilde{h}_t^{\text{LM}}$，$s_{\text{LM}}$ | 语言先验 $\log p(x_t \mid x_{1:t-1})$ |
| Speaker Encoder | $v_{1:T}$ | $\hat{\text{id}}$ | 身份表示（预训练，冻结）|
| Text Head（PoE）| $s_{\text{vis}} + \alpha s_{\text{LM}}$ | $\hat{x}_t$ | $\mathcal{L}_{\text{CE}}$ on posterior logits |
| FM Head（DiT）| $\tilde{v}_t,\, \tilde{h}_t^{\text{LM}},\, \hat{\text{id}}$ | $a_t$ | $\mathcal{L}_{\text{FM}}$（OT-CFM）|

### 8.4 与原架构的对比

**原架构**（因果倒置，缺失 identity）：
$$v_t \xrightarrow{\text{AV-HuBERT}} \tilde{v}_t \xrightarrow{\text{LM on video}} \tilde{h}_t \xrightarrow{\text{text head}} \hat{x}_t, \quad \text{FM cond} = \operatorname{sg}(\tilde{h}_t)$$

问题：视觉似然与语言先验混合于同一 LM 前向；身份缺失；FM 条件与文本预测强绑定。

**新架构**（因果正确，双路解耦）：
- 文本路径：PoE 组合 $s_{\text{vis}}$ 和 $s_{\text{LM}}$，得到最优文本后验
- 音频路径：FM 条件化在连续特征 $(\tilde{v}_t, \tilde{h}_t^{\text{LM}}, \hat{\text{id}})$ 上，不经过离散 token

---

## 9. 训练目标

### 9.1 Evidence Lower Bound

对对数似然 $\log p(a_{1:T},\, v_{1:T} \mid \hat{\text{id}})$ 做变分下界，引入推断网络 $q_\phi(x_{1:T} \mid v_{1:T})$：

$$\log p(a_{1:T}, v_{1:T} \mid \hat{\text{id}}) \;\geq\; \mathbb{E}_{q_\phi}\!\left[\log p(a_{1:T} \mid x_{1:T},\, \hat{\text{id}}) + \log p(v_{1:T} \mid x_{1:T}) + \log p(x_{1:T}) - \log q_\phi(x_{1:T} \mid v_{1:T})\right]$$

### 9.2 实用简化（Teacher Forcing）

用 GT 文本做 teacher forcing（$q_\phi$ 退化为 delta 分布在 $x^*$ 上），ELBO 退化为双 loss：

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{FM}}\bigl(\tilde{v}_t,\; \tilde{h}_t^{\text{LM}},\; \hat{\text{id}}\bigr)}_{\text{OT-CFM loss，连续条件}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{CE}}\bigl(s_{\text{vis}} + \alpha\, s_{\text{LM}},\; x^*\bigr)}_{\text{对后验 logits 的 CE loss}}$$

**注意**：训练时 LM 的输入为 GT 文本（teacher forcing），故 $\tilde{h}_t^{\text{LM}}$ 基于 $x^*_{1:t-1}$ 计算；推理时切换为自回归输入 $\hat{x}_{1:t-1}$，引入的 exposure bias 与标准 LM 解码一致。

### 9.3 梯度流控制

$$\mathcal{L}_{\text{FM}} = \left\| v_\theta\!\left(x_t,\; \operatorname{sg}(\tilde{v}_t),\; \operatorname{sg}(\tilde{h}_t^{\text{LM}}),\; \hat{\text{id}},\; t\right) - (x_1 - x_0) \right\|^2$$

stop-gradient 施加在 $\tilde{v}_t$ 和 $\tilde{h}_t^{\text{LM}}$ 上，确保 FM loss 不回传到视觉编码器和 LM，两者仅由 $\mathcal{L}_{\text{CE}}$ 驱动。

---

## 10. 关键设计选择

### 10.1 $x_t$ 的离散 token 空间

离散文本 token 作为 $x_t$ 的优势：Product of Experts 直接可用（logit 相加）；与预训练 LM 词表对齐；CTC/CE 监督天然支持。连续 phoneme embedding 理论上更平滑，但后验计算更复杂，且与 LM 预训练不兼容。

### 10.2 视觉模型的监督方式

- **CTC loss**：无需帧级对齐，直接建模 $p(x_t \mid v_t)$ 的序列后验，实现最简单
- **帧级 CE loss**：需 MFA forced alignment，得到精确 per-frame 后验，与 PoE 公式对齐更直接

### 10.3 身份提取方案

| 方案 | 实现 | 参考工作 |
|------|------|---------|
| 时序平均面部特征 | 对全序列 face embedding 求均值 | 适用于单说话人片段 |
| 参考帧 | 第 0 帧或随机帧过 face encoder | LipVoicer |
| 预训练 speaker encoder | GE2E / ECAPA-TDNN，256 维 | SLD-L2S |

### 10.4 LM 权重 $\alpha$ 的作用

$$\alpha = 0 \;\Rightarrow\; \text{纯视觉识别（无语言先验）}$$
$$\alpha \to \infty \;\Rightarrow\; \text{纯语言模型（忽视视觉输入）}$$

最优 $\alpha$ 通过在验证集上搜索 WER 确定，等价于 ASR beam search 的 LM 权重调优，无需重训练。

---

*最后更新：2026-05-20*
