# 当前生成音频架构说明

本文记录当前阶段实际用于生成/评测音频的 AVSR-to-audio 架构，重点解释输入 latent、condition 组成、`recon` 路线的数学原理，以及它和原始 Flow Matching sample 路线的区别。

## 1. 一句话概览

当前效果最好的路线不是采样式 FM denoise，而是：

```text
原视频/唇部视频
  -> Auto-AVSR 视觉 encoder latent
  -> SmolLM2 文本 hidden
  -> Mimi 目标音频 latent
  -> 3s 音色 prompt / timbre condition
  -> DiT 条件端点回归 recon
  -> residual base + residual correction
  -> Mimi decoder 还原 waveform
```

也就是说，当前交付路径更接近一个条件 latent regression/reconstruction 模型：

```text
condition c_{1:T}  ->  predicted normalized Mimi latent y_hat_{1:T}
```

而不是从随机噪声一步步采样出音频 latent。

## 2. 当前最好模型

当前记录中最好的 full val1000 结果来自这个配置：

```text
configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts.yaml
```

主 checkpoint：

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts_v1/step_001500.pt
```

residual base checkpoint：

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_from4000_recon_textjson_wordts_v1/step_005000.pt
```

核心配置：

| 项 | 当前值 | 含义 |
| --- | --- | --- |
| `visual_feature_name` | `avsr_enc_lipavsr.npy` | 新 lip-AVSR 处理后的 Auto-AVSR encoder feature |
| `text_source` | `text_json` | GT transcript/word timestamp 文本源 |
| `text_alignment_mode` | `word_timestamps` | 按词级时间戳对齐 LM hidden |
| `timbre_condition_name` | `timbre_cond.npy` | 前 3s Mimi latent 的均值/方差统计 |
| `audio_prompt_frames` | `38` | 前 38 个 Mimi latent frame，约 3.04s |
| `audio_prompt_dim` | `512` | Mimi latent 维度 |
| `audio_prompt_pool_cond` | `true` | prompt token 平均池化后加到每帧 condition |
| `energy_condition_mode` | `pred` | 预测 per-frame energy 作为额外 condition |
| `n_dit_layers` | `6` | DiT block 数 |
| `use_cross_attn` | `true` | 对 condition token 做 cross-attention |
| `loss_fm_weight` | `0.0` | 当前不训练 FM velocity loss |
| `lambda_recon` | `1.0` | 主损失是 deterministic recon MSE |
| `lambda_sample_corr` | `0.2` | per-sample corr 辅助损失 |
| `lambda_prompt_timbre_stats` | `0.05` | 约束生成段统计接近 prompt 统计 |

full val1000 指标：

| Text source | Eval corr | Eval MSE | Eval MAE |
| --- | ---: | ---: | ---: |
| `text_json` | `0.58184180` | `0.66763106` | `0.60181897` |
| `avsr_text_lipavsr.txt` | `0.57833471` | `0.67162028` | `0.60371083` |

指标和听音样本都跳过前 38 帧，也就是跳过前约 3 秒音色 prompt 区间。

## 3. 离线输入与 latent

每条 clip 当前主要依赖这些离线文件：

| 文件 | 形状 | 来源 | 用途 |
| --- | --- | --- | --- |
| `lip_avsr.npy` | roughly `(T_v, 96, 96)` | 人脸/嘴部预处理后的灰度唇部图 | Auto-AVSR 输入 |
| `avsr_enc_lipavsr.npy` | `(T_v, 768)` | Auto-AVSR visual encoder | 视觉 condition |
| `avsr_text_lipavsr.txt` | text | Auto-AVSR CTC decode | 可替代 GT 文本的预测文本源 |
| `text.json` | transcript + word timestamps | 数据集标注 | 当前最好模型默认文本源 |
| `smollm2_h_text_json.npy` | `(L, 960)` | SmolLM2 hidden cache | 文本 condition |
| `smollm2_h_lipavsr.npy` | `(L, 960)` | SmolLM2 encode `avsr_text_lipavsr.txt` | lip-AVSR 文本实验 |
| `speaker_emb.npy` | `(256,)` | 人脸侧 speaker/identity embedding | 弱身份 condition |
| `latent.npz` | `(T_a, 512)` | Mimi encoder encode GT audio | 训练目标 audio latent |
| `timbre_cond.npy` | `(1024,)` | 前 38 帧 normalized Mimi latent 的 mean/std | 全局音色 condition |

这里的目标音频 latent 是 Mimi continuous latent。设原始 Mimi latent 为：

```text
z_{1:T_a},    z_t in R^512
```

训练前会用全局统计做 normalize：

```text
y_t = (z_t - mu) / sigma
```

模型训练和评测里的 MSE/corr 都是在 normalized latent `y` 上计算。最终输出音频时再反归一化：

```text
z_hat_t = y_hat_t * sigma + mu
```

然后交给 Mimi decoder 还原 waveform。

## 4. Condition 如何构造

Mimi audio latent 的帧率约 12.5 Hz；Auto-AVSR visual encoder feature 约 25 Hz，所以视觉 latent 会按 2 倍下采样：

```text
v_t = AVSR_enc_lipavsr[2t]          v_t in R^768
```

文本侧先用 SmolLM2 得到 token hidden：

```text
l_j in R^960,    j = 1 ... L
```

然后根据 word timestamp 找到 audio latent frame `t` 对应的文本 token index：

```text
h_t = l_{a(t)}                      h_t in R^960
```

如果不用 word timestamp，也可以 uniform resample；当前最好模型使用 `word_timestamps`。

speaker embedding 展开到每一帧：

```text
s_t = s                             s in R^256
```

前 38 帧 normalized Mimi latent 作为音色 prompt：

```text
P = 38
A = [y_1, ..., y_P]                 A in R^{P x 512}
```

`timbre_cond.npy` 是这个 prompt 的统计量：

```text
q = concat(mean(A), std(A))          q in R^1024
```

它也会被展开到每一帧：

```text
q_t = q
```

同时，前 38 帧 prompt token 会走一条更强的路径：

```text
p_i = W_a A_i                       p_i in R^512
p_bar = mean_i(p_i)
```

当前配置里 `audio_prompt_pool_cond=true`，所以每帧 condition 会加上 `p_bar`。prompt token 本身也作为 cross-attention 的 condition token。

如果打开预测 energy condition，模型先预测一条 per-frame log-RMS energy：

```text
e_hat_t = g(v_t, h_t, s, q, A)
```

当前 residual 配置中，这个 energy 通常由 frozen residual base 侧预测，再作为 extra condition 拼进主模型。

于是每一帧的基础 condition 可以写成：

```text
r_t = concat(v_t, h_t, s, q, e_hat_t)
c_t = W_c r_t + p_bar               c_t in R^512
```

整段 condition 为：

```text
C = [c_1, ..., c_T]                 C in R^{T_a x 512}
```

## 5. DiT Head

核心网络是 `FMHeadAVSR`，底层是一个 DiT-like Transformer head：

```text
x_tilde_{1:T}, C, tau  ->  D_theta(x_tilde, C, tau)
```

其中：

| 符号 | 含义 |
| --- | --- |
| `x_tilde` | 输入给 DiT 的 latent token，维度 `(T_a, 512)` |
| `C` | 每帧对齐后的 condition token，维度 `(T_a, 512)` |
| `tau` | FM 时间变量/端点变量 |
| `D_theta` | 6 层 DiT block + final projection |

DiT 内部大致做：

```text
u_0 = x_tilde + W_token C + pos
global = time_emb(tau) + mean(C)
u_{k+1} = DiTBlock_k(u_k, global, cond_tokens)
output = W_out(LN(u_K))
```

这里的 `cond_tokens` 在当前配置下主要包含 audio prompt token；由于 `use_cross_attn=true`，DiT block 可以通过 cross-attention 读取 prompt 信息。

## 6. 原始 FM 数学原理

代码最早的 FM/OT-CFM 训练目标是：从噪声 `x_0` 到真实 latent `x_1` 学一条直线路径的 velocity。

设真实目标 normalized Mimi latent 为：

```text
x_1 = y
```

随机噪声：

```text
x_0 ~ N(0, I)
```

随机采样时间：

```text
tau ~ Uniform(0, 1)
```

直线路径上的点：

```text
x_tau = (1 - tau) x_0 + tau x_1
```

目标 velocity：

```text
u_tau = x_1 - x_0
```

FM loss：

```text
L_FM = E || D_theta(x_tau, C, tau) - (x_1 - x_0) ||^2
```

如果走 sample 路线，推理时会从随机噪声开始：

```text
x^{0} ~ N(0, I)
x^{k+1} = x^k + Delta tau * D_theta(x^k, C, tau_k)
```

多步 Euler 积分后得到 `x^K`，再作为预测 latent。

但是当前最好实验已经不使用这条路线作为主结果。当前配置里：

```text
loss_fm_weight = 0.0
```

所以 FM velocity loss 被关掉了；sample 路线虽然代码还在，但当前实验里 sample corr 只有约 `0.35`，明显低于 recon 的约 `0.58`，因此没有继续作为交付主线。

## 7. 当前 recon 的数学原理

`recon` 是 deterministic endpoint prediction。它不从随机噪声采样，也没有 denoise loop。

实现上，`reconstruct_from_cond` 做的是：

```text
x_tilde = 0
tau = 1
y_hat_raw = D_theta(0, C, 1)
```

也就是说，把 DiT 当成一个条件函数：

```text
f_theta(C) = D_theta(0, C, 1)
```

直接从 condition 预测完整 normalized Mimi latent：

```text
y_hat = f_theta(C)
```

训练时对应的 recon loss 是：

```text
L_recon = MSE(y_hat, y)
        = || f_theta(C) - y ||^2
```

当前配置主项就是：

```text
L = 1.0 * L_recon
  + 0.2 * L_sample_corr
  + 0.05 * L_prompt_timbre_stats
```

其中 `L_sample_corr` 是每条样本内部的 Pearson corr loss，跳过前 38 帧后计算：

```text
L_sample_corr = 1 - corr(flat(y_hat_{P+1:T}), flat(y_{P+1:T}))
```

`L_prompt_timbre_stats` 约束生成段的统计量不要偏离前 3 秒 prompt 的整体音色统计：

```text
L_prompt_timbre_stats
  = MSE(mean(y_hat_{P+1:T}), mean(y_{1:P}))
  + MSE(std(y_hat_{P+1:T}), std(y_{1:P}))
```

这就是当前 `recon` 的本质：不是 diffusion/flow sampling，而是条件端点回归。它借用了 DiT/FM head 的网络结构，但训练目标已经变成“从多模态条件直接重建 Mimi latent”。

## 8. Residual Recon

当前最好模型不是单个 recon head，而是 residual refinement：

```text
y_hat_base = f_base(C)
delta_hat  = f_residual(C)
y_hat      = y_hat_base + delta_hat
```

其中：

| 部分 | 作用 |
| --- | --- |
| `f_base` | frozen residual base model，提供一个已有的 latent recon |
| `f_residual` | 当前训练/评测 checkpoint，预测 correction |
| final output | base recon 和 residual correction 相加 |

代码里的组合逻辑是：

```text
compose_endpoint_prediction(raw_prediction, baseline)
```

如果存在 `baseline`，最终就是：

```text
baseline + raw_prediction
```

所以当前模型的实际输出不是“主 ckpt 单独预测整段 latent”，而是：

```text
final normalized latent = frozen base recon + residual correction
```

这也是为什么需要同时记录主 checkpoint 和 residual base checkpoint。

## 9. 推理/生成流程

对一条已处理好的 eval clip，当前 `--use_recon` 流程如下：

1. 读取 `avsr_enc_lipavsr.npy`，下采样到 audio latent 帧率，得到 `v_{1:T}`。
2. 读取文本源：
   - 当前最好默认 `text_json`；
   - 可切换到 `avsr_text_lipavsr.txt`。
3. 读取或生成对应 SmolLM2 hidden cache，并按 word timestamps 对齐到 `h_{1:T}`。
4. 读取 `speaker_emb.npy`。
5. 读取 `timbre_cond.npy` 和前 38 帧 `audio_prompt`。
6. 如配置需要，预测 per-frame energy condition。
7. frozen base 先做一次：

   ```text
   y_hat_base = f_base(C)
   ```

8. 主 residual model 做：

   ```text
   delta_hat = f_residual(C)
   ```

9. 合成 final normalized latent：

   ```text
   y_hat = y_hat_base + delta_hat
   ```

10. 反归一化：

    ```text
    z_hat = y_hat * sigma + mu
    ```

11. Mimi decoder 输出 24 kHz waveform。

12. 保存 wav 时，如果设置 `--wav_start_frame 38`，会裁掉前约 3 秒 prompt 区间。

## 10. 为什么听音要裁掉前 3 秒

当前音色控制使用的是同一条样本 GT audio 的前 38 帧 Mimi latent：

```text
A = y_{1:38}
```

这等价于给模型一个约 3 秒的真实音频 prompt。它对验证“音色条件是否能帮助后续生成”很有用，但它也意味着：

1. 前 3 秒不是模型真正从视频生成出来的未知内容，而是被用作 condition。
2. 如果听音文件包含前 3 秒，开头会非常接近 GT，后面才体现模型生成质量。
3. 为了公平听后续生成段，当前 listen/eval wav 使用 `--wav_start_frame 38` 裁掉前 3 秒。

因此当前结果应理解为：

```text
给定同 clip 前 3 秒音频音色 prompt，预测后续音频 latent。
```

如果输入是完全无声的 raw mp4，严格来说无法得到真实 `timbre_cond.npy` 和真实 `audio_prompt`。之前 raw mp4 一条龙计时里可以用 zero placeholder 跑通流程，但那只代表工程耗时，不代表当前最好质量。

## 11. Corr 指标

评测里的 corr 是 normalized Mimi latent 上的 Pearson correlation。对每条 clip，先跳过 `metric_start_frame=38`，然后展平成一维：

```text
p = flatten(y_hat_{39:T})
g = flatten(y_{39:T})
```

去均值：

```text
p_c = p - mean(p)
g_c = g - mean(g)
```

计算：

```text
corr = sum(p_c * g_c) / sqrt(sum(p_c^2) * sum(g_c^2))
```

最终 `mean_corr` 是对 val clips 的逐条 corr 取平均。它衡量 predicted latent 和 GT latent 的整体线性一致性，但不完全等价于主观音质；例如音色细节、噪声感和 vocoder/latent 解码误差仍然需要听音确认。

## 12. 文本条件的结论

当前结果显示文本不是主要瓶颈。

同一个最终 prompt-stats residual 模型：

```text
text_json corr      = 0.58184180
lip-AVSR text corr  = 0.57833471
drop                = 0.00350709
```

这个下降可测，但不严重，约 0.6% relative。结合更早的 no-text/shuffle-text 观察，可以说当前架构主要由视觉 latent 和音色 prompt 驱动，SmolLM2 文本 hidden 的边际贡献较小。

这不代表文本信息理论上没用；更准确的说法是：在当前 condition fusion、Mimi latent recon 目标和 Auto-AVSR 视觉特征下，文本准确性不是限制 `0.58 -> 0.60+` 的主要因素。

## 13. 当前上限和局限

当前模型的主要局限是：

1. `recon` 是 deterministic regression，天然容易预测平均化的 latent，细粒度音色和高频质感会被抹平。
2. `speaker_emb.npy` 来自人脸侧身份向量，不是强语音 speaker embedding，对真实音色控制有限。
3. 真正有效的音色入口依赖同 clip 前 3 秒 GT audio prompt，因此更像上界/诊断设置，不是纯 silent-video inference 设置。
4. sample/FM 路线目前明显落后，sample corr 约 `0.35`，短期内看不到超过 recon 的趋势。
5. raw mp4 端到端耗时目前主要卡在人脸检测、对齐和唇部 crop；FM head 与 Mimi decode 本身相对较快。

## 14. 关键代码位置

| 文件 | 内容 |
| --- | --- |
| `src/streaminlip/v2/fm_head.py` | DiT/FM head、`forward_train`、`reconstruct_from_cond`、`sample` |
| `scripts/train_fm_avsr.py` | 训练入口、loss 组合、residual 组合、energy condition |
| `scripts/eval_fm_avsr.py` | eval/listen 入口、`--use_recon`、Mimi decode、corr 指标 |
| `src/streaminlip/fm_avsr_dataset.py` | 读取 visual/text/speaker/Mimi latent/timbre/audio prompt |
| `scripts/extract_avsr_enc.py` | 从 `lip_avsr.npy` 提取 `avsr_enc_lipavsr.npy` 和 `avsr_text_lipavsr.txt` |
| `scripts/extract_smollm2_h.py` | 为不同 text source 提取 SmolLM2 hidden cache |
| `scripts/extract_timbre_cond.py` | 从前 38 帧 Mimi latent 提取 `timbre_cond.npy` |

## 15. 最终理解

当前架构可以概括为：

```text
多模态条件 C = video latent + text hidden + face identity + 3s audio prompt + predicted energy

base recon:
  y_base = f_base(C)

residual recon:
  delta = f_residual(C)

final latent:
  y_hat = y_base + delta

audio:
  wav_hat = MimiDecoder(denorm(y_hat))
```

它保留了 FM/DiT 的网络结构，但当前最好结果来自 `recon` 端点回归目标。数学上，`recon` 就是学习一个条件映射 `C -> y`，用前 3 秒 GT Mimi latent 提供音色 anchor，再预测后续 Mimi latent，最后通过 Mimi decoder 合成 waveform。
