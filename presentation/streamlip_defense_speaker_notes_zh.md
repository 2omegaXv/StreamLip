# StreamLip Defense Speaker Notes

## 1. Title
开场先不要讲模型细节，直接说明任务：我们想从被静音、被屏蔽或音频缺失的视频里恢复说话声音。核心不是做字幕，而是恢复和嘴型同步的语音。

## 2. Opening Problem
这一页把问题抛出来：如果访谈或对话视频的音频缺失，只有脸和嘴型，我们如何恢复与画面同步的语音？强调这是一个音频恢复问题，不是单纯识别文字。

## 3. Why Not Video -> Text -> TTS?
解释 cascade 的问题：文字会丢掉音色、韵律、发音细节，也很难保证 frame-level audio-video sync。识别错误还会在 TTS 阶段放大，所以我们需要直接处理视频到音频表示的关系。

## 4. Core Idea
给出本项目的主张：text 只是弱语义和对齐条件，主体路线是 lip/video 加 timbre 直接到 Mimi audio latent，再解码成 waveform。

## 5. Final System Overview
用架构图讲系统。左侧是视觉、文本、说话人/音色条件，右侧预测 normalized Mimi latent，然后 Mimi decoder 输出 waveform。重点说所有条件都对 recon head 可见。

## 6. Data and Target
讲数据不是只存文本，而是每个 clip 都构造成视觉、文本 hidden、speaker、Mimi target、prompt/timbre 的配套表示。训练集 59144，验证集固定 1000。

## 7. Audio-Latent Formulation
解释为什么用 Mimi latent。我们不是直接回归 waveform，也不是从 transcript 做 TTS，而是在 codec-compatible 的连续音频 latent 空间里训练和评估。

## 8. Residual Endpoint Reconstruction
讲最终放弃随机 flow matching sampling。固定 x_tilde=0、tau=1，直接做 endpoint reconstruction，并用 frozen base 加 residual correction 提升结果。训练目标由三部分组成：latent MSE 负责数值重构，相关性项鼓励整体时序形状一致，prompt statistics 项约束输出的均值/方差接近参考音频的音色统计。

## 9. Self-Trained Visual Text Branch
这一页讲 StreamLip V5，但不要把它变成项目中心。它证明文本分支可以内部训练。指标是 WER 29.2、word accuracy 70.8，训练得可以，但最终只是弱 text conditioning。

## 10. Text Is Necessary, But Need Not Be Perfect
这里讲最关键实验：完全去掉 text condition 会明显下降，所以 text 必须有；但 text 不需要非常准，GT text 到 decoded text 只掉 0.0035 corr。即使文字错误较高，人耳仍可能听出大概，因为 lip/audio latent 条件还在。

## 11. Main Experimental Progression
用柱状图讲从 10k visual prior 到最终 59k prompt-stats residual 的进步。重点是 data scale、timbre/audio prompt、residual recon 是主要收益来源。

## 12. Trump Demo Case
这里只讲一个 demo case：输入 silent Trump video，给一段 reference audio，输出合成后的 voiced video。不要展开 pipeline 细节，用它证明系统可以用于缺失音频的视频恢复。

## 13. Takeaway
最后收束：恢复缺失音频不能只靠 text cascade，必须直接重建 audio latent；text 有用但应是弱条件；同步、音色和声学细节需要 lip/audio/timbre 条件保留在 reconstructor 中。
