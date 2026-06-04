
# Raw Video AVSR Recon Pipeline Usage

这个脚本用于把一条外部 `.mov/.mp4` 一条龙跑成当前 recon 模型的生成视频。

核心点：

- 使用新版 `reprocess_worker_avsr.py` 生成 `lip_avsr.npy`
- 使用 `lip_avsr.npy` 提取 `avsr_enc_lipavsr.npy` 和 `avsr_text_lipavsr.txt`
- 使用原视频音频前 3.04 秒作为 `audio_prompt` / `timbre_cond`
- 输出时裁掉前 3.04 秒，只看生成段

## Command

必须用 repo 的 `.venv`：

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/hrx.mov \
  --exp hrx_reprocess_avsr \
  --force
```

默认会调用当前分支内的：

```text
scripts/reprocess_worker_avsr.py
```

该 worker 仍依赖 Auto-AVSR 的 mediapipe crop 代码；脚本会优先查找当前
worktree 的 `third_party/auto_avsr/...`，找不到时回退到主 checkout 的同名路径。

另一个例子：

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/trump.mov \
  --exp trump_reprocess_avsr \
  --force
```

## Output

所有文件会写到：

```text
eval_out/<exp>/
```

关键输出：

| 文件 | 含义 |
| --- | --- |
| `<exp>_224_25fps.mp4` | 标准化输入视频，224x224、25fps、24kHz mono |
| `<exp>_pred_prompt3s_post3s.mp4` | 生成音频 mux 回视频，已裁掉前 3.04 秒 |
| `<exp>_gt_mimi_post3s.mp4` | GT audio 经 Mimi encode/decode 后的参考视频 |
| `recon_lipavsr_prompt3s/0000_pred.wav` | 生成音频 wav |
| `recon_lipavsr_prompt3s/0000_gt.wav` | Mimi GT 参考 wav |
| `recon_lipavsr_prompt3s/metrics.json` | latent corr/MSE/MAE |
| `processed/custom/<exp>/00001/lip_avsr.npy` | 新 AVSR 兼容灰度嘴部 crop |
| `processed/custom/<exp>/00001/avsr_text_lipavsr.txt` | Auto-AVSR 识别文本 |
| `vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4` | aligned face + lip_avsr crop 可视化 |

## Notes

外部视频没有 LRS3 的 GT transcript 和 word timestamps，所以脚本使用：

```text
text_source = lipavsr
text_alignment_mode = uniform
```

这和验证集最强的 `text_json + word_timestamps` 不完全一样。外部视频效果主要受视频域偏移、lip crop 质量、AVSR 文本质量和音色 prompt 影响。

## GUI

旧的 GUI 启动方式仍保留：

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.worktrees/fm-avsr-cleanup/scripts/gradio_avsr_gui.py \
  --port 7860
```

GUI 内部调用同一个 `scripts/run_raw_video_avsr_recon_pipeline.py`。
