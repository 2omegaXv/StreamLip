# FM AVSR Final Status - 2026-06-04

本文是当前分支的收口说明。更长的实验过程记录见：

- `doc/fm_avsr_timbre_condition_2026-06-03.md`
- `doc/fm_avsr_timbre_condition_risks_2026-06-04.md`
- `doc/fm_avsr_audio_generation_architecture.md`
- `doc/raw_video_avsr_recon_pipeline_usage.md`

## Final Route

当前可交付路线是 deterministic recon，不是 FM sample/denoise：

```text
lip_avsr.npy
  -> Auto-AVSR encoder: avsr_enc_lipavsr.npy
  -> SmolLM2 text hidden
  -> speaker_emb.npy
  -> first 38 Mimi frames as audio_prompt + timbre_cond.npy
  -> frozen base recon + residual correction
  -> normalized Mimi latent
  -> Mimi decoder
  -> waveform / mp4
```

核心数学形式：

```text
y_base  = f_base(C)
delta   = f_residual(C)
y_hat   = y_base + delta
wav_hat = MimiDecoder(denorm(y_hat))
```

`reconstruct_from_cond()` 固定使用：

```text
x_tilde = 0
tau = 1
y_raw = D_theta(0, C, 1)
```

所以它是条件端点回归，不包含随机噪声表征，也没有 denoise loop。

## Best Checkpoint

Final config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts.yaml
```

Main checkpoint:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts_v1/step_001500.pt
```

Residual base:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_from4000_recon_textjson_wordts_v1/step_005000.pt
```

Important config fields:

| Key | Value |
| --- | --- |
| `visual_feature_name` | `avsr_enc_lipavsr.npy` |
| `text_source` | `text_json` for in-distribution eval |
| `text_alignment_mode` | `word_timestamps` for in-distribution eval |
| `timbre_condition_name` | `timbre_cond.npy` |
| `audio_prompt_frames` | `38` |
| `audio_prompt_pool_cond` | `true` |
| `energy_condition_mode` | `pred` |
| `use_cross_attn` | `true` |
| `loss_fm_weight` | `0.0` |
| `lambda_recon` | `1.0` |
| `lambda_sample_corr` | `0.2` |
| `lambda_prompt_timbre_stats` | `0.05` |

## Final Metrics

Full val1000, `metric_start_frame=38`:

| Setup | Corr | MSE | MAE |
| --- | ---: | ---: | ---: |
| final prompt-stats residual, `text_json` | `0.58184180` | `0.66763106` | `0.60181897` |
| same checkpoint, `avsr_text_lipavsr.txt` | `0.57833471` | `0.67162028` | `0.60371083` |

Replacing GT text with lip-AVSR text drops only about `0.00351` corr on the
same LRS3 validation clips. This supports the current conclusion that text
accuracy is not the main bottleneck in this model; visual and timbre conditions
dominate.

## Important Data Convention

The final visual path must use the AVSR-compatible lip crop:

```text
lip_avsr.npy: (T, 96, 96) uint8 grayscale
```

Then:

```text
scripts/extract_avsr_enc.py \
  --input_name lip_avsr.npy \
  --output_name avsr_enc_lipavsr.npy \
  --text_output_name avsr_text_lipavsr.txt
```

Do not use old RGB `lip.npy` as the Auto-AVSR input for final results.

For external raw videos, use:

```text
scripts/reprocess_worker_avsr.py
```

This generates `lip_avsr.npy` with the same Auto-AVSR-compatible crop logic.
The worker depends on Auto-AVSR's mediapipe crop implementation and checks both
the current worktree and the main checkout for that external dependency.

## Raw Video Pipeline

The one-command external video path is:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/hrx.mov \
  --exp hrx_reprocess_avsr \
  --force
```

Outputs go to:

```text
eval_out/<exp>/
```

Key artifacts:

| Artifact | Meaning |
| --- | --- |
| `<exp>_224_25fps.mp4` | standardized input |
| `<exp>_pred_prompt3s_post3s.mp4` | predicted audio muxed to video, first 3.04s removed |
| `<exp>_gt_mimi_post3s.mp4` | Mimi GT reference |
| `recon_lipavsr_prompt3s/metrics.json` | latent metrics |
| `processed/custom/<exp>/00001/lip_avsr.npy` | AVSR-compatible lip crop |
| `vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4` | face/lip visualization |

External videos do not have LRS3 GT word timestamps, so this path uses:

```text
text_source = lipavsr
text_alignment_mode = uniform
```

That is different from the best in-distribution validation setup.

## External Video Checks

Two external `.mov` videos were checked with real first-3s audio prompt.

### Trump

Old path accidentally used RGB `lip.npy` as the Auto-AVSR input:

```text
corr = 0.30353260
```

Correct `reprocess_worker_avsr.py` path:

```text
corr = 0.38691377
```

The AVSR text also became much more plausible:

```text
OUR COUNTRY IS WINNING AND IN FACT WE'RE WINNING SO MUCH ...
```

Output:

```text
eval_out/trump_raw_prompt_pipeline/trump_pred_prompt3s_post3s_reprocess_avsr.mp4
eval_out/trump_raw_prompt_pipeline/vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4
```

Splitting the old RGB-path full video into three shorter segments improved corr
from `0.3035` to `0.3185`, so length is a factor but not the main issue.

### HRX

Old RGB `lip.npy` path:

```text
corr = 0.25592437
```

Correct `reprocess_worker_avsr.py` path:

```text
corr = 0.27240886
```

Output:

```text
eval_out/hrx_raw_prompt_pipeline/hrx_pred_prompt3s_post3s_reprocess_avsr.mp4
eval_out/hrx_raw_prompt_pipeline/vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4
```

## Interpretation

The external-video results are below LRS3 validation results. The strongest
current explanation is not just AVSR text quality. It is a combination of:

- external video domain shift,
- visual crop / Auto-AVSR feature quality,
- missing GT word timestamps,
- deterministic recon averaging,
- same-clip first-3s prompt being an upper-bound diagnostic rather than a fully
  production-style speaker control path.

Using the correct AVSR-compatible crop is nevertheless important and measurably
improves raw-video behavior, especially on the trump example.

## What To Keep As Core

Core docs:

- `doc/fm_avsr_final_status_2026-06-04.md`
- `doc/fm_avsr_audio_generation_architecture.md`
- `doc/raw_video_avsr_recon_pipeline_usage.md`
- `doc/fm_avsr_timbre_condition_2026-06-03.md`

Core scripts:

- `scripts/train_fm_avsr.py`
- `scripts/eval_fm_avsr.py`
- `scripts/extract_smollm2_h.py`
- `scripts/run_raw_video_avsr_recon_pipeline.py`
- `scripts/run_preprocess_worker_no_flash_attn.py`

Core source/test changes:

- `src/streaminlip/fm_avsr_dataset.py`
- `tests/test_fm_avsr_dataset.py`
- `tests/test_eval_fm_avsr.py` if touched in the final diff
- `scripts/reprocess_worker_avsr.py`

## Remaining Caveats

- Same-clip first 3 seconds are used as timbre/audio prompt; listening/eval wavs
  remove those first 38 frames.
- For silent raw video, this exact quality path cannot recover true speaker
  timbre unless an external prompt or speaker condition is provided.
- Sampling/FM path remains much worse than recon; sample corr stayed around
  `0.35` in the latest checks.
- The codebase still contains many experiment configs/scripts that should be
  archived or left untracked rather than promoted into the final branch.
