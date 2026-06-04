# FM-AVSR Timbre Fix Experiments - 2026-06-04

## Goal

Fix the current timbre/control problem while keeping validation correlation above
the historical best. Each experiment must stay under 1 hour. Runs with abnormal
or clearly weak metrics should be stopped early.

Success criteria:

- reduce the sequence-level prompt-copying path in the model design;
- keep or improve validation `val_recon_corr` over the current best run;
- record every experiment here with config, checkpoint, elapsed time, and result;
- commit key code/config/doc checkpoints.

## Historical Best

The previous report recorded `corr = 0.58184180` from the final text-json eval.
The local training validation logs show a stronger current baseline:

| Run | Step | val_recon_corr | Notes |
| --- | ---: | ---: | --- |
| `lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts_v1` | 1500 | `0.58431531` | current best full run |
| `lipavsr_59144_timbre3s_audioprompt38_pool_residual_ctctopk_from1000_recon_textjson_wordts_v1` | 1500 | `0.58427079` | CTC top-k auxiliary |
| `lipavsr_59144_timbre3s_audioprompt38_pool_statpool_residual_from1000_recon_textjson_wordts_v1` | 1250 | `0.58426972` | stat-pool enabled, still exposes prompt tokens |
| `lipavsr_59144_timbre3s_audioprompt38_pool_learnedpool_residual_samplecorr02_from1000_recon_textjson_wordts_v1` | 1500 | `0.58426568` | learned pool enabled, still exposes prompt tokens |
| `lipavsr_59144_timbre3s_audioprompt38_pool_residual_samplecorr02_from1000_recon_textjson_wordts_v1` | 1500 | `0.58426899` | no prompt-stats loss |

Working threshold for this stage: beat `0.58431531` on the same
`pretrain_len80_260_lipavsr_val1000_seed43` validation split, not merely the
older report number.

## Root-Cause Hypothesis

The current timbre condition has two parts:

- fixed/global `timbre_cond.npy = concat(mean(prefix), std(prefix))`;
- sequence `audio_prompt.npy` with shape `(38, 512)`, about 3.04 seconds of
  normalized Mimi frames.

Even when `audio_prompt_pool_cond`, `audio_prompt_stat_pool_cond`, or
`audio_prompt_learned_pool_cond` is enabled, the raw projected prompt sequence is
still concatenated into cross-attention condition tokens whenever
`audio_prompt_frames > 0`. This is a direct content-copying route. The model can
use the prompt as a short audio prefix instead of a speaker-only style
condition.

First code fix: add `no_audio_prompt_cross_attn` so experiments can keep
pooled/stat timbre conditioning but hide the raw temporal audio-prompt tokens
from DiT cross-attention.

## Experiment Log

### E0: Structural No-Prompt-Token Switch

Code change:

- `src/streaminlip/v2/fm_head.py`
  - adds `audio_prompt_cross_attn` with default `True`;
  - when disabled, prompt tokens can still affect pooled/stat/learned conditions
    but are not appended to cross-attention tokens.
- `scripts/train_fm_avsr.py` and `scripts/eval_fm_avsr.py`
  - add `no_audio_prompt_cross_attn`.

Verification:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python -m unittest \
  tests.test_timbre_condition.TimbreConditionTest.test_audio_prompt_tokens_can_be_excluded_from_cross_attention \
  tests.test_eval_fm_avsr.EvalFMAVSRTest.test_parse_args_loads_no_audio_prompt_cross_attn_from_config \
  tests.test_fm_avsr_dataset.FMAVSRDatasetTest.test_parse_args_loads_no_audio_prompt_cross_attn_from_config

Ran 3 tests in 0.161s
OK
```

Status: code path ready for training experiments.

### E1: Disable Raw Prompt Cross-Attention, Keep Mean Pool

Planned config:

- start from current best residual config;
- set `no_audio_prompt_cross_attn: true`;
- keep `audio_prompt_pool_cond: true`;
- keep `lambda_sample_corr: 0.2`;
- do not use `lambda_prompt_timbre_stats` in first pass, because matching the
  predicted post-prompt latent distribution to the reference prompt may
  reinforce prompt copying;
- max 1500 steps, eval every 250, expected runtime under 1 hour.

Early-stop rule:

- stop if validation corr at 500 steps is below `0.5830`;
- stop immediately on NaN, exploding loss, or severe train/val regression.

Result:

| Step | val_recon_corr | train_recon_corr | elapsed | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1250 | `0.5646` | not used | under 5 min | early-stopped |

Run directory:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_nopromptxattn_residual_samplecorr02_from1000_recon_textjson_wordts_v1
```

Conclusion:

Fully hiding raw prompt tokens removes the prompt-copying route, but it also
removes a condition that the current checkpoint depends on heavily. This is not
a viable final direction because it loses about `0.020` validation corr against
the current best.

### E2: Keep Prompt Tokens, Move Reconstruction Loss After Prompt

Hypothesis:

The direct copying bug is encouraged because training reconstructs the same
first 38 Mimi frames that are also given as `audio_prompt.npy`. The validation
metric already skips the first 38 frames, but the main reconstruction loss still
starts at frame 0 in the strongest run. Setting `loss_start_frame: 38` keeps the
prompt tokens available for speaker/style control while removing the direct
loss reward for copying the prompt region.

Planned config:

- resume from the current best prompt-stats checkpoint at step 1500;
- keep raw prompt cross-attention enabled;
- set `loss_start_frame: 38`;
- keep `metric_start_frame: 38`;
- train only 500 additional steps, max runtime under 1 hour;
- early-stop if the 1750-step validation corr drops below `0.5830`.

Config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts.yaml
```

Run directory:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts_v1
```

Training-validation result:

| Step | val_recon_corr | val_recon_mse | val_recon_mae | train_recon_corr | elapsed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1750 | `0.58433689` | `0.64905658` | `0.59181344` | `0.60431403` | `184.35 s` |
| 2000 | `0.58434393` | `0.64904572` | `0.59179956` | `0.61712486` | `353.11 s` |

Checkpoint:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts_v1/step_002000.pt
```

This beats the training-validation historical best `0.58431531` by
`+0.00002862` on the same validation split. It also reduces the apparent
train/val overfit gap: the previous best had `train_recon_corr = 0.75502914`,
while E2 has `train_recon_corr = 0.61712486` at its best checkpoint.

Metrics-only eval on 1000 clips with the same eval script:

| Model | corr | mse | mae | metric_start_frame |
| --- | ---: | ---: | ---: | ---: |
| previous best step 1500 | `0.58184180` | `0.66763106` | `0.60181897` | `38` |
| E2 loss_start_frame=38 step 2000 | `0.58186685` | `0.66760399` | `0.60178062` | `38` |

Eval command:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python scripts/eval_fm_avsr.py \
  --config configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts.yaml \
  --ckpt runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts_v1/step_002000.pt \
  --use_recon --metrics_only --n 1000 \
  --output_dir eval_out/timbre_fix_e2_lossstart38_val1000 \
  --metrics_json eval_out/timbre_fix_e2_lossstart38_val1000/metrics.json
```

Conclusion:

E2 is the current best quantitative checkpoint. It does not remove the temporal
audio-prompt condition, so it is not a full architectural fix for speaker-only
timbre. It does remove the direct training reward for reconstructing the same
first 3.04 seconds that are supplied as `audio_prompt.npy`, and it keeps
validation correlation above the historical best. For listening exports, the
first 3.04 seconds should still be cropped until a fixed-size speaker-only
timbre condition or random-reference/dropout training is implemented.

Trump silent-reference listening check:

The E2 checkpoint was run on the existing preprocessed Trump silent-reference
demo to avoid reintroducing face/AVSR preprocessing variability. The output
keeps the same post-prompt export convention, so the first 3.04 seconds are
cropped from the visible/listenable MP4.

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python scripts/eval_fm_avsr.py \
  --config configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts.yaml \
  --ckpt runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts_v1/step_002000.pt \
  --data_root eval_out/trump_silent_ref_demo_full/processed \
  --clip_list eval_out/trump_silent_ref_demo_full/clip_list.txt \
  --text_source lipavsr --text_alignment_mode uniform \
  --output_dir eval_out/trump_silent_ref_demo_full_e2_lossstart38/recon_lipavsr_prompt3s \
  --n 1 --use_recon --audio_prompt_name audio_prompt.npy \
  --wav_start_frame 38 --metric_start_frame 38 \
  --metrics_json eval_out/trump_silent_ref_demo_full_e2_lossstart38/recon_lipavsr_prompt3s/metrics.json
```

Artifacts:

```text
eval_out/trump_silent_ref_demo_full_e2_lossstart38/recon_lipavsr_prompt3s/0000_pred.wav
eval_out/trump_silent_ref_demo_full_e2_lossstart38/trump_silent_ref_demo_full_e2_lossstart38_pred_post3s.mp4
```

The muxed MP4 duration is `23.52 s`, matching the existing full silent-ref
demo after removing the first `3.04 s` prompt region.

### E3: Pooled Prompt Token for Cross-Attention

Hypothesis:

E1 showed that removing audio-prompt cross-attention entirely loses too much
correlation. E3 keeps a prompt cross-attention path but replaces the temporal
`(38, 512)` prompt-token sequence with one mean-pooled prompt token. This
reduces the sequence-level content-copying path while preserving a style-like
reference token.

Code/config:

- `src/streaminlip/v2/fm_head.py`
  - adds `audio_prompt_cross_attn_pool`;
  - when enabled, DiT cross-attention sees only
    `mean(audio_prompt_proj(audio_prompt), dim=time)` as one token.
- `scripts/train_fm_avsr.py` and `scripts/eval_fm_avsr.py`
  - add CLI/config support for `audio_prompt_cross_attn_pool`.

Config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_poolxattn_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts.yaml
```

Run directory:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_poolxattn_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts_v1
```

Result:

| Step | val_recon_corr | val_recon_mse | val_recon_mae | train_recon_corr | elapsed | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2250 | `0.56620605` | `0.66995936` | `0.60439114` | `0.57895392` | `182.76 s` | stop |

Conclusion:

This confirms the current high-corr checkpoint still depends on temporal
audio-prompt cross-attention. Pooling the prompt token is conceptually cleaner
for timbre control but causes a large validation drop (`-0.0181` versus E2).
Therefore E3 is not a deployable final model. The final checkpoint for now
remains E2, with first-3.04s export cropping and the documented limitation that
the prompt representation is not yet a pure speaker embedding.

### E4: Disable Prompt Cross-Attention at E2 Evaluation Only

Hypothesis:

Maybe the checkpoint only needs raw prompt tokens during training, and inference
can remove them while preserving the learned frame/global timbre condition.
This would be an immediate deploy-time fix for prompt content leakage.

Eval command:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python scripts/eval_fm_avsr.py \
  --config configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts.yaml \
  --ckpt runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts_v1/step_002000.pt \
  --use_recon --metrics_only --n 1000 --no_audio_prompt_cross_attn \
  --output_dir eval_out/timbre_fix_e4_e2_no_promptxattn_eval_val1000 \
  --metrics_json eval_out/timbre_fix_e4_e2_no_promptxattn_eval_val1000/metrics.json
```

Result:

| Eval mode | corr | mse | mae | Decision |
| --- | ---: | ---: | ---: | --- |
| E2 normal | `0.58186685` | `0.66760399` | `0.60178062` | keep |
| E2 eval-only no prompt x-attn | `0.55201229` | `0.70036560` | `0.62153375` | reject |

Conclusion:

Inference also depends on raw temporal prompt tokens. A deploy-time switch that
removes prompt cross-attention is not viable. This strengthens the diagnosis:
the present best checkpoint uses prompt sequence information as a strong
condition, not just as speaker identity. The practical handoff remains E2 with
post-prompt export cropping; a true fix requires retraining around a
speaker-only/fixed-size timbre embedding or randomized reference-window
training that prevents prompt-content copying.

### E5: Same-Parent Reference Prompt During Training

Hypothesis:

Most training clips have another clip under the same `pretrain/<video_id>/`
parent (`59035/59144` training clips are in multi-clip parents). Using a
same-parent neighbor as the audio prompt should preserve rough speaker/session
style while breaking the exact equality between the prompt and the target
opening content.

Code/config:

- `FMAVSRDataset(audio_prompt_ref_mode="same_parent_next")`
  - default `self_prefix` keeps previous behavior;
  - `same_parent_next` loads the prompt latent from the next clip under the same
    parent directory, falling back to self for singletons.
- `scripts/train_fm_avsr.py`
  - adds `--audio_prompt_ref_mode`.

Config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_sameparentprompt_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts.yaml
```

Run directory:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_sameparentprompt_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts_v1
```

Result:

| Step | val_recon_corr | val_recon_mse | val_recon_mae | train_recon_corr | elapsed | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2250 | `0.57217276` | `0.66357255` | `0.59945439` | `0.58781838` | `180.90 s` | stop |

Conclusion:

Using a same-parent reference prompt is cleaner than same-clip prefix prompting,
but it still drops far below E2. This is another negative result: the current
checkpoint cannot be converted into a speaker-only/timbre-only prompt model by
short fine-tuning from E2. The deployable checkpoint remains E2, and a true
speaker-only solution likely needs longer training with randomized references
from the start or a separate pretrained speaker/timbre encoder.

### E6: Fine-Tune E2 With Prompt Cross-Attention Disabled

Hypothesis:

E4 showed a direct eval-time removal of raw prompt cross-attention drops corr to
`0.5520`. A short fine-tune from E2 might migrate the useful speaker/style
information into `audio_prompt_pool_cond` and `timbre_cond` while removing the
sequence-level content-copy route.

Config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_nopromptxattn_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts.yaml
```

Run directory:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_nopromptxattn_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts_v1
```

Result:

| Step | val_recon_corr | val_recon_mse | val_recon_mae | train_recon_corr | elapsed | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2250 | `0.56430187` | `0.67164683` | `0.60580907` | `0.58734846` | `184.49 s` | stop |

Conclusion:

Short fine-tuning cannot recover the E2 score after removing raw prompt
cross-attention. This repeats the E1/E4 conclusion with a stronger
initialization and lower LR: the current high-corr model uses temporal prompt
tokens as a major condition, not only as pooled speaker statistics.

### E7: Same-Clip Non-Prefix Prompt Window

Hypothesis:

The practical silent-ref workflow often uses an unmasked segment from the same
video, not necessarily the target opening. Training with a same-clip non-prefix
window should preserve speaker/session information while breaking the direct
copying path for the first 3.04 s. The dataset mode
`audio_prompt_ref_mode=self_random_window` currently takes a deterministic
post-prefix window starting at frame 38 when the clip is long enough, falling
back to the prefix for short clips.

Code/config:

- `FMAVSRDataset(audio_prompt_ref_mode="self_random_window")`
  - uses the same clip as reference;
  - starts the prompt window at frame 38 when possible;
  - keeps shape `(audio_prompt_frames, 512)`.
- `scripts/train_fm_avsr.py`
  - accepts the new ref mode from CLI/config.

Config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_selfwindowprompt_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts.yaml
```

Run directory:

```text
runs/fm_avsr/lipavsr_59144_timbre3s_selfwindowprompt_promptstats005_residual_samplecorr02_lossstart38_from2000_recon_textjson_wordts_v1
```

Result:

| Step | val_recon_corr | val_recon_mse | val_recon_mae | train_recon_corr | elapsed | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2250 | `0.57978476` | `0.65472100` | `0.59459168` | `0.60845828` | `158.70 s` | stop |

Conclusion:

This is the best prompt-cleaning direction so far: it is much closer to E2 than
pooled-only, no-prompt-cross-attn, or same-parent prompting. However, it still
drops about `0.0046` validation corr versus E2 (`0.58434393`) and therefore
does not satisfy the current success gate. The result suggests that same-video
non-prefix prompt training is worth a longer run or from-scratch schedule, but
the current short adaptation is not enough to claim the timbre issue is fixed
while beating the historical best.
