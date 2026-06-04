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
