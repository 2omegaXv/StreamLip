# FM-AVSR Timbre Condition Experiments

Date: 2026-06-03

## Question

The current best FM-AVSR recon model has intelligible speech but weak individual
timbre. The generated voice often collapses to broad male/female traits, which
suggests the existing face-derived `speaker_emb.npy` is not enough for detailed
voice identity.

This run tests whether an explicit audio-prompt timbre condition improves
validation latent correlation, with target `val_recon_corr > 0.6`.

## Condition

`timbre_cond.npy` is a lightweight first-pass audio prompt condition derived from
existing Mimi latents:

- source: first 3 seconds of normalized `latent.npz`
- frame rate: 12.5 Hz
- prompt frames: 38
- vector: `concat(mean(prefix_latent), std(prefix_latent))`
- dim: 1024
- script: `scripts/extract_timbre_cond.py`

The first 3 seconds are a same-clip audio prompt, so reported experiment metrics
use `metric_start_frame: 38` to avoid scoring the prompt segment itself.

## Fixed Architecture

All runs use the current best deterministic recon setup:

- visual prior: `avsr_enc_lipavsr.npy`
- text: `text_json`
- text alignment: `word_timestamps`
- extra condition: predicted log-RMS energy
- loss: `loss_fm_weight: 0`, `lambda_recon: 1`, `lambda_energy: 0.1`
- timbre model change: concatenate a global 1024-d timbre vector to every latent
  frame before `cond_proj`

## Splits

Validation is shared:

- `configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt`

Train splits:

- 1k: `configs/eval_splits/pretrain_lipavsr_train1000_seed44_excl_val1000_seed43.txt`
- 10k: `configs/eval_splits/pretrain_len80_260_lipavsr_train10000_seed43.txt`
- 30k: `configs/eval_splits/pretrain_lipavsr_train30000_seed44_excl_val1000_seed43.txt`

Timbre extraction lists:

- 1k+val: `configs/eval_splits/pretrain_lipavsr_train1000_plus_val1000_seed44_seed43.txt`
- 30k+val: `configs/eval_splits/pretrain_lipavsr_train30000_plus_val1000_seed44_seed43.txt`

## Baselines

Existing full-segment baseline results:

| Train | Config | Best val recon corr |
| --- | --- | ---: |
| 10k | `configs/fm_avsr_len80_260_lipavsr_10000_pred_energy_recon_textjson_wordts.yaml` | 0.51326406 |
| 30k | `configs/fm_avsr_lipavsr_30000_pred_energy_recon_textjson_wordts.yaml` | 0.53649477 |

Need to re-evaluate baseline checkpoints with `metric_start_frame=38` for a
strict same-prompt comparison.

## Planned Runs

| Run | Config | Purpose |
| --- | --- | --- |
| 1k baseline | `configs/fm_avsr_lipavsr_1000_pred_energy_recon_textjson_wordts.yaml` | small-data no-timbre control |
| 1k timbre | `configs/fm_avsr_lipavsr_1000_timbre3s_pred_energy_recon_textjson_wordts.yaml` | quick direction check |
| 10k timbre | `configs/fm_avsr_len80_260_lipavsr_10000_timbre3s_pred_energy_recon_textjson_wordts.yaml` | compare to existing 10k baseline |
| 30k timbre | `configs/fm_avsr_lipavsr_30000_timbre3s_pred_energy_recon_textjson_wordts.yaml` | scale-up target, stop if plateau below 0.6 |

Each run should stay below 1 hour. Monitor `val_metrics.csv` every 500 steps and
stop when `val_recon_corr` plateaus or drops after the current best checkpoint.

## Results

### 1k Strict Prompt-Skipped Metrics

Both 1k runs used `metric_start_frame: 38`, so the first 3-second audio prompt
segment is excluded from reported validation metrics.

| Run | Best step | Best val recon corr | Elapsed at best | Final val recon corr |
| --- | ---: | ---: | ---: | ---: |
| 1k baseline | 500 | 0.40286663 | 355.8389s | 0.32069633 |
| 1k timbre | 500 | 0.42360439 | 307.7510s | 0.34726166 |

Initial result: timbre conditioning is positive on 1k, but small
(`+0.02073776` corr). Both runs overfit after 500 steps, so larger-data runs
should watch the 500/1000 step validation points closely and stop when
`val_recon_corr` drops after the current best.

### 10k Strict Prompt-Skipped Metrics

The 10k timbre run was interrupted once after step 917 and resumed from
`step_000500.pt`. The optimizer state is not restored by the current training
script, but the schedule is fixed LR, so the main caveat is duplicated train
rows between 501-917 in `metrics.csv`; `val_metrics.csv` remains clean.

| Run | Best step | Best val recon corr | Elapsed at best | Final val recon corr |
| --- | ---: | ---: | ---: | ---: |
| 10k baseline | 1500 | 0.51326406 | 574.5242s | 0.50417887 |
| 10k timbre | 1000 | 0.53949885 | 263.2073s after resume | 0.52915170 |

10k result: timbre conditioning improves best validation recon corr by
`+0.02623479` over the existing 10k baseline. It still does not reach 0.6.
The best step is 1000; 1500 and 2000 decline, so this run should use
`step_001000.pt` for downstream eval/listening.
