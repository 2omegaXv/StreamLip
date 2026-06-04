# FM-AVSR Timbre Condition Experiments

Date: 2026-06-03

## Question

The current best FM-AVSR recon model has intelligible speech but weak individual
timbre. The generated voice often collapses to broad male/female traits, which
suggests the existing face-derived `speaker_emb.npy` is not enough for detailed
voice identity.

This run tests whether explicit audio-derived timbre conditioning can improve
validation latent correlation. The target for this investigation is
`eval mean_corr > 0.6` on the shared 1k validation split.

## Conditions

Two same-clip audio prompt conditions were tested. Both are derived from the
first 3 seconds of normalized Mimi latents, and all reported prompt-conditioned
metrics skip those first 38 frames with `metric_start_frame: 38`.

### Global Mean/Std Timbre

`timbre_cond.npy` is a lightweight per-clip summary:

- source: first 3 seconds of normalized `latent.npz`
- frame rate: 12.5 Hz
- prompt frames: 38
- vector: `concat(mean(prefix_latent), std(prefix_latent))`
- dim: 1024
- script: `scripts/extract_timbre_cond.py`

The model concatenates this global 1024-d vector to every frame before
`cond_proj`.

### Audio Prompt Tokens

`audio_prompt` is a stronger prompt path added after the mean/std condition:

- source: first 38 normalized Mimi latent frames
- shape per clip: `(38, 512)`
- zero padded only if a clip is shorter than 38 frames
- projected by `audio_prompt_proj`
- injected as cross-attention condition tokens

This is an oracle-style same-clip prompt experiment. It answers whether a
reference voice segment helps the latent predictor when the reference is
available. It does not yet solve the harder deployment case of choosing an
external same-speaker prompt.

## Fixed Setup

All main runs use the current deterministic recon setup:

- visual prior: `avsr_enc_lipavsr.npy`
- text: `text_json`
- text alignment: `word_timestamps`
- extra condition: predicted log-RMS energy
- loss: `loss_fm_weight: 0`, `lambda_recon: 1`, `lambda_energy: 0.1`
- validation: `configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt`

Train splits:

- 1k: `configs/eval_splits/pretrain_lipavsr_train1000_seed44_excl_val1000_seed43.txt`
- 10k: `configs/eval_splits/pretrain_len80_260_lipavsr_train10000_seed43.txt`
- 30k: `configs/eval_splits/pretrain_lipavsr_train30000_seed44_excl_val1000_seed43.txt`

## Results

### 1k and 10k Direction Checks

| Run | Best/eval step | Strict val corr | Notes |
| --- | ---: | ---: | --- |
| 1k baseline | 500 | 0.40286663 | `metric_start_frame=38` |
| 1k mean/std timbre | 500 | 0.42360439 | +0.02073776 over 1k baseline |
| 10k baseline | 1500 | 0.50868050 | offline strict eval |
| 10k mean/std timbre | 1000 | 0.53999515 | +0.03131465 over 10k baseline |

The 1k and 10k checks showed consistent positive movement from explicit timbre
conditioning, but the scores stayed well below 0.6. The 10k timbre run peaked
around 1000 steps and then declined, so scale-up runs were monitored with early
stopping in mind.

### 30k Scale-Up

| Run | Step | Eval corr | Eval MSE | Eval MAE | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 30k no-timbre baseline | 2000 | 0.53298334 | 0.72177195 | 0.63384026 | strict offline eval, `metric_start_frame=38` |
| 30k mean/std timbre | 2500 | 0.56328889 | 0.68858632 | 0.61404544 | full 1k eval |
| 30k mean/std + corr loss | 3000 | 0.56697732 | n/a | n/a | training-val only; `lambda_recon_corr=0.2` continuation |
| 30k mean/std + audio prompt tokens | 2500 | 0.56893905 | 0.68224197 | 0.61025584 | full 1k eval |
| 30k mean/std + audio prompt tokens | 3000 | 0.56967886 | 0.68265299 | 0.60941165 | full 1k eval, 2500-step continuation |
| 30k mean/std + audio prompt tokens + pooled prompt cond | 2500 | 0.56971665 | 0.68110923 | 0.61001512 | full 1k eval |
| 30k pooled prompt residual from step2500 | 1500 | 0.57236865 | 0.67809615 | 0.60854835 | full 1k eval; frozen pooled prompt baseline |
| 30k pooled prompt residual + corr loss | 2500 | 0.57255184 | 0.67793650 | 0.60844724 | full 1k eval; continuation from residual step1500 |
| 30k audio prompt tokens + loss start 38 | 1000 | n/a | n/a | n/a | early stopped; training-val corr 0.54083031 |
| 30k pooled prompt, 8 DiT layers | 2000 | n/a | n/a | n/a | early stopped; training-val corr 0.56734583 |
| 30k audio prompt tokens, shifted condition | 2500 | 0.00798798 | 1.28502175 | 0.87642337 | full 1k negative control, `condition_shift=1` |

The best verified full-eval result is `0.57255184` from the residual refinement
plus a light corr-loss continuation. This is:

- +0.03956850 over the strict 30k no-timbre baseline
- +0.00926294 over the 30k mean/std timbre model
- +0.00283519 over the pooled audio prompt model
- still below the target `0.6`

The shifted-condition negative control drops to approximately zero correlation,
which confirms that the prompt path is being used by the model rather than being
ignored.

### Audio Prompt Token Training Curve

Run:
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_recon_textjson_wordts_v1`

| Step | Val recon corr | Elapsed |
| ---: | ---: | ---: |
| 500 | 0.51615003 | 323.5024s |
| 1000 | 0.54662385 | 621.8489s |
| 1500 | 0.56017650 | 930.7614s |
| 2000 | 0.56724529 | 1234.7743s |
| 2500 | 0.57115404 | 1541.1965s |
| 3000 | 0.57309073 | 321.1879s continuation |

The 2500 to 3000 continuation only improved full 1k eval from `0.56893905` to
`0.56967886`, so simply extending the same run is not likely to close the gap to
0.6 quickly.

### Pooled Audio Prompt Condition

Run:
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_pool_recon_textjson_wordts_v1`

This variant keeps the cross-attention prompt tokens and also adds their mean
pooled projection into the frame condition, so prompt information reaches both
the per-frame condition stream and the DiT global modulation path.

| Step | Val recon corr | Elapsed |
| ---: | ---: | ---: |
| 500 | 0.51763669 | 329.7789s |
| 1000 | 0.54430954 | 625.7661s |
| 1500 | 0.55867916 | 923.1617s |
| 2000 | 0.56680157 | 1228.7904s |
| 2500 | 0.57077152 | 1511.6859s |

Full 1k eval at step2500 is `0.56971665`. This is only `+0.00077760` over the
token-only prompt step2500 full eval, and only `+0.00003779` over the token-only
step3000 full eval. It is therefore a measurable but marginal improvement.

### Prompt-Skipped Reconstruction Loss

Run:
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_lossstart38_recon_textjson_wordts_v1`

This variant keeps the token-only audio prompt architecture but sets
`loss_start_frame: 38`, so deterministic reconstruction losses skip the same
prompt segment that eval skips.

| Step | Val recon corr | Elapsed |
| ---: | ---: | ---: |
| 500 | 0.51479245 | 337.1739s |
| 1000 | 0.54083031 | 644.0767s |

It was early stopped after step1000 because it trailed the token-only prompt run
at both shared checkpoints (`0.5148` vs `0.5162` at 500, `0.5408` vs `0.5466` at
1000). Skipping the prompt segment in the loss did not help.

### Larger 8-Layer DiT Head

Run:
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_pool_8l_recon_textjson_wordts_v1`

This variant keeps pooled audio prompt conditioning and increases the FM head
from 6 to 8 DiT layers. Parameter count increases from 39.8M to 51.4M.

| Step | Val recon corr | Elapsed |
| ---: | ---: | ---: |
| 500 | 0.51769584 | 337.9654s |
| 1000 | 0.54591195 | 650.9076s |
| 1500 | 0.55976608 | 961.1462s |
| 2000 | 0.56734583 | 1269.1585s |

It was early stopped after step2000. The curve is not meaningfully ahead of the
6-layer pooled prompt run, and it remains below the token-only prompt run at
1500 while only matching it at 2000. Extra depth alone did not show a path
toward 0.6.

### Residual Refinement From Pooled Audio Prompt

Run:
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_pool_residual_from2500_recon_textjson_wordts_v1`

This variant freezes
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_pool_recon_textjson_wordts_v1/step_002500.pt`
as a baseline and trains a second 6-layer head to predict an additive residual:
`pred = baseline + residual`. It keeps the same timbre mean/std condition,
38-frame audio prompt tokens, and pooled prompt condition. `lambda_energy` is
set to zero because the energy condition comes from the frozen baseline in this
mode.

| Step | Val recon corr | Elapsed |
| ---: | ---: | ---: |
| 500 | 0.57307028 | 340.2173s |
| 1000 | 0.57356842 | 678.2311s |
| 1500 | 0.57384256 | 988.4774s |

Full 1k eval at step1500 is `0.57236865`, with MSE `0.67809615` and MAE
`0.60854835`. Residual refinement gives the current best full-eval score and
improves the previous best by `+0.00265201`, but the curve is nearly saturated
by step1500. This is a useful correction layer, not enough by itself to close
the remaining gap to 0.6.

### Residual Corr-Loss Continuation

Run:
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_pool_residual_corr01_from1500_recon_textjson_wordts_v1`

This continuation resumes the residual checkpoint at step1500 and adds
`lambda_recon_corr=0.1` with `lr=5e-5`. The objective is still mostly MSE, but
with a small direct pressure on the same global Pearson correlation used by the
eval metric.

| Step | Val recon corr | Elapsed |
| ---: | ---: | ---: |
| 2000 | 0.57419035 | 356.4885s |
| 2500 | 0.57420011 | 703.2253s |

Full 1k eval at step2500 is `0.57255184`, with MSE `0.67793650` and MAE
`0.60844724`. This is only `+0.00018318` over the residual step1500 full eval,
so corr loss is directionally positive but effectively saturated in this setup.

### Lip-AVSR Data Availability and 50k Scale-Up

Before the data scale-up, the encoded `avsr_enc_lipavsr.npy` pool was the main
bottleneck:

- `avsr_enc_lipavsr.npy`: 32,238 clips
- current val1000 all had `avsr_enc_lipavsr.npy`
- trainable `avsr_enc_lipavsr.npy` after excluding current val1000: 31,238 clips
- current 30k split already covered all but 1,238 of those trainable encoded clips

A naive larger split built from older length filters was not usable for
lip-AVSR training because many entries lacked `avsr_enc_lipavsr.npy` and some
also lacked text hidden states or timbre conditions. The scale-up therefore used
a ready-candidate filter before running the Auto-AVSR encoder.

New 20k increment:

- split: `configs/eval_splits/pretrain_lipavsr_missing_avsr_enc_ready_train20000_excl_val1000_seed43.txt`
- candidate requirements before encoding: `lip_avsr.npy`, `latent.npz`,
  `speaker_emb.npy`, `smollm2_h_text_json.npy`, `audio.wav`, `text.json`
- val overlap: 0
- old 30k overlap: 0
- duplicates: 0
- `timbre_cond.npy`: generated for all 20,000 clips
- `avsr_enc_lipavsr.npy`: generated for all 20,000 clips
- Auto-AVSR encode result: `Done: 20000  Skipped: 0  Errors: 0`
- encode time: about 0.4h

Combined 50k split:

- split: `configs/eval_splits/pretrain_lipavsr_train50000_seed44_plus_ready_missing20000_excl_val1000_seed43.txt`
- train clips: 50,000 unique
- val clips: 1,000 unique
- train/val overlap: 0

Sample checks on the new increment showed expected file shapes, e.g.
`lip_avsr.npy` at 25Hz, `avsr_enc_lipavsr.npy` as `(T, 768)` float16,
`latent.npz` as about `(T/2, 512)` float16, and `timbre_cond.npy` as `(1024,)`
float16. A debug smoke run of the 50k continuation config resumed the 30k pooled
prompt checkpoint at step2500 and completed step2501 successfully with debug
val recon corr `0.5694`.

The 50k continuation config is
`configs/fm_avsr_lipavsr_50000_timbre3s_audioprompt38_pool_from2500_recon_textjson_wordts.yaml`.
It resumes
`runs/fm_avsr/lipavsr_30000_timbre3s_audioprompt38_pool_recon_textjson_wordts_v1/step_002500.pt`,
uses the same timbre and audio prompt conditioning, switches to the 50k split,
and lowers the continuation learning rate to `1e-4`.

Remaining ready-candidate scale-up:

- split: `configs/eval_splits/pretrain_lipavsr_remaining9144_excl_train50_val1000_seed43.txt`
- candidates: 9,144 clips not in the 50k train split and not in val1000
- all had `lip_avsr.npy`, `latent.npz`, `speaker_emb.npy`,
  `smollm2_h_text_json.npy`, `audio.wav`, and `text.json`
- `timbre_cond.npy`: generated for all 9,144 clips
- `avsr_enc_lipavsr.npy`: generated for all 9,144 clips
- Auto-AVSR encode result: `Done: 9144  Skipped: 0  Errors: 0`
- encode time: about 0.2h

Combined 59,144 split:

- split: `configs/eval_splits/pretrain_lipavsr_train59144_seed44_plus_ready_remaining9144_excl_val1000_seed43.txt`
- train clips: 59,144 unique
- val clips: 1,000 unique
- train/val overlap: 0
- sample checks showed `avsr_enc_lipavsr.npy` as `(T, 768)` and
  `timbre_cond.npy` as `(1024,)`

### 50k/59k Continuation Results

| Run | Step | Eval corr | Eval MSE | Eval MAE | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 50k pooled prompt continuation | 3500 | 0.58008041 | 0.66951530 | 0.60291130 | resumed 30k pooled prompt step2500 |
| 50k pooled prompt continuation | 4000 | 0.58013729 | 0.66926031 | 0.60292721 | saturated; best 50k base |
| 50k pooled prompt + GT energy eval | 4000 | 0.58030405 | 0.66907275 | 0.60281535 | diagnostic upper-bound for energy condition |
| 50k residual from step4000 | 1000 | 0.58091919 | 0.66841430 | 0.60245221 | best verified full eval in this note |
| 50k residual + timbre-stats loss | 1500 | 0.58105725 | 0.66844833 | 0.60244348 | `lambda_timbre_stats=0.2`, `lambda_recon_corr=0.05` |
| 50k residual + frozen-base latent condition | 500 | 0.58072395 | 0.66876355 | 0.60250444 | residual head also receives frozen base recon latent |
| 59k pooled prompt continuation | 5000 | 0.58143902 | 0.66799185 | 0.60200530 | resumed 50k base step4000, lr `5e-5` |
| 59k residual from step5000 | 1000 | 0.58167589 | 0.66776208 | 0.60182909 | previous best full eval |
| 59k residual + denoise loss | 1250 | 0.58173362 | 0.66768414 | 0.60181177 | `lambda_denoise=0.1`, `denoise_t=[0, 0.1]` |
| 59k residual + denoise loss | 1500 | 0.58176819 | 0.66763995 | 0.60180507 | effectively saturated |
| 59k residual + CTC top-k condition | 1500 | 0.58177180 | 0.66764077 | 0.60178911 | `ctc_condition_mode=topk`; effectively tied with denoise |
| 59k residual + per-sample corr loss | 1500 | 0.58177344 | 0.66763517 | 0.60181215 | `lambda_sample_corr=0.2`; effectively tied with CTC/denoise |

The 50k scale-up improved the best full eval from `0.57255184` to
`0.58091919`. Residual timbre-stat continuation added only `+0.00013806`
over the 50k residual checkpoint and slightly worsened MSE, so it is not a
meaningful path toward `0.6`.

The 59k scale-up is directionally positive but still small. It improved the
base full eval from `0.58013729` to `0.58143902`, and residual refinement on
that base reached `0.58167589`. Short denoise, CTC top-k, and per-sample corr
continuations all land around `0.58177`, so the current deterministic residual
head is still far below the target `0.6`.

The denoise continuation resumed
`runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_residual_from5000_recon_textjson_wordts_v1/step_001000.pt`
with frozen base
`runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_from4000_recon_textjson_wordts_v1/step_005000.pt`,
`lr=5e-5`, `lambda_recon=1.0`, `lambda_denoise=0.1`, and
`denoise_t_min/max=0/0.1`. It stayed within the 1-hour budget. The small
training-val improvement did not translate into a meaningful full-eval jump.

While validating this run, an eval config bug was found: `eval_fm_avsr.py`
accepted `--residual_base_ckpt`, but did not load `residual_base_ckpt` from YAML
configs. This made residual checkpoints evaluate as raw residuals unless the
base checkpoint was passed explicitly on the command line. The bad symptom was
a false full-eval corr around `0.038`. The parser now loads
`residual_base_ckpt` from config and is covered by a unit test. The valid
denoise rows above are the `fixbase` evals, whose logs explicitly loaded the
frozen residual baseline.

The GT-energy diagnostic moved the 50k base only from `0.58013729` to
`0.58030405`. This rules out predicted log-RMS energy as the main bottleneck in
the current best architecture.

The frozen-base latent condition test added the frozen baseline reconstruction
as an extra 512-d per-frame condition for the residual head. It reached
`0.58072395` at step500, below the plain residual checkpoint (`0.58091919`) and
the timbre-stats continuation (`0.58105725`). Exposing the base prediction
directly to the residual head therefore did not unlock additional validation
correlation.

### CTC Top-k Condition Continuation

Run:
`runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_residual_ctctopk_from1000_recon_textjson_wordts_v1`

Config:
`configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_residual_ctctopk_from1000_recon_textjson_wordts.yaml`

This continuation adds Auto-AVSR CTC top-k posterior tokens as an extra
frame-level condition while keeping the same timbre mean/std condition,
38-frame audio prompt tokens, pooled prompt condition, and residual baseline.
Because CTC top-k expands `cond_proj`, the run uses a partial checkpoint load:
all old tensors with matching shape are loaded, the old `cond_proj.weight`
columns are copied, and the new CTC columns are zero-initialized. The frozen
residual base uses the same partial load, so its output is unchanged when the
new CTC inputs are not used.

Training:

- resume: 59k residual step1000
- residual base: 59k pooled prompt step5000
- `ctc_condition_mode: topk`
- `ctc_topk: 4`
- `ctc_token_emb_dim: 32`
- `lr: 5e-5`
- max extra steps: 500
- small validation recon corr: `0.58427014` at step1250,
  `0.58427079` at step1500

Full val1000 eval at step1500:

- metrics:
  `eval_out/lipavsr_59144_timbre3s_audioprompt38_pool_residual_ctctopk_step1500_fixval1000_metrics/metrics.json`
- `mean_corr: 0.58177180`
- `mean_mse: 0.66764077`
- `mean_mae: 0.60178911`

This is only `+0.00000361` over the denoise step1500 full eval
(`0.58176819`). Per-clip comparison is split almost exactly evenly
(`492` clips better, `508` worse), so the CTC top-k condition is effectively a
tie rather than a meaningful improvement. It does not move the system toward
the `0.6` target.

While running this eval, another config safety issue was found: training YAMLs
contain both `clip_list` and `val_clip_list`, but `eval_fm_avsr.py` originally
used `clip_list`, which silently evaluates the train split unless the val split
is passed explicitly on the command line. Eval config loading now maps
`val_clip_list` to `clip_list` when the user did not explicitly provide
`--clip_list`, and this behavior is covered by unit tests.

### Per-Sample Corr-Loss Continuation

Run:
`runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_residual_samplecorr02_from1000_recon_textjson_wordts_v1`

Config:
`configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_residual_samplecorr02_from1000_recon_textjson_wordts.yaml`

This continuation adds `masked_sample_corr_loss`, which optimizes the mean
per-clip Pearson correlation after the 38-frame audio prompt. The earlier
`masked_corr_loss` was batch-global, while the reported validation metric is an
average of per-clip correlations. This test checks whether matching the training
auxiliary loss to the eval aggregation helps.

Training:

- resume: 59k residual step1000
- residual base: 59k pooled prompt step5000
- `lambda_sample_corr: 0.2`
- `lr: 5e-5`
- max extra steps: 500
- small validation recon corr: `0.58424845` at step1250,
  `0.58426899` at step1500

Full val1000 eval at step1500:

- metrics:
  `eval_out/lipavsr_59144_timbre3s_audioprompt38_pool_residual_samplecorr02_step1500_val1000_metrics/metrics.json`
- `mean_corr: 0.58177344`
- `mean_mse: 0.66763517`
- `mean_mae: 0.60181215`

This is only `+0.00000164` over the CTC top-k step1500 full eval
(`0.58177180`). Per-clip comparison is again split almost exactly evenly
(`509` clips better, `491` worse), with p10/p90 deltas of about `-0.000392` and
`+0.000385`. The loss is implemented and useful as a diagnostic, but this
weight and continuation do not produce a meaningful move toward `0.6`.

### Prompt Calibration Diagnostics

A one-off full-val diagnostic tested whether the first 38 prompt frames could be
used to linearly calibrate the rest of the predicted latent without additional
training. All metrics used the 50k residual step1000 checkpoint and
`metric_start_frame=38`.

| Calibration | Eval corr | Eval MSE | Eval MAE |
| --- | ---: | ---: | ---: |
| raw residual prediction | 0.58091919 | 0.66841430 | 0.60245221 |
| clip mean-shift, alpha 0.25 | 0.58078388 | 0.66857146 | 0.60255020 |
| clip mean-shift, alpha 1.0 | 0.57779765 | 0.67218273 | 0.60470650 |
| clip per-dim affine, alpha 0.25 | 0.58062730 | 0.66894736 | 0.60264453 |
| clip per-dim affine, alpha 1.0 | 0.57656243 | 0.67530594 | 0.60596099 |
| leaky full-val post-frame affine, alpha 1.0 | 0.58113608 | 0.66799752 | 0.60256379 |
| global prompt-frame affine, alpha 1.0 | 0.58070137 | 0.66890699 | 0.60258606 |

The same-clip prompt is already being used by the model, but its residual error
is not a simple per-clip mean/std/affine mismatch. Even a leaky affine fitted on
the full validation set only reaches `0.58113608`. This makes further linear
prompt calibration or energy tuning unlikely to close the remaining gap.

### Timbre-Stats Loss

Added `masked_timbre_stats_loss` in `scripts/train_fm_avsr.py`. It matches the
per-sample post-prompt latent mean and std over valid frames:

```text
loss_timbre_stats = mse(mean(pred), mean(target)) + mse(std(pred), std(target))
```

The implementation is controlled by `--lambda_timbre_stats`, logged in
`metrics.csv`, saved in checkpoints, and covered by unit tests. The 50k residual
continuation config is
`configs/fm_avsr_lipavsr_50000_timbre3s_audioprompt38_pool_residual_timbrestats_from1000_recon_textjson_wordts.yaml`.

Training:

- resume: 50k residual step1000
- residual base: 50k pooled prompt step4000
- `lr: 5e-5`
- `lambda_timbre_stats: 0.2`
- `lambda_recon_corr: 0.05`
- `loss_start_frame: 38`
- max extra steps: 500
- validation recon corr at step1500: `0.5824`

Full eval at step1500 was `0.58105725`. This is a weak positive on corr but too
small to justify further tuning of this exact loss.

## Interpretation

Manual timbre control is practical in this codebase. The mean/std prompt is a
simple global condition, the stronger token prompt gives a measurable but small
additional gain, and residual refinement on top of the best prompt model gives a
further small correction. A light corr loss, GT energy, prompt affine
calibration, and post-prompt timbre-stat matching all add almost nothing after
residual training. Passing the frozen base reconstruction back in as a residual
condition also fails to improve full eval. Adding CTC top-k posterior tokens as
an extra per-frame condition is also effectively a tie, improving full-val corr
by only `0.00000361`. A per-sample corr loss aligned to the full-eval
aggregation is also effectively tied, improving full-val corr by only
`0.00000164` over CTC top-k. Adding the remaining 9,144 ready lip-AVSR clips is
the strongest positive move in the latest batch, but it only raises full eval to
`0.58167589`, and the best short continuations only raise it to about
`0.58177`. The remaining gap to 0.6 is likely not just "missing speaker
identity"; the deterministic MSE-style latent head is still averaging over
speaker and spectral detail that is not recoverable from the current condition
fusion.

The current best prompt is same-clip and should be treated as an upper-bound
style diagnostic. A production-style voice control path should next test
same-speaker external prompts, stronger prompt fusion, an explicit speaker /
prompt consistency loss, or a stronger sampled/denoising generative objective
instead of continuing to tune deterministic recon losses.

## Verification

Code support for `timbre_cond` and `audio_prompt` was covered by unit tests:

- `uv run python -m unittest tests.test_fm_avsr_dataset tests.test_timbre_condition tests.test_eval_fm_avsr tests.test_fm_head_temporal_condition -v`
- `uv run python -m py_compile scripts/train_fm_avsr.py scripts/eval_fm_avsr.py src/streaminlip/fm_avsr_dataset.py src/streaminlip/v2/fm_head.py`
- `uv run python -m unittest tests.test_eval_fm_avsr tests.test_fm_avsr_dataset -v`
- `uv run python -m py_compile scripts/eval_fm_avsr.py scripts/train_fm_avsr.py`
- `uv run python -m unittest tests.test_fm_avsr_dataset.FMAVSRDatasetTest.test_load_fm_head_state_can_expand_cond_projection tests.test_fm_avsr_dataset.FMAVSRDatasetTest.test_partial_load_keeps_old_fm_output_when_new_ctc_condition_is_zero tests.test_eval_fm_avsr.EvalFMAVSRTest.test_parse_args_loads_allow_partial_resume_from_config -v`
- `uv run python -m unittest tests.test_eval_fm_avsr.EvalFMAVSRTest.test_parse_args_prefers_val_clip_list_from_train_config tests.test_eval_fm_avsr.EvalFMAVSRTest.test_parse_args_cli_clip_list_overrides_config_val_clip_list tests.test_eval_fm_avsr.EvalFMAVSRTest.test_parse_args_loads_allow_partial_resume_from_config -v`
- `uv run python -m unittest tests.test_fm_avsr_dataset.FMAVSRDatasetTest.test_masked_sample_corr_loss_averages_per_clip_correlation tests.test_fm_avsr_dataset.FMAVSRDatasetTest.test_combine_training_losses_can_include_sample_corr_loss -v`
- `uv run python -m unittest tests.test_eval_fm_avsr tests.test_fm_avsr_dataset -v`
- `uv run python -m py_compile scripts/train_fm_avsr.py scripts/eval_fm_avsr.py`

All individual training/eval runs in this note were kept under the 1-hour limit.
