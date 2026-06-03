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

## Interpretation

Manual timbre control is practical in this codebase. The mean/std prompt is a
simple global condition, the stronger token prompt gives a measurable but small
additional gain, and residual refinement on top of the best prompt model gives a
further small correction. A light corr loss adds almost nothing after residual
training. The remaining gap to 0.6 is likely not just "missing speaker
identity"; the model is still bottlenecked by the visual/text-to-Mimi latent
prediction problem and by the amount of newly encoded lip-AVSR data available to
the FM head.

The current best prompt is same-clip and should be treated as an upper-bound
style diagnostic. A production-style voice control path should next test
same-speaker external prompts, stronger prompt fusion, or an explicit speaker /
prompt consistency loss.

## Verification

Code support for `timbre_cond` and `audio_prompt` was covered by unit tests:

- `python -m unittest tests.test_timbre_condition tests.test_fm_avsr_dataset -v`
- `python -m unittest tests.test_eval_fm_avsr tests.test_fm_head_temporal_condition -v`
- `python -m py_compile src/streaminlip/v2/fm_head.py src/streaminlip/fm_avsr_dataset.py scripts/train_fm_avsr.py scripts/eval_fm_avsr.py`

All individual training/eval runs in this note were kept under the 1-hour limit.
