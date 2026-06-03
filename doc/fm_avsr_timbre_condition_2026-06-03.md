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
| 30k audio prompt tokens, shifted condition | 2500 | 0.00798798 | 1.28502175 | 0.87642337 | full 1k negative control, `condition_shift=1` |

The best verified full-eval result is `0.56893905` from the audio prompt token
model. This is:

- +0.03595571 over the strict 30k no-timbre baseline
- +0.00565016 over the 30k mean/std timbre model
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

The curve was still increasing at 2500 steps, but the improvement had slowed to
about +0.0039 over the previous 500-step interval. A short continuation to 3000
steps is the next low-risk check.

## Interpretation

Manual timbre control is practical in this codebase. The mean/std prompt is a
simple global condition and the stronger token prompt gives a measurable but
small additional gain. The remaining gap to 0.6 is likely not just "missing
speaker identity"; the model is still bottlenecked by the visual/text-to-Mimi
latent prediction problem and by how strongly the decoder can use a short
reference prompt.

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
