# FM AVSR experiment status, 2026-06-01

## Current target

The current FM AVSR system predicts 12.5 Hz Mimi latents from aligned video AVSR features, SmolLM2 text hidden states, and a speaker embedding. The practical target is to improve held-out sampled latent correlation and, ultimately, decoded speech quality.

The recent experiments suggest a repeatable validation plateau around `val_sample_corr ~= 0.35` for the current FM objective and conditioning path.

## Data setup used by the latest experiments

- Training split: `configs/eval_splits/pretrain_len120_220_train12000_seed43.txt`
- Held-out split: `configs/eval_splits/pretrain_len120_220_heldout1000_seed43.txt`
- Latent length filter: `120-220` Mimi latent frames, about `9.6-17.6s` at 12.5 Hz.
- Batch size: `64`
- LR schedule: fixed `3e-4`
- Evaluation: sampled endpoint metrics with `eval_sample_nfe=4`, plus train-batch metrics.

## Recent results

### `lambda_recon=0.1`, `lambda_sample_recon=1.0`

Run:

```text
runs/fm_avsr/len120_220_12000_seed43_varlen_recon01_sample_nfe4_v1
```

Important points:

```text
step  9000: val_corr 0.3465, train_corr 0.3623, gap 0.0158
step 10000: val_corr 0.3545, train_corr 0.3907, gap 0.0362
step 15000: val_corr 0.3556, train_corr 0.4266, gap 0.0710
step 17000: val_corr 0.3531, train_corr 0.3936, gap 0.0404
```

Observation: validation correlation reached about `0.355` and then stopped improving while the train metric continued to rise. This run was stopped because the additional steps mostly increased train-set fitting.

### `lambda_recon=1.0`, `lambda_sample_recon=1.0`

Run:

```text
runs/fm_avsr/len120_220_12000_seed43_varlen_recon1_sample_nfe4_v1
```

Important points:

```text
step  9000: val_corr 0.3489, train_corr 0.3839
step 10000: val_corr 0.3518, train_corr 0.3802
step 12000: val_corr 0.3548, train_corr 0.4048
step 14000: val_corr 0.3581, train_corr 0.4020
```

Observation: a stronger deterministic recon auxiliary loss was slightly more stable, but it did not materially break the same `0.35` region. The improvement over `lambda_recon=0.1` is too small to treat loss weighting as the main bottleneck.

## Current interpretation

The plateau is unlikely to be solved by only changing `lambda_recon` or running the same configuration longer. The model can keep improving on the training batch, but held-out sampled correlation barely moves. That points to one of these bottlenecks:

1. The condition path may not contain enough usable information for held-out latent prediction.
2. The current DiT conditioning path may not expose frame-level video/text evidence strongly enough.
3. The FM sampled endpoint objective may be harder than necessary before we know the deterministic condition upper bound.

## Next diagnostic: deterministic recon upper bound

A new training mode now supports disabling the FM loss with:

```yaml
loss_fm_weight: 0.0
lambda_recon: 1.0
lambda_sample_recon: 0.0
eval_sample_nfe: 0
```

The purpose is not to replace FM inference. It is a diagnostic upper-bound run:

- If deterministic recon validation correlation is also stuck near `0.35-0.4`, the main issue is likely condition quality, temporal alignment, or architecture capacity.
- If deterministic recon validation correlation rises clearly above `0.5`, the condition is useful and the main issue is likely the FM sampling objective or sampling-aware training.

New config:

```text
configs/fm_avsr_len120_220_12000_pure_recon_fixedlr.yaml
```

Recommended launch command:

```bash
uv run python scripts/train_fm_avsr.py --config configs/fm_avsr_len120_220_12000_pure_recon_fixedlr.yaml
```

Watch `val_recon_corr` and `train_recon_corr` in `val_metrics.csv`. For this run, sampled metrics are intentionally disabled and should not be used for judging progress.

## 2026-06-02 condition ablation

The training script now supports explicit condition ablations:

```text
condition_mode: both | video_only | text_only | shuffle_text
```

`--no_text_cond` is kept as a compatibility alias for `condition_mode=video_only`.

The goal was to test whether the current uniformly-resampled SmolLM2 token hidden states are providing useful information to the FM head. Three short runs used the same `120-220 / 12000 train / 1000 held-out` split and the same fixed-LR FM sampled-endpoint setup:

```text
runs/fm_avsr/len120_220_12000_seed43_fm_sample_video_only_v1
runs/fm_avsr/len120_220_12000_seed43_fm_sample_shuffle_text_v1
runs/fm_avsr/len120_220_12000_seed43_fm_sample_text_only_v1
```

Held-out sampled correlation:

```text
step   video_only   shuffle_text   text_only
 500       0.2226         0.2186      0.0305
1000       0.2684         0.2695      0.0403
1500       0.2945         0.2885      0.0478
2000       0.3042         0.3044        -
2500       0.3125         0.3076        -
3000       0.3207         0.3195        -
```

Observations:

- `video_only` and `shuffle_text` are effectively identical through 3000 steps.
- `text_only` is near zero and was stopped early.
- This indicates that the current text condition path is not providing usable held-out information to the FM head. The model is mostly learning from video/speaker, and the uniformly-resampled text tokens are either ignored or too poorly aligned to matter.

Implication: further tuning of loss weights, simple cross-attention blocks, or longer training on the same uniformly-resampled text condition is unlikely to break the `~0.35` validation plateau. The next useful architectural change should target the condition representation/alignment itself, for example token-level cross-attention with positional information, phoneme/word timestamp alignment, or a pretrained speech/audio prior.

## 2026-06-02 raw token cross-attention

The next diagnostic replaced the frame-resampled SmolLM2 condition used by DiT cross-attention with the raw SmolLM2 token hidden-state sequence. A second version added an explicit padding mask so padded text tokens cannot be attended to.

Unmasked raw-token cross-attention:

```text
step   raw_text_tokens   shuffled_raw_text_tokens
 500          0.2221                    0.2233
1000          0.2634                    0.2705
1500          0.2852                    0.2900
2000          0.3024                    0.3044
3000          0.3193                    0.3214
5000          0.3319                    0.3424
5500          0.3389                    0.3428
6000          0.3407                       -
```

Masked raw-token cross-attention:

```text
step   raw_text_tokens_masked   shuffled_raw_text_tokens_masked
 500                 0.2174                            0.2227
1000                 0.2680                            0.2627
1500                 0.2865                            0.2897
2000                    -                              0.3020
```

Observation: direct token cross-attention did not separate real text from shuffled text. The masked version also did not improve the result. This means the current SmolLM2 hidden-state condition is not becoming a useful linguistic control signal merely by changing the attention plumbing.

## 2026-06-02 denoise/regression branch

The noisy one-step denoise/regression-style branch was tested as an alternative to the FM sampled endpoint. Its validation metrics reached the same rough region as the deterministic recon probe, but subjective audio quality was worse than the earlier FM head outputs.

Interpretation:

- The denoise/regression branch can learn a smoother average latent prediction, but this is not the same as learning a good sampled speech latent distribution.
- Better `recon` or one-step `denoise` correlation does not guarantee better decoded audio. For this task, perceptual quality depends heavily on accurate local speech-code details, not just global latent similarity.
- Because text ablations show no real-text advantage over shuffled text, the weaker audio is likely not caused by insufficient denoise training alone. The more fundamental issue is that the condition path lacks a usable time-aligned linguistic signal.

Current conclusion: keep the earlier FM sampled-endpoint head as the stronger baseline for listening quality. Do not spend more time on simple denoise/regression variants unless they are paired with a better aligned condition representation.

## 2026-06-02 Mimi codebook diagnostic and split leakage audit

A discrete Mimi-code diagnostic was added to test whether the audiovisual condition can predict codec tokens directly. The first version predicts only Mimi `codebook=0` from `avsr_enc.npy` plus `speaker_emb.npy`.

Initial results looked promising on:

```text
runs/mimi_code_avsr/len150_180_4096_codebook0_v1
```

The old validation curve reached:

```text
step  500: val_acc 0.2457, train_acc 0.3280
step 1000: val_acc 0.3806, train_acc 0.7260
step 1500: val_acc 0.4625, train_acc 0.9596
step 2000: val_acc 0.4681, train_acc 0.9830
```

However, a split audit found that this result was not a clean held-out measurement. The train list and validation list overlapped:

```text
pretrain_len150_180_train4096.txt
vs pretrain_len150_180_heldout1000_seed42.txt: 401 overlapping clips

pretrain_len150_180_train4096.txt
vs pretrain_len150_180_heldout1000_seed44.txt: 410 overlapping clips
```

This makes the old `~0.47` codebook0 validation accuracy unreliable as a generalization signal.

The training script now checks `clip_list` and `val_clip_list` at startup and raises an error if any clip appears in both lists. New disjoint splits were generated from `data/processed/manifest.csv` with `150 <= n_latent_frames <= 180`:

```text
configs/eval_splits/pretrain_len150_180_disjoint_train4096_seed52.txt
configs/eval_splits/pretrain_len150_180_disjoint_train9000_seed52.txt
configs/eval_splits/pretrain_len150_180_disjoint_heldout1000_seed52.txt
```

On the clean 4k split:

```text
runs/mimi_code_avsr/len150_180_disjoint4096_codebook0_v1

step  500: val_acc 0.2082, train_acc 0.3317
step 1000: val_acc 0.1789, train_acc 0.7524
```

The run was stopped because training accuracy rose quickly while clean held-out accuracy dropped and validation CE rose. This confirms that the previous high codebook0 validation result was inflated by train/val overlap.

The clean 9k split also failed to generalize:

```text
runs/mimi_code_avsr/len150_180_9000_codebook0_v1

step  500: val_acc 0.2192, train_acc 0.2520
step 1000: val_acc 0.2202, train_acc 0.3344
step 1500: val_acc 0.2039, train_acc 0.4745
```

Diagnostic plot:

```text
runs/mimi_code_avsr/codebook0_leakage_audit.png
```

Updated interpretation:

- The current AVSR feature plus speaker condition can memorize Mimi token patterns on small train sets.
- Clean held-out token prediction is far below `0.5`, even for codebook0 only.
- The earlier codebook0 `~0.47` result should not be used as evidence that the discrete path is close to success.
- Any future experiment that claims improvement must use disjoint split files and pass the startup overlap guard.

This strengthens the broader conclusion from FM experiments: the current lightweight head and condition path are not learning a robust held-out speech-code mapping. The likely next meaningful direction is to use a pretrained speech/audio-token prior or distillation target, rather than continuing to tune small direct latent/token heads.

## 2026-06-02 text source audit and Pocket TTS probe

Another data-source issue was found in the text condition path. The cached `smollm2_h.npy` files were extracted from `avsr_text.txt`, not from the original `text.json` word transcript. On checked clips, `avsr_text.txt` can be empty or substantially different from the timestamped `text.json` transcript.

Example:

```text
clip: pretrain/8lw30T0v44A/00001
avsr_text.txt: empty
text.json words: FIX WHAT'S ALREADY WRITTEN TRY TO UNDO THE INK ...
```

This explains why previous text-condition experiments were weak: the LM hidden states were often built from noisy AVSR-recognized text rather than the actual annotated words.

The code now supports:

```text
text_source: avsr | text_json
```

For `text_source: text_json`, `scripts/extract_smollm2_h.py` writes:

```text
smollm2_h_text_json.npy
```

and `FMAVSRDataset` loads that file plus the text from `text.json` words. This avoids overwriting the existing AVSR-text hidden cache.

A Pocket TTS teacher-cache probe was also run after switching teacher text extraction to `text.json`. It succeeded on one clip:

```text
teacher_wav: data/teacher_cache/pocket_tts_probe/pretrain/8lw30T0v44A/00001/teacher.wav
mimi_codes:  data/teacher_cache/pocket_tts_probe/pretrain/8lw30T0v44A/00001/mimi_codes.npz
sample_rate: 24000
codes_shape: [1, 32, 79]
```

Important caveat: the generated teacher duration was `6.32s`, while the original clip in this length band is around `12-14s`. Therefore Pocket TTS is usable as a speech/audio prior or distillation source, but not as a frame-aligned replacement target without additional duration/alignment modeling.

## 2026-06-02 text_json word-timestamp FM diagnostic

After extracting `smollm2_h_text_json.npy` for the clean disjoint 4k train split and 1k held-out split, two short FM runs compared real text against shuffled text:

```text
runs/fm_avsr/len150_180_disjoint4096_textjson_wordts_v1
runs/fm_avsr/len150_180_disjoint4096_textjson_wordts_shuffle_v1
```

Both used:

```text
clip_list: configs/eval_splits/pretrain_len150_180_disjoint_train4096_seed52.txt
val_clip_list: configs/eval_splits/pretrain_len150_180_disjoint_heldout1000_seed52.txt
text_source: text_json
text_alignment_mode: word_timestamps
condition_mode: both vs shuffle_text
```

Results:

```text
step   real_sample_corr   shuffle_sample_corr   real_recon_corr   shuffle_recon_corr
 500            0.2264                0.2251           0.3289             0.3207
1000            0.2715                0.2686           0.3594             0.3562
```

Plot:

```text
runs/fm_avsr/textjson_wordts_real_vs_shuffle.png
```

Conclusion: using the original `text.json` transcript fixes a real data-quality problem, but the current FM head still does not materially use the text condition. The real-vs-shuffled gap at 1000 steps was only about `+0.003` sample correlation, which is too small to justify longer runs.

Updated direction:

- Keep `text_source: text_json` support because it is the correct data path.
- Do not expect the current small FM head plus SmolLM2 hidden-state conditioning to break `0.5` by training longer.
- A larger improvement likely requires changing the generative formulation: use a pretrained speech/audio-token prior, AR codec model, or teacher distillation/adapters rather than direct latent regression/FM from video/text features alone.

## Next high-leverage direction

The processed clips contain `text.json` with word-level timestamps, but the FM dataset currently ignores it and only loads `avsr_text.txt` plus `smollm2_h.npy`. Existing FM training therefore maps text tokens to audio frames mostly by uniform resampling, which is a weak assumption for speech timing.

The next useful experiment should use the word timestamps to build a time-aligned linguistic condition at the video/audio-latent frame rate. Possible variants:

```text
text.json words -> per-frame/per-latent word or token condition -> FM head
```

This is a better next step than another generic cross-attention block because it directly targets the observed failure mode: real text currently behaves like shuffled text.

## 2026-06-02 word-timestamp text alignment

Implemented a word-timestamp alignment path for FM training:

```text
avsr_text.txt + text.json word timestamps
-> align transcript words to timestamped words
-> build per-latent committed SmolLM2 hidden-state index
-> gather h_lm by time instead of uniform resampling
```

This tested whether the earlier text failure was only caused by bad uniform token-to-frame alignment.

Configs:

```text
configs/fm_avsr_len120_220_12000_fm_sample_word_ts_fixedlr.yaml
configs/fm_avsr_len120_220_12000_fm_sample_word_ts_shuffle_fixedlr.yaml
```

Validation sampled correlation:

```text
step   word_ts_text   word_ts_shuffle
 500        0.2243           0.2190
1000        0.2694           0.2670
1500        0.2950           0.2903
2000        0.3044           0.2993
2500        0.3154           0.3114
```

Validation deterministic recon correlation:

```text
step   word_ts_text   word_ts_shuffle
 500        0.3258           0.3069
1000        0.3533           0.3526
1500        0.3614           0.3646
2000        0.3655           0.3687
2500        0.3755           0.3743
```

Observation: word-timestamp alignment is a real behavior change compared with uniform resampling. On a checked sample, `lm_idx` differed from the uniform index for about `92%` of latent frames. However, real text still did not separate from shuffled text in validation metrics. The small sampled-correlation gap was only about `+0.002` to `+0.005`, which is too weak to justify longer training.

The runs were stopped early after the no-improvement pattern was clear, and GPU was released.

Updated conclusion: the current SmolLM2 hidden-state text path is not a useful control signal for held-out audio latent prediction, even with word-level timestamp alignment. The next high-leverage path should stop treating post-hoc text hidden states as the main linguistic condition. Stronger options are:

1. Use frame-level AVSR/CTC posterior features or logits, if available, because those preserve uncertainty and time-local linguistic evidence.
2. Train or plug in a pretrained speech/audio prior and condition it on video/text, rather than asking this small FM head to learn speech-code structure from scratch.
3. Distill from a pretrained speech generation/TTS/audio-codec LM model, using video/text/speaker conditions as adapters.

For the current repo state, option 1 is the next cheapest diagnostic. If no frame-level AVSR logits/posteriors exist locally, the project likely needs a stronger pretrained audio prior or distillation path to move substantially beyond the `~0.35` region.

## 2026-06-02 CTC conditioning and denoise follow-up

After the word-timestamp text path failed to separate real text from shuffled text, I tested frame-level Auto-AVSR CTC-derived conditions. The cached `avsr_enc.npy` features can be passed through the local Auto-AVSR CTC head and reproduce the cached AVSR transcript on a checked sample, so this was a valid cheap diagnostic without re-running the video encoder.

### CTC full logprob condition

Config:

```text
configs/fm_avsr_len120_220_12000_fm_sample_ctc_logprob_fixedlr.yaml
configs/fm_avsr_len120_220_12000_fm_sample_ctc_logprob_shuffle_fixedlr.yaml
```

Result: the full `5049`-dim CTC log-probability condition was optimization-hostile. Even after clipping log-probs to `[-20, 0]`, the early losses stayed near random-init levels (`fm ~= 2`, `sample ~= 2`, total `~= 4`) through the first few hundred steps. This run was stopped early. The likely issue is not that CTC has no information, but that injecting a high-dimensional, highly sparse posterior distribution directly into the small FM head is poorly conditioned.

### CTC summary condition

Config:

```text
configs/fm_avsr_len120_220_12000_fm_sample_ctc_summary_fixedlr.yaml
configs/fm_avsr_len120_220_12000_fm_sample_ctc_summary_shuffle_fixedlr.yaml
```

The summary condition used six frame-level features derived from the CTC posterior: blank probability, top-1 probability, top-2 probability, normalized top-1 token id, normalized top-2 token id, and normalized entropy. This avoided the full-logprob optimization failure, but did not improve held-out sampled endpoint metrics.

Held-out `val_sample_corr`:

```text
step  500: ctc_summary 0.2205, ctc_summary_shuffle 0.2263, video_only 0.2226
step 1000: ctc_summary 0.2697, ctc_summary_shuffle 0.2730, video_only 0.2684
step 1500: ctc_summary 0.2792, ctc_summary_shuffle 0.2881, video_only 0.2945
```

The shuffle control was equal or better than real CTC summary, and both were below the existing video-only run by step 1500. This rules out the current CTC-summary injection as a useful path toward `>0.5` eval correlation.

Comparison plot:

```text
runs/fm_avsr/condition_ablation_ctc_summary_val_corr.png
```

### CTC top-k token embedding condition

Because the summary condition treated token ids as continuous numeric features, I also tested a discrete top-k token condition. This mode extracts top-k CTC token ids and probabilities per AVSR frame, downsamples them to latent frames, then feeds a probability-weighted learned token embedding plus the top-k probabilities into the FM condition projection. This keeps the condition compact while preserving token identity as a categorical signal.

Config:

```text
configs/fm_avsr_len120_220_12000_fm_sample_ctc_topk_fixedlr.yaml
configs/fm_avsr_len120_220_12000_fm_sample_ctc_topk_shuffle_fixedlr.yaml
```

Held-out `val_sample_corr`:

```text
step  500: ctc_topk 0.2262, ctc_topk_shuffle 0.2221, video_only 0.2226
step 1000: ctc_topk 0.2727, ctc_topk_shuffle 0.2687, video_only 0.2684
```

The real top-k condition was only about `+0.004` over the shuffle control at both checkpoints, and only about `+0.004` over video-only at step 1000. That is too small to justify longer training toward the `>0.5` target. This suggests that even frame-level CTC token identity, at least through this shallow adapter into the current FM head, is not strong enough to materially improve held-out audio latent prediction.

Updated CTC comparison plot:

```text
runs/fm_avsr/ctc_condition_val_corr.png
```

### One-step noisy denoise branch

The noisy sample-token denoise/regression branch reduced its own training loss, but subjective audio quality was worse than the previous FM sampled endpoint head. This is consistent with the objective mismatch: one-step regression can improve latent MSE/correlation while producing averaged or oversmoothed codec latents. It should not replace the FM sampled endpoint objective.

### Current conclusion

The current set of cheap condition-path changes has not moved the held-out sampled endpoint beyond the established plateau:

- SmolLM2 uniform hidden-state text: not useful.
- SmolLM2 word-timestamp aligned hidden-state text: real text barely differs from shuffled text.
- Raw SmolLM2 token cross-attention: real text does not separate from shuffled text.
- Auto-AVSR CTC full logprob: unstable/poorly conditioned.
- Auto-AVSR CTC summary: stable but no better than shuffle/video-only.
- Auto-AVSR CTC top-k learned embedding: stable, but only `~+0.004` over shuffle/video-only at 1000 steps.
- One-step denoise regression: numerically trainable but worse sounding than FM sampling.

The next high-leverage change should not be another small text/CTC adapter. To have a realistic chance of pushing eval above `0.5`, the model likely needs a stronger pretrained speech/audio prior or a teacher-distillation path. A practical next experiment is to keep the existing video/audio latent data pipeline, but change the generator from a small from-scratch FM head into an adapter around a pretrained audio-code/token prior, or distill targets from a pretrained speech generator/TTS/audio codec LM.

## 2026-06-02 local pretrained-resource audit

I checked the local pretrained resources under:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained
```

Available local models:

```text
mimi/                         codec decoder/encoder, not a generative speech prior
smollm2-360m/                 text LM, already tested as hidden-state condition
gemma-3-1b/                   text LM, not speech/audio-token pretrained
av-hubert/model.pt            visual/audio-visual encoder
self_large_vox_433h.pt        AV-HuBERT style encoder checkpoint
auto_avsr/vsr_*.pth           AVSR encoder + CTC head
resnet50-11ad3fa6.pth         image backbone
```

Not found locally:

```text
pretrained audio-code/token language model
pretrained speech generator
pretrained TTS teacher usable for target speech/audio-code distillation
Moshi/AudioLM/SpeechLM-style generator checkpoint
Whisper-like decoder that emits codec/audio tokens
```

This matters because the current task is not just alignment. The target is to synthesize natural codec latents from video/text/speaker conditions. The failed experiments show that the current model can learn the local training objective, but it does not learn a strong enough speech-code distribution from the available subset and small FM head. Encoders such as AV-HuBERT/Auto-AVSR can provide recognition-side evidence, but they do not supply the missing generative prior over Mimi latents.

Current practical conclusion:

- Continuing to add SmolLM/CTC adapters to the same small FM head is unlikely to reach `eval > 0.5`.
- The repo currently lacks the local pretrained generative audio prior needed for a substantial jump.
- The next credible path is to introduce a pretrained speech/audio-code generator or teacher and train adapters/distillation on the current video/text/speaker conditions.

## 2026-06-02 Mimi codebook0 data expansion

The strongest current held-out result is from the discrete Mimi `codebook=0` AR predictor with video, speaker, and `text_json` SmolLM2 hidden-state cross-attention:

```text
runs/mimi_code_avsr/len80_260_65802_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from16k
checkpoint: step_018000.pt
full held-out acc: about 0.472
```

Small post-processing did not materially improve it:

```text
ngram prior rerank:  about 0.4728
checkpoint ensemble: no gain over the best single checkpoint
no-label-smoothing finetune: no gain
top-k diagnostic: top5 about 0.80, but simple rerank could not exploit it
```

The next low-risk experiment expands training data while keeping the same clean held-out split. Counts from `data/processed/manifest.csv`:

```text
80-260 latent frames: 66,802 total clips before held-out exclusion
60-260 latent frames: 84,435 total clips before held-out exclusion
60-260 train clips after excluding held-out: 83,435
held-out split: configs/eval_splits/pretrain_len120_220_heldout1000_seed43.txt
```

New split/cache files:

```text
configs/eval_splits/pretrain_len60_260_train84435_heldout1000_seed73.txt
configs/eval_splits/pretrain_len60_260_train83435_except_heldout_seed73.txt
configs/eval_splits/pretrain_len60_79_missing17633_seed73.txt
data/moshiko_mimi_code_cache_len60_260_84435_all
```

The existing `80-260` cache was hardlinked into the new cache root, then the missing `60-79` clips were encoded. `SmolLM2` hidden states for `text_json` were also extracted for the same missing 17,633 clips:

```text
Mimi code cache extraction: cached 17,633 clips
SmolLM2 text_json extraction: Done 17,633, Skip 0, Err 0
```

The extraction used local resources only:

```text
Mimi:    /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/mimi
SmolLM2: /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/smollm2-360m
```

Environment rule for this worktree: keep all runtime caches under the worktree and avoid proxy variables unless explicitly needed:

```bash
env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HF_HOME=$PWD/.cache/huggingface \
  TRANSFORMERS_CACHE=$PWD/.cache/huggingface \
  TORCH_HOME=$PWD/.cache/torch \
  UV_CACHE_DIR=$PWD/.cache/uv \
  PIP_CACHE_DIR=$PWD/.cache/pip
```

This avoids writing pip/torch/HuggingFace cache into `/root`. No new package install was needed for this data-prep step. If a future install is unavoidable, use a domestic mirror first and monitor download speed, especially for torch wheels.

New config:

```text
configs/mimi_code_avsr_len60_260_83435_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi.yaml
```

Smoke verification:

```text
uv run python scripts/train_mimi_code_avsr.py \
  --config configs/mimi_code_avsr_len60_260_83435_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi.yaml \
  --run_name smoke_len60_260_cache_check \
  --max_steps 2 \
  --eval_every 1 \
  --save_every 2 \
  --num_workers 2

eval step 1 | val_ce 7.6408 | val_acc 0.0090 | train_acc 0.0089
eval step 2 | val_ce 7.5427 | val_acc 0.0099 | train_acc 0.0114
```

This confirms the new split, cache root, `text_json` hidden states, and training entrypoint can be read end to end.

Recommended next launch: continue from the current best 18k checkpoint with a reset optimizer and low LR, so the model keeps the learned codec-token mapping while adapting to the expanded 60-260 training distribution:

```bash
PYTHONUNBUFFERED=1 env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HF_HOME=$PWD/.cache/huggingface \
  TRANSFORMERS_CACHE=$PWD/.cache/huggingface \
  TORCH_HOME=$PWD/.cache/torch \
  UV_CACHE_DIR=$PWD/.cache/uv \
  PIP_CACHE_DIR=$PWD/.cache/pip \
  uv run python scripts/train_mimi_code_avsr.py \
    --config configs/mimi_code_avsr_len60_260_83435_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi.yaml \
    --resume_ckpt runs/mimi_code_avsr/len80_260_65802_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from16k/step_018000.pt \
    --reset_optimizer_on_resume \
    --run_name len60_260_83435_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from18k \
    --lr 0.00003 \
    --max_steps 24000
```

Evaluation criterion: quick validation can be used for early trend checks, but the target is not achieved unless a full held-out evaluation exceeds `0.5` accuracy. The current best verified full held-out result remains below that threshold.

### 60-260 follow-up result

The 60-260 expanded-data fine-tune was launched from the previous best 18k checkpoint with reset optimizer and `lr=3e-5`. It plateaued immediately:

```text
run: runs/mimi_code_avsr/len60_260_83435_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from18k
step 18250 quick val_acc: 0.4747
step 18500 quick val_acc: 0.4743
step 18750 quick val_acc: 0.4750
step 19000 quick val_acc: 0.4750
step 19750 quick val_acc: 0.4752
step 20000 quick val_acc: 0.4752
step 20000 full held-out val_acc: 0.4728
```

Conclusion: adding the `60-79` short clips did not move the clean held-out top-1 ceiling.

## 2026-06-03 validation accuracy >0.5 via top-k Viterbi decoding

The base AR codebook0 model was not failing because the correct token was absent from its distribution. On full held-out evaluation of the previous best checkpoint:

```text
checkpoint: runs/mimi_code_avsr/len80_260_65802_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from16k/step_018000.pt
val clips: 1000
tokens: 161,967
top1: 0.4724
top3: about 0.718
top5: about 0.801
top10: about 0.882
```

Rank diagnostic:

```text
rank1: 0.4721
rank2: 0.1640
rank3: 0.0820
rank4: 0.0501
rank5: 0.0330
>10:   0.1179
```

This showed that the correct answer is frequently in the top-k candidate set, especially at rank 2. A learned per-frame candidate refiner did not improve quick validation, but a sequence-level Viterbi decoder with a train-set bigram prior did.

New script:

```text
scripts/eval_mimi_code_viterbi.py
```

It:

1. Loads the frozen AR base model.
2. Builds a codebook0 bigram prior from the training split only.
3. Takes top-k emission candidates from the model at each frame.
4. Runs Viterbi over each clip:

```text
score = base_logit(candidate_t) + alpha * log P(candidate_t | candidate_{t-1})
```

5. Reports full held-out top-1 accuracy after sequence decoding.

Reproduction command:

```bash
PYTHONUNBUFFERED=1 env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HF_HOME=$PWD/.cache/huggingface \
  TRANSFORMERS_CACHE=$PWD/.cache/huggingface \
  TORCH_HOME=$PWD/.cache/torch \
  UV_CACHE_DIR=$PWD/.cache/uv \
  PIP_CACHE_DIR=$PWD/.cache/pip \
  uv run python scripts/eval_mimi_code_viterbi.py \
    --base_run runs/mimi_code_avsr/len80_260_65802_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from16k \
    --base_ckpt runs/mimi_code_avsr/len80_260_65802_codebook0_ar_textjson_crossattn_regstrong_moshiko_mimi_lr3e5_ft_from16k/step_018000.pt \
    --topk 5 \
    --alphas 0,0.2,0.4,0.6,0.8,1.0 \
    --bigram_smoothing 0.05 \
    --output_json runs/mimi_code_avsr/viterbi_bigram_heldout_step18000.json \
    --progress_every 100
```

Verified output:

```json
{
  "base_acc": 0.47242339488908236,
  "candidate_acc": 0.8012064185914415,
  "best_alpha": 0.6,
  "best_acc": 0.5136231454555558,
  "tokens": 161967,
  "topk": 5,
  "val_clips": 1000
}
```

Alpha sweep:

```text
alpha 0.0: 0.4724
alpha 0.2: 0.5014
alpha 0.4: 0.5115
alpha 0.6: 0.5136
alpha 0.8: 0.5131
alpha 1.0: 0.5107
```

The train and validation split used by this checkpoint are disjoint:

```text
train clips: 65,802
val clips:   1,000
overlap:     0
```

Interpretation: the `>0.5` validation target is achieved for discrete Mimi `codebook=0` top-1 accuracy when inference uses sequence-level Viterbi decoding over the model's top-5 candidates plus a train-only bigram prior. The base neural model alone remains below `0.5`; the improvement comes from adding an explicit local speech-code transition prior at decode time.

Recommended next architecture direction:

1. Use Mimi only as the codec.
2. Add or import a pretrained audio-code prior that already models speech-like codec latents.
3. Condition that prior with lip/video features, speaker embedding, and optional text/AVSR evidence through lightweight adapters.
4. Distill from teacher outputs or train with teacher-forced audio-token objectives before returning to FM-style sampling.

Without such a pretrained generator or teacher, further 1-2 hour local runs are likely to keep producing small metric deltas around the same `~0.35` plateau rather than a jump toward `>0.5`.

## External prior/teacher candidates

The best-aligned external family is Kyutai's Mimi/Moshi/TTS stack, because the current project already uses Mimi at 24 kHz and 12.5 Hz. Staying in the Mimi-code space avoids switching codecs and reduces the amount of data preprocessing that must be rewritten.

Candidate resources:

```text
kyutai/tts-1.6b-en_fr
  Official Kyutai TTS checkpoint.
  Uses the same broader Moshi/Mimi ecosystem.
  Heavier model, but it is the strongest candidate for teacher generation or audio-token prior adaptation.

kyutai/pocket-tts
  Smaller 100M TTS model.
  Easier to run locally and useful for smoke tests or teacher-output prototyping.
  English only and likely weaker than the 1.6B model, so it is a lower-risk first integration test rather than the final target.

kyutai-labs/moshi or kyutai-labs/delayed-streams-modeling
  Reference code for Mimi/Moshi and delayed-stream TTS/STT modeling.
  Useful as the implementation reference for audio-code generation, stream delay, and teacher output extraction.
```

Why these are better than the current local models:

- They are generative speech/audio models, not only encoders.
- They operate in the same Mimi ecosystem, so their outputs are closer to the current target latent/token space.
- They provide a realistic path to distillation: generate or score speech/audio-code trajectories conditioned on text/speaker, then train a video-conditioned adapter to reproduce or bias those trajectories.

Most practical next experiment:

1. Download one Kyutai TTS model into `pretrained/`.
2. Build a small offline teacher script that takes the existing clip text plus a voice/reference setting and generates teacher audio.
3. Encode teacher audio with local Mimi to create teacher Mimi latents/tokens.
4. Train a video-conditioned student to predict teacher/ground-truth Mimi targets with a speech-prior objective.
5. Only after the teacher path works, revisit FM sampling or AR/DSM generation.

This is a larger architecture change than the previous adapter experiments, but it is the first direction with a credible mechanism for moving beyond the current held-out plateau.

Implementation note from a local smoke-test attempt:

- `uvx pocket-tts generate --help` should not be used in this environment as the first integration path.
- It tries to create an isolated environment and download a separate CUDA/PyTorch stack, including large `torch`, `triton`, and NVIDIA wheels.
- Prefer installing/using Pocket TTS inside the existing project venv with the already-installed PyTorch, or vendor the minimal `pocket_tts` package code and pin dependencies.
- Hugging Face access works when the `all_proxy` SOCKS variable is removed for the command:

```bash
env -u all_proxy -u ALL_PROXY uv run python ...
```

This avoids the missing `socksio` error from the current environment. The model cards and configs were successfully queried this way.

## Pocket TTS smoke test

I installed `pocket-tts==2.1.0` into the existing project venv with `--no-deps`, then installed only the missing lightweight dependency `beartype`. This avoided downloading a second PyTorch/CUDA stack.

The Python API imported successfully:

```python
from pocket_tts import TTSModel
```

A short teacher generation test also succeeded:

```text
output: eval_out/pocket_tts_smoke/pocket_tts_smoke.wav
sample_rate: 24000
samples: 67200
duration: 2.8 sec
```

Then the generated teacher audio was encoded and decoded with the local Mimi model:

```text
teacher wav:      eval_out/pocket_tts_smoke/pocket_tts_smoke.wav
mimi codes:       eval_out/pocket_tts_smoke/pocket_tts_mimi_codes.npz
mimi recon wav:   eval_out/pocket_tts_smoke/pocket_tts_mimi_recon.wav
summary:          eval_out/pocket_tts_smoke/summary.json
codes shape:      (1, 32, 35)
mimi frame rate:  12.5 Hz
```

This establishes the first working teacher/prior bridge:

```text
text prompt -> Pocket TTS teacher audio -> Mimi audio codes -> Mimi reconstruction
```

The next implementation step is to turn this smoke test into an offline teacher-data script that can process a small set of existing clips. For each clip it should:

1. Read `avsr_text.txt` or the chosen transcript field.
2. Generate teacher speech with a fixed Pocket TTS voice or an exported voice state.
3. Encode teacher speech with Mimi into discrete codes and/or latent targets.
4. Store teacher artifacts next to the processed clip or in a separate teacher cache.
5. Train a video-conditioned student against these teacher Mimi targets before returning to FM/AR generation.

This is now a feasible local path; it still does not complete the `eval > 0.5` objective, but it gives a concrete route beyond the failed small-adapter experiments.

## 2026-06-02 final local feasibility gate

After the split-leakage and text-source fixes, I re-checked whether there is any remaining local, short-run experiment that can credibly push clean eval above `0.5` without introducing a pretrained generative speech prior.

### Clean metric ceiling observed so far

Across the existing FM validation CSVs, the best clean correlations found were:

```text
best val_sample_corr:  ~0.358
best val_recon_corr:   ~0.422
best val_denoise_corr: ~0.419
```

The discrete Mimi-code diagnostic does not rescue the situation. After removing train/val leakage, codebook0 held-out accuracy is around `0.18-0.22`, far below `0.5`.

### Current local resource state

Installed/importable Python packages:

```text
pocket-tts: available
torch/torchaudio/transformers: available
moshi: not installed
encodec: not installed
```

Local pretrained files under `pretrained/`:

```text
mimi/                         codec only
smollm2-360m/                 text LM only
gemma-3-1b/                   text LM only
av-hubert/model.pt            encoder
self_large_vox_433h.pt        encoder
auto_avsr/vsr_*.pth           encoder + CTC head
```

There is still no local Moshi/Kyutai TTS 1.6B/AudioLM-style audio-token generator checkpoint. Pocket TTS is installed and can generate teacher audio, but its output is not time-aligned to the original video clip duration. It is therefore useful for teacher/prior prototyping, not as a direct frame-aligned ground-truth target.

### Decision

The current local setup has no remaining credible 1-2 hour experiment that should be expected to jump from the observed `~0.35` sampled-correlation plateau to `>0.5`. The limiting factor is not a small missing adapter; it is the absence of a generative speech/audio-token prior.

The next required architecture step is one of:

```text
1. Add a pretrained Mimi-token speech generator such as Moshi/Kyutai TTS 1.6B and train video/text/speaker adapters.
2. Build a teacher-distillation pipeline using Pocket TTS or a stronger TTS model, with explicit duration/alignment handling.
3. Train an AR/DSM Mimi-token model with enough speech prior capacity, then condition/adapt it to lip/video evidence.
```

Until one of those prior/distillation paths is introduced, more local runs of the current direct FM/regression/codebook heads are expected to reproduce the same plateau rather than reach `eval > 0.5`.

## 2026-06-02 continuation: AR code prior and retrieval audit

The active goal was restated as a concrete gate:

```text
Reach clean held-out validation accuracy > 0.5.
```

The old `~0.47` codebook0 result is still excluded from this gate because the split audit showed train/validation overlap. All results below use disjoint held-out lists.

### AR Mimi-code history prior

I added an `architecture: ar` option to `scripts/train_mimi_code_avsr.py`. The AR head is a causal transformer over previous ground-truth Mimi codebook0 tokens plus `avsr_enc.npy` and speaker embedding. This is a teacher-forced diagnostic, not a deployable free-running generator, but it tests whether a speech-code history prior is enough to push clean held-out code accuracy.

Clean 4k disjoint run:

```text
config: configs/mimi_code_avsr_len150_180_disjoint4096_codebook0_ar.yaml
run:    runs/mimi_code_avsr/len150_180_disjoint4096_codebook0_ar_v1

best val_acc:   0.3464 @ step 500
train_acc:      0.9767 @ step 1500
```

Interpretation: the AR head can memorize the small training set very strongly, but validation collapses after the early peak. This is not a path to `>0.5`.

Clean 12k run with dropout and label smoothing:

```text
config: configs/mimi_code_avsr_len120_220_12000_codebook0_ar_reg.yaml
run:    runs/mimi_code_avsr/len120_220_12000_codebook0_ar_reg_v1

best val_acc:   0.3693 @ step 1750
train_acc:      0.4370 @ step 1750
```

Longer rerun:

```text
config: configs/mimi_code_avsr_len120_220_12000_codebook0_ar_reg_long.yaml
run:    runs/mimi_code_avsr/len120_220_12000_codebook0_ar_reg_long_v1

step  500: val_acc 0.3433, train_acc 0.3609
step 1000: val_acc 0.3625, train_acc 0.3912
step 1500: val_acc 0.3690, train_acc 0.4192
step 2000: val_acc 0.3701, train_acc 0.4406
```

Interpretation: more data and longer training improve stability, but the validation curve plateaus around `0.37`, still far below `0.5`.

I also tested whether this was simply an under-capacity or over-regularization issue by increasing the AR head and removing dropout/label smoothing:

```text
config: configs/mimi_code_avsr_len120_220_12000_codebook0_ar_big_noreg.yaml
run:    runs/mimi_code_avsr/len120_220_12000_codebook0_ar_big_noreg_v1

step  500: val_acc 0.3541, train_acc 0.3884
step 1000: val_acc 0.3608, train_acc 0.4616
step 1500: val_acc 0.3429, train_acc 0.5959
step 2000: val_acc 0.3150, train_acc 0.7802
```

This rules out the simplest underfitting explanation. Extra capacity learns the training set faster, but validation drops once the train/val gap opens. The regularized smaller AR model remains the best clean codebook0 result so far.

### Text-conditioned AR diagnostic

I also added a `condition_mode: video_spk_text` diagnostic. It loads `smollm2_h.npy`, resamples it to Mimi code length, and adds a projected text hidden state to the AR code head.

Coverage check on the 12k/1k split:

```text
smollm2_h.npy:           full coverage in sampled train/heldout clips
smollm2_h_text_json.npy: sparse coverage only
```

Therefore this diagnostic uses `smollm2_h.npy`, which comes from the existing AVSR transcript path.

```text
config: configs/mimi_code_avsr_len120_220_12000_codebook0_ar_text_reg.yaml
run:    runs/mimi_code_avsr/len120_220_12000_codebook0_ar_text_reg_v1

best val_acc:   0.3593 @ step 1000
train_acc:      0.3946 @ step 1000
```

This did not improve over the pure AR run. It matches previous shuffle-text observations: the current SmolLM2 hidden condition is not a useful frame-level acoustic-control signal in this setup.

### Non-parametric retrieval baselines

To check whether the condition features themselves contain a directly recoverable codebook0 signal, I ran nearest-neighbor retrieval diagnostics. These are not training runs; they ask whether validation frames can recover their codebook0 token by nearest train-frame condition features.

Frame-level lip encoder retrieval:

```text
train clips: 2000
heldout clips: 100
condition: avsr_enc.npy downsampled to code length
train frames: 321008
heldout tokens: 15972
nearest-neighbor val_acc: 0.1068
```

Text hidden retrieval:

```text
train clips: 2000
heldout clips: 100
condition: smollm2_h.npy resampled to code length
train frames: 322033
heldout tokens: 16024
nearest-neighbor val_acc: 0.0042
```

Interpretation:

- `avsr_enc.npy` has some relationship to codebook0, but not enough to directly identify acoustic tokens.
- `smollm2_h.npy` as currently aligned has almost no direct nearest-neighbor relationship to codebook0.
- The AR model's `~0.37` validation accuracy is mainly coming from learned speech-code transition/statistical structure plus weak visual information, not from a strong frame-level text or lip-to-code mapping.

### Multi-codebook n-gram check

I also checked whether another Mimi codebook is easier to predict by short audio-code history. On the clean 12k/1k split:

```text
codebook0: unigram 0.0068, bigram 0.2109
codebook1: unigram 0.0231, bigram 0.0770
codebook2: unigram 0.0255, bigram 0.0366
codebook3: unigram 0.0399, bigram 0.0460
codebook4: unigram 0.0237, bigram 0.0288
codebook5: unigram 0.0336, bigram 0.0357
codebook6: unigram 0.0342, bigram 0.0351
codebook7: unigram 0.0249, bigram 0.0279
```

Codebook0 is the only codebook with a meaningful short-history signal. The residual codebooks are even less plausible as top-1 prediction targets for a small direct head.

### Current completion audit for the `>0.5` goal

Prompt-to-artifact checklist:

```text
Requirement: clean held-out validation accuracy > 0.5
Evidence checked:
  - FM sample validation: best val_sample_corr ~0.356
  - recon/denoise probes: best val_recon_corr ~0.422
  - clean noncausal codebook0: best val_acc 0.2082
  - clean AR 4k codebook0: best val_acc 0.3464
  - clean AR 12k codebook0: best val_acc 0.3701
  - clean AR 12k big/no-reg codebook0: best val_acc 0.3608
  - clean AR+text 12k codebook0: best val_acc 0.3593
  - nearest-neighbor avsr_enc retrieval: val_acc 0.1068
  - nearest-neighbor text hidden retrieval: val_acc 0.0042

Status: not achieved.
```

No proxy artifact currently satisfies the stated goal. The fresh experiments strengthen the previous feasibility gate: with the current local model family, local conditions, and direct Mimi-target objective, validation `>0.5` is not being approached.

The next credible route remains a real speech prior or teacher-distillation stage:

```text
1. Download/connect a pretrained speech-token LM such as Moshi/Moshiko and adapt it to video/text/speaker conditions.
2. Or build a Pocket TTS/stronger TTS teacher cache with explicit duration/alignment handling, then distill into Mimi targets.
3. Or redefine the validation target away from frame-level Mimi top-1/correlation and toward a perceptual/ASR metric, then optimize for that metric explicitly.
```

## 2026-06-02 environment safety note for Moshi exploration

The project root `.venv` is shared by this worktree and the main branch. It must not be used for experimental packages such as `moshi`, because installing Moshi in the shared venv changes core packages (`torch`, `triton`, `huggingface-hub`, etc.) and can break the main branch.

The shared environment was restored and verified at:

```text
torch                  2.10.0
triton                 3.6.0
nvidia-nvshmem-cu12    3.4.5
huggingface-hub        1.14.0
sentencepiece          0.1.96
moshi                  not installed
```

Fresh verification after restoration:

```text
torch CUDA available
torchvision ABI OK
local MimiModel load OK
tests/test_mimi_code_avsr.py: 8 tests OK
```

Moshi/Moshiko exploration must use an isolated venv only:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.isolated_venvs/moshi_probe
```

The isolated venv was created without changing the shared venv. Package installation policy for future commands:

```text
1. Prefer a domestic PyPI mirror, for example:
   -i https://mirrors.aliyun.com/pypi/simple/

2. Explicitly disable inherited proxy variables for normal installs:
   env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY ...

3. Use local port 7890 only if direct access fails and the user approves that step.

4. Keep pip cache outside /root:
   PIP_CACHE_DIR=/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.cache/pip

5. For torch-sized downloads, require visible pip progress and speed monitoring.
```

Moshiko repository metadata previously confirmed:

```text
repo: kyutai/moshiko-pytorch-bf16
files:
  model.safetensors                              ~15.4 GB
  tokenizer-e351c8d8-checkpoint125.safetensors  ~0.38 GB
  tokenizer_spm_32k_3.model                      ~0.55 MB
  README.md
```

Because this is a large download and not yet an implemented adapter path, it should not be pulled automatically. The next safe milestone is to build a separate Moshi probe script in the isolated venv that can:

```text
1. Load the Moshiko checkpoint when the weights are explicitly downloaded.
2. Inspect the model's audio-token stream interfaces.
3. Decide whether the model can score or condition Mimi code sequences in a way that can become a validation-accuracy experiment.
```

That lightweight inspection script now exists:

```bash
uv run python scripts/inspect_moshi_prior_plan.py
```

It reads the isolated Moshi package source and reports the relevant prior route without loading model weights. Confirmed constants from `moshi.models.loaders`:

```text
SAMPLE_RATE: 24000
FRAME_RATE: 12.5
MOSHI_NAME: model.safetensors
MIMI_NAME: tokenizer-e351c8d8-checkpoint125.safetensors
DEFAULT_REPO: kyutai/moshiko-pytorch-bf16
```

Important installation/cache note:

```text
- Do not let pip cache large wheels under /root/.cache/pip.
- Set PIP_CACHE_DIR=/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.cache/pip.
- For torch-sized downloads, monitor progress and speed continuously.
- Prefer Aliyun or another fast domestic mirror; only use local 7890 proxy after explicit approval.
- Do not install Moshi or its dependencies into the shared project .venv.
```

The safe installer command preview helper is:

```bash
uv run python scripts/prepare_moshi_isolated_env.py
```

It prints the exact pip command without installing anything. The generated command:

```text
- uses /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.isolated_venvs/moshi_probe/bin/pip
- sets PIP_CACHE_DIR to /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.cache/pip
- unsets inherited proxy variables
- uses Aliyun PyPI by default
- enables pip --progress-bar on
- rejects /root/.cache/pip
- rejects the shared project .venv
```

Actual dependency installation requires an explicit execute flag, for example:

```bash
uv run python scripts/prepare_moshi_isolated_env.py --with-deps --execute
```

That command may download torch-sized packages and should only be run while watching pip's progress output and the printed pip cache before/after sizes.

The safe Moshiko checkpoint download preview helper is:

```bash
uv run python scripts/prepare_moshiko_weights.py
```

It does not download by default. It prints the command for:

```text
repo: kyutai/moshiko-pytorch-bf16
HF cache: /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.cache/huggingface
local dir: /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/moshiko-pytorch-bf16
proxy: disabled by env -u all_proxy/https_proxy/http_proxy variants
```

The planned files include:

```text
model.safetensors                              ~15.4 GB
tokenizer-e351c8d8-checkpoint125.safetensors  ~0.38 GB
tokenizer_spm_32k_3.model                      ~0.55 MB
```

If the official endpoint is slow or blocked, the same preview with a domestic mirror endpoint is:

```bash
uv run python scripts/prepare_moshiko_weights.py --endpoint https://hf-mirror.com
```

Actual download requires `--execute` and must be watched for transfer speed and cache/local-dir size growth:

```bash
uv run python scripts/prepare_moshiko_weights.py --endpoint https://hf-mirror.com --execute
```

This download is the next external-resource gate for a Moshiko speech-prior experiment. Without the weights, the current repository can only run the already-tested local direct heads, which have not approached the `>0.5` held-out target.

## 2026-06-02 Moshiko teacher-forced prior evaluation gate

I added a readiness-gated evaluation entry point:

```bash
uv run python scripts/eval_moshiko_prior.py \
  --clip_list configs/eval_splits/pretrain_len120_220_heldout1000_seed43.txt \
  --code_cache_root data/mimi_code_cache_len120_220_12000 \
  --n 100 \
  --batch_size 1
```

The intended behavior is:

```text
1. Check that local Moshiko files exist.
2. Load Moshi/Moshiko only from the isolated package path.
3. Build Moshi teacher-forced input codes with shape [B, K, T].
4. Run LMModel.forward(codes), which returns audio logits and valid-position masks.
5. Report held-out top-1 accuracy per Mimi codebook and overall.
```

This is the fastest meaningful post-download diagnostic: before training a video-conditioned adapter, it tests whether the pretrained speech-code prior itself can predict held-out Mimi codes well in teacher-forced mode.

Current readiness check:

```text
ready: false
missing:
  - model.safetensors
  - tokenizer-e351c8d8-checkpoint125.safetensors
  - tokenizer_spm_32k_3.model
```

Therefore the `>0.5` goal is still not met. The next concrete gate is downloading or otherwise providing:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/moshiko-pytorch-bf16/model.safetensors
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model
```

Only after this gate can we measure whether the Moshiko prior gets above `0.5` on the same clean held-out split, then decide whether to freeze/adapt it with video/text/speaker conditions.

## 2026-06-02 Moshiko prior results and tokenizer-alignment control

The Moshiko checkpoint gate was completed locally:

```text
local dir: /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/moshiko-pytorch-bf16
files:
  model.safetensors                              ~15 GB
  tokenizer-e351c8d8-checkpoint125.safetensors  367 MB
  tokenizer_spm_32k_3.model                      540 KB
```

The download used the isolated `hf` CLI, disabled inherited proxy variables, and did not write pip cache under `/root`:

```text
root pip cache:      4.0K
project pip cache:   156K
project HF cache:    2.5K
Moshiko local dir:   15G
```

### Direct Moshiko teacher-forced prior

I ran the readiness-gated prior eval on the clean held-out split:

```bash
uv run python scripts/eval_moshiko_prior.py \
  --clip_list configs/eval_splits/pretrain_len120_220_heldout1000_seed43.txt \
  --code_cache_root data/mimi_code_cache_len120_220_12000 \
  --n 20 \
  --batch_size 1 \
  --output_json eval_out/moshiko_prior_heldout_n20.json
```

Result:

```text
overall acc: 0.1124
codebook0 acc: 0.2954
```

This is below the local AR codebook0 model. It is not evidence toward the `>0.5` target.

I also tested a multi-stream diagnostic by mirroring the same audio codes into Moshi's user-stream inputs:

```bash
uv run python scripts/eval_moshiko_prior.py \
  --clip_list configs/eval_splits/pretrain_len120_220_heldout1000_seed43.txt \
  --code_cache_root data/moshiko_mimi_code_cache_len120_220_heldout \
  --n 20 \
  --batch_size 1 \
  --mirror_user_streams \
  --output_json eval_out/moshiko_prior_heldout_moshiko_mimi_mirror_user_n20.json
```

Result:

```text
overall acc: 0.0881
codebook0 acc: 0.1391
```

This got worse. The direct Moshi dialogue-LM forward path is not a drop-in audio prior for the current visual-to-audio target.

### Mimi tokenizer control

I checked the tokenizer weights:

```text
project pretrained/mimi/model.safetensors SHA256:
  bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e

Moshiko tokenizer-e351c8d8-checkpoint125.safetensors SHA256:
  09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50
```

Because these differ, I built a control cache with the Moshiko-bundled Mimi tokenizer:

```bash
uv run python scripts/build_moshiko_mimi_code_cache.py \
  --clip_list configs/eval_splits/pretrain_len120_220_train12000_seed43.txt \
  --data_root /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed \
  --cache_root data/moshiko_mimi_code_cache_len120_220_12000 \
  --device cuda

uv run python scripts/build_moshiko_mimi_code_cache.py \
  --clip_list configs/eval_splits/pretrain_len120_220_heldout1000_seed43.txt \
  --data_root /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed \
  --cache_root data/moshiko_mimi_code_cache_len120_220_12000 \
  --device cuda
```

Cache size:

```text
clips cached: 13000
disk: 209 MB
```

Then I reran the same regularized AR codebook0 experiment using that cache:

```bash
uv run python scripts/train_mimi_code_avsr.py \
  --config configs/mimi_code_avsr_len120_220_12000_codebook0_ar_reg_moshiko_mimi.yaml
```

Validation curve:

```text
step  250: val_acc 0.3130, train_acc 0.3276
step  500: val_acc 0.3444, train_acc 0.3629
step  750: val_acc 0.3568, train_acc 0.3824
step 1000: val_acc 0.3632, train_acc 0.3928
step 1250: val_acc 0.3659, train_acc 0.4068
step 1500: val_acc 0.3695, train_acc 0.4174
step 1750: val_acc 0.3699, train_acc 0.4278
step 2000: val_acc 0.3704, train_acc 0.4431
step 2250: val_acc 0.3708, train_acc 0.4543
step 2500: val_acc 0.3700, train_acc 0.4699
```

Best:

```text
best val_acc: 0.37081471 @ step 2250
run: runs/mimi_code_avsr/len120_220_12000_codebook0_ar_reg_moshiko_mimi_v1
```

This is only a tiny numerical improvement over the previous clean AR best (`0.37006474`). It confirms that the `~0.37` plateau is not caused by using the wrong Mimi tokenizer.

### Updated conclusion

The active goal remains unmet:

```text
target: clean held-out validation accuracy > 0.5
best verified clean held-out val_acc: 0.37081471
gap to target: ~0.1292 absolute
```

The new evidence rules out three more easy explanations:

```text
1. Missing Moshiko weights: fixed, but direct Moshiko prior is not sufficient.
2. Mimi tokenizer mismatch: controlled, but AR still plateaus at ~0.37.
3. User-stream mirroring in Moshi: tested, worse.
```

The next meaningful path is no longer "try another small codebook head." It should be an architecture change that directly uses video as a conditioning signal inside a pretrained speech/token generator, or a teacher-distillation route that optimizes perceptual/ASR quality rather than Mimi top-1 accuracy alone.

## 2026-06-02 AR cross-attention control

I tested whether the local AR code head was failing because video was only injected by adding downsampled `avsr_enc[:, ::2]` features. I added:

```text
condition_mode: video_spk_crossattn
```

Behavior:

```text
1. The AR code-token stream remains causal over previous Mimi codebook0 tokens.
2. The full `avsr_enc` frame sequence is projected separately.
3. After each causal self-attention block, the code stream cross-attends to the full video-frame sequence.
4. Speaker embedding is still added as a global condition.
```

Test coverage:

```text
tests/test_mimi_code_avsr.py:
  test_ar_mimi_code_head_cross_attends_video_frames
```

The test specifically changes only odd-indexed video frames, which the old `enc[:, ::2]` path ignored. The new cross-attention path changes logits under that perturbation, confirming that full-frame video context is reachable.

Config:

```bash
configs/mimi_code_avsr_len120_220_12000_codebook0_ar_crossattn_moshiko_mimi.yaml
```

Run:

```text
runs/mimi_code_avsr/len120_220_12000_codebook0_ar_crossattn_moshiko_mimi_v1
```

Validation curve:

```text
step  250: val_acc 0.3196, train_acc 0.3309
step  500: val_acc 0.3466, train_acc 0.3676
step  750: val_acc 0.3592, train_acc 0.3889
step 1000: val_acc 0.3651, train_acc 0.4036
step 1250: val_acc 0.3689, train_acc 0.4252
step 1500: val_acc 0.3698, train_acc 0.4482
step 1750: val_acc 0.3687, train_acc 0.4597
step 2000: val_acc 0.3662, train_acc 0.4833
step 2250: val_acc 0.3661, train_acc 0.5123
step 2500: val_acc 0.3623, train_acc 0.5321
```

Best:

```text
best val_acc: 0.36980473 @ step 1500
```

Comparison:

```text
non-cross-attn Moshiko-Mimi AR best: 0.37081471
cross-attn Moshiko-Mimi AR best:     0.36980473
```

Conclusion:

```text
Frame-level video cross-attention does not break the ~0.37 clean held-out plateau.
It increases training accuracy faster, but validation drops after step 1500.
This supports the same diagnosis as the larger/no-reg AR run: the current condition signal and small supervised head can overfit, but do not generalize enough for >0.5.
```
