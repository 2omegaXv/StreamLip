# FM AVSR lip_avsr visual-prior experiment log

Date: 2026-06-03

Goal:

```text
Use the newly processed lip_avsr.npy visual input to rebuild Auto-AVSR visual
features and rerun a controlled text+image -> audio experiment. Keep each
training run within about 1 hour, monitor validation corr, early-stop abnormal
or clearly weak runs, and aim to move eval corr past 0.5.
```

Current reference point before this branch of experiments:

```text
Old visual feature: avsr_enc.npy extracted from lip.npy
Best checkpoint: runs/fm_avsr/len120_220_32272_pred_energy_recon_textjson_wordts_v1/step_002000.pt
Val heldout1000 corr: 0.4896
Heldout100 metrics-only corr: 0.4933
Heldout6 wav samples: eval_out/len120_220_32272_textjson_wordts_step2000_heldout6_wav
```

Lessons to keep from previous experiments:

```text
Do not repeat small deterministic-loss tweaks around the same visual features:
cross-attn, corr-loss, PCA replacement, and PCA auxiliary did not move the main
metric meaningfully.

Do not overwrite avsr_enc.npy. The new visual feature must be a parallel A/B
artifact, e.g. avsr_enc_lipavsr.npy, so old results remain reproducible.

Use the same text_json + word_timestamps + predicted-energy setup as the current
best baseline unless the visual-prior experiment itself proves weak.
```

Planned experiment:

```text
1. Build a 10k train + 1k val split from clips that have lip_avsr.npy and all
   required supervision/conditioning files.
2. Extract Auto-AVSR encoder features from lip_avsr.npy into
   avsr_enc_lipavsr.npy.
3. Train the existing pred-energy deterministic recon setup with
   visual_feature_name=avsr_enc_lipavsr.npy.
4. Monitor 500/1000/1500/2000-step validation corr and stop if the curve is
   clearly below the old 12k baseline or unstable.
5. If promising, run heldout metrics and generate wav samples.
```

Implementation notes:

```text
lip_avsr.npy is not a latent. It is a newly processed grayscale visual input:
roughly (T, 96, 96) uint8 per clip.

Auto-AVSR encoder output is the latent-like visual condition used by FM AVSR.
For the A/B run, lip_avsr.npy is encoded to avsr_enc_lipavsr.npy:
(T', 768) float16 per clip. The old avsr_enc.npy files are left untouched.

Code support added:
- scripts/extract_avsr_enc.py can take --data_root, --clip_list,
  --input_name, --output_name, and --text_output_name.
- src/streaminlip/auto_avsr.py accepts grayscale lip_avsr.npy frames.
- FMAVSRDataset/train/eval accept visual_feature_name.
```

Split choice:

```text
The strict old len120_220 pool with complete lip_avsr coverage was only 7373
clips, so it could not support the requested 10k train experiment. I widened
the window to len80_260, where the complete pool was enough for 10k train +
1k val.

Train split: configs/eval_splits/pretrain_len80_260_lipavsr_train10000_seed43.txt
Val split:   configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt
Extraction:  configs/eval_splits/pretrain_len80_260_lipavsr_train11000_seed43.txt
```

Extraction result:

```text
Input:  lip_avsr.npy
Output: avsr_enc_lipavsr.npy
Text:   avsr_text_lipavsr.txt
Auto-AVSR checkpoint:
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth

Bulk extraction over train+val completed in about 0.2h:
Done: 10999
Skipped: 1
Errors: 0

The skipped item was the smoke-test clip that already had
avsr_enc_lipavsr.npy. Final coverage was 10000/10000 train and 1000/1000 val.
```

Training config and checkpoint:

```text
Config:
configs/fm_avsr_len80_260_lipavsr_10000_pred_energy_recon_textjson_wordts.yaml

Run:
runs/fm_avsr/len80_260_lipavsr_10000_pred_energy_recon_textjson_wordts_v1

Condition setup:
condition_mode=both
text_source=text_json
text_alignment_mode=word_timestamps
visual_feature_name=avsr_enc_lipavsr.npy
energy_condition_mode=pred
lambda_recon=1.0
lambda_energy=0.1
loss_fm_weight=0.0
```

Validation curve:

```text
step  val_recon_corr  val_energy_corr  elapsed_seconds
 500  0.49619544      0.78907274       199.1565
1000  0.51232149      0.79662524       389.4225
1500  0.51326406      0.79905108       574.5242
2000  0.50417887      0.80467921       748.6273
```

Best checkpoint:

```text
runs/fm_avsr/len80_260_lipavsr_10000_pred_energy_recon_textjson_wordts_v1/step_001500.pt
```

External eval from the best checkpoint:

```text
Matched val1000 metrics-only:
eval_out/len80_260_lipavsr_textjson_wordts_step1500_val1000_metrics/metrics.json
n=1000
mean_corr=0.5072512165
mean_mse=0.7721213100
mean_mae=0.6560713623

Condition-shift val1000 negative control:
eval_out/len80_260_lipavsr_textjson_wordts_step1500_val1000_shift1_metrics/metrics.json
n=1000
condition_shift=1
mean_corr=0.0229793445
mean_mse=1.3055599350
mean_mae=0.8837756377

The shifted-condition drop indicates the matched score is condition-grounded,
not just a dataset-average latent predictor.
```

Audio samples:

```text
Path:
eval_out/len80_260_lipavsr_textjson_wordts_step1500_val6_wav

Files:
0000_pred.wav ... 0005_pred.wav
0000_gt.wav   ... 0005_gt.wav

Sample metrics:
n=6
mean_corr=0.5435053983
mean_mse=0.6031611959
mean_mae=0.5825375070
```

Conclusion:

```text
The new lip_avsr visual input is worth pursuing. It crosses 0.5 on the 1k val
split within a sub-1h training run and beats the old visual-feature reference
point. The next useful step is listening to the generated samples, then scaling
the same setup beyond 10k if the audio quality is acceptable.
```

30k scale-up plan:

```text
After the 10k run sounded worth pursuing, the next experiment scales the same
condition/loss setup to 30k train clips.

The strict len80_260 pool with complete lip_avsr coverage was only about 20k
clips, so the 30k split uses all lengths that have the required files. Dataset
loading caps very long clips at _MAX_TA=400 latent frames.

Train split:
configs/eval_splits/pretrain_lipavsr_train30000_seed44_excl_val1000_seed43.txt

Validation split, kept identical to the 10k run for comparison:
configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt

Selected preprocessing list:
configs/eval_splits/pretrain_lipavsr_train30000_plus_val1000_seed44_seed43.txt

Train/val overlap: 0

Train length bins:
lt60=4472
60_79=4739
80_260=16663
261_320=1421
321_400=1066
gt400=1639

Coverage over 30000 train + 1000 val:
lip_avsr.npy: 31000/31000
avsr_enc_lipavsr.npy: 31000/31000
latent.npz: 31000/31000
speaker_emb.npy: 31000/31000
text.json: 31000/31000
smollm2_h_text_json.npy: 31000/31000
audio.wav: 31000/31000
```

30k preprocessing result:

```text
SmolLM2 text hidden extraction:
clip list:
configs/eval_splits/pretrain_lipavsr_train30000_plus_val1000_missing_textjson_h.txt
Done: 8598
Skip: 0
Err: 0

Auto-AVSR visual extraction:
clip list:
configs/eval_splits/pretrain_lipavsr_train30000_plus_val1000_missing_avsr_enc_lipavsr.txt
Input:  lip_avsr.npy
Output: avsr_enc_lipavsr.npy
Text:   avsr_text_lipavsr.txt
Done: 21238
Skipped: 0
Errors: 0
Total time: 0.5h
```

30k training config:

```text
Config:
configs/fm_avsr_lipavsr_30000_pred_energy_recon_textjson_wordts.yaml

Run:
runs/fm_avsr/lipavsr_30000_pred_energy_recon_textjson_wordts_v1

Condition setup remains matched to the 10k run:
condition_mode=both
text_source=text_json
text_alignment_mode=word_timestamps
visual_feature_name=avsr_enc_lipavsr.npy
energy_condition_mode=pred
lambda_recon=1.0
lambda_energy=0.1
loss_fm_weight=0.0

The run is capped at 3000 steps with validation/checkpointing every 500 steps
and should be stopped early if the validation curve is unstable or clearly below
the 10k reference.
```

30k training result:

```text
Training was run with the config above and stopped early after the 2500-step
validation point because the curve had plateaued. The hard cap was 3000 steps,
but step 2500 was already slightly below step 2000.

Validation curve:

step  val_recon_corr  val_energy_corr  elapsed_seconds
 500  0.49262037      0.77075606       340.3996
1000  0.51954315      0.78461363       607.2626
1500  0.52965798      0.79076741       910.1417
2000  0.53649477      0.79796781       1210.5889
2500  0.53534534      0.79401128       1551.6804

Best checkpoint:
runs/fm_avsr/lipavsr_30000_pred_energy_recon_textjson_wordts_v1/step_002000.pt

Training stayed under the 1h experiment limit. The run was explicitly stopped
after step 2500 once step 2000 remained the best checkpoint.
```

30k external eval from the best checkpoint:

```text
Matched val1000 metrics-only:
eval_out/lipavsr_30000_textjson_wordts_step2000_val1000_metrics/metrics.json
n=1000
mean_corr=0.5308107045
mean_mse=0.7437959659
mean_mae=0.6433063798

Condition-shift val1000 negative control:
eval_out/lipavsr_30000_textjson_wordts_step2000_val1000_shift1_metrics/metrics.json
n=1000
condition_shift=1
mean_corr=0.0243571066
mean_mse=1.2840104176
mean_mae=0.8762072835

The matched-vs-shifted gap remains large, so the 30k result is still grounded
in the matched visual/text/speaker condition rather than a condition-agnostic
average latent prediction.
```

30k audio samples:

```text
Path:
eval_out/lipavsr_30000_textjson_wordts_step2000_val6_wav

Files:
0000_pred.wav ... 0005_pred.wav
0000_gt.wav   ... 0005_gt.wav

Sample metrics:
n=6
mean_corr=0.5501924323
mean_mse=0.5943195671
mean_mae=0.5799260835
```

10k vs 30k comparison:

```text
Both rows use the same val1000 split and the same text_json + word_timestamps +
lip_avsr -> avsr_enc_lipavsr visual condition setup.

train_size  best_step  val_csv_corr  external_corr  external_mse  external_mae
10000       1500       0.51326406    0.5072512165   0.7721213100  0.6560713623
30000       2000       0.53649477    0.5308107045   0.7437959659  0.6433063798

Delta, 30k - 10k:
val_csv_corr:  +0.02323071
external_corr: +0.02355949
external_mse:  -0.02832534
external_mae:  -0.01276498
```

30k conclusion:

```text
Scaling the lip_avsr visual-prior experiment from 10k to 30k train clips gives
a real improvement on the matched val1000 split. The best external metrics-only
corr moves from 0.5073 to 0.5308, while the shifted-condition control stays near
zero. This supports continuing with more data or cleaner length-distribution
experiments, rather than reverting to the old visual features.

Caveat: the 30k train set is not a strict same-length-distribution expansion of
the 10k len80_260 train set. It includes short and long clips because complete
lip_avsr coverage could not provide 30k clips inside len80_260 alone.
```
