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
