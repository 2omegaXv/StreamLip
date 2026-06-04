# FM-AVSR Audio Reconstruction

This branch keeps the current FM-AVSR deterministic recon pipeline and the
raw-video demo path. The active script surface is intentionally small:

```text
scripts/train_fm_avsr.py
scripts/eval_fm_avsr.py
scripts/extract_avsr_enc.py
scripts/extract_smollm2_h.py
scripts/extract_speaker_emb.py
scripts/extract_timbre_cond.py
scripts/preprocess_lrs3.py
scripts/preprocess_worker.py
scripts/reprocess_worker_avsr.py
scripts/run_preprocess_worker_no_flash_attn.py
scripts/run_raw_video_avsr_recon_pipeline.py
scripts/gradio_avsr_gui.py
```

Legacy v2/v3/v4, Mimi-code, teacher-cache, and sweep scripts are archived under
`archive/scripts/`.

## Environment

Use the shared virtualenv from the main checkout:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python --version
```

Run commands from this worktree:

```bash
cd /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.worktrees/fm-avsr-cleanup
```

The raw-video path expects these local assets to exist:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/mimi
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/smollm2-360m
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/resnet50-11ad3fa6.pth
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed/latent_norm_stats.npz
```

The default pipeline and GUI use the current timbre-fix recon checkpoint:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts.yaml
runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts_v1/step_002000.pt
```

## Raw Video Pipeline

### Video With Audio

Run one input video end to end. The first 3.04 seconds of the input audio are
used as the same-clip timbre/audio prompt and are removed from the listening
output:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/trump.mov \
  --exp trump_cleanup_verify \
  --force
```

The script performs:

```text
raw mp4/mov
-> 224x224 25fps video + 24kHz mono audio
-> face.npz/audio.wav/lip.npy
-> lip_avsr.npy
-> Mimi latent
-> avsr_enc_lipavsr.npy + avsr_text_lipavsr.txt
-> smollm2_h_lipavsr.npy
-> speaker_emb.npy + timbre_cond.npy
-> FM-AVSR recon
-> post-3.04s generated mp4
```

Important outputs are written under `eval_out/<exp>/`:

```text
<exp>_pred_prompt3s_post3s.mp4
<exp>_gt_mimi_post3s.mp4
recon_lipavsr_prompt3s/0000_pred.wav
recon_lipavsr_prompt3s/0000_gt.wav
recon_lipavsr_prompt3s/metrics.json
vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4
vis_reprocess_avsr/lip_avsr_crop_with_audio.mp4
```

The first 3.04 seconds are used as same-clip audio/timbre prompt and are removed
from the exported listening videos.

### Silent Reference Demo

Use the checked-in silent/reference example. It is the only
documented silent-mode demo for this branch.

```text
data/assets/trump_silent_ref_demo/trump_silent_input_no_tail3s.mp4
data/assets/trump_silent_ref_demo/trump_ref_tail3s.mp4
data/assets/trump_silent_ref_demo/trump_silent_ref_demo_full_pred_post3s.mp4
```

The silent input is `data/trump.mov` with the final 3 seconds removed and all
audio stripped. The reference file is the final 3 seconds of the same source
video, kept with audio for timbre/audio-prompt conditioning.

Recommended usage: if the original audio is only partially masked, use an
unmasked segment from the same video as `--ref_audio`. The first 3.04 seconds of
the reference must contain valid speech/audio, because the current model uses
only those first 38 Mimi frames as the audio prompt and timbre condition. This
lets the pipeline recover the missing silent video content with the same
speaker/timbre style instead of requiring a separate speaker reference.

Current hack / TODO: the model can copy the reference prompt audio into the
first generated seconds. Until this is fixed in the model, silent-mode exports
drop the first 3.04 seconds and produce a post-prompt listening video.

Reproduce the generated post-3.04s output:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/assets/trump_silent_ref_demo/trump_silent_input_no_tail3s.mp4 \
  --ref_audio data/assets/trump_silent_ref_demo/trump_ref_tail3s.mp4 \
  --silent_input \
  --exp trump_silent_ref_demo_full \
  --force
```

Expected generated video:

```text
eval_out/trump_silent_ref_demo_full/trump_silent_ref_demo_full_pred_post3s.mp4
```

The timbre-fix checkpoint was also verified on the same preprocessed Trump
silent-reference example:

```text
eval_out/trump_silent_ref_demo_full_e2_lossstart38/trump_silent_ref_demo_full_e2_lossstart38_pred_post3s.mp4
```

## GUI

Start the Gradio UI:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/gradio_avsr_gui.py \
  --port 7860
```

Open:

```text
http://0.0.0.0:7860
```

The GUI calls the same `scripts/run_raw_video_avsr_recon_pipeline.py` backend.

## Verified Example

The cleanup branch was verified with:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/trump.mov \
  --exp trump_cleanup_verify \
  --force
```

Generated artifacts:

```text
eval_out/trump_cleanup_verify/trump_cleanup_verify_pred_prompt3s_post3s.mp4
eval_out/trump_cleanup_verify/trump_cleanup_verify_gt_mimi_post3s.mp4
eval_out/trump_cleanup_verify/vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4
eval_out/trump_cleanup_verify/recon_lipavsr_prompt3s/metrics.json
```

Observed metrics:

```text
T_a: 371
metric_start_frame: 38
mean_corr: 0.3878001476
mean_mse: 0.8064498901
mean_mae: 0.6968652010
```

## Tests

Core validation command:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python -m unittest \
  tests.test_fm_avsr_dataset \
  tests.test_eval_fm_avsr \
  tests.test_timbre_condition \
  tests.test_fm_head_temporal_condition
```
