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

## Raw Video Pipeline

Run one input video end to end:

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

