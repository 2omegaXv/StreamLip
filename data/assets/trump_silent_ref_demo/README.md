# Trump Silent Reference Demo

This folder contains a full-length silent-input/reference-audio demo for the
final AVSR reconstruction pipeline.

Source video:

```text
data/trump.mov
```

Assets:

| File | Duration | Description |
| --- | ---: | --- |
| `trump_silent_input_no_tail3s.mp4` | 26.567 s | `data/trump.mov` with the final 3 seconds removed and all audio removed. |
| `trump_ref_tail3s.mp4` | 3.015 s | The final 3 seconds of `data/trump.mov`, kept with audio, used as reference audio/timbre prompt. |
| `trump_silent_ref_demo_full_pred_full.mp4` | 26.560 s | Generated silent-mode output with synthesized audio muxed onto the silent input video. |

Reproduce the output:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  --input data/assets/trump_silent_ref_demo/trump_silent_input_no_tail3s.mp4 \
  --ref_audio data/assets/trump_silent_ref_demo/trump_ref_tail3s.mp4 \
  --silent_input \
  --exp trump_silent_ref_demo_full \
  --force
```

The pipeline output is:

```text
eval_out/trump_silent_ref_demo_full/trump_silent_ref_demo_full_pred_full.mp4
```

The checked-in copy is:

```text
data/assets/trump_silent_ref_demo/trump_silent_ref_demo_full_pred_full.mp4
```
