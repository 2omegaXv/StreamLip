# Trump Silent Reference Demo

This folder contains a silent-input/reference-audio demo for the final AVSR
reconstruction pipeline.

Source video note:

```text
The assets were prepared from a local Trump source video. The raw source
`data/trump.mov` is not committed; use the checked-in files below to reproduce
the demo.
```

Assets:

| File | Duration | Description |
| --- | ---: | --- |
| `trump_silent_input_no_tail3s.mp4` | 26.567 s | Source video with the final 3 seconds removed and all audio removed. |
| `trump_ref_tail3s.mp4` | 3.015 s | The final 3 seconds of the source video, kept with audio, used as reference audio/timbre prompt. |
| `trump_silent_ref_demo_full_pred_post3s.mp4` | 23.552 s | Generated silent-mode output after dropping the first 3.04 seconds. |

Current hack / TODO: the model can copy the reference prompt audio into the
first generated seconds. Until this is fixed in the model, silent-mode exports
drop the first 3.04 seconds and provide a post-prompt listening video.

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
eval_out/trump_silent_ref_demo_full/trump_silent_ref_demo_full_pred_post3s.mp4
```

The checked-in copy is:

```text
data/assets/trump_silent_ref_demo/trump_silent_ref_demo_full_pred_post3s.mp4
```
