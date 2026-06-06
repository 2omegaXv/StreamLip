# StreamLip Demo Videos

This directory contains small generated-output examples that are safe to keep in git.
They are intended for quick visual inspection after cloning the repository; they are
not raw training data.

| File | Duration | Notes |
| --- | ---: | --- |
| `0000_0001_pred_orig_post3s.mp4` | 5.52 s | Generated output, post-prompt region only. |
| `0001_0003_pred_orig_post3s.mp4` | 17.60 s | Generated output, post-prompt region only. |
| `0003_0017_pred_orig_post3s.mp4` | 3.80 s | Generated output, post-prompt region only. |
| `0018_pred_orig.mp4` | 4.52 s | Generated output. |
| `pred_orig_video_no_audio_input_post3s.mp4` | 3.04 s | Generated output for a no-audio input demo. |

The files are 224x224 mp4 exports produced by the current StreamLip raw-video
pipeline. Larger raw source videos such as `data/trump.mov` are treated as local
development assets and are not committed.
