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
| `hrx_pred_prompt3s_post3s_reprocess_avsr.mp4` | 15.84 s | HRX generated output with current visual preprocessing. |

The files are 224x224 mp4 exports produced by the current StreamLip raw-video
pipeline. Larger raw source videos such as `data/trump.mov` are treated as local
development assets and are not committed.
