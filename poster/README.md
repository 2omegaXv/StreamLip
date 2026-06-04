# FM-AVSR Poster

This folder contains the editable one-page poster for the current FM-AVSR
reconstruction system.

## Files

- `fm_avsr_poster.pptx`: editable poster deck.
- `fm_avsr_poster.pdf`: exported PDF for submission or quick viewing.
- `fm_avsr_poster_preview.png`: raster preview of the poster.
- `build_poster.py`: regenerates the PPTX from the template and checked-in
  assets.
- `Poster Template.pptx`: copied source template used for slide size and
  poster format.
- `assets/`: Trump demo frames, waveforms, and reused report figures.

## Rebuild

```bash
./poster/build.sh
```

The Trump example uses the checked-in silent-reference sample:

```text
data/assets/trump_silent_ref_demo/trump_silent_input_no_tail3s.mp4
data/assets/trump_silent_ref_demo/trump_ref_tail3s.mp4
data/assets/trump_silent_ref_demo/trump_silent_ref_demo_full_pred_post3s.mp4
```

The poster text example is taken from the processed Trump AVSR transcript:

```text
eval_out/trump_raw_prompt_pipeline/processed/custom/trump/00001/avsr_text_lipavsr.txt
```

It includes the phrase:

```text
OUR COUNTRY IS WINNING ... WE'RE WINNING SO MUCH ... WE WINNING TOO MUCH ...
```
