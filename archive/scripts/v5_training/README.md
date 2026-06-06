# StreamLip V5 Training Archive

This directory preserves the research/training scripts from the original V5 development commits on `main`.

## Preserved Files

- `train_v5_avsr.py`: trains StreamLip V5, an LM-based VSR branch that consumes frozen Auto-AVSR visual speech features and injects them into the LM through gated cross-attention.
- `finetune_lm.py`: fine-tunes the LM on LRS3 transcripts before V5 training.
- `prepare_lm_text.py`: collects LRS3 transcript text for LM adaptation.
- `eval_compare.py`: compares StreamLip V5 and Auto-AVSR WER under shared evaluation settings.
- `reprocess_avsr.py`: schedules Auto-AVSR-compatible lip crop reprocessing jobs for large LRS3 splits.
- `sweep_lm.sh`: records the LM fine-tuning sweep used during V5 development.

## Status

These files are archived rather than exposed as the default release surface. They are valuable for reproducing and explaining the V5 training process, but several paths still reflect the original research environment (`pretrained/...`, `runs/v5/...`). The cleaned demo/release path uses:

- `scripts/run_raw_video_avsr_recon_pipeline.py`
- `scripts/gradio_avsr_gui.py`
- `scripts/extract_v5_text.py`
- `scripts/decode_v5.py`
- checkpoints restored under `ckpt/`

The original V5 system note is preserved at `archive/docs_legacy/v5_training/v5_system.md`.
