# Codebase Cleanup Plan - 2026-06-04

This is a conservative cleanup plan for the FM AVSR branch. It does not require
deleting experiment artifacts immediately. The goal is to make the final branch
reviewable by promoting only the core path and documenting the rest.

## Cleanup Principle

Use a whitelist. Do not try to commit everything currently visible in
`git status`.

Keep:

- final recon model code,
- final data/text condition support,
- raw-video one-command pipeline,
- focused tests,
- final docs.

Do not promote:

- bulk experiment outputs,
- exploratory configs,
- raw media,
- cache files,
- unrelated Moshi/Mimi-code experiments,
- large third-party/vendor directories unless they are already intentionally
  tracked.

## Core Files To Keep

### Documentation

Keep these docs in the final branch:

```text
doc/fm_avsr_final_status_2026-06-04.md
doc/fm_avsr_audio_generation_architecture.md
doc/raw_video_avsr_recon_pipeline_usage.md
doc/fm_avsr_timbre_condition_2026-06-03.md
doc/fm_avsr_lipavsr_visual_prior_2026-06-03.md
```

`doc/fm_avsr_experiment_status_2026-06-01.md` is useful historical context but
large. Keep it only if the final branch is intended to preserve the full
experiment trail.

### Core Config

Keep the final config:

```text
configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts.yaml
```

Keep final split files required by that config:

```text
configs/eval_splits/pretrain_lipavsr_train59144_seed44_plus_ready_remaining9144_excl_val1000_seed43.txt
configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt
```

Archive or leave untracked most exploratory configs:

```text
configs/fm_avsr_len120_220_*.yaml
configs/fm_avsr_len150_180_*.yaml
configs/mimi_code_avsr_*.yaml
configs/eval_splits/pretrain_len120_220_*.txt
configs/eval_splits/pretrain_len150_180_*.txt
```

### Core Source

Keep these source changes:

```text
src/streaminlip/fm_avsr_dataset.py
src/streaminlip/v2/fm_head.py
```

`fm_head.py` may not currently show as modified in this worktree, but it is part
of the conceptual core and should remain covered by tests.

### Core Scripts

Keep:

```text
scripts/train_fm_avsr.py
scripts/eval_fm_avsr.py
scripts/extract_smollm2_h.py
scripts/extract_avsr_enc.py
scripts/extract_timbre_cond.py
scripts/run_raw_video_avsr_recon_pipeline.py
scripts/run_preprocess_worker_no_flash_attn.py
scripts/reprocess_worker_avsr.py
scripts/gradio_avsr_gui.py
```

`scripts/reprocess_worker_avsr.py` is now local to this branch. It looks for
Auto-AVSR's mediapipe crop implementation in either:

```text
third_party/auto_avsr/preparation/detectors/mediapipe
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/third_party/auto_avsr/preparation/detectors/mediapipe
```

The full `third_party/auto_avsr` checkout is intentionally treated as an
external dependency, not as core branch code.

### Tests

Keep focused tests for the final behavior:

```text
tests/test_fm_avsr_dataset.py
tests/test_eval_fm_avsr.py
tests/test_timbre_condition.py
```

If tests for `lipavsr` text source live only in
`tests/test_fm_avsr_dataset.py`, that file should be kept. Tests unrelated to
the final FM AVSR path should not be promoted into this branch.

## Files To Leave Untracked Or Ignore

Do not commit these generated/local artifacts:

```text
eval_out/
runs/
.cache/
data/*.mov
data/*.mp4
data/processed/
external_model_cards/
data/mimi_code_cache*/
data/moshiko_mimi_code_cache*/
data/teacher_cache/
third_party/auto_avsr/
```

During the cleanup pass, untracked exploratory configs, Moshi/Mimi-code scripts,
and related tests were moved out of this worktree to:

```text
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.worktrees/fm-avsr-cleanup-untracked-archive-20260604
```

The current worktree has local media and generated outputs under `data/` and
`eval_out/`. They are useful for inspection but should not be part of the code
branch.

Do not commit temporary Python caches:

```text
__pycache__/
*.pyc
```

## Tracked Legacy Archive

Tracked historical command files were moved out of the primary command surface
and into explicit archive folders:

```text
archive/scripts/legacy_pipeline/
archive/scripts/mimi_code/
archive/scripts/sweeps/
archive/tests/
```

This keeps `scripts/` focused on the active FM AVSR recon/raw-video path while
preserving older v2/v3/v4, Mimi-code, teacher-cache, and sweep scripts for
provenance.

## Suggested Git Review Commands

Show only likely final files:

```bash
git status --short -- \
  doc/fm_avsr_final_status_2026-06-04.md \
  doc/fm_avsr_audio_generation_architecture.md \
  doc/raw_video_avsr_recon_pipeline_usage.md \
  doc/fm_avsr_timbre_condition_2026-06-03.md \
  doc/fm_avsr_lipavsr_visual_prior_2026-06-03.md \
  configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_residual_samplecorr02_from1000_recon_textjson_wordts.yaml \
  configs/eval_splits/pretrain_lipavsr_train59144_seed44_plus_ready_remaining9144_excl_val1000_seed43.txt \
  configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt \
  src/streaminlip/fm_avsr_dataset.py \
  scripts/train_fm_avsr.py \
  scripts/eval_fm_avsr.py \
  scripts/extract_smollm2_h.py \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  scripts/run_preprocess_worker_no_flash_attn.py \
  tests/test_fm_avsr_dataset.py \
  tests/test_eval_fm_avsr.py
```

Check for generated files accidentally staged:

```bash
git status --short | rg '^(A|M|\\?\\?) (eval_out|runs|data/.*\\.(mov|mp4|wav)|\\.cache|external_model_cards)'
```

## Verification Before Final Commit

Run:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python -m py_compile \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  scripts/run_preprocess_worker_no_flash_attn.py \
  scripts/reprocess_worker_avsr.py \
  scripts/gradio_avsr_gui.py \
  scripts/train_fm_avsr.py \
  scripts/eval_fm_avsr.py \
  scripts/extract_smollm2_h.py \
  src/streaminlip/fm_avsr_dataset.py
```

Run focused tests:

```bash
/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python -m unittest \
  tests.test_eval_fm_avsr \
  tests.test_fm_avsr_dataset
```

Run diff whitespace check:

```bash
git diff --check -- \
  doc \
  scripts/run_raw_video_avsr_recon_pipeline.py \
  scripts/run_preprocess_worker_no_flash_attn.py \
  scripts/train_fm_avsr.py \
  scripts/eval_fm_avsr.py \
  scripts/extract_smollm2_h.py \
  src/streaminlip/fm_avsr_dataset.py \
  tests/test_fm_avsr_dataset.py \
  tests/test_eval_fm_avsr.py
```

## Recommended Next Step

Before committing, rerun the verification commands above and stage only the
whitelist files. Leave generated data, raw media, external model cards, and the
full Auto-AVSR checkout untracked or ignored.
