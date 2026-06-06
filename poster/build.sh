#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python"

cd "$ROOT"

"$PYTHON" poster/build_poster.py
libreoffice --headless --convert-to pdf --outdir poster poster/fm_avsr_poster.pptx
pdftoppm -png -singlefile -r 160 poster/fm_avsr_poster.pdf poster/fm_avsr_poster_preview
