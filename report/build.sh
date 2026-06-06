#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PDF_NAME="fm_avsr_final_report_2026-06-05.pdf"
BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"

cleanup() {
  if [[ "${KEEP_BUILD:-0}" != "1" && -d "$BUILD_DIR" ]]; then
    rm -rf "$BUILD_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$BUILD_DIR" main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" main.tex
else
  echo "No LaTeX compiler found. Install latexmk or pdflatex, then run ./build.sh again." >&2
  exit 1
fi

cp "$BUILD_DIR/main.pdf" "$PDF_NAME"
echo "Built report/$PDF_NAME"
