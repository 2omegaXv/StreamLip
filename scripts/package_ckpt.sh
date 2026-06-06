#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OUT_DIR="${OUT_DIR:-release}"
ARCHIVE_NAME="${ARCHIVE_NAME:-streamlip_ckpt_${DATE_TAG}.tar}"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"
SHA_PATH="${ARCHIVE_PATH}.sha256"
MANIFEST_PATH="${OUT_DIR}/streamlip_ckpt_${DATE_TAG}_manifest.txt"
PART_SIZE="${PART_SIZE:-256M}"

mkdir -p "$OUT_DIR"

python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

"$python_bin" scripts/check_env.py --skip-imports --skip-cuda

find ckpt -type f \
  ! -path '*/.cache/*' \
  ! -name '*.lock' \
  ! -name '*.metadata' \
  -printf '%P\t%s\n' | sort > "$MANIFEST_PATH"

tar \
  --exclude='*/.cache/*' \
  --exclude='*.lock' \
  --exclude='*.metadata' \
  -cf "$ARCHIVE_PATH" ckpt
sha256sum "$ARCHIVE_PATH" > "$SHA_PATH"
rm -f "${ARCHIVE_PATH}".part*
split -b "$PART_SIZE" -d -a 3 "$ARCHIVE_PATH" "${ARCHIVE_PATH}.part"

du -h "$ARCHIVE_PATH"
echo "Archive:  $ARCHIVE_PATH"
echo "SHA256:   $SHA_PATH"
echo "Manifest: $MANIFEST_PATH"
echo "Parts:    ${ARCHIVE_PATH}.part* (${PART_SIZE} chunks)"
