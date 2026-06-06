#!/usr/bin/env python3
"""Strip a StreamLip V5 training checkpoint for inference release.

Training checkpoints may include optimizer state under ``opt``. That state is
not needed by ``scripts/extract_v5_text.py`` or the raw-video pipeline, which
only load ``step`` and ``model``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an inference-only StreamLip V5 checkpoint."
    )
    parser.add_argument("--input", required=True, help="Training checkpoint path.")
    parser.add_argument("--output", required=True, help="Inference checkpoint path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")

    ckpt = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    if "model" not in ckpt:
        raise KeyError(f"{src} does not contain a 'model' key")

    out = {
        "step": int(ckpt.get("step", 0)),
        "model": ckpt["model"],
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

    before = src.stat().st_size / 1024**3
    after = dst.stat().st_size / 1024**3
    print(f"source: {src} ({before:.2f} GiB)")
    print(f"output: {dst} ({after:.2f} GiB)")
    print(f"kept keys: {sorted(out.keys())}")
    print(f"dropped keys: {sorted(set(ckpt.keys()) - set(out.keys()))}")


if __name__ == "__main__":
    main()
