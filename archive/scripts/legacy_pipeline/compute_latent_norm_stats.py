#!/usr/bin/env python
"""Compute per-channel normalization stats for valid 12.5Hz Mimi latents."""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    output = Path(args.output) if args.output else root / "latent_norm_stats.npz"
    rows = [r for r in csv.DictReader(open(root / "manifest.csv")) if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]

    sum_x = np.zeros(512, dtype=np.float64)
    sum_x2 = np.zeros(512, dtype=np.float64)
    count = 0
    skipped = 0
    errors = 0

    for r in tqdm(rows, desc="latent stats", unit="clip", dynamic_ncols=True):
        clip = root / r["path"]
        try:
            enc_len = np.load(clip / "avsr_enc.npy", mmap_mode="r").shape[0]
            lat = np.load(clip / "latent.npz")["latent"].astype("float32")
        except Exception:
            errors += 1
            continue
        ratio = enc_len / max(lat.shape[0], 1)
        if not (1.5 <= ratio <= 2.5):
            skipped += 1
            continue
        sum_x += lat.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(lat, dtype=np.float64).sum(axis=0)
        count += lat.shape[0]

    if count == 0:
        raise RuntimeError("No valid 12.5Hz latent frames found.")

    mean = (sum_x / count).astype("float32")
    var = np.maximum(sum_x2 / count - np.square(sum_x / count), 1e-8)
    std = np.sqrt(var).astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mean=mean, std=std, count=np.array(count, dtype=np.int64))
    print(f"wrote {output}")
    print(f"frames={count} skipped_clips={skipped} error_clips={errors}")
    print(f"mean_abs_mean={float(np.abs(mean).mean()):.6f} std_mean={float(std.mean()):.6f}")


if __name__ == "__main__":
    main()
