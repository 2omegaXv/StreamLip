#!/usr/bin/env python
"""Audit processed Mimi latent frame rates.

For 25fps visual features and the local pretrained Mimi config, valid FM
targets are 12.5Hz latents, so enc_len / latent_len should be close to 2.
Ratios close to 1 indicate stale 25Hz quantizer latents that must be
re-extracted.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def classify_ratio(ratio: float) -> str:
    if 0.75 <= ratio <= 1.25:
        return "bad_25hz"
    if 1.5 <= ratio <= 2.5:
        return "ok_12p5hz"
    return "unexpected"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    manifest = root / "manifest.csv"
    rows = [r for r in csv.DictReader(open(manifest)) if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]

    counts = Counter()
    out_rows = []
    examples = {}

    for r in rows:
        clip = root / r["path"]
        try:
            enc_len = np.load(clip / "avsr_enc.npy", mmap_mode="r").shape[0]
            lat_len = np.load(clip / "latent.npz")["latent"].shape[0]
            ratio = enc_len / max(lat_len, 1)
            status = classify_ratio(ratio)
        except Exception as exc:
            enc_len = lat_len = 0
            ratio = 0.0
            status = "missing_or_error"
            r["error"] = str(exc)

        counts[status] += 1
        examples.setdefault(status, [])
        if len(examples[status]) < 5:
            examples[status].append((r["path"], enc_len, lat_len, round(ratio, 3)))
        out_rows.append({
            "path": r["path"],
            "status": status,
            "enc_len": enc_len,
            "latent_len": lat_len,
            "ratio": f"{ratio:.6f}",
        })

    print(f"split={args.split} clips={len(rows)}")
    for k in ["ok_12p5hz", "bad_25hz", "unexpected", "missing_or_error"]:
        print(f"{k}: {counts[k]}")
        for ex in examples.get(k, []):
            print(f"  {ex}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "status", "enc_len", "latent_len", "ratio"])
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
