"""
Pre-extract AV-HuBERT features for all LRS3 clips.

Reads lip.npy (T, 96, 96, 3) uint8, converts RGB→grayscale via averaging
through the patched 3-channel stem, and saves avhubert.npy (T, 768) float16.

Usage:
  cd /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A
  .venv/bin/python scripts/extract_avhubert.py --split pretrain --gpu 0
  .venv/bin/python scripts/extract_avhubert.py --split trainval --gpu 0
  .venv/bin/python scripts/extract_avhubert.py --split test     --gpu 0
"""
import argparse
import csv
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.streaminlip.av_hubert import AVHuBERTExtractor

AVHUBERT_CKPT = "pretrained/av-hubert/model.pt"
DATA_ROOT     = Path("data/processed")
BATCH_FRAMES  = 500   # frames per GPU batch (tune for VRAM)


def extract_clip(extractor: AVHuBERTExtractor, clip_dir: Path, device: str, force: bool = False) -> bool:
    out_path = clip_dir / "avhubert.npy"
    if out_path.exists() and not force:
        # Validate: try loading to catch corrupted files
        try:
            arr = np.load(str(out_path))
            if arr.ndim == 2 and arr.shape[1] == 768:
                return False  # valid, skip
        except Exception:
            pass  # corrupted, re-extract

    lip_path = clip_dir / "lip.npy"
    if not lip_path.exists():
        return False

    lip = np.load(str(lip_path))                          # (T, H, W, C) uint8
    T = lip.shape[0]

    # (T, H, W, 3) → (T, 3, H, W) float in [-1, 1]
    frames = torch.from_numpy(lip).float().permute(0, 3, 1, 2) / 127.5 - 1.0

    # Process in chunks to avoid OOM on long clips
    all_feats = []
    for start in range(0, T, BATCH_FRAMES):
        chunk = frames[start:start + BATCH_FRAMES].unsqueeze(0).to(device)  # (1, t, 3, H, W)
        with torch.no_grad():
            feat = extractor(chunk)   # (1, t, 768)
        all_feats.append(feat.squeeze(0).cpu().half().numpy())

    feats = np.concatenate(all_feats, axis=0)  # (T, 768) float16
    np.save(str(out_path), feats)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="trainval", choices=["pretrain", "trainval", "test"])
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-extract even if avhubert.npy exists")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading AV-HuBERT on {device}...")
    extractor = AVHuBERTExtractor(AVHUBERT_CKPT, device=device)
    extractor.eval()

    # Load clip list from manifest
    clips = []
    with open(DATA_ROOT / "manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] != args.split:
                continue
            clips.append(DATA_ROOT / row["path"])
            if args.limit and len(clips) >= args.limit:
                break

    print(f"Split: {args.split}  |  Clips: {len(clips)}")

    done = skip = err = 0
    for clip_dir in tqdm(clips, desc=f"extract {args.split}"):
        try:
            if extract_clip(extractor, clip_dir, device, force=args.force):
                done += 1
            else:
                skip += 1
        except Exception as e:
            tqdm.write(f"  ERR {clip_dir.name}: {e}")
            err += 1

    print(f"\nDone: {done}  Skipped (already exist): {skip}  Errors: {err}")


if __name__ == "__main__":
    main()
