"""
Pre-extract AV-HuBERT CNN frontend features (Transformer input) for all LRS3 clips.

Saves: avhubert_pre.npy  (T, 768) float16 per clip.
This is the output of post_extract_proj, i.e., the input to the 12-layer Transformer.
Training can then run LoRA on the Transformer while skipping the CNN frontend.

Data ratio: lip.npy ~4MB → avhubert_pre.npy ~0.24MB (18x smaller)
Total pretrain: ~27GB (fits in 100GB RAM for memory-mapped fast access)

Usage:
  python scripts/extract_avhubert_pre.py --split pretrain
  python scripts/extract_avhubert_pre.py --split trainval
  python scripts/extract_avhubert_pre.py --split test
"""
import argparse
import csv
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/av_hubert"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/av_hubert/avhubert"))

from streaminlip.av_hubert import load_avhubert

AVHUBERT_CKPT = "pretrained/self_large_vox_433h.pt"
DATA_ROOT     = Path("data/processed")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AVHuBERTPreExtractor(nn.Module):
    """
    Runs only the CNN frontend + post_extract_proj of AV-HuBERT.
    Output: (B, T, 768) — Transformer input, before dropout and positional conv.
    """
    def __init__(self, avhubert_model: nn.Module):
        super().__init__()
        self.model = avhubert_model

    @torch.no_grad()
    def forward(self, lip_frames: torch.Tensor) -> torch.Tensor:
        """
        lip_frames: (B, T, 3, 96, 96) float32, ImageNet normalised
        returns:    (B, T, 768) float32
        """
        B, T = lip_frames.shape[:2]
        x = lip_frames.permute(0, 2, 1, 3, 4)   # (B, 3, T, 96, 96)
        x = x.to(next(self.model.parameters()).dtype)

        m = self.model
        # video CNN frontend: (B, 768, T)
        feat_v = m.forward_features(x, modality='video')
        # audio placeholder zeros (same shape, different channel dim)
        feat_a = torch.zeros(B, m.encoder_embed_dim, T,
                             device=x.device, dtype=x.dtype)

        if m.modality_fuse == 'concat':
            features = torch.cat([feat_a, feat_v], dim=1)  # (B, 1536, T)
        else:
            features = feat_a + feat_v                      # (B, 768, T)

        features = features.transpose(1, 2)   # (B, T, 1536 or 768)
        features = m.layer_norm(features)

        if m.post_extract_proj is not None:
            features = m.post_extract_proj(features)       # (B, T, 768)

        return features.float()


def extract_clip(extractor: AVHuBERTPreExtractor, clip_dir: Path,
                 device: str, force: bool = False) -> bool:
    out_path = clip_dir / "avhubert_pre.npy"
    if out_path.exists() and not force:
        try:
            arr = np.load(str(out_path))
            if arr.ndim == 2 and arr.shape[1] == 768:
                return False
        except Exception:
            pass

    lip_path = clip_dir / "lip.npy"
    if not lip_path.exists():
        return False

    lip = np.load(str(lip_path))               # (T, H, W, 3) uint8
    lip = lip.astype(np.float32) / 255.0
    lip = (lip - IMAGENET_MEAN) / IMAGENET_STD
    frames = torch.from_numpy(lip).permute(0, 3, 1, 2)  # (T, 3, H, W)
    frames = frames.unsqueeze(0).to(device)              # (1, T, 3, H, W)

    feat = extractor(frames).squeeze(0).cpu().half().numpy()  # (T, 768) fp16
    np.save(str(out_path), feat)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="pretrain",
                   choices=["pretrain", "trainval", "test"])
    p.add_argument("--gpu",    type=int, default=0)
    p.add_argument("--limit",  type=int, default=None)
    p.add_argument("--force",  action="store_true")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading AV-HuBERT CNN frontend on {device} ...")
    model = load_avhubert(AVHUBERT_CKPT, device=device)
    model.eval()
    extractor = AVHuBERTPreExtractor(model).to(device)

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
    for clip_dir in tqdm(clips, desc=f"extract_pre {args.split}"):
        try:
            if extract_clip(extractor, clip_dir, device, force=args.force):
                done += 1
            else:
                skip += 1
        except Exception as e:
            tqdm.write(f"  ERR {clip_dir.name}: {e}")
            err += 1

    print(f"\nDone: {done}  Skipped: {skip}  Errors: {err}")


if __name__ == "__main__":
    main()
