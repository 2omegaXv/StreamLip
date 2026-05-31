"""
Pre-extract speaker embeddings from face.npz for all LRS3 clips.

Saves: speaker_emb.npy  (256,) float16 per clip.
Eliminates face.npz JPEG decode + ResNet50 forward from DataLoader hot path.

Usage:
  python scripts/extract_speaker_emb.py --split pretrain
"""
import argparse, csv, sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v2.speaker_encoder import SpeakerEncoder

DATA_ROOT = Path("data/processed")
CHUNK_SIZE = 6
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_clip(encoder, clip_dir, device, force=False):
    out_path = clip_dir / "speaker_emb.npy"
    if out_path.exists() and not force:
        try:
            arr = np.load(str(out_path))
            if arr.shape == (256,): return False
        except: pass

    face_path = clip_dir / "face.npz"
    if not face_path.exists(): return False

    import cv2
    f = np.load(str(face_path))
    data, offsets = f["data"], f["offsets"]
    n = min(CHUNK_SIZE, len(offsets) - 1)
    frames = []
    for i in range(n):
        buf = data[offsets[i]:offsets[i+1]].tobytes()
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face = np.stack(frames).mean(0).astype(np.float32) / 255.0
    face = (face - IMAGENET_MEAN) / IMAGENET_STD
    face_t = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = encoder(face_t).squeeze(0).cpu().half().numpy()  # (256,)
    np.save(str(out_path), emb)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="pretrain",
                   choices=["pretrain", "trainval", "test"])
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resnet50_weights", default="pretrained/resnet50-11ad3fa6.pth")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading SpeakerEncoder on {device}...")
    encoder = SpeakerEncoder(weights_path=args.resnet50_weights).to(device).eval()

    clips = []
    with open(DATA_ROOT / "manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] != args.split: continue
            clips.append(DATA_ROOT / row["path"])
            if args.limit and len(clips) >= args.limit: break

    print(f"Split: {args.split}  |  Clips: {len(clips)}")
    done = skip = err = 0
    for clip_dir in tqdm(clips):
        try:
            if extract_clip(encoder, clip_dir, device, args.force):
                done += 1
            else:
                skip += 1
        except Exception as e:
            tqdm.write(f"  ERR {clip_dir.name}: {e}")
            err += 1

    print(f"\nDone: {done}  Skipped: {skip}  Errors: {err}")


if __name__ == "__main__":
    main()
