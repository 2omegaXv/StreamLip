"""
Pre-extract Auto-AVSR encoder features (enc_feat) for all LRS3 clips.

Input:  lip_avsr.npy  (T, 96, 96) uint8 灰度 — Auto-AVSR 兼容预处理
Saves:  avsr_enc.npy  (T, 768) float16 per clip.

Usage:
  python scripts/extract_avsr_enc.py --split pretrain
  python scripts/extract_avsr_enc.py --split trainval
  python scripts/extract_avsr_enc.py --split test
"""
import argparse, csv, sys, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.auto_avsr import AutoAVSRInferencer

DATA_ROOT  = Path("data/processed")
AVSR_MEAN  = 0.421
AVSR_STD   = 0.165
CROP_SIZE  = 88


def preprocess_lip_avsr(lip_np: np.ndarray) -> torch.Tensor:
    """(T, 96, 96) uint8 灰度 → (T, 1, 88, 88) float32，AVSR 归一化。"""
    x = torch.from_numpy(lip_np).float() / 255.0   # (T, 96, 96)
    margin = (x.shape[1] - CROP_SIZE) // 2
    x = x[:, margin:margin + CROP_SIZE, margin:margin + CROP_SIZE]
    x = (x - AVSR_MEAN) / AVSR_STD
    return x.unsqueeze(1)   # (T, 1, 88, 88)


@torch.no_grad()
def encode_frames(asr: AutoAVSRInferencer, frames: torch.Tensor,
                  device: str) -> torch.Tensor:
    """(T, 1, 88, 88) → (T, 768)，直接调 Auto-AVSR 内部组件。"""
    x = frames.unsqueeze(0).to(device)          # (1, T, 1, 88, 88)
    x = asr.model.frontend(x)                   # (1, T, 512)
    x = asr.model.proj_encoder(x)               # (1, T, 768)
    enc, _ = asr.model.encoder(x, None)         # (1, T, 768)
    return enc.squeeze(0)                        # (T, 768)


@torch.no_grad()
def encode_batch_frames(asr: AutoAVSRInferencer,
                        batch: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    """批量编码，pad 到最长，返回每条的 (T_i, 768)。"""
    lens  = [x.shape[0] for x in batch]
    max_T = max(lens)
    B     = len(batch)
    padded = torch.zeros(B, max_T, 1, CROP_SIZE, CROP_SIZE,
                         dtype=batch[0].dtype, device=device)
    for i, (x, l) in enumerate(zip(batch, lens)):
        padded[i, :l] = x.to(device)
    x = asr.model.frontend(padded)              # (B, T, 512)
    x = asr.model.proj_encoder(x)              # (B, T, 768)
    enc, _ = asr.model.encoder(x, None)        # (B, T, 768)
    return [enc[i, :l].cpu() for i, l in enumerate(lens)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="pretrain",
                   choices=["pretrain", "trainval", "test"])
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--force",      action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--avsr_ckpt",  default="pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading Auto-AVSR on {device}...")
    asr = AutoAVSRInferencer(args.avsr_ckpt, device=device)
    for p_ in asr.parameters(): p_.requires_grad_(False)
    asr.eval()

    clips = []
    with open(DATA_ROOT / "manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] != args.split: continue
            clips.append(DATA_ROOT / row["path"])
            if args.limit and len(clips) >= args.limit: break

    print(f"Split: {args.split}  |  Clips: {len(clips)}")
    done = skip = err = 0
    import time; t_start = time.perf_counter()
    LOG_EVERY = 500

    def save_enc(clip: Path, enc: torch.Tensor):
        enc_np = enc.float().clamp(-65504, 65504).half().numpy()
        np.save(str(clip / "avsr_enc.npy"), enc_np)

    batch_clips: list[Path] = []
    batch_frames: list[torch.Tensor] = []

    def flush_batch():
        nonlocal done, err
        if not batch_clips: return
        try:
            encs = encode_batch_frames(asr, batch_frames, device)
            for clip, enc in zip(batch_clips, encs):
                save_enc(clip, enc)
                done += 1
        except RuntimeError as e:
            if "memory" not in str(e).lower(): raise
            torch.cuda.empty_cache()
            for clip, frames in zip(batch_clips, batch_frames):
                try:
                    enc = encode_frames(asr, frames, device)
                    save_enc(clip, enc.cpu())
                    done += 1
                except Exception as e2:
                    tqdm.write(f"  ERR {clip.name}: {e2}"); err += 1
        except Exception as e:
            tqdm.write(f"  batch ERR: {e}"); err += len(batch_clips)
        batch_clips.clear(); batch_frames.clear()

    pbar = tqdm(clips, desc="extract_avsr_enc")
    for clip in pbar:
        out_path  = clip / "avsr_enc.npy"
        lip_path  = clip / "lip_avsr.npy"

        # 跳过：没有新预处理结果
        if not lip_path.exists():
            err += 1
            pbar.set_postfix(done=done, skip=skip, err=err)
            continue

        # 跳过：已提取且比 lip_avsr.npy 更新
        if out_path.exists() and not args.force:
            if out_path.stat().st_mtime >= lip_path.stat().st_mtime:
                try:
                    arr = np.load(str(out_path), mmap_mode="r")
                    if arr.ndim == 2 and arr.shape[1] == 768:
                        skip += 1
                        pbar.set_postfix(done=done, skip=skip, err=err)
                        continue
                except Exception:
                    pass

        try:
            lip    = np.load(str(lip_path))                # (T, 96, 96) uint8
            frames = preprocess_lip_avsr(lip)              # (T, 1, 88, 88)
            batch_clips.append(clip)
            batch_frames.append(frames)
            if len(batch_clips) >= args.batch_size:
                flush_batch()
        except Exception as e:
            tqdm.write(f"  load ERR {clip.name}: {e}"); err += 1

        total_done = done + skip
        if total_done > 0 and total_done % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t_start
            rate    = total_done / elapsed
            eta_h   = (len(clips) - total_done) / rate / 3600
            print(f"[{total_done:6d}/{len(clips)}] done={done} skip={skip} err={err} "
                  f"| {rate:.1f} clips/s | ETA {eta_h:.1f}h", flush=True)
        pbar.set_postfix(done=done, skip=skip, err=err)

    flush_batch()
    elapsed = time.perf_counter() - t_start
    print(f"\nDone={done}  Skip={skip}  Err={err}  Time={elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()

