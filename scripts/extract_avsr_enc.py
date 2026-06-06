"""
Pre-extract Auto-AVSR encoder features (enc_feat) for LRS3 clips.

Saves: avsr_enc.npy  (T', 768) float16 per clip.
T' ≈ T/4 (ConformerEncoder downsamples ~4x from 25fps → ~6.25fps).

This allows FM head training to load enc_feat directly without running
Auto-AVSR online, reducing training time from hours/epoch to minutes/epoch.

Usage:
  python scripts/extract_avsr_enc.py --split pretrain
  python scripts/extract_avsr_enc.py --split trainval
  python scripts/extract_avsr_enc.py --split test
  python scripts/extract_avsr_enc.py --clip_list configs/eval_splits/foo.txt \
    --input_name lip_avsr.npy --output_name avsr_enc_lipavsr.npy
"""
import argparse, csv, sys, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.auto_avsr import AutoAVSRInferencer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("data/processed")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="pretrain",
                   choices=["pretrain", "trainval", "test"])
    p.add_argument("--gpu",    type=int, default=0)
    p.add_argument("--limit",  type=int, default=None)
    p.add_argument("--force",  action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--avsr_ckpt", default=str(REPO_ROOT / "ckpt/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth"))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--clip_list", default=None,
                   help="Optional file with clip paths relative to data_root or absolute paths.")
    p.add_argument("--input_name", default="lip.npy",
                   help="Per-clip visual input file, e.g. lip.npy or lip_avsr.npy.")
    p.add_argument("--output_name", default="avsr_enc.npy",
                   help="Per-clip output feature file; avoid overwriting old features for A/B tests.")
    p.add_argument("--text_output_name", default="avsr_text.txt",
                   help="CTC greedy text output file. Use a distinct name for A/B feature extraction.")
    args = p.parse_args()
    data_root = Path(args.data_root)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading Auto-AVSR on {device}...")
    asr = AutoAVSRInferencer(args.avsr_ckpt, device=device)
    for p_ in asr.parameters(): p_.requires_grad_(False)
    asr.eval()

    clips = []
    if args.clip_list:
        for line in Path(args.clip_list).read_text().splitlines():
            rel = line.strip()
            if not rel or rel.startswith("#"):
                continue
            clip = Path(rel)
            clips.append(clip if clip.is_absolute() else data_root / rel)
            if args.limit and len(clips) >= args.limit:
                break
    else:
        with open(data_root / "manifest.csv") as f:
            for row in csv.DictReader(f):
                if row["split"] != args.split: continue
                clips.append(data_root / row["path"])
                if args.limit and len(clips) >= args.limit: break

    print(
        f"Split: {args.split}  |  Clips: {len(clips)}  "
        f"| input={args.input_name} | output={args.output_name}"
    )
    done = skip = err = 0
    batch_clips, batch_frames = [], []
    import time
    t_start = time.perf_counter()
    LOG_EVERY = 500  # print summary every N clips processed

    def save_one(clip, enc, log_probs):
        enc_np = enc.float().cpu().clamp(-65504, 65504).half().numpy()
        np.save(str(clip / args.output_name), enc_np)
        ids = log_probs.argmax(-1).tolist()
        blank, prev, col = 0, 0, []
        for t in ids:
            if t != blank and t != prev: col.append(t)
            prev = t
        pred_ids = torch.tensor(col, dtype=torch.long)
        text = asr.text_transform.post_process(pred_ids).replace("<eos>","").strip()
        (clip / args.text_output_name).write_text(text)

    def flush_batch():
        nonlocal done, err
        if not batch_clips: return
        try:
            results = asr.encode_batch(batch_frames)
            for clip, (enc, lp) in zip(batch_clips, results):
                save_one(clip, enc, lp)
                done += 1
        except RuntimeError as e:
            if "out of memory" not in str(e): raise
            torch.cuda.empty_cache()
            # OOM fallback: process one by one
            for clip, frames in zip(batch_clips, batch_frames):
                try:
                    enc, lp = asr._encode(frames)
                    save_one(clip, enc, lp)
                    done += 1
                except Exception as e2:
                    tqdm.write(f"  single ERR {clip.name}: {e2}"); err += 1
        except Exception as e:
            tqdm.write(f"  batch ERR: {e}"); err += len(batch_clips)
        batch_clips.clear(); batch_frames.clear()

    pbar = tqdm(clips, desc="extract_avsr")
    for clip in pbar:
        out_path = clip / args.output_name
        if out_path.exists() and not args.force:
            try:
                arr = np.load(str(out_path))
                if arr.ndim == 2 and arr.shape[1] == 768:
                    skip += 1
                    pbar.set_postfix(done=done, skip=skip, err=err)
                    continue
            except: pass

        lip_path = clip / args.input_name
        if not lip_path.exists(): err += 1; continue

        try:
            lip = np.load(str(lip_path))
            batch_clips.append(clip)
            batch_frames.append(torch.from_numpy(lip))
            if len(batch_clips) >= args.batch_size:
                flush_batch()
        except Exception as e:
            tqdm.write(f"  load ERR {clip.name}: {e}"); err += 1

        # Periodic summary log
        total_done = done + skip
        if total_done > 0 and total_done % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t_start
            rate = total_done / elapsed
            eta_h = (len(clips) - total_done) / rate / 3600
            print(f"[{total_done:6d}/{len(clips)}] done={done} skip={skip} err={err} "
                  f"| {rate:.1f} clips/s | ETA {eta_h:.1f}h", flush=True)
        pbar.set_postfix(done=done, skip=skip, err=err)

    flush_batch()
    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {done}  Skipped: {skip}  Errors: {err}  "
          f"Total time: {elapsed/3600:.1f}h")

    print(f"\nDone: {done}  Skipped: {skip}  Errors: {err}")


if __name__ == "__main__":
    main()
