"""
LRS3 预处理主脚本

两阶段：
  1. 视频处理（并行 subprocess worker，无 transformers）→ lip.npy / face.mp4 / audio.wav / text.json
  2. Mimi latent 提取（主进程 GPU）→ latent.npz

用法：
  python scripts/preprocess_lrs3.py --split test --workers 4 --gpu 0
  python scripts/preprocess_lrs3.py --split test --workers 4 --gpu 0 --limit 3  # 调试
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
try:
    import torchaudio
except OSError:
    torchaudio = None  # fallback: use scipy in extract_latent
from tqdm import tqdm
from transformers import MimiModel

LRS3_ROOT  = Path("/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3")
OUT_ROOT   = Path("data/processed")
MIMI_PATH  = Path("pretrained/mimi")
WORKER_PY  = Path("scripts/preprocess_worker.py")
PYTHON     = str(Path(sys.executable))

TARGET_FPS = 25
MIMI_SR    = 24000


# ── Mimi latent 提取 ─────────────────────────────────────────────────────────

def build_mimi(device, mimi_path):
    mimi = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(device).eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi


@torch.no_grad()
def extract_latent(mimi, wav_path, device):
    from scipy.io import wavfile
    import scipy.signal
    sr, wav_np = wavfile.read(str(wav_path))
    wav_f = wav_np.astype(np.float32) / 32768.0
    if wav_f.ndim > 1:
        wav_f = wav_f.mean(axis=1)
    if sr != MIMI_SR:
        n_out = int(len(wav_f) * MIMI_SR / sr)
        wav_f = scipy.signal.resample(wav_f, n_out)
    wav = torch.from_numpy(wav_f).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
    codes = mimi.encode(wav).audio_codes                             # (1, 32, T_a) int64
    x_q   = mimi.quantizer.decode(codes)                             # (1, 512, T_a) quantized
    lat   = x_q.transpose(1,2).squeeze(0).float().cpu()             # (T_a, 512) float32
    lat   = lat.clamp(-65504, 65504)                                  # safe float16 range
    return lat.half().numpy()                                         # (T_a, 512) float16


def extract_latents_gpu(clip_dirs, device, mimi_path, force=False, total=None):
    mimi = build_mimi(device, mimi_path)
    errors = []
    for d in tqdm(clip_dirs, desc="Mimi latent", unit="clip", total=total, dynamic_ncols=True):
        latent_path = d / "latent.npz"
        audio_path  = d / "audio.wav"
        if latent_path.exists() and not force:
            continue
        if not audio_path.exists():
            errors.append(f"missing audio: {d}")
            continue
        try:
            latent = extract_latent(mimi, audio_path, device)
            np.savez(str(latent_path), latent=latent)
        except Exception as e:
            errors.append(f"{d}: {e}")
    return errors


# ── 并行 subprocess worker ───────────────────────────────────────────────────

def _stream_worker(job_file: str, fa_device: str, pbar, all_results: list, lock):
    """启动 worker 子进程，流式读取每个 clip 完成时输出的 JSON 行，实时更新进度条"""
    proc = subprocess.Popen(
        [PYTHON, str(WORKER_PY), "--job_file", job_file, "--fa_device", fa_device],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            res = json.loads(line)
            with lock:
                all_results.append(res)
                pbar.update(1)
                if not res.get("ok") and not res.get("skipped"):
                    tqdm.write(f"  ERR {res.get('path','?')}: {res.get('error','')}")
        except Exception:
            pass
    proc.wait()


def parallel_video_processing(worker_args, n_workers, tmpdir, fa_device):
    """把 worker_args 分成 n_workers 批，每批启动一个 subprocess，实时进度条"""
    batches = [[] for _ in range(n_workers)]
    for i, a in enumerate(worker_args):
        batches[i % n_workers].append({"mp4": a[0], "txt": a[1], "out": a[2]})

    job_files = []
    for i, batch in enumerate(batches):
        if not batch:
            continue
        jf = str(Path(tmpdir) / f"job_{i:03d}.json")
        Path(jf).write_text(json.dumps(batch))
        job_files.append(jf)

    import concurrent.futures, threading
    all_results = []
    lock = threading.Lock()
    pbar = tqdm(total=len(worker_args), desc="视频裁剪", unit="clip")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_stream_worker, jf, fa_device, pbar, all_results, lock)
                for jf in job_files]
        concurrent.futures.wait(futs)
    pbar.close()
    return all_results


# ── manifest ─────────────────────────────────────────────────────────────────

def build_manifest(splits, out_root):
    rows = []
    for split in splits:
        split_dir = out_root / split
        if not split_dir.exists():
            continue
        for speaker_dir in sorted(split_dir.iterdir()):
            for clip_dir in sorted(speaker_dir.iterdir()):
                if not all([(clip_dir/"text.json").exists(),
                            (clip_dir/"lip.npy").exists(),
                            (clip_dir/"latent.npz").exists()]):
                    continue
                meta   = json.loads((clip_dir/"text.json").read_text())
                latent = np.load(str(clip_dir/"latent.npz"))["latent"]
                rows.append({
                    "split":     split,
                    "speaker_id": speaker_dir.name,
                    "clip_id":   clip_dir.name,
                    "path":      str(clip_dir.relative_to(out_root)),
                    "n_frames":  meta["n_frames"],
                    "duration_sec": round(meta["n_frames"] / TARGET_FPS, 3),
                    "n_words":   len(meta["words"]),
                    "n_latent_frames": len(latent),
                })
    if not rows:
        print("manifest: 0 clips，跳过")
        return
    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest: {len(rows)} clips → {manifest_path}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def collect_clips(lrs3_root, split):
    split_dir = lrs3_root / split / split
    if not split_dir.exists():
        raise FileNotFoundError(f"找不到: {split_dir}")
    clips = []
    for speaker_dir in sorted(split_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for mp4 in sorted(speaker_dir.glob("*.mp4")):
            clips.append((mp4, mp4.with_suffix(".txt")))
    return clips


def iter_processed_clip_dirs_from_manifest(out_root, split, limit=None, num_shards=1, shard_id=0):
    manifest_path = out_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest: {manifest_path}")
    n = 0
    seen = 0
    for row in csv.DictReader(open(manifest_path)):
        if row["split"] != split:
            continue
        if seen % num_shards != shard_id:
            seen += 1
            continue
        yield out_root / row["path"]
        seen += 1
        n += 1
        if limit and n >= limit:
            break


def count_manifest_split(out_root, split, limit=None, num_shards=1, shard_id=0):
    manifest_path = out_root / "manifest.csv"
    n = 0
    seen = 0
    for row in csv.DictReader(open(manifest_path)):
        if row["split"] != split:
            continue
        if seen % num_shards != shard_id:
            seen += 1
            continue
        n += 1
        seen += 1
        if limit and n >= limit:
            break
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    choices=["pretrain","trainval","test"], required=True)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--gpu",      type=int, default=0)
    parser.add_argument("--fa_device", default=None,
                        help="face_alignment 设备，默认与 --gpu 一致，如 cuda:1 或 cpu")
    parser.add_argument("--lrs3_root", default=str(LRS3_ROOT))
    parser.add_argument("--out_root",  default=str(OUT_ROOT))
    parser.add_argument("--mimi_path", default=str(MIMI_PATH))
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--latent_only", action="store_true",
                        help="Only re-extract latent.npz from existing processed audio.wav files listed in manifest.csv.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split latent_only manifest work into this many disjoint shards.")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for latent_only, in [0, num_shards).")
    parser.add_argument("--force_latent", action="store_true",
                        help="Re-extract latent.npz even if it already exists (use after VQ fix)")
    args = parser.parse_args()

    lrs3_root = Path(args.lrs3_root)
    out_root  = Path(args.out_root)
    mimi_path = Path(args.mimi_path)
    device    = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    fa_device = args.fa_device or (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")
    print(f"Split: {args.split}  Workers: {args.workers}  Mimi: {device}  FAN: {fa_device}", flush=True)

    if args.latent_only:
        total = count_manifest_split(
            out_root, args.split, args.limit, args.num_shards, args.shard_id
        )
        clip_dirs = iter_processed_clip_dirs_from_manifest(
            out_root, args.split, args.limit, args.num_shards, args.shard_id
        )
        print(
            f"latent_only: 从 manifest 流式处理 {total} 个 clip "
            f"(shard {args.shard_id}/{args.num_shards})",
            flush=True,
        )
        print(f"\n提取 Mimi latent（{device}）...", flush=True)
        latent_errors = extract_latents_gpu(
            clip_dirs, device, mimi_path, force=args.force_latent, total=total
        )
        if latent_errors:
            print(f"latent 错误 {len(latent_errors)} 个:")
            for e in latent_errors[:10]:
                print(f"  {e}")
        print("完成。", flush=True)
        return

    # 1. 收集 clip 列表
    clips = collect_clips(lrs3_root, args.split)
    if args.limit:
        clips = clips[:args.limit]
    print(f"共 {len(clips)} 个 clip", flush=True)

    # 2. 构建 worker 参数，跳过已完成的 clip（从本地目录判断，不用 NAS stat）
    done = set()
    split_dir = out_root / args.split
    if split_dir.exists():
        for spk in split_dir.iterdir():
            if not spk.is_dir():
                continue
            for clip in spk.iterdir():
                if (clip / "lip.npy").exists():
                    done.add((spk.name, clip.name))  # (speaker_id, clip_stem)

    worker_args = []
    for mp4, txt in clips:
        parts   = mp4.relative_to(lrs3_root / args.split / args.split).parts
        out_dir = out_root / args.split / parts[0] / mp4.stem
        if (parts[0], mp4.stem) in done:
            continue
        worker_args.append((str(mp4), str(txt), str(out_dir)))

    print(f"已完成: {len(done)}  待处理: {len(worker_args)}", flush=True)
    if not worker_args:
        print("全部已完成，跳到 Mimi latent 阶段", flush=True)
        clip_dirs = [d for spk in split_dir.iterdir() if spk.is_dir()
                     for d in spk.iterdir() if (d / "lip.npy").exists()]
    else:
        # 并行视频处理
        with tempfile.TemporaryDirectory() as tmpdir:
            results = parallel_video_processing(worker_args, args.workers, tmpdir, fa_device)

        ok    = sum(1 for r in results if r.get("ok") and not r.get("skipped"))
        skip  = sum(1 for r in results if r.get("skipped"))
        err   = sum(1 for r in results if not r.get("ok"))
        clip_dirs = [Path(r["path"]) for r in results if r.get("ok")]
        # 加上之前已完成的 clip
        for spk in split_dir.iterdir():
            if not spk.is_dir():
                continue
            for d in spk.iterdir():
                if (d / "lip.npy").exists() and d not in clip_dirs:
                    clip_dirs.append(d)
        print(f"视频完成: ok={ok}, skip={skip}, err={err}", flush=True)

    # 4. Mimi latent
    print(f"\n提取 Mimi latent（{device}）...", flush=True)
    latent_errors = extract_latents_gpu(clip_dirs, device, mimi_path, force=args.force_latent)
    if latent_errors:
        print(f"latent 错误 {len(latent_errors)} 个:")
        for e in latent_errors[:10]:
            print(f"  {e}")

    # 5. manifest
    build_manifest([args.split], out_root)
    print("完成。")


if __name__ == "__main__":
    main()
