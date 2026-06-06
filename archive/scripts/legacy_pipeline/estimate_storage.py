"""
估算 LRS3 全量数据集在不同存储方案下的空间、预处理时间、训练读取时间。
基于已处理的 3868 个 pretrain clip 采样统计。

用法：
  .venv/bin/python3 scripts/estimate_storage.py
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import io

PROCESSED_DIR = Path("data/processed/pretrain")
SAMPLE_N = 200  # 采样 clip 数量

# LRS3 全量 clip 数（来自文献/官网统计）
TOTAL_CLIPS = {
    "pretrain":  118516,
    "trainval":   31982,
    "test":        1321,
}
TOTAL_ALL = sum(TOTAL_CLIPS.values())

# 训练时每 epoch 读取估算（假设 batch=32, 每 clip 读一次/epoch）
TRAINING_WORKERS = 8
EPOCHS = 50


def sample_clips(processed_dir, n):
    all_clips = []
    for spk in processed_dir.iterdir():
        if not spk.is_dir():
            continue
        for clip in spk.iterdir():
            if (clip / "lip.npy").exists() and (clip / "text.json").exists():
                all_clips.append(clip)
    return random.sample(all_clips, min(n, len(all_clips)))


def measure_clip(clip_dir):
    """对单个 clip 测量各指标"""
    lip_path  = clip_dir / "lip.npy"
    face_path = clip_dir / "face.npy"
    audio_path = clip_dir / "audio.wav"

    lip_arr  = np.load(str(lip_path))              # (T, 96, 96, 3)
    face_arr = np.load(str(face_path)) if face_path.exists() else None
    T = lip_arr.shape[0]

    result = {
        "n_frames": T,
        "lip_raw_bytes":  lip_arr.nbytes,
        "face_raw_bytes": face_arr.nbytes if face_arr is not None else 0,
        "audio_bytes":    audio_path.stat().st_size if audio_path.exists() else 0,
    }

    # ── 压缩测试（lip）──────────────────────────────────────────────────────
    # savez_compressed
    buf = io.BytesIO()
    t0 = time.time()
    np.savez_compressed(buf, frames=lip_arr)
    result["lip_npz_write_s"]  = time.time() - t0
    result["lip_npz_bytes"]    = buf.tell()
    buf.seek(0)
    t0 = time.time()
    np.load(buf)["frames"]
    result["lip_npz_read_s"]   = time.time() - t0

    # ── 压缩测试（face，如果存在）────────────────────────────────────────────
    if face_arr is not None:
        # savez_compressed
        buf2 = io.BytesIO()
        t0 = time.time()
        np.savez_compressed(buf2, frames=face_arr)
        result["face_npz_write_s"] = time.time() - t0
        result["face_npz_bytes"]   = buf2.tell()
        buf2.seek(0)
        t0 = time.time()
        np.load(buf2)["frames"]
        result["face_npz_read_s"]  = time.time() - t0

        # JPEG 序列
        jbufs = []
        t0 = time.time()
        for f in face_arr:
            b = io.BytesIO()
            Image.fromarray(f).save(b, format="JPEG", quality=85)
            jbufs.append(b)
        result["face_jpeg_write_s"] = time.time() - t0
        result["face_jpeg_bytes"]   = sum(b.tell() for b in jbufs)
        t0 = time.time()
        for b in jbufs:
            b.seek(0)
            np.array(Image.open(b))
        result["face_jpeg_read_s"]  = time.time() - t0

        # 1 帧 JPEG
        b1 = io.BytesIO()
        t0 = time.time()
        Image.fromarray(face_arr[len(face_arr)//2]).save(b1, format="JPEG", quality=90)
        result["face_1jpg_write_s"] = time.time() - t0
        result["face_1jpg_bytes"]   = b1.tell()
    else:
        for k in ["face_npz_write_s","face_npz_bytes","face_npz_read_s",
                  "face_jpeg_write_s","face_jpeg_bytes","face_jpeg_read_s",
                  "face_1jpg_write_s","face_1jpg_bytes"]:
            result[k] = 0

    return result


def fmt_gb(b):
    return f"{b/1e9:.1f}GB"

def fmt_h(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}min"
    return f"{s/3600:.1f}h"


def main():
    print(f"采样 {SAMPLE_N} 个 clip 中...")
    clips = sample_clips(PROCESSED_DIR, SAMPLE_N)
    print(f"实际采样: {len(clips)} 个\n")

    stats = []
    for i, clip in enumerate(clips):
        if i % 20 == 0:
            print(f"  测量 {i}/{len(clips)}...")
        stats.append(measure_clip(clip))

    # ── 统计均值 ──────────────────────────────────────────────────────────────
    def avg(key):
        vals = [s[key] for s in stats if s.get(key, 0) > 0]
        return sum(vals) / len(vals) if vals else 0

    n_frames_avg    = avg("n_frames")
    has_face        = avg("face_raw_bytes") > 0

    lip_raw         = avg("lip_raw_bytes")
    face_raw        = avg("face_raw_bytes")
    lip_npz         = avg("lip_npz_bytes")
    face_npz        = avg("face_npz_bytes")
    face_jpeg       = avg("face_jpeg_bytes")
    face_1jpg       = avg("face_1jpg_bytes")
    audio           = avg("audio_bytes")

    lip_npz_write   = avg("lip_npz_write_s")
    lip_npz_read    = avg("lip_npz_read_s")
    face_npz_write  = avg("face_npz_write_s")
    face_npz_read   = avg("face_npz_read_s")
    face_jpeg_write = avg("face_jpeg_write_s")
    face_jpeg_read  = avg("face_jpeg_read_s")
    face_1jpg_write = avg("face_1jpg_write_s")

    print(f"\n{'='*60}")
    print(f"平均帧数: {n_frames_avg:.0f} 帧/clip  face数据: {'有' if has_face else '无'}")
    print(f"{'='*60}\n")

    # ── 方案定义 ──────────────────────────────────────────────────────────────
    # (名称, 每clip字节数, 预处理额外写入时间/s, 训练读取时间/s)
    # 基准：当前方案 = lip.npy(raw) + face.npy(raw)
    schemes = [
        ("当前: lip.npy + face.npy",
         lip_raw + face_raw + audio,
         0, 0),  # baseline，无额外开销
        ("lip.npz + face.npy",
         lip_npz + face_raw + audio,
         lip_npz_write, lip_npz_read),
        ("lip.npz + face.npz",
         lip_npz + face_npz + audio,
         lip_npz_write + face_npz_write,
         lip_npz_read + face_npz_read),
        ("lip.npz + face JPEG序列",
         lip_npz + face_jpeg + audio,
         lip_npz_write + face_jpeg_write,
         lip_npz_read + face_jpeg_read),
        ("lip.npz + face 1帧JPEG",
         lip_npz + face_1jpg + audio,
         lip_npz_write + face_1jpg_write,
         lip_npz_read + 0.001),
        ("lip.npz（无face）",
         lip_npz + audio,
         lip_npz_write, lip_npz_read),
        ("lip.npy（无face）",
         lip_raw + audio,
         0, 0),
    ]

    print(f"{'方案':<28} {'全量存储':>10} {'预处理+时间':>12} {'训练读取+时间/epoch':>20}")
    print("-" * 76)

    baseline_bytes, _, _ = schemes[0][1], schemes[0][2], schemes[0][3]

    for name, bytes_per_clip, extra_write, extra_read in schemes:
        total_bytes  = bytes_per_clip * TOTAL_ALL
        # 预处理额外时间（10 workers并行）
        preproc_extra = extra_write * TOTAL_ALL / 10
        # 训练读取额外时间（8 workers并行，每epoch）
        train_extra   = extra_read  * TOTAL_ALL / TRAINING_WORKERS

        saved = baseline_bytes * TOTAL_ALL - total_bytes
        saved_str = f"(-{fmt_gb(saved)})" if saved > 0 else ""

        print(f"{name:<28} {fmt_gb(total_bytes):>10} {saved_str:<12}"
              f"  +{fmt_h(preproc_extra):>8}      +{fmt_h(train_extra):>8}/epoch")

    print(f"\n注：")
    print(f"  全量 clip 数: pretrain {TOTAL_CLIPS['pretrain']:,} + "
          f"trainval {TOTAL_CLIPS['trainval']:,} + test {TOTAL_CLIPS['test']:,} = {TOTAL_ALL:,}")
    print(f"  预处理并行: 10 workers  训练 DataLoader workers: {TRAINING_WORKERS}")
    print(f"  latent.npz 未计入（已压缩，每clip约 100KB，全量约 15GB）")


if __name__ == "__main__":
    main()
