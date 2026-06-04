"""
DataLoader 测试脚本。

测试内容：
  1. 单样本加载（形状、dtype、值域）
  2. DataLoader 多 worker 批量加载
  3. 加载速度 benchmark
  4. 可选：带 tokenizer 的帧级标签

用法：
  .venv/bin/python3 scripts/test_dataloader.py
  .venv/bin/python3 scripts/test_dataloader.py --split test --workers 4 --batch_size 8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import LRS3Dataset, collate_variable_length


def fmt(x): return f"{x:.3f}"


def test_single_sample(args):
    print("=" * 60)
    print("1. 单样本加载测试")
    print("=" * 60)

    ds = LRS3Dataset(
        processed_root=args.processed_root,
        split=args.split,
        tokenizer=None,
        clip_mode="window",
        window_frames=150,
        load_face=True,
    )
    print(f"  Dataset 大小: {len(ds)} clips")

    t0 = time.time()
    sample = ds[0]
    elapsed = time.time() - t0

    print(f"  加载耗时: {elapsed:.3f}s")
    print()

    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<14} shape={str(tuple(v.shape)):<22} dtype={v.dtype}")
        else:
            print(f"  {k:<14} = {v}")

    # 值域检查
    lip = sample["lip"]
    print()
    print("  值域检查（归一化后）：")
    print(f"    lip   min={lip.min():.2f}  max={lip.max():.2f}  mean={lip.mean():.3f}")
    if "face" in sample:
        face = sample["face"]
        print(f"    face  min={face.min():.2f}  max={face.max():.2f}  mean={face.mean():.3f}")

    # 对齐检查
    T   = sample["lip"].shape[0]
    T_a = sample["latent"].shape[0]
    assert T_a == T // 2, f"T_a={T_a} ≠ T//2={T//2}"
    print(f"\n  对齐检查：T={T}, T_a={T_a}（T_a == T//2 ✓）")
    print()


def test_dataloader_speed(args):
    print("=" * 60)
    print("2. DataLoader 速度测试")
    print("=" * 60)

    ds = LRS3Dataset(
        processed_root=args.processed_root,
        split=args.split,
        tokenizer=None,
        clip_mode="window",
        window_frames=150,
        load_face=True,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    print(f"  batch_size={args.batch_size}  num_workers={args.workers}")
    print(f"  共 {len(loader)} 个 batch，测前 {args.n_batches} 个")
    print()

    # 热身
    it = iter(loader)
    _ = next(it)
    print("  热身完成，开始计时...")

    t0 = time.time()
    for i in range(args.n_batches):
        batch = next(it)

    elapsed = time.time() - t0
    clips_per_sec = args.n_batches * args.batch_size / elapsed

    b = batch
    print(f"  {args.n_batches} 个 batch 耗时: {elapsed:.2f}s")
    print(f"  吞吐量: {clips_per_sec:.1f} clips/s")
    print()
    print("  最后一个 batch 形状：")
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:<14} {tuple(v.shape)}")
    print()


def test_full_clip_mode(args):
    print("=" * 60)
    print("3. full clip 模式（可变长度，自定义 collate）")
    print("=" * 60)

    ds = LRS3Dataset(
        processed_root=args.processed_root,
        split=args.split,
        clip_mode="full",
        load_face=False,   # 全 clip 模式跳过 face，加快测试
    )

    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        collate_fn=collate_variable_length,
        shuffle=False,
    )

    t0 = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - t0

    print(f"  加载耗时: {elapsed:.2f}s")
    print(f"  lip    shape: {tuple(batch['lip'].shape)}")
    print(f"  latent shape: {tuple(batch['latent'].shape)}")
    print(f"  mask   shape: {tuple(batch['mask'].shape)}")
    print(f"  mask   有效帧比例: {batch['mask'].float().mean():.2f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", default="data/processed")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_batches",  type=int, default=20)
    args = parser.parse_args()

    test_single_sample(args)
    test_dataloader_speed(args)
    test_full_clip_mode(args)

    print("全部测试通过 ✓")


if __name__ == "__main__":
    main()
