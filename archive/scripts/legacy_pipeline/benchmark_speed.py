"""
StreamLip inference speed benchmark.

Measures per-chunk latency (200ms = 5 frames) with random weights,
as a proxy for real-time feasibility before training.

Usage:
    cd /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A
    .venv/bin/python scripts/benchmark_speed.py [--device cuda] [--dtype bf16]
"""
import argparse
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.streaminlip.model import StreamLip


def benchmark(device: str, dtype: torch.dtype, n_warmup: int = 20, n_runs: int = 50):
    print(f"\n{'='*60}")
    print(f"StreamLip speed benchmark (random init)")
    print(f"  device: {device}  dtype: {dtype}")
    print(f"{'='*60}")

    # Build model with random weights (no pretrained download needed)
    print("Building model...")
    model = StreamLip(random_init=True, lora_rank=16)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    counts = model.param_counts()
    print(f"  Total params:     {counts['total']/1e6:.1f}M")
    print(f"  Trainable (LoRA+adapter): {counts['trainable']/1e6:.1f}M")
    print(f"  Frozen:           {counts['frozen']/1e6:.1f}M")

    # Simulate one streaming chunk:
    #   - 5 visual frames  (200ms look-ahead window)
    #   - 4 text tokens    (~average tokens per 200ms at 120 wpm)
    B = 1
    T_vis = 5      # one chunk
    T_text = 4
    av_feats  = torch.randn(B, T_vis, 1024, device=device, dtype=dtype)
    input_ids = torch.randint(0, 262144, (B, T_text), device=device)

    print(f"\nInput shapes:  av_feats={tuple(av_feats.shape)}, input_ids={tuple(input_ids.shape)}")

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(av_feats, input_ids)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({n_runs} runs)...")
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(av_feats, input_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.9)]
    p99 = latencies[-1]
    mean = sum(latencies) / len(latencies)

    print(f"\n{'─'*40}")
    print(f"  Mean:   {mean:.1f} ms / chunk")
    print(f"  P50:    {p50:.1f} ms / chunk")
    print(f"  P90:    {p90:.1f} ms / chunk")
    print(f"  P99:    {p99:.1f} ms / chunk")
    print(f"{'─'*40}")
    print(f"  Budget: 50ms  →  {'✓ OK' if p90 < 50 else '✗ OVER BUDGET'}")
    print(f"  Output logits shape: {tuple(logits.shape)}")

    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    benchmark(args.device, dtype_map[args.dtype])
