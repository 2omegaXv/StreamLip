"""
重处理 LRS3 嘴唇裁剪（Auto-AVSR 兼容版）。

从已有 manifest 中找出尚未用新流程处理的 clip，
通过 subprocess 并行启动 reprocess_worker_avsr.py 重跑裁剪，
输出 lip.npy (T, 96, 96) uint8 灰度，覆盖旧的 (T,96,96,3) RGB 版本。

用法：
  python scripts/reprocess_avsr.py --split pretrain --workers 4
  python scripts/reprocess_avsr.py --split pretrain --workers 4 --limit 20  # 调试
"""

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

LRS3_ROOT  = Path("/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3")
OUT_ROOT   = Path("data/processed")
WORKER_PY  = Path("scripts/reprocess_worker_avsr.py")
PYTHON     = str(Path(sys.executable))


def collect_all(split: str, out_root: Path, limit: int = None) -> list:
    """从 manifest 读出全部 clip，不做文件检查（让 worker 自己跳过已完成的）。"""
    manifest = out_root / "manifest.csv"
    with open(manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if limit:
        rows = rows[:limit]

    lrs3_split = LRS3_ROOT / split / split
    jobs = []
    for r in rows:
        clip_dir = out_root / r["path"]
        speaker  = clip_dir.parts[-2]
        clip_id  = clip_dir.parts[-1]
        mp4      = lrs3_split / speaker / f"{clip_id}.mp4"
        jobs.append({"mp4": str(mp4), "out": str(clip_dir)})
    print(f"split={split}  total={len(jobs)}", flush=True)
    return jobs


def _stream_worker(job_file: str, fa_device: str, pbar, all_results: list,
                   lock, start_delay: float = 0.0):
    if start_delay > 0:
        time.sleep(start_delay)   # 错开启动，避免同时抢 GPU init 和 NFS 读模型
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",     choices=["pretrain", "trainval", "test"], required=True)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--fa_device", default=None,
                        help="默认 cuda:0，多 GPU 可传 cuda:1 等")
    parser.add_argument("--out_root",  default=str(OUT_ROOT))
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    fa_device = args.fa_device or "cuda:0"
    out_root  = Path(args.out_root)

    jobs = collect_all(args.split, out_root, args.limit)
    if not jobs:
        print("manifest 为空，退出。")
        return

    # 分批分给 workers
    with tempfile.TemporaryDirectory(dir=".") as tmpdir:
        batches = [[] for _ in range(args.workers)]
        for i, job in enumerate(jobs):
            batches[i % args.workers].append(job)

        job_files = []
        for i, batch in enumerate(batches):
            if not batch:
                continue
            jf = str(Path(tmpdir) / f"job_{i:03d}.json")
            Path(jf).write_text(json.dumps(batch))
            job_files.append(jf)

        all_results = []
        lock = threading.Lock()
        pbar = tqdm(total=len(jobs), desc="重处理", unit="clip")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [
                pool.submit(_stream_worker, jf, fa_device, pbar, all_results, lock,
                            start_delay=i * 30.0)   # 每个 worker 错开 30s 启动
                for i, jf in enumerate(job_files)
            ]
            concurrent.futures.wait(futs)
        pbar.close()

    ok   = sum(1 for r in all_results if r.get("ok") and not r.get("skipped"))
    skip = sum(1 for r in all_results if r.get("skipped"))
    fail = sum(1 for r in all_results if not r.get("ok"))
    print(f"\n完成: ok={ok}  skip={skip}  fail={fail}")


if __name__ == "__main__":
    main()
