"""收集 LRS3 pretrain+trainval 转写文本，写入 data/processed/lrs3_text.txt，每行一条。
不包含 test split，避免评估时数据泄露。"""
import csv, json
from pathlib import Path
from tqdm import tqdm

DATA_ROOT  = Path("data/processed")
USE_SPLITS = {"pretrain", "trainval"}

lines = []
with open(DATA_ROOT / "manifest.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] in USE_SPLITS]

for r in tqdm(rows, desc="reading text.json"):
    p = DATA_ROOT / r["path"] / "text.json"
    if not p.exists():
        continue
    try:
        t = json.loads(p.read_text()).get("transcript", "").strip().lower()
        if t:
            lines.append(t)
    except Exception:
        pass

out = DATA_ROOT / "lrs3_text.txt"
out.write_text("\n".join(lines))
print(f"{len(lines)} clips ({'+'.join(USE_SPLITS)}) → {out}  ({out.stat().st_size/1e6:.1f} MB)")

