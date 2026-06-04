"""
Decode / WER evaluation for StreamLipOffline.

Uses greedy generation (model.generate_text) on the full clip.

Usage:
  python scripts/decode_offline.py --ckpt runs/offline/phase1/step_001000.pt --n 20
"""
import argparse, json, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.offline import StreamLipOffline
from streaminlip.offline.dataset import LRS3DatasetOffline


def wer(ref: str, hyp: str) -> float:
    r, h = ref.upper().split(), hyp.upper().split()
    if not r: return 0.0
    d = list(range(len(h) + 1))
    for rw in r:
        d2 = [d[0] + 1]
        for j, hw in enumerate(h):
            d2.append(min(d[j+1]+1, d2[j]+1, d[j]+(0 if rw==hw else 1)))
        d = d2
    return d[-1] / len(r)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          default="runs/offline/phase1/step_001000.pt")
    p.add_argument("--data_root",     default="data/processed")
    p.add_argument("--avhubert_ckpt", default="pretrained/av-hubert/model.pt")
    p.add_argument("--gemma_path",    default="pretrained/gemma-3-1b")
    p.add_argument("--lora_rank",     type=int, default=16)
    p.add_argument("--split",         default="pretrain")
    p.add_argument("--subset",        default="test")
    p.add_argument("--n",             type=int, default=20)
    p.add_argument("--max_frames",    type=int, default=150)
    p.add_argument("--max_new_tokens",type=int, default=64)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.gemma_path)

    print("Loading model...")
    model = StreamLipOffline(args.avhubert_ckpt, args.gemma_path,
                             lora_rank=args.lora_rank)
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.cross_attn_layers.load_state_dict(ckpt["cross_attn_layers"])
    if "lm_lora" in ckpt:
        model.lm.load_state_dict(ckpt["lm_lora"], strict=False)
    model = model.to(device).bfloat16().eval()
    print(f"Loaded: {args.ckpt}\n")

    ds = LRS3DatasetOffline(
        args.data_root, args.split, args.gemma_path,
        max_frames=args.max_frames, deterministic=True, subset=args.subset,
    )

    total_wer, n_eval = 0.0, 0
    with torch.no_grad():
        for i in range(min(args.n, len(ds))):
            item     = ds[i]
            clip_dir = ds.clips[i]
            meta     = json.loads((clip_dir / "text.json").read_text())
            gt_ref   = meta.get("transcript", "").strip().lower()
            if not gt_ref:
                continue

            vis  = item["visual"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            ids  = model.generate_text(vis, max_new_tokens=args.max_new_tokens)
            pred = tokenizer.decode(ids[0], skip_special_tokens=True).strip()

            w = wer(gt_ref, pred)
            total_wer += w
            n_eval    += 1

            print(f"[{i}] REF : {gt_ref[:80]}")
            print(f"[{i}] PRED: {pred[:80]}")
            print(f"[{i}] WER : {w:.1%}")
            print()

    if n_eval:
        print(f"Mean WER ({n_eval} clips): {total_wer / n_eval:.1%}")


if __name__ == "__main__":
    main()
