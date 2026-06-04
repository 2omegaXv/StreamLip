"""
Decode test for StreamLipV3.

Same evaluation logic as decode_v2.py; uses GT word timestamps for per-word
majority-vote decoding. Model forward() is teacher-forced on GT clean_ids.

Usage:
  python scripts/decode_v3.py --ckpt runs/v3/phase1/step_006000.pt --n 20
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v3 import StreamLipV3
from streaminlip.v2.data.dataset import LRS3DatasetV2, SIL_ID, FPS

# ── shared decode utilities (identical to decode_v2.py) ──────────────────────

def majority_token(token_ids: list[int]) -> int:
    counts = {}
    for t in token_ids:
        if t != SIL_ID:
            counts[t] = counts.get(t, 0) + 1
    return max(counts, key=counts.get) if counts else SIL_ID


def decode_by_word_boundaries(pred_ids: list[int], words: list[dict],
                               tok: AutoTokenizer) -> str:
    out_parts = []
    for w in words:
        f0 = int(w["start"] * FPS)
        f1 = min(int(w["end"] * FPS), len(pred_ids))
        if f0 >= f1:
            continue
        best = majority_token(pred_ids[f0:f1])
        if best != SIL_ID:
            predicted = tok.decode([best], skip_special_tokens=True).strip()
            if predicted:
                out_parts.append(predicted)
    return " ".join(out_parts)


def ctc_decode(pred_ids: list[int], tok: AutoTokenizer) -> str:
    collapsed = [pred_ids[0]] + [b for a, b in zip(pred_ids, pred_ids[1:]) if a != b]
    tokens    = [t for t in collapsed if t != SIL_ID]
    return tok.decode(tokens, skip_special_tokens=True) if tokens else ""


def wer(ref: str, hyp: str) -> float:
    ref_w = ref.upper().split()
    hyp_w = hyp.upper().split()
    if not ref_w:
        return 0.0
    d = list(range(len(hyp_w) + 1))
    for i, rw in enumerate(ref_w):
        d2 = [i + 1]
        for j, hw in enumerate(hyp_w):
            d2.append(min(d[j+1] + 1, d2[j] + 1, d[j] + (0 if rw == hw else 1)))
        d = d2
    return d[-1] / len(ref_w)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",               default="runs/v3/phase1/step_006000.pt")
    p.add_argument("--data_root",          default="data/processed")
    p.add_argument("--smollm2_path",       default="pretrained/smollm2-360m")
    p.add_argument("--avhubert_ckpt",      default="pretrained/av-hubert/model.pt")
    p.add_argument("--resnet50_weights",   default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--split",              default="pretrain")
    p.add_argument("--subset",             default="test")
    p.add_argument("--n",                  type=int, default=20)
    p.add_argument("--cross_attn_every_n", type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.smollm2_path)

    print("Loading model...")
    model = StreamLipV3(
        avhubert_ckpt=args.avhubert_ckpt,
        smollm2_path=args.smollm2_path,
        resnet50_weights=args.resnet50_weights,
        cross_attn_every_n=args.cross_attn_every_n,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.visual_encoder.load_state_dict(ckpt["visual_encoder"])
    model.sil_head.load_state_dict(ckpt["sil_head"])
    model.lm.load_state_dict(ckpt["lm"])
    model = model.to(device).bfloat16().eval()
    print(f"Loaded: {args.ckpt}\n")

    ds = LRS3DatasetV2(args.data_root, args.split, args.smollm2_path,
                       subset=args.subset, deterministic=True)

    total_wer, n_eval = 0.0, 0
    with torch.no_grad():
        for i in range(min(args.n, len(ds))):
            item     = ds[i]
            clip_dir = ds.clips[i]
            meta     = json.loads((clip_dir / "text.json").read_text())
            gt_ref   = meta["transcript"].strip()
            words    = meta.get("words", [])
            if not gt_ref:
                continue

            vis         = item["visual"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            clean_ids   = item["clean_ids"].unsqueeze(0).to(device)
            clean_mask  = torch.ones(1, clean_ids.shape[1], dtype=torch.long, device=device)
            lm_idx_text = item["lm_idx_text"].unsqueeze(0).to(device)
            lm_idx_fm   = item["lm_idx_fm"].unsqueeze(0).to(device)
            flabs       = item["frame_labels"].unsqueeze(0).to(device)
            mask        = item["mask"].unsqueeze(0).to(device)
            face        = item["face"].unsqueeze(0).to(device, dtype=torch.bfloat16)

            out      = model(visual=vis, clean_ids=clean_ids, clean_mask=clean_mask,
                             lm_idx_text=lm_idx_text, lm_idx_fm=lm_idx_fm,
                             face=face, frame_labels=flabs, mask=mask, latent=None)
            pred_ids = out["posterior"].argmax(-1).squeeze(0).tolist()

            T    = len(pred_ids)
            ws   = [w for w in words if int(w["start"] * FPS) < T]
            pred_text = decode_by_word_boundaries(pred_ids, ws, tok) if ws \
                        else ctc_decode(pred_ids, tok)
            gt_text   = gt_ref if not ws else " ".join(w["word"] for w in ws)

            w_score = wer(gt_text, pred_text)
            total_wer += w_score
            n_eval    += 1

            print(f"[{i}] REF : {gt_ref[:80]}")
            print(f"[{i}] PRED: {pred_text[:80]}")
            print(f"[{i}] WER : {w_score:.1%}")
            print()

    if n_eval:
        print(f"Mean WER ({n_eval} clips): {total_wer / n_eval:.1%}")


if __name__ == "__main__":
    main()
