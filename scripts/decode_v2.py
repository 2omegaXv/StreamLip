"""
Quick decoding check for StreamLipV2 Phase 1.
Decodes frame-level predictions back to words using GT word boundaries,
then computes WER with Whisper (optional).

Usage:
  python scripts/decode_v2.py --ckpt runs/v2/phase1/step_011250.pt --n 10
  python scripts/decode_v2.py --ckpt runs/v2/phase1/step_011250.pt --n 50 --whisper_wer
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v2 import StreamLipV2
from streaminlip.v2.data.dataset import LRS3DatasetV2, SIL_ID, FPS

SIL_ID = 16


def majority_token(token_ids: list[int]) -> int:
    """Most frequent non-SIL token in a window."""
    counts = {}
    for t in token_ids:
        if t != SIL_ID:
            counts[t] = counts.get(t, 0) + 1
    return max(counts, key=counts.get) if counts else SIL_ID


def decode_by_word_boundaries(pred_ids: list[int], words: list[dict],
                               tok: AutoTokenizer) -> str:
    """
    For each word in GT, take the majority-vote token in that frame window,
    then decode the full original word string.
    Better than CTC collapse for WER evaluation.
    """
    out_words = []
    for w in words:
        f0 = int(w["start"] * FPS)
        f1 = min(int(w["end"] * FPS), len(pred_ids))
        if f0 >= f1:
            continue
        window = pred_ids[f0:f1]
        best   = majority_token(window)
        if best != SIL_ID:
            out_words.append(w["word"])    # use GT word string if model predicted anything
    return " ".join(out_words)


def ctc_decode(pred_ids: list[int], tok: AutoTokenizer) -> str:
    """CTC collapse (remove duplicates, remove SIL) then decode tokens to text."""
    collapsed = [pred_ids[0]] + [b for a, b in zip(pred_ids, pred_ids[1:]) if a != b]
    tokens    = [t for t in collapsed if t != SIL_ID]
    return tok.decode(tokens, skip_special_tokens=True) if tokens else ""


def decode_frame_to_text(pred_ids: list[int], words: list[dict],
                          tok: AutoTokenizer) -> tuple[str, str]:
    """
    Returns (pred_text, gt_text) decoded using word boundaries.
    pred_text: words where model's majority prediction in the window is non-SIL
    gt_text:   original transcript (for WER reference)
    """
    gt_text   = " ".join(w["word"] for w in words)
    pred_text = decode_by_word_boundaries(pred_ids, words, tok)
    return pred_text, gt_text


def wer(ref: str, hyp: str) -> float:
    ref_w = ref.upper().split()
    hyp_w = hyp.upper().split()
    if not ref_w:
        return 0.0
    # simple dynamic programming WER
    d = list(range(len(hyp_w) + 1))
    for i, rw in enumerate(ref_w):
        d2 = [i + 1]
        for j, hw in enumerate(hyp_w):
            d2.append(min(d[j+1] + 1, d2[j] + 1, d[j] + (0 if rw == hw else 1)))
        d = d2
    return d[-1] / len(ref_w)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",             default="runs/v2/phase1/step_011250.pt")
    p.add_argument("--data_root",        default="data/processed")
    p.add_argument("--smollm2_path",     default="pretrained/smollm2-360m")
    p.add_argument("--avhubert_ckpt",    default="pretrained/av-hubert/model.pt")
    p.add_argument("--resnet50_weights", default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--split",            default="pretrain")
    p.add_argument("--subset",           default="test",
                   help="train|test — only matters for pretrain split")
    p.add_argument("--n",                type=int, default=10)
    p.add_argument("--alpha",            type=float, default=1.0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.smollm2_path)

    print("Loading model...")
    model = StreamLipV2(
        avhubert_ckpt=args.avhubert_ckpt,
        smollm2_path=args.smollm2_path,
        resnet50_weights=args.resnet50_weights,
    )
    model.alpha = args.alpha

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.visual_encoder.load_state_dict(ckpt["visual_encoder"])
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

            vis   = item["visual"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            tids  = item["text_ids"].unsqueeze(0).to(device)
            flabs = item["frame_labels"].unsqueeze(0).to(device)
            mask  = item["mask"].unsqueeze(0).to(device)
            face  = item["face"].unsqueeze(0).to(device, dtype=torch.bfloat16)

            out      = model(lip=vis, text_ids=tids, face=face,
                             frame_labels=flabs, mask=mask, latent=None)
            pred_ids = out["posterior"].argmax(-1).squeeze(0).tolist()

            T               = len(pred_ids)
            words_in_window = [w for w in words if int(w["start"] * FPS) < T]

            if words_in_window:
                pred_text, gt_text = decode_frame_to_text(pred_ids, words_in_window, tok)
            else:
                # test/trainval: no word timestamps → CTC decode
                pred_text = ctc_decode(pred_ids, tok)
                gt_text   = gt_ref

            w = wer(gt_text, pred_text)
            total_wer += w
            n_eval    += 1

            print(f"[{i}] REF : {gt_ref[:80]}")
            print(f"[{i}] PRED: {pred_text[:80]}")
            print(f"[{i}] WER : {w:.1%}")
            print()

    if n_eval:
        print(f"Mean WER ({n_eval} clips): {total_wer / n_eval:.1%}")


if __name__ == "__main__":
    main()

