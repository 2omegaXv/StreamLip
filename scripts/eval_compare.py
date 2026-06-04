"""
比较 StreamLipV5 与 Auto-AVSR 在相同 test clips 上的 WER。

Usage:
  python scripts/eval_compare.py --ckpt runs/v5/.../step_007000.pt --n_clips 50
  python scripts/eval_compare.py --ckpt runs/v5/.../step_007000.pt --n_clips 200 --beam 40
"""
import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np
import torch
import jiwer

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))  # 让 scripts/ 下的模块可 import

DATA_ROOT = Path("data/processed")
AVSR_CKPT = "pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def calc_wer(refs, hyps):
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return jiwer.wer(refs, hyps,
                     reference_transform=transform,
                     hypothesis_transform=transform)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--smollm2_path",  default="pretrained/olmo-1b-lrs3-ep2")
    p.add_argument("--cross_attn_every_n", type=int, default=4)
    p.add_argument("--split",         default="test")
    p.add_argument("--n_clips",       type=int, default=50)
    p.add_argument("--beam",          type=int, default=10)
    p.add_argument("--show",          type=int, default=10)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--no_ctc_len",    action="store_true",
                   help="禁用 CTC 估长度，回退到语速粗估（用于对比实验）")
    return p.parse_args()


def main():
    args   = parse_args()
    import random
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 找 clips（有 avsr_enc.npy 且有 lip_avsr.npy） ────────────────────────
    cache_file = DATA_ROOT / f"clip_cache_{args.split}.txt"
    if cache_file.exists() and cache_file.stat().st_size > 0:
        all_clips = [Path(p) for p in cache_file.read_text().splitlines() if p]
    else:
        with open(DATA_ROOT / "manifest.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
        all_clips = [DATA_ROOT / r["path"] for r in rows]

    # 过滤有效 clip，随机打散后取 n_clips
    valid_clips = [c for c in all_clips
                   if (c / "avsr_enc.npy").exists() and (c / "lip_avsr.npy").exists()]
    random.shuffle(valid_clips)
    clips = valid_clips[:args.n_clips]

    print(f"clips: {len(clips)}  split={args.split}  beam={args.beam}")
    if not clips:
        print("No clips found. Exiting.")
        return

    # ── GT ───────────────────────────────────────────────────────────────────
    refs = []
    for c in clips:
        meta = json.loads((c / "text.json").read_text())
        words = meta.get("words", [])
        gt = " ".join(w["word"].lower() for w in words) if words \
             else meta.get("transcript", "").strip().lower()
        refs.append(normalize(gt))

    # ── Auto-AVSR ─────────────────────────────────────────────────────────────
    print(f"\n[1/2] Running Auto-AVSR (beam={args.beam})...")
    from streaminlip.auto_avsr import AutoAVSRInferencer
    asr = AutoAVSRInferencer(AVSR_CKPT, device=device, beam_size=args.beam)
    asr.eval()

    hyps_avsr = []
    for i, c in enumerate(clips):
        lip = np.load(str(c / "lip_avsr.npy"))           # (T, 96, 96) uint8
        rgb = np.stack([lip, lip, lip], axis=-1)          # (T, 96, 96, 3)
        with torch.no_grad():
            hyp = asr.infer(torch.from_numpy(rgb))
        hyps_avsr.append(normalize(hyp))
        if (i + 1) % 10 == 0:
            print(f"  avsr {i+1}/{len(clips)}", flush=True)

    wer_avsr = calc_wer(refs, hyps_avsr)

    # ── StreamLipV5 ───────────────────────────────────────────────────────────
    print(f"\n[2/2] Running StreamLipV5 (beam={args.beam})...")
    from streaminlip.v5 import StreamLipV5
    from decode_v5 import offline_decode
    from transformers import AutoTokenizer

    model = StreamLipV5(
        avsr_ckpt=AVSR_CKPT,
        smollm2_path=args.smollm2_path,
        lora_rank=0,
        cross_attn_every_n=args.cross_attn_every_n,
    ).to(device).bfloat16()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.smollm2_path,
                                        clean_up_tokenization_spaces=False)
    print(f"  Loaded step {ckpt['step']}")

    hyps_v5 = []
    for i, c in enumerate(clips):
        feat   = np.load(str(c / "avsr_enc.npy"), mmap_mode="r")
        visual = torch.from_numpy(feat.copy()).unsqueeze(0)
        with torch.no_grad():
            hyp = offline_decode(model, visual, tok,
                                 max_new_tokens=80,
                                 rep_penalty=1.3,
                                 num_beams=args.beam,
                                 device=device,
                                 use_ctc_len=not args.no_ctc_len)
        hyps_v5.append(normalize(hyp))
        if (i + 1) % 10 == 0:
            print(f"  v5 {i+1}/{len(clips)}", flush=True)

    wer_v5 = calc_wer(refs, hyps_v5)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    v5_label = "StreamLipV5" if args.no_ctc_len else "StreamLipV5+CTC"
    print(f"{'':30s}  {'Auto-AVSR':>12}  {v5_label:>15}")
    print(f"{'─'*60}")
    print(f"{'WER':30s}  {wer_avsr*100:>11.1f}%  {wer_v5*100:>14.1f}%")
    print(f"{'Word Acc':30s}  {(1-wer_avsr)*100:>11.1f}%  {(1-wer_v5)*100:>14.1f}%")
    print(f"{'─'*60}")
    print(f"clips={len(clips)}  beam={args.beam}  step={ckpt['step']}")

    if args.show > 0:
        print(f"\n{'─'*60}")
        print("Sample comparisons:")
        for i in range(min(args.show, len(clips))):
            print(f"\nGT:    {refs[i][:80]}")
            print(f"AVSR:  {hyps_avsr[i][:80]}")
            print(f"V5:    {hyps_v5[i][:80]}")


if __name__ == "__main__":
    main()
