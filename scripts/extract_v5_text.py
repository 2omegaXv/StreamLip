"""
extract_v5_text.py — 批量对指定 clip list 跑 StreamLip V5 推理，
每个 clip 写入 {clip_dir}/streamlip_v5_text.txt。

Usage:
  uv run python scripts/extract_v5_text.py \
    --clip_list configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt \
    --v5_ckpt ckpt/v5/streamlip_v5_olmo_step_002000.pt \
    --v5_lm_path ckpt/streamlip-v5-lm \
    --data_root data/processed
"""
import argparse, sys, re
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

AVSR_CKPT = str(REPO_ROOT / "ckpt/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_list",   required=True,
                   help="一行一条相对路径（相对于 data_root），如 val1000 list")
    p.add_argument("--data_root", default=str(REPO_ROOT / "data/processed"))
    p.add_argument("--v5_ckpt",     required=True)
    p.add_argument("--v5_lm_path", default=str(REPO_ROOT / "ckpt/streamlip-v5-lm"))
    p.add_argument("--avsr_ckpt",   default=AVSR_CKPT)
    p.add_argument("--cross_attn_every_n", type=int, default=4)
    p.add_argument("--beam",        type=int, default=3)
    p.add_argument("--input_name", default="avsr_enc.npy",
                   help="Per-clip visual latent file, e.g. avsr_enc.npy or avsr_enc_lipavsr.npy")
    p.add_argument("--output_name", default="streamlip_v5_text.txt",
                   help="每个 clip 目录下写入的文件名")
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)

    clips = [data_root / l.strip()
             for l in Path(args.clip_list).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    clips = [c for c in clips if (c / args.input_name).exists()]
    if not args.overwrite:
        clips = [c for c in clips if not (c / args.output_name).exists()]
    print(f"clips to process: {len(clips)}")
    if not clips:
        print("All done.")
        return

    from streaminlip.v5 import StreamLipV5
    from decode_v5 import offline_decode
    from transformers import AutoTokenizer

    model = StreamLipV5(
        avsr_ckpt=args.avsr_ckpt,
        smollm2_path=args.v5_lm_path,
        lora_rank=0,
        cross_attn_every_n=args.cross_attn_every_n,
    ).to(device).bfloat16()
    ckpt = torch.load(args.v5_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.v5_lm_path,
                                        clean_up_tokenization_spaces=False)
    print(f"Loaded V5 step {ckpt['step']}")

    for i, c in enumerate(clips):
        feat   = np.load(str(c / args.input_name), mmap_mode="r")
        visual = torch.from_numpy(feat.copy()).unsqueeze(0)
        with torch.no_grad():
            hyp = offline_decode(model, visual, tok,
                                 max_new_tokens=80, rep_penalty=1.3,
                                 num_beams=args.beam, device=device,
                                 use_ctc_len=True)
        text = normalize(hyp)
        (c / args.output_name).write_text(text)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(clips)}", flush=True)

    print(f"Done. Written to {args.output_name}")


if __name__ == "__main__":
    main()
