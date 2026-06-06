"""
Pre-extract SmolLM2 token-level hidden states for all clips.

Saves: {clip_dir}/smollm2_h.npy  shape (L, 960) float16
  where L = number of LM tokens (incl. BOS)

Training loads this and resamples to T_a frames at runtime,
eliminating SmolLM2 from the training loop entirely.

Usage:
  python scripts/extract_smollm2_h.py --batch_size 256
"""
import argparse, csv, os, sys
from pathlib import Path

_TMPDIR = Path(__file__).parent.parent / ".tmp"
_TMPDIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_TMPDIR)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.fm_avsr_dataset import read_clip_text, smollm2_hidden_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default="data/processed")
    p.add_argument("--smollm2_path", default="pretrained/smollm2-360m")
    p.add_argument("--split",        default="pretrain")
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--clip_list",    default=None,
                   help="Optional file with one processed clip path per line.")
    p.add_argument("--text_source",  choices=["avsr", "text_json", "lipavsr", "v5"], default="avsr")
    p.add_argument("--overwrite",    action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root   = Path(args.data_root)

    print("Loading SmolLM2...")
    tok = AutoTokenizer.from_pretrained(args.smollm2_path)
    lm  = AutoModelForCausalLM.from_pretrained(
        args.smollm2_path, dtype=torch.bfloat16).to(device).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    bos = lm.config.bos_token_id or 0
    print("SmolLM2 loaded.")

    # collect clips that have text inputs
    if args.clip_list:
        clips = [
            root / line.strip()
            for line in Path(args.clip_list).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        cache = root / f"_fm_avsr_{args.split}.txt"
        if cache.exists() and args.text_source == "avsr":
            clips = [root / p for p in cache.read_text().split()]
        else:
            clips = [root / r["path"]
                     for r in csv.DictReader(open(root / "manifest.csv"))
                     if r["split"] == args.split
                     and (root / r["path"] / "text.json").exists()]

    if not args.overwrite:
        clips = [c for c in clips if not smollm2_hidden_path(c, args.text_source).exists()]
    print(f"Clips to process: {len(clips)}")

    done = skip = err = 0
    for i in tqdm(range(0, len(clips), args.batch_size), desc="extract_smollm2"):
        batch_clips = clips[i : i + args.batch_size]

        # Tokenize
        all_tokens = []
        kept_clips = []
        for c in batch_clips:
            txt = read_clip_text(c, args.text_source)
            if not txt:
                skip += 1
                continue
            tokens = [bos]
            for w in txt.upper().split():
                tokens += tok.encode(" " + w.lower(), add_special_tokens=False)
            all_tokens.append(tokens)
            kept_clips.append(c)
        if not all_tokens:
            continue

        max_L = max(len(t) for t in all_tokens)
        B     = len(kept_clips)
        ids   = torch.zeros(B, max_L, dtype=torch.long,  device=device)
        amask = torch.zeros(B, max_L, dtype=torch.long,  device=device)
        for b, tokens in enumerate(all_tokens):
            L = len(tokens)
            ids[b, :L]   = torch.tensor(tokens, dtype=torch.long, device=device)
            amask[b, :L] = 1

        try:
            with torch.no_grad():
                h = lm(input_ids=ids, attention_mask=amask,
                       output_hidden_states=True).hidden_states[-1]  # (B, max_L, 960)
        except torch.cuda.OutOfMemoryError:
            # fallback: smaller sub-batches
            torch.cuda.empty_cache()
            sub = max(1, args.batch_size // 4)
            for j in range(0, B, sub):
                sub_clips  = kept_clips[j : j + sub]
                sub_tokens = all_tokens[j : j + sub]
                sub_L = max(len(t) for t in sub_tokens)
                bs    = len(sub_clips)
                sub_ids   = torch.zeros(bs, sub_L, dtype=torch.long,  device=device)
                sub_amask = torch.zeros(bs, sub_L, dtype=torch.long,  device=device)
                for b, tokens in enumerate(sub_tokens):
                    L = len(tokens)
                    sub_ids[b, :L]   = torch.tensor(tokens, dtype=torch.long, device=device)
                    sub_amask[b, :L] = 1
                with torch.no_grad():
                    sub_h = lm(input_ids=sub_ids, attention_mask=sub_amask,
                                output_hidden_states=True).hidden_states[-1]
                for k, (c, tokens) in enumerate(zip(sub_clips, sub_tokens)):
                    L   = len(tokens)
                    out = sub_h[k, :L].float().cpu().numpy().astype(np.float16)
                    np.save(str(smollm2_hidden_path(c, args.text_source)), out)
                    done += 1
            continue

        for b, (c, tokens) in enumerate(zip(kept_clips, all_tokens)):
            L   = len(tokens)
            out = h[b, :L].float().cpu().numpy().astype(np.float16)  # (L, 960)
            np.save(str(smollm2_hidden_path(c, args.text_source)), out)
            done += 1

    print(f"Done: {done}  Skip: {skip}  Err: {err}")


if __name__ == "__main__":
    main()
