"""
extract_v5_timestamps.py — 批量对 val 1000 clips 用 V5 forced align 产生词级时间戳，
写入 {clip_dir}/streamlip_v5_timestamps.json，格式与 text.json["words"] 相同：
  [{"word": "hello", "start": 0.12, "end": 0.45}, ...]

Usage:
  uv run python scripts/extract_v5_timestamps.py \
    --clip_list configs/eval_splits/pretrain_len80_260_lipavsr_val1000_seed43.txt \
    --data_root /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from torchaudio.functional import forced_align, merge_tokens

REPO   = Path(__file__).resolve().parent.parent  # scripts/ → fm-avsr-cleanup
DL_V2A = Path("/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A")
sys.path.insert(0, str(DL_V2A / "src"))
sys.path.insert(0, str(DL_V2A / "scripts"))

SPM_PATH   = str(DL_V2A / "third_party/auto_avsr/spm/unigram/unigram5000.model")
AVSR_CKPT  = str(DL_V2A / "pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth")
V5_CKPT    = str(DL_V2A / "runs/v5/v5_olmo_lr3e-6_ep50_eos_frame400/step_004500.pt")
V5_LM_PATH = str(DL_V2A / "pretrained/olmo-1b-lrs3-lr3e-5_ep2")
FPS = 25.0


def detect_pause_splits(log_probs, blank=0, min_pause_frames=10):
    ids = log_probs.argmax(-1).cpu().tolist()
    T   = len(ids)
    segments, seg_start = [], None
    for i in range(T + 1):
        is_blank = (i == T) or (ids[i] == blank)
        if not is_blank and seg_start is None:
            seg_start = i
        elif is_blank and seg_start is not None:
            segments.append((seg_start, i - 1))
            seg_start = None
    if not segments:
        return [(0, T - 1)]
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] - 1 < min_pause_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def ctc_forced_align_with_pauses(log_probs, token_ids, blank=0, min_pause_frames=10):
    device = log_probs.device
    speech_segs = detect_pause_splits(log_probs, blank=blank,
                                      min_pause_frames=min_pause_frames)
    ids_greedy = log_probs.argmax(-1).cpu().tolist()

    seg_token_counts = []
    for s, e in speech_segs:
        collapsed, prev = [], blank
        for v in ids_greedy[s:e+1]:
            if v != blank and v != prev:
                collapsed.append(v)
            prev = v
        seg_token_counts.append(len(collapsed))

    total_greedy = sum(seg_token_counts)
    N = len(token_ids)
    if total_greedy == 0:
        alloc = [N] + [0] * (len(speech_segs) - 1)
    else:
        alloc = [round(c / total_greedy * N) for c in seg_token_counts]
        diff  = N - sum(alloc)
        if diff != 0:
            biggest = max(range(len(alloc)), key=lambda i: seg_token_counts[i])
            alloc[biggest] += diff

    all_spans, tok_offset = [], 0
    for (s, e), n_tok in zip(speech_segs, alloc):
        seg_toks = token_ids[tok_offset: tok_offset + n_tok]
        tok_offset += n_tok
        if not seg_toks:
            continue
        lp_seg = log_probs[s:e+1].unsqueeze(0).float()
        T_seg  = e - s + 1
        tgt    = torch.tensor(seg_toks, dtype=torch.int32, device=device)
        il     = torch.tensor([T_seg], dtype=torch.int32, device=device)
        tl     = torch.tensor([len(seg_toks)], dtype=torch.int32, device=device)
        try:
            al, sc = forced_align(lp_seg, tgt.unsqueeze(0), il, tl, blank=blank)
            token_spans = merge_tokens(al[0], sc[0], blank=blank)
        except Exception:
            token_spans = None
        for k in range(len(seg_toks)):
            if token_spans is not None and k < len(token_spans):
                ts = token_spans[k]
                all_spans.append((s + ts.start, s + ts.end))
            else:
                f = s + k * max(1, T_seg // len(seg_toks))
                all_spans.append((f, f + 1))
    return all_spans


def spans_to_word_timestamps(pieces, spans, fps=FPS):
    words, cur_pieces, cur_spans = [], [], []
    for piece, span in zip(pieces, spans):
        if piece.startswith("▁") and cur_pieces:
            words.append({"word":  "".join(cur_pieces).lower(),
                           "start": round(cur_spans[0][0] / fps, 3),
                           "end":   round((cur_spans[-1][1] - 1) / fps, 3)})
            cur_pieces, cur_spans = [], []
        cur_pieces.append(piece.lstrip("▁"))
        cur_spans.append(span)
    if cur_pieces:
        words.append({"word":  "".join(cur_pieces).lower(),
                       "start": round(cur_spans[0][0] / fps, 3),
                       "end":   round((cur_spans[-1][1] - 1) / fps, 3)})
    return words


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_list",    required=True)
    p.add_argument("--data_root",
                   default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed")
    p.add_argument("--output_name",  default="streamlip_v5_timestamps.json")
    p.add_argument("--beam",         type=int, default=3)
    p.add_argument("--min_pause",    type=int, default=10)
    p.add_argument("--overwrite",    action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)

    sp = spm.SentencePieceProcessor(); sp.Load(SPM_PATH)

    clips = [data_root / l.strip()
             for l in Path(args.clip_list).read_text().splitlines() if l.strip()]
    clips = [c for c in clips if (c / "avsr_enc.npy").exists()]
    if not args.overwrite:
        clips = [c for c in clips if not (c / args.output_name).exists()]
    print(f"clips to process: {len(clips)}")
    if not clips:
        print("All done."); return

    from streaminlip.v5 import StreamLipV5
    from decode_v5 import offline_decode
    from transformers import AutoTokenizer

    model = StreamLipV5(avsr_ckpt=AVSR_CKPT, smollm2_path=V5_LM_PATH,
                        lora_rank=0, cross_attn_every_n=4).to(device).bfloat16()
    ckpt  = torch.load(V5_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False); model.eval()
    tok   = AutoTokenizer.from_pretrained(V5_LM_PATH, clean_up_tokenization_spaces=False)
    print(f"V5 loaded (step {ckpt['step']})")

    for i, c in enumerate(clips):
        feat   = np.load(str(c / "avsr_enc.npy"), mmap_mode="r")
        visual = torch.from_numpy(feat.copy()).unsqueeze(0)

        with torch.no_grad():
            v5_text = offline_decode(model, visual, tok, max_new_tokens=80,
                                     rep_penalty=1.3, num_beams=args.beam,
                                     device=device, use_ctc_len=True)
            enc = model._encode(visual.to(device, dtype=model.proj.weight.dtype))
            lp  = torch.nn.functional.log_softmax(
                  model.ctc_lo(enc[0].to(model.ctc_lo.weight.dtype)).float(), dim=-1)

        text_norm = re.sub(r"[^a-z\s']", " ", v5_text.lower()).strip()
        pieces    = sp.EncodeAsPieces(text_norm.upper())
        token_ids = sp.EncodeAsIds(text_norm.upper())

        if not token_ids:
            (c / args.output_name).write_text("[]")
            continue

        try:
            spans = ctc_forced_align_with_pauses(lp, token_ids, blank=0,
                                                  min_pause_frames=args.min_pause)
            words = spans_to_word_timestamps(pieces, spans)
        except Exception as e:
            print(f"  [{i}] error: {e}")
            words = [{"word": w, "start": 0.0, "end": 0.0}
                     for w in text_norm.split()]

        # 时间戳平移：让第一个词从 0s 开始，与 GT 对齐方式一致
        if words and words[0]["start"] > 0:
            shift = words[0]["start"]
            words = [{"word": w["word"],
                       "start": round(max(0.0, w["start"] - shift), 3),
                       "end":   round(max(0.0, w["end"]   - shift), 3)}
                     for w in words]

        (c / args.output_name).write_text(json.dumps(words, ensure_ascii=False))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(clips)}", flush=True)

    print(f"Done. Written to {args.output_name}")


if __name__ == "__main__":
    main()
