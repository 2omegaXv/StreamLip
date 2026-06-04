"""
V5 streaming greedy-decode evaluation.

Streaming protocol (consistent with training):
  For each chunk (CHUNK_SIZE=6 frames, 240ms):
    1. AV-HuBERT re-encodes ALL frames seen so far (growing prefix)
    2. SIL head decides whether this chunk contains speech
    3. If speech: LM predicts ONE new token given current vis_feats + text history
  This matches training: each token is predicted with only the video context
  available at that chunk step.

Position IDs (cross-attn mode): natural [0..L-1] for text tokens.
"""
import argparse, json, re, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CHUNK_SIZE = 6   # 240ms @ 25fps, matches training

from streaminlip.v5 import StreamLipV5
from streaminlip.v5.data.dataset import LRS3DatasetV5, FPS
from transformers import AutoTokenizer
import numpy as np
import jiwer


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def streaming_decode(
    model:          StreamLipV5,
    visual:         torch.Tensor,
    tokenizer,
    max_new_tokens: int   = 80,
    device:         str   = "cuda",
    rep_penalty:    float = 1.3,
    sil_threshold:  float = 0.5,   # 保留参数兼容旧调用，已不使用
    use_ctc_len:    bool  = True,
) -> str:
    """每 CHUNK_SIZE 帧预测一个 token，streaming 模式。"""
    model.eval()
    dtype    = model.proj.weight.dtype
    T_full   = visual.shape[1]
    eos_id   = tokenizer.eos_token_id or -1
    bos_id   = tokenizer.bos_token_id or eos_id
    gen_ids  = [bos_id]
    seen_toks = {}

    # CTC 估算本段应生成的 token 数上界
    if use_ctc_len and hasattr(model, "ctc_lo"):
        feat_full = visual.to(device, dtype=dtype)
        max_new_tokens = model.ctc_len_estimate(feat_full)


    for chunk_end in range(CHUNK_SIZE, T_full + 1, CHUNK_SIZE):
        if len(gen_ids) - 1 >= max_new_tokens:
            break

        vis_prefix = visual[:, :chunk_end, :].to(device, dtype=dtype)
        feat = model._encode(vis_prefix)

        if model.cross_attn_mode:
            model._vis_feats_buf = [
                (buf.detach().to(dtype) if buf is not None else None)
                for buf in model._layer_bufs
            ]
            model._layer_bufs = [None] * len(model._layer_bufs)

        L         = len(gen_ids)
        input_ids = torch.tensor([gen_ids], dtype=torch.long, device=device)
        text_emb  = model._embed_tokens(input_ids)
        attn_mask = torch.ones(1, L, dtype=torch.long, device=device)

        if model.cross_attn_mode:
            try:
                out = model.lm(inputs_embeds=text_emb, attention_mask=attn_mask)
            finally:
                model._vis_feats_buf = None
        else:
            vis_tokens = model.proj(feat)
            T_cur      = vis_tokens.shape[1]
            video_pos  = torch.arange(T_cur, device=device).unsqueeze(0)
            text_pos   = torch.arange(T_cur, T_cur + L, device=device).unsqueeze(0)
            seq        = torch.cat([vis_tokens, text_emb], dim=1)
            pos_ids    = torch.cat([video_pos, text_pos],  dim=1)
            vis_mask   = torch.ones(1, T_cur, dtype=torch.long, device=device)
            full_mask  = torch.cat([vis_mask, attn_mask], dim=1)
            out        = model.lm(inputs_embeds=seq, position_ids=pos_ids,
                                  attention_mask=full_mask)

        logits = out.logits[0, -1, :].float()
        for tok_id, cnt in seen_toks.items():
            logits[tok_id] /= (rep_penalty ** cnt)

        next_id = int(logits.argmax())
        if next_id == eos_id:
            break

        gen_ids.append(next_id)
        seen_toks[next_id] = seen_toks.get(next_id, 0) + 1

    decoded = tokenizer.decode(gen_ids[1:], skip_special_tokens=True)
    return decoded.strip()


# keep old greedy_decode for backward compat / offline ablation
@torch.no_grad()
def greedy_decode(model, visual, tokenizer, max_new_tokens=80, device="cuda",
                  rep_penalty=1.3, cfg_scale=1.0):
    """Offline decode (full video at once). Use streaming_decode for eval."""
    return streaming_decode(model, visual, tokenizer, max_new_tokens, device, rep_penalty)


@torch.no_grad()
def offline_decode(model, visual, tokenizer, max_new_tokens=80, device="cuda",
                   rep_penalty=1.3, num_beams=10, length_penalty=0.6,
                   use_ctc_len=True):
    """
    完全对标 Auto-AVSR 推理方式：
      - Beam search（默认 beam=10，对标 Auto-AVSR beam=40）
      - Cross-attn 模式: encode 填充 _layer_bufs，CA hooks 在整个生成过程持续注入
      - Prefix 模式: visual tokens 拼接后 beam search
    """
    model.eval()
    dtype  = model.proj.weight.dtype
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id

    feat = model._encode(visual.to(device, dtype=dtype))  # 填充 _layer_bufs
    T    = feat.shape[1]
    # CTC 估算优先，否则按语速粗估（~3 words/sec，25fps，1.5 tokens/word）
    if use_ctc_len and hasattr(model, "ctc_lo"):
        max_toks = model.ctc_len_estimate(feat)
    else:
        max_toks = min(max_new_tokens, max(8, int(T / 25 * 3 * 1.5)))

    if model.cross_attn_mode:
        # CA 模式：_vis_feats_buf 在整个 generate 过程中保持有效
        def _safe_p(buf):
            if buf is None: return None
            while isinstance(buf, (tuple, list)): buf = buf[0]
            return buf.to(dtype) if isinstance(buf, torch.Tensor) else None
        model._vis_feats_buf = [_safe_p(b) for b in model._layer_bufs]
        # BOS embedding 作为起点
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        try:
            out = model.lm.generate(
                input_ids=bos,
                max_new_tokens=max_toks,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=rep_penalty,
                no_repeat_ngram_size=4,
                early_stopping=True,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )
        finally:
            model._vis_feats_buf = None
        # 去掉 BOS
        gen_ids = out[0][1:].tolist()
    else:
        # Prefix 模式：prepend visual tokens，beam search
        vis_tokens = model.proj(feat)          # (1, T, lm_dim)
        bos_emb    = model._embed_tokens(
            torch.tensor([[bos_id]], dtype=torch.long, device=device))
        inputs_emb = torch.cat([vis_tokens, bos_emb], dim=1)  # (1, T+1, lm_dim)
        out = model.lm.generate(
            inputs_embeds=inputs_emb,
            max_new_tokens=max_toks,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=rep_penalty,
            no_repeat_ngram_size=4,
            early_stopping=True,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
        gen_ids = out[0].tolist()

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",               required=True)
    p.add_argument("--data_root",          default="data/processed")
    p.add_argument("--smollm2_path",       default="pretrained/smollm2-360m")
    p.add_argument("--avhubert_ckpt", default="pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth")
    p.add_argument("--cross_attn_every_n", type=int, default=0)
    p.add_argument("--split",              default="pretrain")
    p.add_argument("--n_clips",            type=int, default=500)
    p.add_argument("--max_new_tokens",     type=int,   default=80)
    p.add_argument("--cfg_scale",          type=float, default=2.0)
    p.add_argument("--rep_penalty",        type=float, default=1.3)
    p.add_argument("--show",               type=int, default=20)
    p.add_argument("--offline",            action="store_true",
                   help="全段 encode 一次再 beam search 生成（对标 Auto-AVSR 推理）")
    p.add_argument("--num_beams",          type=int, default=10,
                   help="beam search 宽度（Auto-AVSR 用 40，默认 10 速度更快）")
    p.add_argument("--limit_ds",           type=int, default=5000,
                   help="dataset 初始化时最多扫多少条（避免 NFS 慢扫描）")
    p.add_argument("--prefix_eval",        action="store_true",
                   help="截断评测：只评估 PRED 前 min(len(GT), len(PRED)) 个词，消除重复惩罚")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── model ─────────────────────────────────────────────────────────────────
    model = StreamLipV5(
        avsr_ckpt=args.avhubert_ckpt,
        smollm2_path=args.smollm2_path,
        lora_rank=0,
        cross_attn_every_n=args.cross_attn_every_n,
    ).to(device).bfloat16()
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded step {ckpt['step']}")

    tok = AutoTokenizer.from_pretrained(args.smollm2_path)

    # ── val clips ─────────────────────────────────────────────────────────────
    ds = LRS3DatasetV5(args.data_root, args.split, args.smollm2_path,
                       max_frames=150, subset="train", limit=args.limit_ds,
                       deterministic=True)
    if args.split == "pretrain":
        # pretrain: 取末尾 val subset（与训练时 random_split 一致）
        from torch.utils.data import random_split
        val_n   = min(args.n_clips, max(1, len(ds) // 10))
        train_n = len(ds) - val_n
        _, val_ds = random_split(ds, [train_n, val_n],
                                 generator=torch.Generator().manual_seed(42))
        clips = [ds.clips[i] for i in val_ds.indices[:args.n_clips]]
    else:
        # test/trainval: 直接用所有 clips
        clips = ds.clips[:args.n_clips]
    print(f"Decoding {len(clips)} clips …")

    hyps, refs = [], []
    shown = 0

    for clip_dir in clips:
        # 优先加载新预处理特征 avsr_enc.npy (768)，兼容旧 avhubert_pre.npy (1024)
        for fname in ["avsr_enc.npy", "avhubert_pre.npy"]:
            pre_path = clip_dir / fname
            if pre_path.exists():
                break
        else:
            continue
        meta  = json.loads((clip_dir / "text.json").read_text())
        words = meta.get("words", [])
        if words:
            gt_text = " ".join(w["word"].lower() for w in words)
        else:
            gt_text = meta.get("transcript", "").strip().lower()
        if not gt_text:
            continue

        feat = np.load(str(pre_path), mmap_mode="r")
        T    = len(feat)
        if T < 6:
            continue
        visual = torch.from_numpy(feat.copy()).unsqueeze(0)     # (1, T, D)

        decode_fn = offline_decode if args.offline else streaming_decode
        kw = dict(max_new_tokens=args.max_new_tokens, rep_penalty=args.rep_penalty, device=device)
        if args.offline:
            kw["num_beams"] = args.num_beams
        pred_text = decode_fn(model, visual, tok, **kw)

        hyp_n = normalize(pred_text)
        ref_n = normalize(gt_text)

        if args.prefix_eval:
            # 截断评测：pred 截到 GT 词数，消除尾部重复的干扰
            ref_words = ref_n.split()
            hyp_words = hyp_n.split()
            hyp_n = " ".join(hyp_words[:len(ref_words)])

        hyps.append(hyp_n)
        refs.append(ref_n)

        if shown < args.show:
            print(f"\nGT:   {gt_text}")
            print(f"PRED: {pred_text}")
            shown += 1

    # ── WER ───────────────────────────────────────────────────────────────────
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    wer  = jiwer.wer(refs, hyps,
                     reference_transform=transform,
                     hypothesis_transform=transform)
    print(f"\n{'─'*50}")
    print(f"clips decoded : {len(hyps)}")
    print(f"WER           : {wer*100:.1f}%")
    print(f"word accuracy : {(1-wer)*100:.1f}%")


if __name__ == "__main__":
    main()
