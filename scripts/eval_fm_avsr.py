"""
Inference & evaluation for FMHeadAVSR.

For each test clip:
  1. Load pre-extracted avsr_enc, smollm2_h, speaker_emb
  2. FMHeadAVSR.forward_inference() → pred_latent (T_a, 512)
  3. Mimi decode → waveform @ 24kHz
  4. Save pred.wav (and optionally gt.wav)

Usage:
  python scripts/eval_fm_avsr.py \
      --ckpt runs/fm_avsr/fm_avsr_with_text/step_026580.pt \
      --n 20 --output_dir eval_out/with_text --save_gt

  # no-text ablation
  python scripts/eval_fm_avsr.py \
      --ckpt runs/fm_avsr/fm_avsr_no_text/step_026580.pt \
      --no_text_cond --n 20 --output_dir eval_out/no_text --save_gt
"""
import argparse, json, sys, wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v2.fm_head import FMHead as _FMBase, SinusoidalTimeEmb, DiTBlock
from streaminlip.fm_avsr_dataset import (
    FMAVSRDataset, _MAX_TA, _MAX_L, validate_latent_frame_rate,
    denormalize_latent, build_word_timestamp_lm_indices,
)
from transformers import MimiModel


class FMHeadAVSR(_FMBase):
    DIM = 512
    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        use_cross_attn=False,
        use_text_token_cross_attn=False,
        extra_cond_dim=0,
        ctc_vocab_size=0,
        ctc_topk=0,
        ctc_token_emb_dim=0,
    ):
        super().__init__(
            n_layers=n_layers,
            n_heads=n_heads,
            use_cross_attn=use_cross_attn,
            use_text_token_cross_attn=use_text_token_cross_attn,
            extra_cond_dim=extra_cond_dim,
            ctc_vocab_size=ctc_vocab_size,
            ctc_topk=ctc_topk,
            ctc_token_emb_dim=ctc_token_emb_dim,
        )
        ctc_cond_dim = ctc_token_emb_dim + ctc_topk if ctc_topk > 0 else 0
        self.cond_proj = nn.Linear(768 + 960 + 256 + extra_cond_dim + ctc_cond_dim, self.DIM)


def save_wav(audio: np.ndarray, path: str, sr: int = 24000):
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(pcm.tobytes())


def mimi_decode(mimi, latent_np, device):
    """latent_np: (T_a, 512) float32 → waveform numpy float32"""
    x = torch.from_numpy(latent_np).to(device).float().T.unsqueeze(0)  # (1, 512, T_a)
    x = mimi.upsample(x)
    x = mimi.decoder_transformer(x.transpose(1, 2)).last_hidden_state
    x = mimi.decoder(x.transpose(1, 2))
    return x.squeeze().float().cpu().numpy()


def explicit_cli_keys(argv=None):
    argv = argv or sys.argv
    keys = set()
    for item in argv[1:]:
        if not item.startswith("--"):
            continue
        key = item[2:].split("=", 1)[0].replace("-", "_")
        if key:
            keys.add(key)
    return keys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default=None,
                   help="Optional YAML config. Known eval keys are applied; train-only keys are ignored.")
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--data_root",     default="data/processed")
    p.add_argument("--mimi_path",     default="pretrained/mimi")
    p.add_argument("--smollm2_path",  default="pretrained/smollm2-360m",
                   help="Tokenizer path used for word-timestamp text alignment.")
    p.add_argument("--split",         default="pretrain")
    p.add_argument("--clip_list",     default=None,
                   help="Optional file with one processed clip path per line.")
    p.add_argument("--n",             type=int, default=20)
    p.add_argument("--output_dir",    default="eval_out/with_text")
    p.add_argument("--no_text_cond",  action="store_true",
                   help="Deprecated alias for --condition_mode video_only.")
    p.add_argument("--condition_mode",
                   choices=["both", "video_only", "text_only", "shuffle_text"],
                   default="both",
                   help="Condition ablation mode for video/text contribution checks.")
    p.add_argument("--text_alignment_mode",
                   choices=["uniform", "word_timestamps"],
                   default="uniform",
                   help="How to align SmolLM2 hidden states to latent frames.")
    p.add_argument("--ctc_condition_mode",
                   choices=[
                       "none",
                       "logprob",
                       "shuffle_logprob",
                       "summary",
                       "shuffle_summary",
                       "topk",
                       "shuffle_topk",
                   ],
                   default="none",
                   help="Optional Auto-AVSR CTC posterior condition from avsr_enc.")
    p.add_argument("--auto_avsr_ckpt",
                   default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth")
    p.add_argument("--ctc_vocab_size", type=int, default=5049)
    p.add_argument("--ctc_topk", type=int, default=4)
    p.add_argument("--ctc_token_emb_dim", type=int, default=32)
    p.add_argument("--save_gt",       action="store_true")
    p.add_argument("--nfe",           type=int, default=10)
    p.add_argument("--n_dit_layers",  type=int, default=6)
    p.add_argument("--use_cross_attn", action="store_true",
                   help="Build FMHeadAVSR with DiT condition cross-attention.")
    p.add_argument("--use_text_token_cross_attn", action="store_true",
                   help="Use raw SmolLM token sequence as DiT cross-attention K/V.")
    p.add_argument("--use_recon",     action="store_true",
                   help="Decode deterministic reconstruct_from_cond output instead of FM sampling.")
    p.add_argument("--use_denoise",   action="store_true",
                   help="Decode one-step noisy endpoint denoise output instead of FM sampling.")
    p.add_argument("--denoise_t",     type=float, default=0.0,
                   help="Time embedding used with --use_denoise.")
    p.add_argument("--denoise_seed",  type=int, default=0,
                   help="Random seed for the noisy token used with --use_denoise.")
    cli_keys = explicit_cli_keys()
    args = p.parse_args()
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        config_keys = {"data_root", "mimi_path", "smollm2_path", "split", "clip_list",
                       "no_text_cond", "condition_mode", "text_alignment_mode",
                       "ctc_condition_mode", "auto_avsr_ckpt", "ctc_vocab_size",
                       "ctc_topk", "ctc_token_emb_dim",
                       "n_dit_layers", "use_cross_attn", "use_text_token_cross_attn"}
        for k, v in cfg.items():
            if k in config_keys:
                if k in cli_keys and k != "config":
                    continue
                setattr(args, k, v)
    if args.no_text_cond and "condition_mode" not in cli_keys:
        args.condition_mode = "video_only"
    return args


def resample_h_lm(h_lm_np, T_a, device):
    """(L, 960) → (1, T_a, 960) bfloat16"""
    L, D = h_lm_np.shape
    h = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16).unsqueeze(0)  # (1, L, D)
    idx = torch.clamp(torch.arange(T_a, device=device) * L // max(T_a, 1), 0, L - 1)
    return h[:, idx, :]  # (1, T_a, D)


def gather_h_lm_by_lm_idx(h_lm_np, lm_idx_np, T_a, device):
    """(L, 960) + (T_a,) indices -> (1, T_a, 960) bfloat16."""
    L, D = h_lm_np.shape
    h = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16).unsqueeze(0)
    idx = torch.from_numpy(lm_idx_np[:T_a].astype("int64")).to(device)
    idx = idx.clamp(min=0, max=max(L - 1, 0))
    return h[:, idx, :]


def load_ctc_head(ckpt_path: str, vocab_size: int = 5049):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    head = nn.Linear(768, vocab_size)
    head.weight.data.copy_(sd["ctc.ctc_lo.weight"].float())
    head.bias.data.copy_(sd["ctc.ctc_lo.bias"].float())
    for p in head.parameters():
        p.requires_grad_(False)
    return head.eval()


def ctc_extra_dim(mode: str, vocab_size: int) -> int:
    if mode == "none":
        return 0
    if mode in {"summary", "shuffle_summary"}:
        return 6
    if mode in {"topk", "shuffle_topk"}:
        return 0
    return vocab_size


def ctc_topk_dim(mode: str, topk: int) -> int:
    return topk if mode in {"topk", "shuffle_topk"} else 0


def build_ctc_condition(enc_np, ctc_head, mode, vocab_size, T_a, device, topk=4):
    if ctc_head is None:
        return None, None, None
    enc_t = torch.from_numpy(enc_np).to(device, dtype=torch.float32).unsqueeze(0)
    logits = ctc_head(enc_t)
    if mode in {"logprob", "shuffle_logprob"}:
        cond = F.log_softmax(logits, dim=-1).clamp(min=-20.0, max=0.0)
    elif mode in {"summary", "shuffle_summary"}:
        prob = F.softmax(logits.float(), dim=-1)
        top_prob, top_idx = torch.topk(prob, k=2, dim=-1)
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.tensor(float(np.log(vocab_size)), device=device)
        cond = torch.cat([
            prob[..., :1],
            top_prob[..., :1],
            top_prob[..., 1:2],
            top_idx[..., :1].float() / max(vocab_size - 1, 1),
            top_idx[..., 1:2].float() / max(vocab_size - 1, 1),
            entropy,
        ], dim=-1)
        extra = cond[:, ::2, :][:, :T_a, :]
        if extra.shape[1] < T_a:
            pad = torch.zeros(1, T_a - extra.shape[1], extra.shape[2], device=device)
            extra = torch.cat([extra, pad], dim=1)
        return extra.to(dtype=torch.bfloat16), None, None
    elif mode in {"topk", "shuffle_topk"}:
        prob = F.softmax(logits.float(), dim=-1)
        top_probs, top_ids = torch.topk(prob, k=topk, dim=-1)
        ids = top_ids[:, ::2, :][:, :T_a, :].to(device=device, dtype=torch.long)
        probs = top_probs[:, ::2, :][:, :T_a, :].to(device=device, dtype=torch.bfloat16)
        if ids.shape[1] < T_a:
            pad_ids = torch.zeros(1, T_a - ids.shape[1], ids.shape[2], device=device, dtype=torch.long)
            pad_probs = torch.zeros(1, T_a - probs.shape[1], probs.shape[2], device=device, dtype=torch.bfloat16)
            ids = torch.cat([ids, pad_ids], dim=1)
            probs = torch.cat([probs, pad_probs], dim=1)
        return None, ids, probs
    return None, None, None


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FM head
    print("Loading FMHeadAVSR...")
    fm = FMHeadAVSR(
        n_layers=args.n_dit_layers,
        use_cross_attn=args.use_cross_attn,
        use_text_token_cross_attn=args.use_text_token_cross_attn,
        extra_cond_dim=ctc_extra_dim(args.ctc_condition_mode, args.ctc_vocab_size),
        ctc_vocab_size=args.ctc_vocab_size,
        ctc_topk=ctc_topk_dim(args.ctc_condition_mode, args.ctc_topk),
        ctc_token_emb_dim=args.ctc_token_emb_dim,
    ).to(device).bfloat16().eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    fm.load_state_dict(ckpt["fm_head"])
    print(f"  Loaded {args.ckpt}")

    # Load Mimi
    print("Loading Mimi decoder...")
    mimi = MimiModel.from_pretrained(args.mimi_path, local_files_only=True).to(device).eval()
    tokenizer = None
    if args.text_alignment_mode == "word_timestamps":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.smollm2_path, local_files_only=True)
    ctc_head = None
    if args.ctc_condition_mode != "none":
        ctc_head = load_ctc_head(args.auto_avsr_ckpt, args.ctc_vocab_size).to(device)

    # Test clips: last 2000 of pretrain by default, or an explicit clip list.
    subset = "train" if args.clip_list else "test"
    ds = FMAVSRDataset(args.data_root, args.split, subset=subset, clip_list=args.clip_list)
    clips = ds.clips[:args.n]
    print(f"Evaluating {len(clips)} clips → {out_dir}\n")

    root = Path(args.data_root)
    shuffle_h_lm = None
    if args.condition_mode == "shuffle_text" and clips:
        h_path = clips[-1] / "smollm2_h.npy"
        if h_path.exists():
            shuffle_h_lm = np.load(str(h_path)).astype("float32")
            if shuffle_h_lm.shape[0] > _MAX_L:
                shuffle_h_lm = shuffle_h_lm[:_MAX_L]

    with torch.no_grad():
        for i, c in enumerate(clips):
            enc = np.load(str(c / "avsr_enc.npy")).astype("float32")   # (T, 768)
            lat_gt = np.load(str(c / "latent.npz"))["latent"].astype("float32")  # (T_a, 512)
            lat_gt = validate_latent_frame_rate(lat_gt, enc.shape[0], c)
            spk = np.load(str(c / "speaker_emb.npy")).astype("float32")  # (256,)

            T_a = min(lat_gt.shape[0], _MAX_TA)

            # Visual condition
            enc_t  = torch.from_numpy(enc).to(device, dtype=torch.bfloat16).unsqueeze(0)
            v_down = enc_t[:, ::2, :][:, :T_a, :]  # (1, T_a, 768)
            if v_down.shape[1] < T_a:
                pad = torch.zeros(1, T_a - v_down.shape[1], 768, device=device, dtype=torch.bfloat16)
                v_down = torch.cat([v_down, pad], 1)
            if args.condition_mode == "text_only":
                v_down = torch.zeros_like(v_down)

            # Text condition
            text_tokens = None
            text_token_mask = None
            if args.condition_mode == "video_only":
                h_down = torch.zeros(1, T_a, 960, device=device, dtype=torch.bfloat16)
            else:
                h_path = c / "smollm2_h.npy"
                if args.condition_mode == "shuffle_text" and shuffle_h_lm is not None:
                    h_down = resample_h_lm(shuffle_h_lm, T_a, device)
                    text_tokens = torch.from_numpy(shuffle_h_lm).to(
                        device, dtype=torch.bfloat16
                    ).unsqueeze(0)
                    text_token_mask = torch.ones(
                        1, shuffle_h_lm.shape[0], device=device, dtype=torch.bool
                    )
                elif h_path.exists():
                    h_lm = np.load(str(h_path)).astype("float32")
                    if h_lm.shape[0] > _MAX_L:
                        h_lm = h_lm[:_MAX_L]
                    if args.text_alignment_mode == "word_timestamps":
                        txt = (c / "avsr_text.txt").read_text().strip()
                        meta = json.loads((c / "text.json").read_text())
                        lm_idx = build_word_timestamp_lm_indices(
                            txt, meta.get("words", []), tokenizer, T_a
                        )
                        h_down = gather_h_lm_by_lm_idx(h_lm, lm_idx, T_a, device)
                    else:
                        h_down = resample_h_lm(h_lm, T_a, device)
                    text_tokens = torch.from_numpy(h_lm).to(
                        device, dtype=torch.bfloat16
                    ).unsqueeze(0)
                    text_token_mask = torch.ones(
                        1, h_lm.shape[0], device=device, dtype=torch.bool
                    )
                else:
                    h_down = torch.zeros(1, T_a, 960, device=device, dtype=torch.bfloat16)

            # Speaker
            spk_t = torch.from_numpy(spk).to(device, dtype=torch.bfloat16).unsqueeze(0)
            extra_cond, ctc_ids, ctc_probs = build_ctc_condition(
                enc, ctc_head, args.ctc_condition_mode, args.ctc_vocab_size, T_a, device, args.ctc_topk
            )

            # FM inference → denormalize
            if args.use_recon and args.use_denoise:
                raise ValueError("--use_recon and --use_denoise are mutually exclusive")
            if args.use_recon:
                pred_latent = fm.reconstruct_from_cond(
                    v_down, h_down, spk_t,
                    text_tokens=text_tokens,
                    text_token_mask=text_token_mask,
                    extra_cond=extra_cond,
                    ctc_topk_ids=ctc_ids,
                    ctc_topk_probs=ctc_probs,
                )
            elif args.use_denoise:
                gen = torch.Generator(device=device).manual_seed(args.denoise_seed + i)
                noise = torch.randn(
                    1, T_a, 512, device=device, dtype=torch.bfloat16, generator=gen
                )
                denoise_t = torch.full(
                    (1,), args.denoise_t, device=device, dtype=torch.bfloat16
                )
                pred_latent = fm.denoise_from_noise(
                    v_down, h_down, spk_t, noise, denoise_t,
                    text_tokens=text_tokens,
                    text_token_mask=text_token_mask,
                    extra_cond=extra_cond,
                    ctc_topk_ids=ctc_ids,
                    ctc_topk_probs=ctc_probs,
                )
            else:
                pred_latent = fm.forward_inference(
                    v_down, h_down, spk_t, nfe=args.nfe,
                    text_tokens=text_tokens,
                    text_token_mask=text_token_mask,
                    extra_cond=extra_cond,
                    ctc_topk_ids=ctc_ids,
                    ctc_topk_probs=ctc_probs,
                )
            pred_np = denormalize_latent(pred_latent.squeeze(0).float().cpu().numpy())
            save_wav(mimi_decode(mimi, pred_np, device), str(out_dir / f"{i:04d}_pred.wav"))

            if args.save_gt:
                gt_np = mimi_decode(mimi, lat_gt[:T_a], device)
                save_wav(gt_np, str(out_dir / f"{i:04d}_gt.wav"))

            txt = (c / "avsr_text.txt").read_text().strip()
            print(f"[{i:3d}] {c.name}  T_a={T_a}  ref: {txt[:60]}")

    print(f"\n完成，音频保存至 {out_dir}")


if __name__ == "__main__":
    main()
