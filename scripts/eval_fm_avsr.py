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
import argparse, sys, wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v2.fm_head import FMHead as _FMBase, SinusoidalTimeEmb, DiTBlock
from streaminlip.fm_avsr_dataset import (
    FMAVSRDataset, _MAX_TA, _MAX_L, validate_latent_frame_rate,
    denormalize_latent,
)
from transformers import MimiModel


class FMHeadAVSR(_FMBase):
    DIM = 512
    def __init__(self, n_layers=6, n_heads=8):
        nn.Module.__init__(self)
        COND_DIM = 768 + 960 + 256
        self.cond_proj  = nn.Linear(COND_DIM, self.DIM)
        self.cond_token_proj = nn.Linear(self.DIM, self.DIM)
        self.time_emb   = SinusoidalTimeEmb(self.DIM)
        self.blocks     = nn.ModuleList([DiTBlock(self.DIM, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(self.DIM)
        self.final_proj = nn.Linear(self.DIM, self.DIM)


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default=None,
                   help="Optional YAML config. Known eval keys are applied; train-only keys are ignored.")
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--data_root",     default="data/processed")
    p.add_argument("--mimi_path",     default="pretrained/mimi")
    p.add_argument("--split",         default="pretrain")
    p.add_argument("--clip_list",     default=None,
                   help="Optional file with one processed clip path per line.")
    p.add_argument("--n",             type=int, default=20)
    p.add_argument("--output_dir",    default="eval_out/with_text")
    p.add_argument("--no_text_cond",  action="store_true")
    p.add_argument("--save_gt",       action="store_true")
    p.add_argument("--nfe",           type=int, default=10)
    p.add_argument("--n_dit_layers",  type=int, default=6)
    p.add_argument("--use_recon",     action="store_true",
                   help="Decode deterministic reconstruct_from_cond output instead of FM sampling.")
    args = p.parse_args()
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        config_keys = {"data_root", "mimi_path", "split", "clip_list",
                       "no_text_cond", "n_dit_layers"}
        for k, v in cfg.items():
            if k in config_keys:
                setattr(args, k, v)
    return args


def resample_h_lm(h_lm_np, T_a, device):
    """(L, 960) → (1, T_a, 960) bfloat16"""
    L, D = h_lm_np.shape
    h = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16).unsqueeze(0)  # (1, L, D)
    idx = torch.clamp(torch.arange(T_a, device=device) * L // max(T_a, 1), 0, L - 1)
    return h[:, idx, :]  # (1, T_a, D)


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FM head
    print("Loading FMHeadAVSR...")
    fm = FMHeadAVSR(n_layers=args.n_dit_layers).to(device).bfloat16().eval()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    fm.load_state_dict(ckpt["fm_head"])
    print(f"  Loaded {args.ckpt}")

    # Load Mimi
    print("Loading Mimi decoder...")
    mimi = MimiModel.from_pretrained(args.mimi_path, local_files_only=True).to(device).eval()

    # Test clips: last 2000 of pretrain by default, or an explicit clip list.
    subset = "train" if args.clip_list else "test"
    ds = FMAVSRDataset(args.data_root, args.split, subset=subset, clip_list=args.clip_list)
    clips = ds.clips[:args.n]
    print(f"Evaluating {len(clips)} clips → {out_dir}\n")

    root = Path(args.data_root)

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

            # Text condition
            if args.no_text_cond:
                h_down = torch.zeros(1, T_a, 960, device=device, dtype=torch.bfloat16)
            else:
                h_path = c / "smollm2_h.npy"
                if h_path.exists():
                    h_lm = np.load(str(h_path)).astype("float32")
                    if h_lm.shape[0] > _MAX_L:
                        h_lm = h_lm[:_MAX_L]
                    h_down = resample_h_lm(h_lm, T_a, device)
                else:
                    h_down = torch.zeros(1, T_a, 960, device=device, dtype=torch.bfloat16)

            # Speaker
            spk_t = torch.from_numpy(spk).to(device, dtype=torch.bfloat16).unsqueeze(0)

            # FM inference → denormalize
            if args.use_recon:
                pred_latent = fm.reconstruct_from_cond(v_down, h_down, spk_t)
            else:
                pred_latent = fm.forward_inference(v_down, h_down, spk_t, nfe=args.nfe)
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
