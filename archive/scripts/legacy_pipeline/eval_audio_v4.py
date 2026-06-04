"""
Audio evaluation for StreamLip V4 Phase 2.

For each test clip:
  1. Run model forward → pred_latent (B, T_a, 512)
  2. Decode with Mimi: upsample + decoder → waveform @ 24kHz
  3. Save to output_dir/{i:04d}_pred.wav  (and optionally GT)
  4. Run Whisper ASR on pred audio → compute WER vs GT transcript

Usage:
  python scripts/eval_audio_v4.py \
      --ckpt runs/v4/v4_p2_with_text_lr3e-4_ep30/step_008000.pt \
      --n 20 --output_dir /tmp/eval_audio
"""
import argparse, json, os, sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v4 import StreamLipV4
from streaminlip.v4.data.dataset import LRS3DatasetV4, SIL_ID, FPS
from transformers import MimiModel


def wer(ref: str, hyp: str) -> float:
    ref_w = ref.upper().split()
    hyp_w = hyp.upper().split()
    if not ref_w: return 0.0
    d = list(range(len(hyp_w) + 1))
    for rw in ref_w:
        d2 = [d[0] + 1]
        for j, hw in enumerate(hyp_w):
            d2.append(min(d[j+1]+1, d2[j]+1, d[j]+(0 if rw==hw else 1)))
        d = d2
    return d[-1] / len(ref_w)


def save_wav(audio: np.ndarray, path: str, sr: int = 24000):
    """Save float32 mono audio as 16-bit PCM wav."""
    import wave, struct
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",              required=True)
    p.add_argument("--data_root",         default="data/processed")
    p.add_argument("--smollm2_path",      default="pretrained/smollm2-360m")
    p.add_argument("--avhubert_ckpt",     default="pretrained/av-hubert/model.pt")
    p.add_argument("--mimi_path",         default="pretrained/mimi")
    p.add_argument("--resnet50_weights",  default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--split",             default="pretrain")
    p.add_argument("--subset",            default="test")
    p.add_argument("--n",                 type=int, default=20)
    p.add_argument("--output_dir",        default="eval_audio")
    p.add_argument("--lora_rank",         type=int, default=16)
    p.add_argument("--no_text_cond",      action="store_true")
    p.add_argument("--save_gt",           action="store_true",
                   help="Also save GT audio decoded from GT latent")
    p.add_argument("--whisper_asr",       action="store_true",
                   help="Run Whisper ASR for WER evaluation")
    p.add_argument("--whisper_model",     default="openai/whisper-large-v3")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = StreamLipV4(
        avhubert_ckpt=args.avhubert_ckpt, smollm2_path=args.smollm2_path,
        lora_rank=args.lora_rank, no_text_cond=args.no_text_cond,
        resnet50_weights=args.resnet50_weights,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Load all available keys
    for key in ["visual_encoder", "sil_head", "lm", "fm_head", "speaker_encoder"]:
        if key in ckpt:
            getattr(model, key).load_state_dict(ckpt[key], strict=False)
            print(f"  Loaded {key}")
    model = model.to(device).bfloat16().eval()

    # Load Mimi decoder
    print("Loading Mimi...")
    mimi = MimiModel.from_pretrained(args.mimi_path).to(device).eval()

    # Load Whisper ASR if requested
    asr_pipe = None
    if args.whisper_asr:
        print(f"Loading Whisper ({args.whisper_model})...")
        from transformers import pipeline
        asr_pipe = pipeline("automatic-speech-recognition",
                            model=args.whisper_model, device=device)

    # Dataset
    ds = LRS3DatasetV4(
        args.data_root, args.split, args.smollm2_path,
        load_face=True, load_latent=True, deterministic=True, subset=args.subset,
    )
    print(f"Loaded: {args.ckpt}")
    print(f"Evaluating {min(args.n, len(ds))} clips → {out_dir}\n")

    total_wer, n_eval = 0.0, 0
    with torch.no_grad():
        for i in range(min(args.n, len(ds))):
            item     = ds[i]
            clip_dir = ds.clips[i]
            meta     = json.loads((clip_dir / "text.json").read_text())
            gt_ref   = meta["transcript"].strip()
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
            latent_gt   = item["latent"]  # (T_a, 512) float16, for GT decode

            # Model forward → pred latent via generate()
            pred_latent = model.generate(
                visual=vis, clean_ids=clean_ids, clean_mask=clean_mask,
                lm_idx_fm=lm_idx_fm, face=face, mask=mask, nfe=10,
            )  # (1, T_a, 512)

            def mimi_decode(latent_Ta_512):
                """latent: (T_a, 512) quantized embedding → waveform numpy float32"""
                x = latent_Ta_512.to(device).float()          # (T_a, 512)
                x = x.T.unsqueeze(0)                          # (1, 512, T_a)
                # decode path mirrors _decode_frame: upsample → decoder_transformer → decoder
                x = mimi.upsample(x)                          # (1, 512, 2*T_a)
                x = mimi.decoder_transformer(
                        x.transpose(1, 2)).last_hidden_state  # (1, 2*T_a, 512)
                x = mimi.decoder(x.transpose(1, 2))           # (1, 1, N_samples)
                return x.squeeze().float().cpu().numpy()

            wav_np    = mimi_decode(pred_latent.squeeze(0))
            pred_path = str(out_dir / f"{i:04d}_pred.wav")
            save_wav(wav_np, pred_path)

            if args.save_gt and latent_gt.shape[0] > 0:
                wav_gt = mimi_decode(latent_gt)
                save_wav(wav_gt, str(out_dir / f"{i:04d}_gt.wav"))

            # ASR WER
            w_score = 0.0
            if asr_pipe is not None:
                result   = asr_pipe(pred_path)
                hyp      = result["text"].strip()
                w_score  = wer(gt_ref, hyp)
                total_wer += w_score
                n_eval    += 1
                print(f"[{i:3d}] REF: {gt_ref[:70]}")
                print(f"      HYP: {hyp[:70]}")
                print(f"      WER: {w_score:.1%}  → {pred_path}")
            else:
                print(f"[{i:3d}] saved → {pred_path}  | ref: {gt_ref[:60]}")

    if n_eval > 0:
        print(f"\nMean WER ({n_eval} clips): {total_wer/n_eval:.1%}")
    print(f"\nAudio saved to: {out_dir}")


if __name__ == "__main__":
    main()
