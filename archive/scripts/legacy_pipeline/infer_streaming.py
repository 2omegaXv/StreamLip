"""
Streaming lip-to-speech inference with Auto-AVSR text prior.

For each chunk k (frames 0..k*C):
  1. Auto-AVSR CTC greedy → predicted text (streaming-safe, ~24% WER)
  2. Tokenize + SmolLM2 → h_lm  (~19ms)
  3. AV-HuBERT → vis_feat
  4. FM head → Mimi latent → audio chunk

Outputs concatenated waveform + per-chunk text predictions.

Usage:
  python scripts/infer_streaming.py \
      --ckpt runs/v4/v4_p2_with_text_lr3e-4_ep30/step_006630.pt \
      --clip data/processed/pretrain/gyPoqFcvt9w/00002 \
      --out eval_audio/streaming_test.wav
"""
import argparse, json, sys, time, wave
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.v4 import StreamLipV4
from streaminlip.v4.data.dataset import LRS3DatasetV4, CHUNK_SIZE, SIL_ID
from streaminlip.auto_avsr import AutoAVSRInferencer
from transformers import MimiModel


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def save_wav(arr, path, sr=24000):
    arr = arr / (abs(arr).max() + 1e-6) * 0.9
    pcm = (arr * 32767).astype('int16')
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def text_to_clean_ids(text: str, tok: AutoTokenizer, bos_id: int) -> tuple:
    """Convert predicted text → (clean_ids, lm_idx_fm) for a sequence of T frames."""
    words = text.strip().upper().split()
    tokens = []
    for w in words:
        toks = tok.encode(' ' + w.lower(), add_special_tokens=False)
        tokens.extend(toks)
    clean_ids = np.array([bos_id] + tokens, dtype=np.int64)
    return clean_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",             required=True)
    p.add_argument("--clip",             required=True,
                   help="Path to processed clip dir (has avhubert_pre.npy, lip.npy, latent.npz)")
    p.add_argument("--out",              default="eval_audio/streaming_out.wav")
    p.add_argument("--avhubert_ckpt",    default="pretrained/av-hubert/model.pt")
    p.add_argument("--avsr_ckpt",        default="pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth")
    p.add_argument("--smollm2_path",     default="pretrained/smollm2-360m")
    p.add_argument("--mimi_path",        default="pretrained/mimi")
    p.add_argument("--resnet50_weights", default="pretrained/resnet50-11ad3fa6.pth")
    p.add_argument("--lora_rank",        type=int, default=16)
    p.add_argument("--nfe",              type=int, default=10)
    p.add_argument("--min_chunks",       type=int, default=4,
                   help="Minimum prefix length before generating audio (avoid cold-start)")
    return p.parse_args()


def mimi_decode(mimi, latent_Ta_512, device):
    """(T_a, 512) → waveform numpy"""
    x = torch.from_numpy(latent_Ta_512.astype('float32')).unsqueeze(0).to(device)
    with torch.no_grad():
        up  = mimi.upsample(x.transpose(1, 2))
        dt  = mimi.decoder_transformer(up.transpose(1, 2)).last_hidden_state
        wav = mimi.decoder(dt.transpose(1, 2))
    return wav.squeeze().float().cpu().numpy()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip   = Path(args.clip)

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models...")
    tok = AutoTokenizer.from_pretrained(args.smollm2_path)

    model = StreamLipV4(
        avhubert_ckpt=args.avhubert_ckpt, smollm2_path=args.smollm2_path,
        lora_rank=args.lora_rank, resnet50_weights=args.resnet50_weights,
    )
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    for k in ['visual_encoder','sil_head','lm','fm_head','speaker_encoder']:
        if k in ckpt: getattr(model, k).load_state_dict(ckpt[k], strict=False)
    model = model.to(device).bfloat16().eval()

    asr   = AutoAVSRInferencer(args.avsr_ckpt, device=device)
    mimi  = MimiModel.from_pretrained(args.mimi_path).to(device).eval()

    bos_id = model.lm.model.config.bos_token_id or SIL_ID

    # ── Load clip ────────────────────────────────────────────────────────────
    meta  = json.loads((clip / 'text.json').read_text())
    gt    = meta['transcript']
    lip   = np.load(str(clip / 'lip.npy'))           # (T_full, 96, 96, 3) uint8
    pre   = np.load(str(clip / 'avhubert_pre.npy'))  # (T_full, 768) float16
    face_emb = np.load(str(clip / 'speaker_emb.npy'))

    T_full = len(pre)
    K = T_full // CHUNK_SIZE
    face_t = torch.from_numpy(face_emb.astype('float32')).unsqueeze(0).to(device, dtype=torch.bfloat16)

    print(f"GT: {gt}")
    print(f"T={T_full} frames, {K} chunks @ {CHUNK_SIZE} frames each\n")

    # ── Streaming chunk-by-chunk inference ───────────────────────────────────
    all_audio = []
    timings   = {'avsr': [], 'lm': [], 'fm': []}

    for k in range(args.min_chunks, K + 1):
        T = k * CHUNK_SIZE

        # 1. Auto-AVSR CTC greedy → text
        t0 = time.perf_counter()
        lip_frames = torch.from_numpy(lip[:T])          # (T, 96, 96, 3) uint8
        pred_text  = asr.infer_ctc(lip_frames)
        timings['avsr'].append((time.perf_counter() - t0) * 1000)

        # 2. Text → clean_ids → SmolLM2 → h_lm
        t0 = time.perf_counter()
        clean_ids_np = text_to_clean_ids(pred_text, tok, bos_id)
        L = len(clean_ids_np)

        # Build lm_idx_fm: uniform mapping of T frames to L token positions
        lm_idx_fm_np = np.minimum(
            np.arange(T, dtype=np.int64) * L // T,
            L - 1
        )

        vis_t  = torch.from_numpy(pre[:T].astype('float32')).unsqueeze(0).to(device, dtype=torch.bfloat16)
        ids_t  = torch.from_numpy(clean_ids_np).unsqueeze(0).to(device)
        cmsk_t = torch.ones(1, L, dtype=torch.long, device=device)
        lmf_t  = torch.from_numpy(lm_idx_fm_np).unsqueeze(0).to(device)
        mask_t = torch.ones(1, T, dtype=torch.bool, device=device)

        # LM forward to get h_lm
        with torch.no_grad():
            _, h_lm = model.lm(ids_t, cmsk_t, [vis_t.squeeze(0).unsqueeze(0)] * 12, lmf_t)
        timings['lm'].append((time.perf_counter() - t0) * 1000)

        # Only generate audio for the LAST chunk (new content)
        if k == args.min_chunks:
            chunk_start = 0
        else:
            chunk_start = (k - 1) * CHUNK_SIZE

        T_chunk_end   = T
        T_chunk_start = chunk_start

        # 3. FM head → Mimi latent for this chunk
        t0 = time.perf_counter()
        with torch.no_grad():
            last_feat, layer_feats = model.visual_encoder(vis_t)
            id_vec  = model.speaker_encoder(face_t)
            v_down  = last_feat[:, ::2, :]
            idx_f   = lmf_t.unsqueeze(-1)
            h_lm_fm = h_lm.gather(1, idx_f.expand(1, T, h_lm.shape[-1]))
            h_down  = h_lm_fm[:, ::2, :]
            pred_latent = model.fm_head.forward_inference(v_down, h_down, id_vec, nfe=args.nfe)

        # Take only the new chunk's latent
        T_a_start = T_chunk_start // 2
        T_a_end   = T_chunk_end   // 2
        chunk_latent = pred_latent.squeeze(0)[T_a_start:T_a_end].float().cpu().numpy()
        timings['fm'].append((time.perf_counter() - t0) * 1000)

        # 4. Mimi decode → audio
        chunk_audio = mimi_decode(mimi, chunk_latent, device)
        all_audio.append(chunk_audio)

        print(f"chunk {k:3d}/{K} | text: {pred_text[:60]}")

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_audio = np.concatenate(all_audio)
    save_wav(full_audio, str(out_path))

    print(f"\nSaved: {out_path}  ({len(full_audio)/24000:.1f}s)")
    print(f"Avg timing per chunk:")
    print(f"  Auto-AVSR: {np.mean(timings['avsr']):.0f}ms")
    print(f"  SmolLM2:   {np.mean(timings['lm']):.0f}ms")
    print(f"  FM+decode: {np.mean(timings['fm']):.0f}ms")
    print(f"  Total/chunk: {np.mean(timings['avsr'])+np.mean(timings['lm'])+np.mean(timings['fm']):.0f}ms  (chunk=240ms)")


if __name__ == "__main__":
    main()
