"""
Build lightweight audio-prompt timbre conditions from the first seconds of
normalized Mimi latents.

This avoids adding a new speaker-recognition dependency for the first timbre
conditioning experiment. Each output is a global vector:

  timbre_cond.npy = concat(mean(prefix_latent), std(prefix_latent))  # (1024,)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.fm_avsr_dataset import normalize_latent


def build_mel_stats(
    audio: np.ndarray,
    sample_rate: int,
    prompt_seconds: float,
    n_mels: int = 80,
    mel_transform=None,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        audio = np.zeros((1,), dtype=np.float32)
    n_samples = max(1, int(round(float(prompt_seconds) * int(sample_rate))))
    prefix = np.zeros((n_samples,), dtype=np.float32)
    n = min(n_samples, audio.shape[0])
    prefix[:n] = audio[:n]
    wav = torch.from_numpy(prefix).unsqueeze(0)
    if mel_transform is None:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=int(n_mels),
            power=2.0,
        )
    mel = mel_transform(wav)
    log_mel = torch.log(mel.clamp_min(1e-5)).squeeze(0).transpose(0, 1)
    mean = log_mel.mean(dim=0)
    std = log_mel.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=0).cpu().numpy().astype(np.float32)


def build_timbre_condition(
    latent: np.ndarray,
    prompt_frames: int,
    extra_stats: np.ndarray | None = None,
) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != 512:
        raise ValueError(f"latent must have shape (T, 512), got {latent.shape}")
    n = max(1, min(int(prompt_frames), latent.shape[0]))
    prefix = latent[:n]
    parts = [prefix.mean(axis=0), prefix.std(axis=0)]
    if extra_stats is not None:
        parts.append(np.asarray(extra_stats, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def iter_clips(data_root: Path, split: str, clip_list: str | None):
    if clip_list:
        for rel in Path(clip_list).read_text().splitlines():
            rel = rel.strip()
            if rel:
                yield data_root / rel
        return
    for path in sorted((data_root / split).glob("*/*")):
        if path.is_dir():
            yield path


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(str(path))
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def extract_clip(
    clip: Path,
    output_name: str,
    prompt_frames: int,
    prompt_seconds: float,
    n_mels: int,
    include_mel_stats: bool,
    force: bool,
    mel_cache: dict[tuple[int, int], torchaudio.transforms.MelSpectrogram] | None = None,
) -> str:
    out = clip / output_name
    if out.exists() and not force:
        return "skip"
    latent_path = clip / "latent.npz"
    if not latent_path.exists():
        return "missing"
    latent = np.load(str(latent_path))["latent"].astype(np.float32)
    extra = None
    if include_mel_stats:
        audio_path = clip / "audio.wav"
        if not audio_path.exists():
            return "missing"
        audio, sr = load_audio(audio_path)
        key = (int(sr), int(n_mels))
        mel_transform = None
        if mel_cache is not None:
            mel_transform = mel_cache.get(key)
            if mel_transform is None:
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=key[0],
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    n_mels=key[1],
                    power=2.0,
                )
                mel_cache[key] = mel_transform
        extra = build_mel_stats(audio, sr, prompt_seconds, n_mels, mel_transform)
    cond = build_timbre_condition(normalize_latent(latent), prompt_frames, extra)
    np.save(out, cond.astype(np.float16))
    return "done"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/processed")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--clip_list", default=None)
    p.add_argument("--output_name", default="timbre_cond.npy")
    p.add_argument("--prompt_seconds", type=float, default=3.0)
    p.add_argument("--latent_hz", type=float, default=12.5)
    p.add_argument("--include_mel_stats", action="store_true")
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = Path(args.data_root)
    prompt_frames = max(1, int(round(args.prompt_seconds * args.latent_hz)))
    counts = {"done": 0, "skip": 0, "missing": 0, "error": 0}
    mel_cache = {} if args.include_mel_stats else None
    for clip in iter_clips(root, args.split, args.clip_list):
        try:
            status = extract_clip(
                clip,
                args.output_name,
                prompt_frames,
                args.prompt_seconds,
                args.n_mels,
                args.include_mel_stats,
                args.force,
                mel_cache,
            )
        except Exception as exc:
            counts["error"] += 1
            print(f"[ERR] {clip}: {exc}", flush=True)
            continue
        counts[status] += 1
    print(
        f"Done: {counts['done']}  Skip: {counts['skip']}  "
        f"Missing: {counts['missing']}  Err: {counts['error']}"
    )


if __name__ == "__main__":
    main()
