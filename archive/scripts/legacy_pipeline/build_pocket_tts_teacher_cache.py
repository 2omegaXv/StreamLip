#!/usr/bin/env python3
"""Build a small Pocket TTS teacher cache for FM/codec distillation tests."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from scipy.io import wavfile
from transformers import MimiModel


TTFn = Callable[[str, str, Path], tuple[int, int]]
MimiFn = Callable[[Path, Path], tuple[int, ...]]


def read_clip_list(path: str | Path, n: int = 0, data_root: str | Path | None = None) -> list[Path]:
    data_root = Path(data_root) if data_root is not None else None
    clips = []
    with Path(path).open() as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            clip = Path(item)
            if not clip.is_absolute() and data_root is not None:
                clip = data_root / clip
            clips.append(clip)
            if n > 0 and len(clips) >= n:
                break
    return clips


def cache_dir_for_clip(cache_root: Path, clip: Path) -> Path:
    """Use the last 3 path parts to keep cache paths readable and mostly unique."""
    parts = clip.parts[-3:] if len(clip.parts) >= 3 else clip.parts
    return cache_root.joinpath(*parts)


def read_clip_text(clip: Path) -> str:
    text_json = clip / "text.json"
    if text_json.exists():
        meta = json.loads(text_json.read_text())
        words = []
        for item in meta.get("words", []):
            word = str(item.get("word", "")).strip()
            if word:
                words.append(word)
        text = " ".join(words).strip()
        if text:
            return re.sub(r"\s+", " ", text)
    avsr_text = clip / "avsr_text.txt"
    if avsr_text.exists():
        return re.sub(r"\s+", " ", avsr_text.read_text()).strip()
    return ""


def default_tts_fn(text: str, voice: str, out_wav: Path) -> tuple[int, int]:
    from pocket_tts import TTSModel

    if not hasattr(default_tts_fn, "_model"):
        default_tts_fn._model = TTSModel.load_model(language="english")  # type: ignore[attr-defined]
        default_tts_fn._voice_states = {}  # type: ignore[attr-defined]
    model = default_tts_fn._model  # type: ignore[attr-defined]
    voice_states = default_tts_fn._voice_states  # type: ignore[attr-defined]
    if voice not in voice_states:
        voice_states[voice] = model.get_state_for_audio_prompt(voice)
    audio = model.generate_audio(voice_states[voice], text)
    audio_np = audio.detach().cpu().numpy()
    wavfile.write(out_wav, model.sample_rate, audio_np)
    return int(model.sample_rate), int(audio_np.shape[0])


def make_mimi_fn(mimi_path: str | Path, device: str | None = None) -> MimiFn:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mimi = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(device).eval()

    def encode(wav_path: Path, out_npz: Path) -> tuple[int, ...]:
        audio_np, sr = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        if sr != 24000:
            raise ValueError(f"Expected 24kHz teacher audio, got {sr} for {wav_path}")
        with torch.no_grad():
            x = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
            codes = mimi.encode(x).audio_codes.cpu().numpy()
        np.savez(out_npz, codes=codes)
        return tuple(int(v) for v in codes.shape)

    return encode


def build_teacher_cache(
    clip_list: str | Path,
    cache_root: str | Path,
    data_root: str | Path | None = None,
    n: int = 0,
    voice: str = "alba",
    tts_fn: TTFn = default_tts_fn,
    mimi_fn: MimiFn | None = None,
) -> list[dict]:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.jsonl"
    rows = []
    for clip in read_clip_list(clip_list, n=n, data_root=data_root):
        text = read_clip_text(clip)
        if not text:
            rows.append({"clip": str(clip), "skipped": True, "reason": "empty_text"})
            continue
        item_dir = cache_dir_for_clip(cache_root, clip)
        item_dir.mkdir(parents=True, exist_ok=True)
        teacher_wav = item_dir / "teacher.wav"
        mimi_codes = item_dir / "mimi_codes.npz"

        generated = False
        if not teacher_wav.exists() or not mimi_codes.exists():
            sr, samples = tts_fn(text, voice, teacher_wav)
            if mimi_fn is None:
                raise ValueError("mimi_fn is required when cache is missing")
            codes_shape = mimi_fn(teacher_wav, mimi_codes)
            generated = True
        else:
            audio, sr = sf.read(str(teacher_wav), dtype="float32")
            samples = int(audio.shape[0])
            codes_shape = tuple(int(v) for v in np.load(mimi_codes)["codes"].shape)

        row = {
            "clip": str(clip),
            "text": text,
            "voice": voice,
            "teacher_wav": str(teacher_wav),
            "mimi_codes": str(mimi_codes),
            "sample_rate": int(sr),
            "samples": int(samples),
            "duration_sec": float(samples / sr),
            "codes_shape": list(codes_shape),
            "generated": bool(generated),
        }
        rows.append(row)
        with manifest_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_list", required=True)
    p.add_argument("--data_root", default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/data/processed")
    p.add_argument("--cache_root", default="data/teacher_cache/pocket_tts")
    p.add_argument("--mimi_path", default="/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/pretrained/mimi")
    p.add_argument("--voice", default="alba")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    mimi_fn = make_mimi_fn(args.mimi_path, args.device)
    rows = build_teacher_cache(
        clip_list=args.clip_list,
        cache_root=args.cache_root,
        data_root=args.data_root,
        n=args.n,
        voice=args.voice,
        mimi_fn=mimi_fn,
    )
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
