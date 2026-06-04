import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
import scipy.signal
from tqdm import tqdm
from transformers import MimiModel


MIMI_SR = 24000


def read_clip_list(path: str | Path, data_root: str | Path | None = None, n: int = 0) -> list[Path]:
    base = Path(data_root) if data_root is not None else None
    clips: list[Path] = []
    for line in Path(path).read_text().splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        clip = Path(item)
        if not clip.is_absolute() and base is not None:
            clip = base / clip
        clips.append(clip)
        if n and len(clips) >= n:
            break
    return clips


def cache_path_for_clip(cache_root: str | Path, clip: str | Path) -> Path:
    parts = Path(clip).parts[-3:]
    return Path(cache_root).joinpath(*parts) / "mimi_codes.npz"


def load_wav(path: str | Path, target_sr: int = MIMI_SR) -> torch.Tensor:
    sr, wav_np = wavfile.read(str(path))
    wav_f = wav_np.astype(np.float32)
    if np.issubdtype(wav_np.dtype, np.integer):
        wav_f = wav_f / float(np.iinfo(wav_np.dtype).max)
    if wav_f.ndim > 1:
        wav_f = wav_f.mean(axis=1)
    if sr != target_sr:
        n_out = int(len(wav_f) * target_sr / sr)
        wav_f = scipy.signal.resample(wav_f, n_out).astype(np.float32)
    return torch.from_numpy(wav_f).unsqueeze(0).unsqueeze(0)


def build_mimi(mimi_path: str | Path, device: str):
    mimi = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(device).eval()
    for p in mimi.parameters():
        p.requires_grad_(False)
    return mimi


@torch.no_grad()
def encode_clip(mimi, clip: Path, out_npz: Path, device: str) -> dict:
    wav = load_wav(clip / "audio.wav").to(device)
    codes = mimi.encode(wav).audio_codes.cpu().numpy().astype(np.int64)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, codes=codes)
    return {
        "clip": str(clip),
        "mimi_codes": str(out_npz),
        "codes_shape": list(codes.shape),
    }


def build_code_cache(
    clip_list: str | Path,
    cache_root: str | Path,
    data_root: str | Path | None = None,
    mimi_path: str | Path = "pretrained/mimi",
    n: int = 0,
    device: str | None = None,
    force: bool = False,
) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clips = read_clip_list(clip_list, data_root=data_root, n=n)
    mimi = build_mimi(mimi_path, device)
    rows: list[dict] = []
    manifest = Path(cache_root) / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as f:
        for clip in tqdm(clips, desc="Mimi code cache", unit="clip", dynamic_ncols=True):
            out_npz = cache_path_for_clip(cache_root, clip)
            if out_npz.exists() and not force:
                shape = list(np.load(out_npz)["codes"].shape)
                row = {"clip": str(clip), "mimi_codes": str(out_npz), "codes_shape": shape, "cached": True}
            else:
                row = encode_clip(mimi, clip, out_npz, device)
                row["cached"] = False
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_list", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--data_root", default=None)
    p.add_argument("--mimi_path", default="pretrained/mimi")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = build_code_cache(**vars(args))
    print(f"cached {len(rows)} clips")


if __name__ == "__main__":
    main()
