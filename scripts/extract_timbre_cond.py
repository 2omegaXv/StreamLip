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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from streaminlip.fm_avsr_dataset import normalize_latent


def build_timbre_condition(latent: np.ndarray, prompt_frames: int) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != 512:
        raise ValueError(f"latent must have shape (T, 512), got {latent.shape}")
    n = max(1, min(int(prompt_frames), latent.shape[0]))
    prefix = latent[:n]
    return np.concatenate(
        [prefix.mean(axis=0), prefix.std(axis=0)],
        axis=0,
    ).astype(np.float32)


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


def extract_clip(clip: Path, output_name: str, prompt_frames: int, force: bool) -> str:
    out = clip / output_name
    if out.exists() and not force:
        return "skip"
    latent_path = clip / "latent.npz"
    if not latent_path.exists():
        return "missing"
    latent = np.load(str(latent_path))["latent"].astype(np.float32)
    cond = build_timbre_condition(normalize_latent(latent), prompt_frames)
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
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = Path(args.data_root)
    prompt_frames = max(1, int(round(args.prompt_seconds * args.latent_hz)))
    counts = {"done": 0, "skip": 0, "missing": 0, "error": 0}
    for clip in iter_clips(root, args.split, args.clip_list):
        try:
            status = extract_clip(clip, args.output_name, prompt_frames, args.force)
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
