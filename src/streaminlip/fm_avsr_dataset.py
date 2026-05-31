"""
Minimal dataset for FM head training with Auto-AVSR features.
Loads pre-extracted: avsr_enc.npy + avsr_text.txt + latent.npz + speaker_emb.npy
No heavy preprocessing — just numpy loads.
"""
import csv, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset

_MAX_TA = 400  # ~32s at 12.5Hz; clips beyond this waste memory due to padding
_MAX_L  = 256  # max SmolLM2 tokens; ~30s speech at typical speaking rate

# Per-dim latent normalization stats (computed from training set)
_lat_mean: np.ndarray | None = None
_lat_std:  np.ndarray | None = None
_norm_stats_path: Path | None = None


def set_norm_stats_path(path: str | Path):
    global _lat_mean, _lat_std, _norm_stats_path
    new_path = Path(path)
    if _norm_stats_path != new_path:
        _lat_mean = None
        _lat_std = None
        _norm_stats_path = new_path


def _load_norm_stats():
    global _lat_mean, _lat_std
    if _norm_stats_path is None:
        set_norm_stats_path(Path(__file__).parent.parent.parent / "data/processed/latent_norm_stats.npz")
    if _lat_mean is None and _norm_stats_path is not None and _norm_stats_path.exists():
        stats = np.load(str(_norm_stats_path))
        _lat_mean = stats["mean"].astype("float32")   # (512,)
        _lat_std  = stats["std"].astype("float32")    # (512,)

def normalize_latent(lat: np.ndarray) -> np.ndarray:
    _load_norm_stats()
    if _lat_mean is None:
        return lat
    return (lat - _lat_mean) / _lat_std

def denormalize_latent(lat: np.ndarray) -> np.ndarray:
    _load_norm_stats()
    if _lat_mean is None:
        return lat
    return lat * _lat_std + _lat_mean


def validate_latent_frame_rate(
    lat: np.ndarray,
    enc_len: int,
    clip: str | Path | None = None,
) -> np.ndarray:
    """Require Mimi latents to be the 12.5Hz post-downsample representation."""
    ratio = enc_len / max(lat.shape[0], 1)
    if 0.75 <= ratio <= 1.25:
        where = f" for {clip}" if clip is not None else ""
        raise ValueError(
            f"Found likely 25Hz Mimi latent{where}: enc_len={enc_len}, "
            f"latent_len={lat.shape[0]}, ratio={ratio:.3f}. Re-extract "
            "latent.npz with current Mimi encode/downsample pipeline; do not "
            "fix this by taking lat[::2]."
        )
    if not (1.5 <= ratio <= 2.5):
        where = f" for {clip}" if clip is not None else ""
        raise ValueError(
            f"Unexpected Mimi latent frame rate{where}: enc_len={enc_len}, "
            f"latent_len={lat.shape[0]}, ratio={ratio:.3f}. Expected "
            "enc_len / latent_len around 2.0 for 25Hz video and 12.5Hz Mimi."
        )
    return lat


class FMAVSRDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        split:          str = "pretrain",
        subset:         str = "train",
        test_reserve:   int = 2000,
        limit:          int | None = None,
        clip_list:      str | None = None,
    ):
        root = Path(processed_root)
        set_norm_stats_path(root / "latent_norm_stats.npz")
        if clip_list:
            rels = [line.strip() for line in Path(clip_list).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
            clips = [root / rel for rel in rels]
        else:
            clips = [root / r["path"]
                     for r in csv.DictReader(open(root / "manifest.csv"))
                     if r["split"] == split]

            # Use cache to avoid slow NFS stat per clip
            cache_file = root / f"_fm_avsr_{split}.txt"
            if cache_file.exists():
                valid = set(cache_file.read_text().split())
                clips = [c for c in clips if str(c.relative_to(root)) in valid]
            else:
                clips = [c for c in clips
                         if (c/"avsr_enc.npy").exists()
                         and (c/"avsr_text.txt").exists()
                         and (c/"latent.npz").exists()
                         and (c/"speaker_emb.npy").exists()]
                cache_file.write_text("\n".join(str(c.relative_to(root)) for c in clips))

            if split == "pretrain" and test_reserve > 0:
                clips = clips[:-test_reserve] if subset == "train" else clips[-test_reserve:]

        if limit:
            clips = clips[:limit]
        self.clips = clips
        print(f"[FMAVSRDataset] split={split}/{subset}  clips={len(self.clips)}")

    def __len__(self): return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        # Return numpy arrays — workers pass them via pickle (no shm),
        # collate_fn converts to tensors in main process.
        enc = np.load(str(c/"avsr_enc.npy")).astype("float32")  # (T, 768)
        lat = np.load(str(c/"latent.npz"))["latent"].astype("float32")  # (T_a, 512)
        lat = validate_latent_frame_rate(lat, enc.shape[0], c)
        # Truncate long clips to cap batch padding overhead (p99 T_a=868, max=3390)
        if lat.shape[0] > _MAX_TA:
            lat = lat[:_MAX_TA]
            enc = enc[:_MAX_TA * 2]
        lat = normalize_latent(lat)
        spk = np.load(str(c/"speaker_emb.npy")).astype("float32")  # (256,)
        # Pre-extracted SmolLM2 hidden states (L, 960) float16 — fast NFS load
        h_path = c / "smollm2_h.npy"
        h_lm = np.load(str(h_path)).astype("float32") if h_path.exists() else None
        if h_lm is not None and h_lm.shape[0] > _MAX_L:
            h_lm = h_lm[:_MAX_L]
        txt = (c/"avsr_text.txt").read_text().strip()
        return {"enc": enc, "latent": lat, "speaker": spk, "h_lm": h_lm, "text": txt}


def collate_fn(batch):
    """Returns numpy arrays — no shm needed, workers pass via pickle."""
    max_T  = max(b["enc"].shape[0] for b in batch)
    max_Ta = max(b["latent"].shape[0] for b in batch)
    B = len(batch)
    enc    = np.zeros((B, max_T, 768),  dtype=np.float32)
    latent = np.zeros((B, max_Ta, 512), dtype=np.float32)
    spk    = np.stack([b["speaker"] for b in batch])   # (B, 256)
    texts  = [b["text"] for b in batch]
    # h_lm: (B, max_L, 960) — None if not yet extracted
    has_h_lm = all(b["h_lm"] is not None for b in batch)
    if has_h_lm:
        max_L = max(b["h_lm"].shape[0] for b in batch)
        h_lm  = np.zeros((B, max_L, 960), dtype=np.float32)
        lens_L = np.array([b["h_lm"].shape[0] for b in batch])
        for i, b in enumerate(batch):
            L = b["h_lm"].shape[0]
            h_lm[i, :L] = b["h_lm"]
    else:
        h_lm   = None
        lens_L = None
    for i, b in enumerate(batch):
        T  = b["enc"].shape[0]
        Ta = b["latent"].shape[0]
        enc[i, :T]     = b["enc"]
        latent[i, :Ta] = b["latent"]
    return {"enc": enc, "latent": latent, "speaker": spk,
            "h_lm": h_lm, "lens_L": lens_L, "texts": texts}
