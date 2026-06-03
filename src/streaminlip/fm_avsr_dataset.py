"""
Minimal dataset for FM head training with Auto-AVSR features.
Loads pre-extracted: avsr_enc.npy + avsr_text.txt + latent.npz + speaker_emb.npy
No heavy preprocessing — just numpy loads.
"""
import csv, json, re, difflib, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import wavfile

_MAX_TA = 400  # ~32s at 12.5Hz; clips beyond this waste memory due to padding
_MAX_L  = 256  # max SmolLM2 tokens; ~30s speech at typical speaking rate
_LATENT_HZ = 12.5

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


def compute_log_rms_energy(audio: np.ndarray, n_frames: int, eps: float = 1e-5) -> np.ndarray:
    """Compute per-latent-frame log RMS energy from mono waveform samples."""
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    edges = np.linspace(0, len(audio), n_frames + 1).round().astype(np.int64)
    out = np.zeros((n_frames,), dtype=np.float32)
    for i in range(n_frames):
        start, end = int(edges[i]), int(edges[i + 1])
        frame = audio[start:end]
        if len(frame) == 0:
            out[i] = np.log(eps)
        else:
            out[i] = np.log(float(np.sqrt(np.mean(frame * frame))) + eps)
    return out


def load_log_rms_energy(clip: str | Path, n_frames: int) -> np.ndarray:
    """Load audio.wav and return (n_frames, 1) log-RMS energy."""
    sr, wav = wavfile.read(str(Path(clip) / "audio.wav"))
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / float(np.iinfo(wav.dtype).max)
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return compute_log_rms_energy(wav, n_frames)[:, None]


def _norm_word(word: str) -> str:
    word = re.sub(r"[^a-z0-9]", "", word.lower())
    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]
    return word


def read_clip_text(clip: str | Path, text_source: str = "avsr") -> str:
    clip = Path(clip)
    if text_source == "text_json":
        meta_path = clip / "text.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            words = [
                str(w.get("word", "")).strip()
                for w in meta.get("words", [])
                if str(w.get("word", "")).strip()
            ]
            text = " ".join(words).strip()
            if text:
                return re.sub(r"\s+", " ", text)
    if text_source != "avsr":
        raise ValueError(f"unknown text_source={text_source!r}")
    return re.sub(r"\s+", " ", (clip / "avsr_text.txt").read_text()).strip()


def smollm2_hidden_path(clip: str | Path, text_source: str = "avsr") -> Path:
    name = "smollm2_h_text_json.npy" if text_source == "text_json" else "smollm2_h.npy"
    return Path(clip) / name


def _assign_text_word_times(text_words: list[str], words: list[dict]) -> list[tuple[float, float]]:
    """Align AVSR transcript words to timestamped words, interpolating unmatched spans."""
    timed_norm = [_norm_word(w.get("word", "")) for w in words]
    text_norm = [_norm_word(w) for w in text_words]
    assigned: list[tuple[float, float] | None] = [None] * len(text_words)

    matcher = difflib.SequenceMatcher(a=text_norm, b=timed_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                w = words[j1 + off]
                assigned[i1 + off] = (float(w["start"]), float(w["end"]))
            continue
        if i1 == i2:
            continue
        if j1 < j2:
            start = float(words[j1]["start"])
            end = float(words[j2 - 1]["end"])
        else:
            start = float(words[j1 - 1]["end"]) if j1 > 0 else 0.0
            end = float(words[j1]["start"]) if j1 < len(words) else start
        span = max(end - start, 1e-4)
        n = i2 - i1
        for k in range(n):
            t0 = start + span * k / n
            t1 = start + span * (k + 1) / n
            assigned[i1 + k] = (t0, t1)

    last_end = 0.0
    for i, cur in enumerate(assigned):
        if cur is not None:
            last_end = cur[1]
            continue
        next_start = None
        for nxt in assigned[i + 1:]:
            if nxt is not None:
                next_start = nxt[0]
                break
        end = next_start if next_start is not None else last_end
        assigned[i] = (last_end, max(last_end, end))
    return [(float(s), float(e)) for s, e in assigned]


def build_word_timestamp_lm_indices(
    text: str,
    words: list[dict],
    tokenizer,
    T_a: int,
    latent_hz: float = _LATENT_HZ,
) -> np.ndarray:
    """Map each latent frame to the committed SmolLM2 hidden-state index.

    `smollm2_h.npy` is extracted as [BOS, token1, token2, ...] from
    `avsr_text.txt`. This function uses `text.json` word timestamps to choose
    which hidden-state position should condition each audio-latent frame.
    Silence and gaps hold the last committed token.
    """
    idx = np.zeros(T_a, dtype=np.int64)
    text_words = text.upper().split()
    if T_a <= 0 or not text_words or tokenizer is None:
        return idx

    word_times = _assign_text_word_times(text_words, words or [])
    pos = 1  # 0 is BOS
    for word, (start, end) in zip(text_words, word_times):
        toks = tokenizer.encode(" " + word.lower(), add_special_tokens=False)
        if not toks:
            continue
        dur = max(end - start, 1e-4)
        tok_text = [tokenizer.decode([t]).strip() for t in toks]
        weights = [max(len(t), 1) for t in tok_text]
        total = sum(weights)
        cursor = 0.0
        for n_chars in weights:
            t0 = start + dur * cursor / total
            t1 = start + dur * (cursor + n_chars) / total
            f0 = max(0, min(T_a, int(np.floor(t0 * latent_hz))))
            f1 = max(f0 + 1, min(T_a, int(np.ceil(t1 * latent_hz))))
            idx[f0:f1] = pos
            pos += 1
            cursor += n_chars

    last = 0
    for i, value in enumerate(idx):
        if value > 0:
            last = int(value)
        else:
            idx[i] = last
    return idx


class FMAVSRDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        split:          str = "pretrain",
        subset:         str = "train",
        test_reserve:   int = 2000,
        limit:          int | None = None,
        clip_list:      str | None = None,
        tokenizer=None,
        text_alignment_mode: str = "uniform",
        text_source: str = "avsr",
        visual_feature_name: str = "avsr_enc.npy",
        load_energy: bool = False,
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
        self.tokenizer = tokenizer
        self.text_alignment_mode = text_alignment_mode
        self.text_source = text_source
        self.visual_feature_name = visual_feature_name
        self.load_energy = load_energy
        print(f"[FMAVSRDataset] split={split}/{subset}  clips={len(self.clips)}")

    def __len__(self): return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        # Return numpy arrays — workers pass them via pickle (no shm),
        # collate_fn converts to tensors in main process.
        enc = np.load(str(c / self.visual_feature_name)).astype("float32")  # (T, 768)
        lat = np.load(str(c/"latent.npz"))["latent"].astype("float32")  # (T_a, 512)
        lat = validate_latent_frame_rate(lat, enc.shape[0], c)
        # Truncate long clips to cap batch padding overhead (p99 T_a=868, max=3390)
        if lat.shape[0] > _MAX_TA:
            lat = lat[:_MAX_TA]
            enc = enc[:_MAX_TA * 2]
        lat = normalize_latent(lat)
        spk = np.load(str(c/"speaker_emb.npy")).astype("float32")  # (256,)
        energy = (
            load_log_rms_energy(c, lat.shape[0]).astype("float32")
            if self.load_energy else None
        )
        # Pre-extracted SmolLM2 hidden states (L, 960) float16 — fast NFS load
        h_path = smollm2_hidden_path(c, self.text_source)
        h_lm = np.load(str(h_path)).astype("float32") if h_path.exists() else None
        if h_lm is not None and h_lm.shape[0] > _MAX_L:
            h_lm = h_lm[:_MAX_L]
        txt = read_clip_text(c, self.text_source)
        lm_idx = None
        if self.text_alignment_mode == "word_timestamps" and self.tokenizer is not None:
            meta = json.loads((c / "text.json").read_text())
            lm_idx = build_word_timestamp_lm_indices(
                txt, meta.get("words", []), self.tokenizer, lat.shape[0]
            )
        item = {
            "enc": enc, "latent": lat, "speaker": spk, "h_lm": h_lm,
            "text": txt, "lm_idx": lm_idx,
        }
        if energy is not None:
            item["energy"] = energy
        return item


def collate_fn(batch):
    """Returns numpy arrays — no shm needed, workers pass via pickle."""
    max_T  = max(b["enc"].shape[0] for b in batch)
    max_Ta = max(b["latent"].shape[0] for b in batch)
    B = len(batch)
    enc_lens    = np.array([b["enc"].shape[0] for b in batch], dtype=np.int64)
    latent_lens = np.array([b["latent"].shape[0] for b in batch], dtype=np.int64)
    enc    = np.zeros((B, max_T, 768),  dtype=np.float32)
    latent = np.zeros((B, max_Ta, 512), dtype=np.float32)
    has_energy = all(b.get("energy") is not None for b in batch)
    energy = np.zeros((B, max_Ta, 1), dtype=np.float32) if has_energy else None
    spk    = np.stack([b["speaker"] for b in batch])   # (B, 256)
    texts  = [b["text"] for b in batch]
    has_lm_idx = all(b.get("lm_idx") is not None for b in batch)
    lm_idx = np.zeros((B, max_Ta), dtype=np.int64) if has_lm_idx else None
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
        if has_energy:
            energy[i, :Ta] = b["energy"]
        if has_lm_idx:
            lm_idx[i, :Ta] = b["lm_idx"][:Ta]
    out = {"enc": enc, "latent": latent, "speaker": spk,
            "enc_lens": enc_lens, "latent_lens": latent_lens,
            "h_lm": h_lm, "lens_L": lens_L, "texts": texts,
            "lm_idx": lm_idx}
    if has_energy:
        out["energy"] = energy
    return out
