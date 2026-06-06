"""
LRS3 Dataset V4 for StreamLipV4.

Fast path: if avhubert_pre.npy exists, returns (T, 768) float16 pre-extracted
CNN frontend features — AV-HuBERT Transformer + LoRA runs online inside the model.

Slow path: if avhubert_pre.npy is missing, returns raw lip frames (T, 3, 96, 96).
The model's visual encoder handles both input shapes transparently.

T is snapped to a chunk-aligned prefix: T = k * CHUNK_SIZE, k ~ Uniform[min_k, max_k].
start is always 0 (prefix from beginning), matching streaming re-encode at inference.
"""
import csv
import json
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SIL_ID     = 16    # <empty_output> in SmolLM2 tokenizer
FPS        = 25
CHUNK_SIZE = 6     # 240ms @ 25fps
MIN_CHUNKS = 4     # minimum 24 frames = 960ms (avoid very short clips)

# CTC character vocabulary: a-z (0-25), space (26), blank (27)
CHARS    = "abcdefghijklmnopqrstuvwxyz "
BLANK_ID = 27


class LRS3DatasetV4(Dataset):
    """
    LRS3 dataset with chunk-aligned prefix sampling.

    Fast path (avhubert_pre.npy exists): visual = (T, 768) float16
    Slow path (raw frames only):         visual = (T, 3, 96, 96) float32
    Only clips with avhubert_pre.npy are loaded when pre_only=True.
    """

    def __init__(
        self,
        processed_root: str,
        split:          str,
        tokenizer_path: str,
        max_frames:     int  = 150,
        min_chunks:     int  = MIN_CHUNKS,
        load_face:      bool = False,
        load_latent:    bool = False,
        deterministic:  bool = False,
        subset:         str  = "train",
        test_reserve:   int  = 2000,
        limit:          int | None = None,
        pre_only:       bool = True,   # only load clips with avhubert_pre.npy
    ):
        self.root          = Path(processed_root)
        self.load_face     = load_face
        self.load_latent   = load_latent
        self.deterministic = deterministic
        self.max_chunks    = (max_frames // CHUNK_SIZE)
        self.min_chunks    = min_chunks

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.bos_id    = self.tokenizer.bos_token_id or SIL_ID

        manifest = self.root / "manifest.csv"
        all_clips = []
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                all_clips.append(self.root / row["path"])

        # pre_only filter BEFORE test_reserve split so cache covers full split
        if pre_only:
            cache = self.root / f"_pre_only_{split}.txt"
            if cache.exists():
                valid = set(cache.read_text().split())
                all_clips = [c for c in all_clips
                             if str(c.relative_to(self.root)) in valid]
            else:
                all_clips = [c for c in all_clips
                             if (c / "avhubert_pre.npy").exists()]
                cache.write_text(
                    "\n".join(str(c.relative_to(self.root)) for c in all_clips)
                )

        if split == "pretrain" and test_reserve > 0:
            if subset == "test":
                all_clips = all_clips[-test_reserve:]
            else:
                all_clips = all_clips[:-test_reserve]

        if limit:
            all_clips = all_clips[:limit]
        self.clips = all_clips
        print(f"[LRS3DatasetV4] split={split}/{subset}  clips={len(self.clips)}"
              + (" (pre_only)" if pre_only else ""))

    def __len__(self):
        return len(self.clips)

    def _char_ids(self, words: list, T: int) -> np.ndarray:
        """Character sequence for CTC: words whose start < T/FPS, a-z + space only."""
        t_max = T / FPS
        chars = []
        for w in words:
            if w["start"] >= t_max:
                break
            text = w["word"].lower().strip()
            if chars:
                chars.append(26)  # space between words
            for c in text:
                idx = CHARS.find(c)
                if idx >= 0:
                    chars.append(idx)
        return np.array(chars, dtype=np.int32) if chars else np.zeros(1, dtype=np.int32)

    def _frame_labels(self, words: list, T: int) -> np.ndarray:
        """Assign BPE tokens to frames proportionally by character count (same as V2)."""
        labels = np.full(T, SIL_ID, dtype=np.int64)
        for w in words:
            toks = self.tokenizer.encode(' ' + w["word"].lower(), add_special_tokens=False)
            if not toks:
                continue
            dur         = w["end"] - w["start"]
            tok_strs    = [self.tokenizer.decode([t]).strip() for t in toks]
            char_counts = [max(len(s), 1) for s in tok_strs]
            total_chars = sum(char_counts)
            cumulative  = 0
            for tok_id, n_chars in zip(toks, char_counts):
                t0 = w["start"] + dur * cumulative / total_chars
                t1 = t0 + dur * n_chars / total_chars
                f0 = max(0, int(t0 * FPS))
                f1 = min(T, int(t1 * FPS))
                if f0 < f1:
                    labels[f0:f1] = tok_id
                cumulative += n_chars
        return labels

    def _compute_lm_indices(
        self, frame_labels: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Same as V2: build clean token sequence and per-frame LM gather indices."""
        clean_ids   = [self.bos_id]
        lm_idx_text = np.zeros(T, dtype=np.int64)
        lm_idx_fm   = np.zeros(T, dtype=np.int64)
        t = 0
        while t < T:
            tok = int(frame_labels[t])
            end = t + 1
            while end < T and frame_labels[end] == tok:
                end += 1
            if tok == SIL_ID:
                last = len(clean_ids) - 1
                lm_idx_text[t:end] = last
                lm_idx_fm[t:end]   = last
            else:
                pos_before = len(clean_ids) - 1
                pos_after  = len(clean_ids)
                clean_ids.append(tok)
                lm_idx_text[t:end] = pos_before
                lm_idx_fm[t:end]   = pos_after
            t = end
        return np.array(clean_ids, dtype=np.int64), lm_idx_text, lm_idx_fm

    def __getitem__(self, idx: int) -> dict:
        clip_dir = self.clips[idx]

        # ── Load metadata first so we can bias k sampling ────────────────────
        meta   = json.loads((clip_dir / "text.json").read_text())
        words  = meta.get("words", [])

        # ── Visual input: fast path (pre-extracted) or slow path (raw frames) ──
        pre_path = clip_dir / "avhubert_pre.npy"
        if pre_path.exists():
            pre_full = np.load(str(pre_path), mmap_mode="r")
            T_full   = len(pre_full)
        else:
            lip_raw = np.load(str(clip_dir / "lip.npy"), mmap_mode="r")
            T_full  = len(lip_raw)

        max_k = min(self.max_chunks, T_full // CHUNK_SIZE)
        min_k = min(self.min_chunks, max_k)

        if self.deterministic or min_k >= max_k:
            k = max_k
        else:
            # Bias: prefer k values where the last chunk contains a non-SIL frame.
            # Compute full-length frame labels once (cheap, numpy only).
            labels_full = self._frame_labels(words, max_k * CHUNK_SIZE)
            good_ks = [
                ki for ki in range(min_k, max_k + 1)
                if np.any(labels_full[(ki-1)*CHUNK_SIZE : ki*CHUNK_SIZE] != SIL_ID)
            ]
            if good_ks:
                k = random.choice(good_ks)
            else:
                k = random.randint(min_k, max_k)

        T = k * CHUNK_SIZE

        if pre_path.exists():
            visual = torch.from_numpy(pre_full[:T].copy())       # (T, 768) float16
        else:
            lip    = lip_raw[:T].copy().astype(np.float32) / 255.0
            lip    = (lip - IMAGENET_MEAN) / IMAGENET_STD
            visual = torch.from_numpy(lip).permute(0, 3, 1, 2)   # (T, 3, 96, 96)

        # ── Face / speaker embedding ──────────────────────────────────────────
        if self.load_face:
            emb_path = clip_dir / "speaker_emb.npy"
            if emb_path.exists():
                # Fast path: pre-extracted 256-d speaker embedding (float16)
                face = torch.from_numpy(np.load(str(emb_path)).copy())  # (256,)
            else:
                # Fallback: zeros until speaker_emb.npy is extracted
                face = torch.zeros(256, dtype=torch.float16)
        else:
            face = torch.zeros(3, 256, 256, dtype=torch.float32)

        # ── Frame labels & LM indices (prefix only, start=0) ─────────────────
        labels   = self._frame_labels(words, T)
        clean_ids, lm_idx_text, lm_idx_fm = self._compute_lm_indices(labels, T)
        ctc_ids  = self._char_ids(words, T)

        # ── Latent (Phase 2, prefix only) ─────────────────────────────────────
        if self.load_latent:
            latent = np.load(str(clip_dir / "latent.npz"))["latent"]
            T_a    = T // 2
            latent = torch.from_numpy(latent[:T_a].copy())
        else:
            latent = torch.zeros(0, 512, dtype=torch.float16)

        return {
            "visual":       visual,                              # (T, 3, 96, 96)
            "face":         face,                                # (3, 256, 256)
            "clean_ids":    torch.from_numpy(clean_ids),        # (L,)
            "lm_idx_text":  torch.from_numpy(lm_idx_text),     # (T,)
            "lm_idx_fm":    torch.from_numpy(lm_idx_fm),       # (T,)
            "frame_labels": torch.from_numpy(labels),          # (T,)
            "mask":         torch.ones(T, dtype=torch.bool),
            "ctc_ids":      torch.from_numpy(ctc_ids),         # (C,) variable length
            "latent":       latent,
        }


def collate_fn(batch: list[dict]) -> dict:
    max_T   = max(b["visual"].shape[0] for b in batch)
    max_Ta  = max(b["latent"].shape[0] for b in batch)
    max_L   = max(b["clean_ids"].shape[0] for b in batch)
    max_C   = max(b["ctc_ids"].shape[0] for b in batch)
    B       = len(batch)

    # visual may be (T, 768) float16 [pre-extracted] or (T, 3, 96, 96) float32 [raw]
    vis0       = batch[0]["visual"]
    vis_shape  = vis0.shape[1:]   # (768,) or (3, 96, 96)
    vis_dtype  = vis0.dtype
    visual     = torch.zeros(B, max_T, *vis_shape, dtype=vis_dtype)

    face         = torch.stack([b["face"] for b in batch])
    clean_ids    = torch.full((B, max_L), SIL_ID, dtype=torch.long)
    clean_mask   = torch.zeros(B, max_L, dtype=torch.long)
    lm_idx_text  = torch.zeros(B, max_T, dtype=torch.long)
    lm_idx_fm    = torch.zeros(B, max_T, dtype=torch.long)
    frame_labels = torch.full((B, max_T), SIL_ID, dtype=torch.long)
    mask         = torch.zeros(B, max_T, dtype=torch.bool)
    ctc_ids      = torch.zeros(B, max_C, dtype=torch.long)
    ctc_lens     = torch.zeros(B, dtype=torch.long)
    latent       = (torch.zeros(B, max_Ta, 512, dtype=torch.float16)
                    if max_Ta > 0 else None)

    for i, b in enumerate(batch):
        T  = b["visual"].shape[0]
        L  = b["clean_ids"].shape[0]
        Ta = b["latent"].shape[0]
        visual[i, :T]        = b["visual"]
        clean_ids[i, :L]     = b["clean_ids"]
        clean_mask[i, :L]    = 1
        lm_idx_text[i, :T]   = b["lm_idx_text"]
        lm_idx_fm[i, :T]     = b["lm_idx_fm"]
        frame_labels[i, :T]  = b["frame_labels"]
        mask[i, :T]          = b["mask"]
        C = b["ctc_ids"].shape[0]
        ctc_ids[i, :C]       = b["ctc_ids"]
        ctc_lens[i]          = C
        if latent is not None and Ta > 0:
            latent[i, :Ta]   = b["latent"]

    return {
        "visual": visual, "face": face,
        "clean_ids": clean_ids, "clean_mask": clean_mask,
        "lm_idx_text": lm_idx_text, "lm_idx_fm": lm_idx_fm,
        "frame_labels": frame_labels, "mask": mask,
        "ctc_ids": ctc_ids, "ctc_lens": ctc_lens,
        "latent": latent,
    }
