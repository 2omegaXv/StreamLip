"""
LRS3 Dataset V2 for StreamLipV2.

Per-item output:
  visual:       (T, 768) pre-extracted AV-HuBERT features, or (T, 3, 96, 96) raw lip frames
  face:         (3, 256, 256)   float32, ImageNet normalized  (zeros if load_face=False)
  clean_ids:    (L,)            int64,  [BOS, tok1, tok2, ...] collapsed SIL-free token seq
  lm_idx_text:  (T,)            int64,  per-frame index into clean_ids for text CE loss
                                        points to the LM output BEFORE current token is committed
  lm_idx_fm:    (T,)            int64,  per-frame index into clean_ids for FM conditioning
                                        points to the LM output AFTER current token is committed
  frame_labels: (T,)            int64,  per-frame token labels (CE targets)
  mask:         (T,)            bool,   True = valid frame
  latent:       (T_a, 512)      float16 (empty tensor if load_latent=False)

LM index semantics:
  The LM runs once on clean_ids (L << T) in teacher-forcing mode → h_lm/s_lm (B, L, ...).
  Each frame gathers two positions:
    lm_idx_text[t]: s_lm[lm_idx_text[t]] predicts frame_labels[t]  (text head CE loss)
    lm_idx_fm[t]:   h_lm[lm_idx_fm[t]]  conditions FM on committed word  (FM head)
  SIL frames: both indices hold at the last committed position.
  Non-SIL frames: lm_idx_fm[t] = lm_idx_text[t] + 1.

Frame label generation (pretrain has word timestamps; others are all-SIL):
  SIL = <empty_output> = token 16
  Each word's BPE tokens are distributed across its frame span proportionally by char count.
"""
import csv
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SIL_ID     = 16   # <empty_output> in SmolLM2 tokenizer
FPS        = 25
CHUNK_SIZE = 6


class LRS3DatasetV2(Dataset):
    def __init__(
        self,
        processed_root: str,
        split:          str,
        tokenizer_path: str,
        max_frames:     int  = 150,
        load_face:      bool = False,
        load_latent:    bool = False,
        deterministic:  bool = False,
        subset:         str  = "train",   # "train" | "test" — only for pretrain split
        test_reserve:   int  = 2000,      # last N clips reserved as pretrain-test
        limit:          int | None = None,
    ):
        self.root          = Path(processed_root)
        self.load_face     = load_face
        self.load_latent   = load_latent
        self.deterministic = deterministic
        self.max_frames    = (max_frames // CHUNK_SIZE) * CHUNK_SIZE

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.bos_id    = self.tokenizer.bos_token_id or SIL_ID

        manifest = self.root / "manifest.csv"
        all_clips = []
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                all_clips.append(self.root / row["path"])

        # pretrain split: optionally carve out last N clips as held-out test
        if split == "pretrain" and test_reserve > 0:
            if subset == "test":
                all_clips = all_clips[-test_reserve:]
            else:
                all_clips = all_clips[:-test_reserve]

        if limit:
            all_clips = all_clips[:limit]
        self.clips = all_clips

        print(f"[LRS3DatasetV2] split={split}/{subset}  clips={len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def _frame_labels(self, words: list, T: int, start: int = 0) -> np.ndarray:
        """
        Assign BPE tokens to frames proportionally by character count.
        e.g. MARRIAGE → [▁MAR(37%), RIAGE(63%)] distributed across the word's frames.
        """
        labels = np.full(T, SIL_ID, dtype=np.int64)
        for w in words:
            # Space prefix + lowercase: SmolLM2 is SentencePiece-based; words in
            # running text carry a leading ▁ (space). Encoding without it fragments
            # words that are single tokens in context ('gay'→['g','ay'] vs ' gay'→[' gay']).
            # LRS3 stores all-caps, which also causes fragmentation without lowercase.
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
                f0 = max(0, int(t0 * FPS) - start)
                f1 = min(T, int(t1 * FPS) - start)
                if f0 < f1:
                    labels[f0:f1] = tok_id
                cumulative += n_chars
        return labels

    def _compute_lm_indices(
        self, frame_labels: np.ndarray, T: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the clean token sequence and per-frame LM gather indices.

        Returns:
          clean_ids   (L,)  [BOS, tok1, tok2, ...] — SIL-free, no repeats
          lm_idx_text (T,)  per-frame index for text CE: points to position BEFORE commit
          lm_idx_fm   (T,)  per-frame index for FM cond:  points to position AFTER commit
                            for non-SIL frames: lm_idx_fm = lm_idx_text + 1
                            for SIL frames:     both hold at last committed position
        """
        clean_ids    = [self.bos_id]
        lm_idx_text  = np.zeros(T, dtype=np.int64)
        lm_idx_fm    = np.zeros(T, dtype=np.int64)

        t = 0
        while t < T:
            tok = int(frame_labels[t])
            # find the end of this run
            end = t + 1
            while end < T and frame_labels[end] == tok:
                end += 1

            if tok == SIL_ID:
                # SIL: both indices stay at last committed position
                last = len(clean_ids) - 1
                lm_idx_text[t:end] = last
                lm_idx_fm[t:end]   = last
            else:
                pos_before = len(clean_ids) - 1   # LM output before this token
                pos_after  = len(clean_ids)        # LM output after this token
                clean_ids.append(tok)
                lm_idx_text[t:end] = pos_before
                lm_idx_fm[t:end]   = pos_after

            t = end

        return np.array(clean_ids, dtype=np.int64), lm_idx_text, lm_idx_fm

    def __getitem__(self, idx: int) -> dict:
        clip_dir = self.clips[idx]

        # ── visual input ──────────────────────────────────────────────────────
        # Fast path: pre-extracted AV-HuBERT features (T, 768) float16
        avhubert_cache = clip_dir / "avhubert.npy"
        if avhubert_cache.exists():
            av = np.load(str(avhubert_cache)).astype(np.float32)
            T_full = len(av)
            if T_full <= self.max_frames or self.deterministic:
                T, start = min(T_full, self.max_frames), 0
                visual = torch.from_numpy(av[:T].copy())
            else:
                start  = torch.randint(0, T_full - self.max_frames + 1, ()).item()
                T      = self.max_frames
                visual = torch.from_numpy(av[start:start + T].copy())
        else:
            lip_raw = np.load(str(clip_dir / "lip.npy"), mmap_mode="r")
            T_full  = len(lip_raw)
            if T_full <= self.max_frames or self.deterministic:
                T, start = min(T_full, self.max_frames), 0
                lip = lip_raw[:T].copy()
            else:
                start = torch.randint(0, T_full - self.max_frames + 1, ()).item()
                T     = self.max_frames
                lip   = lip_raw[start:start + T].copy()
            lip    = lip.astype(np.float32) / 255.0
            lip    = (lip - IMAGENET_MEAN) / IMAGENET_STD
            visual = torch.from_numpy(lip).permute(0, 3, 1, 2)

        # ── face (first chunk mean) ───────────────────────────────────────────
        if self.load_face:
            f    = np.load(str(clip_dir / "face.npz"))
            data, offsets = f["data"], f["offsets"]
            n    = min(CHUNK_SIZE, len(offsets) - 1)
            frames = []
            for i in range(n):
                buf = data[offsets[i]:offsets[i+1]].tobytes()
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face = np.stack(frames).mean(0).astype(np.float32) / 255.0  # (256, 256, 3)
            face = (face - IMAGENET_MEAN) / IMAGENET_STD
            face = torch.from_numpy(face).permute(2, 0, 1)              # (3, 256, 256)
        else:
            face = torch.zeros(3, 256, 256, dtype=torch.float32)

        # ── frame labels & LM indices ─────────────────────────────────────────
        meta     = json.loads((clip_dir / "text.json").read_text())
        labels   = self._frame_labels(meta.get("words", []), T, start=start)
        clean_ids, lm_idx_text, lm_idx_fm = self._compute_lm_indices(labels, T)

        # ── latent (Phase 2) ──────────────────────────────────────────────────
        if self.load_latent:
            latent = np.load(str(clip_dir / "latent.npz"))["latent"]
            ta0    = start // 2
            T_a    = T // 2
            latent = torch.from_numpy(latent[ta0:ta0 + T_a].copy())
        else:
            latent = torch.zeros(0, 512, dtype=torch.float16)

        return {
            "visual":       visual[:T],
            "face":         face,
            "clean_ids":    torch.from_numpy(clean_ids),       # (L,)
            "lm_idx_text":  torch.from_numpy(lm_idx_text),    # (T,)
            "lm_idx_fm":    torch.from_numpy(lm_idx_fm),      # (T,)
            "frame_labels": torch.from_numpy(labels),         # (T,)
            "mask":         torch.ones(T, dtype=torch.bool),
            "latent":       latent,
        }


def collate_fn(batch: list[dict]) -> dict:
    max_T  = max(b["visual"].shape[0] for b in batch)
    max_Ta = max(b["latent"].shape[0] for b in batch)
    max_L  = max(b["clean_ids"].shape[0] for b in batch)
    B      = len(batch)
    vis_shape = batch[0]["visual"].shape[1:]

    visual       = torch.zeros(B, max_T, *vis_shape)
    face         = torch.stack([b["face"] for b in batch])
    # clean_ids padded with SIL; the LM attention mask covers only valid positions
    clean_ids    = torch.full((B, max_L), SIL_ID, dtype=torch.long)
    clean_mask   = torch.zeros(B, max_L, dtype=torch.long)
    lm_idx_text  = torch.zeros(B, max_T, dtype=torch.long)
    lm_idx_fm    = torch.zeros(B, max_T, dtype=torch.long)
    frame_labels = torch.full((B, max_T), SIL_ID, dtype=torch.long)
    mask         = torch.zeros(B, max_T, dtype=torch.bool)
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
        if latent is not None and Ta > 0:
            latent[i, :Ta]   = b["latent"]

    return {
        "visual": visual, "face": face,
        "clean_ids": clean_ids, "clean_mask": clean_mask,
        "lm_idx_text": lm_idx_text, "lm_idx_fm": lm_idx_fm,
        "frame_labels": frame_labels, "mask": mask, "latent": latent,
    }
