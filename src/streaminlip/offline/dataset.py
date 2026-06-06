"""
Simplified offline dataset: visual features + transcript text.

No MFA alignment, no frame_labels, no SIL tokens.
Just: avhubert_pre.npy → model K/V, transcript → next-token CE loss.
"""
import csv
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

FPS       = 25
SIL_ID    = 16   # kept for compatibility but not used in Phase 1


class LRS3DatasetOffline(Dataset):
    """
    Each item:
      visual:    (T, 768) float16  — avhubert_pre.npy CNN frontend features
      input_ids: (L,)    int64    — [BOS, tok1, ..., tokN-1]   (LM input)
      labels:    (L,)    int64    — [tok1, ..., tokN, EOS-or-pad]  (CE targets)
    """

    def __init__(
        self,
        processed_root: str,
        split:          str,
        tokenizer_path: str,
        max_frames:     int  = 150,
        max_text_len:   int  = 64,
        deterministic:  bool = False,
        subset:         str  = "train",
        test_reserve:   int  = 2000,
        limit:          int | None = None,
    ):
        self.root          = Path(processed_root)
        self.max_frames    = max_frames
        self.max_text_len  = max_text_len
        self.deterministic = deterministic

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.bos_id    = self.tokenizer.bos_token_id
        self.eos_id    = self.tokenizer.eos_token_id or self.bos_id

        # Build clip list (only those with avhubert_pre.npy)
        cache = self.root / f"_pre_only_{split}.txt"
        all_clips = []
        with open(self.root / "manifest.csv") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    all_clips.append(self.root / row["path"])

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
            all_clips = all_clips[-test_reserve:] if subset == "test" \
                        else all_clips[:-test_reserve]

        if limit:
            all_clips = all_clips[:limit]

        # Filter out clips with missing/empty transcripts
        self.clips = []
        for c in all_clips:
            try:
                meta = json.loads((c / "text.json").read_text())
                if meta.get("transcript", "").strip():
                    self.clips.append(c)
            except Exception:
                pass

        print(f"[LRS3DatasetOffline] split={split}/{subset}  clips={len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip_dir = self.clips[idx]

        # ── Visual features ───────────────────────────────────────────────────
        pre = np.load(str(clip_dir / "avhubert_pre.npy"), mmap_mode="r")
        T_full = len(pre)
        if T_full <= self.max_frames or self.deterministic:
            T = min(T_full, self.max_frames)
            visual = torch.from_numpy(pre[:T].copy())            # (T, 768) fp16
        else:
            start  = torch.randint(0, T_full - self.max_frames + 1, ()).item()
            visual = torch.from_numpy(pre[start:start + self.max_frames].copy())

        # ── Text ──────────────────────────────────────────────────────────────
        meta       = json.loads((clip_dir / "text.json").read_text())
        transcript = meta["transcript"].strip().lower()   # lowercase for Gemma
        tok_ids    = self.tokenizer.encode(transcript, add_special_tokens=False)
        tok_ids    = tok_ids[:self.max_text_len]           # truncate if very long

        # Teacher-forcing format:
        #   input_ids = [BOS, tok1, tok2, ..., tokN]
        #   labels    = [tok1, tok2, ..., tokN, EOS]
        input_ids = torch.tensor([self.bos_id] + tok_ids,            dtype=torch.long)
        labels    = torch.tensor(tok_ids + [self.eos_id],            dtype=torch.long)

        return {"visual": visual, "input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict]) -> dict:
    max_T = max(b["visual"].shape[0] for b in batch)
    max_L = max(b["input_ids"].shape[0] for b in batch)
    B     = len(batch)

    vis_dtype = batch[0]["visual"].dtype
    visual    = torch.zeros(B, max_T, 768, dtype=vis_dtype)
    input_ids = torch.zeros(B, max_L, dtype=torch.long)
    labels    = torch.full((B, max_L), -100, dtype=torch.long)  # -100 = ignore
    attn_mask = torch.zeros(B, max_L, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["visual"].shape[0]
        L = b["input_ids"].shape[0]
        visual[i, :T]    = b["visual"]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L]    = b["labels"]
        attn_mask[i, :L] = 1

    return {"visual": visual, "input_ids": input_ids,
            "labels": labels, "attention_mask": attn_mask}
