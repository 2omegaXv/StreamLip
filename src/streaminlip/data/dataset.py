"""
LRS3 dataset for StreamLip Phase 1.

Each item:
  av_feats:  (T_vis, 1024)  AV-HuBERT features (pre-extracted or online)
  input_ids: (T_text,)      Gemma tokenized text (causal: shifted right)
  labels:    (T_text,)      targets (-100 for padding)

Directory structure (from preprocess_lrs3.py):
  data/processed/{split}/{speaker}/{clip}/
    lip.npy      (T, 96, 96, 3) uint8
    text.json    {transcript, words, n_frames, fps}
    latent.npz   (T_a, 512) float16  [Phase 2]
    avhubert.npy (T, 1024) float16   [pre-extracted, optional]
"""
import csv
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LRS3Dataset(Dataset):
    def __init__(
        self,
        processed_root: str,          # e.g. "data/processed"
        split: str,                    # "pretrain" | "trainval" | "test"
        tokenizer_path: str,           # e.g. "pretrained/gemma-3-1b"
        max_frames: int = 250,         # ~10s at 25fps
        max_text_tokens: int = 128,
        use_cached_avhubert: bool = True,
        limit: int | None = None,      # cap dataset size (debug)
    ):
        self.root = Path(processed_root)
        self.max_frames = max_frames
        self.max_text_tokens = max_text_tokens
        self.use_cached = use_cached_avhubert

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load clip list from manifest.csv (fast, no filesystem scan)
        manifest = self.root / "manifest.csv"
        self.clips = []
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                clip_dir = self.root / row["path"]
                self.clips.append(clip_dir)
                if limit and len(self.clips) >= limit:
                    break

        print(f"[LRS3Dataset] split={split}  clips={len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip_dir = self.clips[idx]

        # ── visual features ────────────────────────────────────────────────
        cached = clip_dir / "avhubert.npy"
        if self.use_cached and cached.exists():
            try:
                av_feats = torch.from_numpy(
                    np.load(str(cached)).astype(np.float32)
                )
            except Exception:
                # corrupted cache: fall back to a random valid sample
                return self.__getitem__((idx + 1) % len(self))
        else:
            lip = np.load(str(clip_dir / "lip.npy"))  # regular load, no mmap on NFS
            lip = torch.from_numpy(lip[:self.max_frames].copy())  # (T, H, W, C)
            lip = lip.float().permute(0, 3, 1, 2) / 127.5 - 1.0  # (T, C, H, W)
            av_feats = lip   # training loop detects shape and runs AV-HuBERT

        if av_feats.shape[0] > self.max_frames:
            av_feats = av_feats[:self.max_frames]

        # ── text ───────────────────────────────────────────────────────────
        meta = json.loads((clip_dir / "text.json").read_text())
        text = meta["transcript"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)   # (T_text,)

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        return {
            "av_feats":  av_feats,
            "input_ids": input_ids,
            "labels":    labels,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad av_feats and input_ids to batch max length."""
    vis_lens = [b["av_feats"].shape[0] for b in batch]
    max_vis  = max(vis_lens)
    vis_dim  = batch[0]["av_feats"].shape[1:]
    av_feats = torch.zeros(len(batch), max_vis, *vis_dim)
    for i, b in enumerate(batch):
        T = b["av_feats"].shape[0]
        av_feats[i, :T] = b["av_feats"]

    text_lens  = [b["input_ids"].shape[0] for b in batch]
    max_text   = max(text_lens)
    pad_id     = batch[0]["input_ids"][-1]
    input_ids  = torch.full((len(batch), max_text), pad_id, dtype=torch.long)
    labels     = torch.full((len(batch), max_text), -100,    dtype=torch.long)
    attn_mask  = torch.zeros(len(batch), max_text, dtype=torch.long)
    for i, b in enumerate(batch):
        T = b["input_ids"].shape[0]
        input_ids[i, :T] = b["input_ids"]
        labels[i, :T]    = b["labels"]
        attn_mask[i, :T] = 1

    return {"av_feats": av_feats, "input_ids": input_ids,
            "labels": labels, "attention_mask": attn_mask}
