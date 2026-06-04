"""
LRS3 Dataset V5.

Returns per-sample:
  visual:     (T, 768) float  pre-extracted AVSR encoder features
  video_pos:  (T,)     int64  frame-index position_ids [0..T-1]
  input_ids:  (L,)     int64  BOS + tok_0..tok_{L-2}
  target_ids: (L,)     int64  tok_0..tok_{L-1}  (-100 for padding)
  text_pos:   (L,)     int64  co-temporal position_ids
  video_mask: (T,)     bool
  text_mask:  (L,)     bool
"""
import csv
import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer

FPS        = 25
CHUNK_SIZE = 6
MIN_CHUNKS = 4


def build_text_sequence(
    words: list, T: int, tokenizer, bos_id: int
) -> tuple[list, list, list, list]:
    """
    Returns (input_ids, target_ids, text_pos, last_chunk_mask).
    target_ids 末尾包含 EOS，让模型学会自然停止。
    """
    eos_id = tokenizer.eos_token_id
    tok_seq  = []
    last_mask = []
    last_start_sec = max(0, T - CHUNK_SIZE) / FPS

    for w in words:
        toks   = tokenizer.encode(" " + w["word"].lower(), add_special_tokens=False)
        f_mid  = (w["start"] + w["end"]) / 2
        in_last = f_mid >= last_start_sec
        tok_seq.extend(toks)
        last_mask.extend([in_last] * len(toks))

    if not tok_seq:
        return [], [], [], []

    # 末尾加 EOS：模型学到"说完最后一个词后预测 EOS"
    tok_seq_with_eos = tok_seq + [eos_id]
    last_mask_with_eos = last_mask + [False]   # EOS 不属于任何 chunk 的语音

    input_ids  = [bos_id] + tok_seq_with_eos[:-1]   # = [bos] + tok_seq
    target_ids = tok_seq_with_eos                     # = tok_0..tok_{N-1} + eos
    text_pos   = list(range(T, T + len(input_ids)))
    return input_ids, target_ids, text_pos, last_mask_with_eos


class LRS3DatasetV5(Dataset):

    def __init__(
        self,
        processed_root: str,
        split:          str,
        tokenizer_path: str,
        max_frames:     int  = 150,
        min_chunks:     int  = MIN_CHUNKS,
        deterministic:  bool = False,
        subset:         str  = "train",
        test_reserve:   int  = 2000,
        limit:          int | None = None,
    ):
        self.root          = Path(processed_root)
        self.max_chunks    = max_frames // CHUNK_SIZE
        self.min_chunks    = min_chunks
        self.deterministic = deterministic

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # OLMo 没有 bos_token，用 eos_token 代替
        self.bos_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

        manifest   = self.root / "manifest.csv"
        cache_file = self.root / f"clip_cache_{split}.txt"

        if cache_file.exists():
            cached = set(cache_file.read_text().splitlines())
            clips, nframes = [], []
            with open(manifest) as f:
                for row in csv.DictReader(f):
                    if row["split"] != split:
                        continue
                    p = str(self.root / row["path"])
                    if p in cached:
                        clips.append(Path(p))
                        nframes.append(int(row["n_frames"]))
            print(f"[LRS3DatasetV5] loaded from cache: {len(clips)} clips")
        else:
            clips, nframes = [], []
            with open(manifest) as f:
                for row in csv.DictReader(f):
                    if row["split"] != split:
                        continue
                    clips.append(self.root / row["path"])
                    nframes.append(int(row["n_frames"]))

        if split == "pretrain" and test_reserve > 0:
            if subset == "test":
                clips, nframes = clips[-test_reserve:], nframes[-test_reserve:]
            else:
                clips, nframes = clips[:-test_reserve], nframes[:-test_reserve]

        if limit:
            clips, nframes = clips[:limit], nframes[:limit]

        if not cache_file.exists():
            def _has_feat(c: Path) -> bool:
                if (c / "avsr_enc.npy").exists(): return True
                if (c / "avsr_pre_enc.npy").exists(): return True
                if (c / "auto_avsr_lip_cnn.npy").exists(): return True
                return False
            ready = [(c, n) for c, n in zip(clips, nframes) if _has_feat(c)]
        else:
            ready = list(zip(clips, nframes))

        self.clips   = [x[0] for x in ready]
        self.lengths = [min(n, self.max_chunks * CHUNK_SIZE) for n in (x[1] for x in ready)]
        print(f"[LRS3DatasetV5] split={split}/{subset}  clips={len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict | None:
        clip_dir = self.clips[idx]
        meta     = json.loads((clip_dir / "text.json").read_text())
        words    = meta.get("words", [])

        # 优先加载预提取特征
        for fname in ["avsr_enc.npy", "avsr_pre_enc.npy", "auto_avsr_lip_cnn.npy"]:
            p = clip_dir / fname
            if p.exists():
                feat_raw = np.load(str(p), mmap_mode="r")
                break
        else:
            return None

        T_full  = len(feat_raw)
        max_k   = min(self.max_chunks, T_full // CHUNK_SIZE)
        min_k   = min(self.min_chunks, max_k)
        if max_k < 1:
            return None

        if self.deterministic or min_k >= max_k:
            k = max_k
        else:
            k = random.randint(min_k, max_k)

        T       = k * CHUNK_SIZE
        t_max   = T / FPS
        words_t = [w for w in words if w["start"] < t_max]
        if not words_t:
            return None

        visual = torch.from_numpy(feat_raw[:T].copy())   # (T, D)

        input_ids, target_ids, text_pos, last_chunk_mask = build_text_sequence(
            words_t, T, self.tokenizer, self.bos_id
        )
        if not input_ids:
            return None

        return {
            "visual":           visual,
            "video_pos":        torch.arange(T, dtype=torch.long),
            "input_ids":        torch.tensor(input_ids,       dtype=torch.long),
            "target_ids":       torch.tensor(target_ids,      dtype=torch.long),
            "text_pos":         torch.tensor(text_pos,        dtype=torch.long),
            "last_chunk_mask":  torch.tensor(last_chunk_mask, dtype=torch.bool),
            "mask":             torch.ones(T, dtype=torch.bool),
        }


def collate_fn(batch: list) -> dict | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    B     = len(batch)
    T_max = max(b["visual"].shape[0]    for b in batch)
    L_max = max(b["input_ids"].shape[0] for b in batch)

    vis0      = batch[0]["visual"]
    vis_shape = vis0.shape[1:]
    visual    = torch.zeros(B, T_max, *vis_shape, dtype=vis0.dtype)
    video_pos   = torch.zeros(B, T_max, dtype=torch.long)
    input_ids   = torch.zeros(B, L_max, dtype=torch.long)   # 0 = safe pad
    target_ids  = torch.full((B, L_max), -100, dtype=torch.long)
    text_pos    = torch.zeros(B, L_max, dtype=torch.long)
    video_mask  = torch.zeros(B, T_max, dtype=torch.bool)
    text_mask   = torch.zeros(B, L_max, dtype=torch.bool)
    last_chunk_mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, b in enumerate(batch):
        T = b["visual"].shape[0]
        L = b["input_ids"].shape[0]
        visual[i, :T]            = b["visual"]
        video_pos[i, :T]         = b["video_pos"]
        input_ids[i, :L]         = b["input_ids"]
        target_ids[i, :L]        = b["target_ids"]
        text_pos[i, :L]          = b["text_pos"]
        video_mask[i, :T]        = b["mask"]
        text_mask[i, :L]         = True
        last_chunk_mask[i, :L]   = b["last_chunk_mask"]

    return {
        "visual":          visual,
        "video_pos":       video_pos,
        "input_ids":       input_ids,
        "target_ids":      target_ids,
        "text_pos":        text_pos,
        "video_mask":      video_mask,
        "text_mask":       text_mask,
        "last_chunk_mask": last_chunk_mask,
    }
