"""
StreamLip Dataset — 加载 data/processed/ 格式的预处理数据。
旧格式规范归档见 archive/docs_legacy/DATA_DESIGN.md。
"""

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── 常量 ─────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SIL_TOKEN_STR = "<empty_output>"   # SmolLM2 tokenizer 中的静音 token


# ── face.npz JPEG 解码工具 ────────────────────────────────────────────────────

def load_face_frames(face_npz_path: str,
                     frame_indices: Optional[list] = None) -> np.ndarray:
    """
    从 face.npz（JPEG 字节流格式）加载人脸帧。
    frame_indices=None 时加载全部帧，否则只加载指定帧（节省 decode 时间）。
    返回 (T, 256, 256, 3) uint8 RGB。
    """
    f = np.load(face_npz_path)
    data, offsets = f["data"], f["offsets"]
    indices = frame_indices if frame_indices is not None else range(len(offsets) - 1)
    frames = []
    for i in indices:
        buf = data[offsets[i]:offsets[i+1]].tobytes()
        frame_bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LRS3Dataset(Dataset):
    """
    LRS3 预处理数据集。

    每个样本返回一个 clip（或从中采样的固定长度窗口）。
    支持两种模式：
      clip_mode="full"    — 返回整个 clip（长度可变，需自定义 collate_fn）
      clip_mode="window"  — 随机采样 window_frames 帧的滑动窗口（可 batch）
    """

    def __init__(
        self,
        processed_root: str,
        split: str,                     # pretrain | trainval | test
        tokenizer=None,                 # transformers tokenizer，用于生成帧级标签
        clip_mode: str = "window",      # "full" | "window"
        window_frames: int = 150,       # clip_mode="window" 时的窗口长度（帧数）
        load_face: bool = True,
        normalize: bool = True,
        manifest_csv: Optional[str] = None,
    ):
        self.root        = Path(processed_root)
        self.split       = split
        self.tokenizer   = tokenizer
        self.clip_mode   = clip_mode
        self.window_frames = window_frames
        self.load_face   = load_face
        self.normalize   = normalize

        # SIL token id
        if tokenizer is not None:
            self.sil_id = tokenizer.convert_tokens_to_ids(SIL_TOKEN_STR)
        else:
            self.sil_id = 0

        # 加载 manifest
        csv_path = manifest_csv or str(self.root / "manifest.csv")
        self.clips = self._load_manifest(csv_path, split)

    def _load_manifest(self, csv_path: str, split: str) -> list[dict]:
        import csv
        clips = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    clips.append(row)
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        meta = self.clips[idx]
        clip_dir = self.root / meta["path"]
        T = int(meta["n_frames"])

        # ── 确定帧范围 ────────────────────────────────────────────────────────
        if self.clip_mode == "window":
            W = self.window_frames
            if T >= W:
                start = random.randint(0, T - W)
                frame_range = slice(start, start + W)
                indices = list(range(start, start + W))
            else:
                # clip 比窗口短：从头取，稍后 pad
                start = 0
                frame_range = slice(None)
                indices = None
            T_out = W
        else:
            frame_range = slice(None)
            indices = None
            T_out = T

        # ── 加载 lip ──────────────────────────────────────────────────────────
        lip = np.load(str(clip_dir / "lip.npy"), mmap_mode="r")[frame_range].copy()
        # (T_out, 96, 96, 3) uint8

        # ── 加载 face ─────────────────────────────────────────────────────────
        face = None
        if self.load_face:
            face_path = clip_dir / "face.npz"
            if face_path.exists():
                face = load_face_frames(str(face_path), indices)
                # (T_out, 256, 256, 3) uint8

        # ── 加载 latent ───────────────────────────────────────────────────────
        T_a_full = T // 2
        if self.clip_mode == "window":
            a_start = start // 2
            a_end   = (start + T_out) // 2
            latent = np.load(str(clip_dir / "latent.npz"))["latent"][a_start:a_end]
        else:
            latent = np.load(str(clip_dir / "latent.npz"))["latent"]
        # (T_a, 512) float16

        # ── 生成帧级文本标签 ───────────────────────────────────────────────────
        text_data = json.loads((clip_dir / "text.json").read_text())
        frame_labels = self._make_frame_labels(text_data["words"], T, T_out,
                                                start if self.clip_mode == "window" else 0)

        # ── 归一化 ────────────────────────────────────────────────────────────
        lip_f = lip.astype(np.float32) / 255.0
        if self.normalize:
            lip_f = (lip_f - IMAGENET_MEAN) / IMAGENET_STD
        lip_t = torch.from_numpy(lip_f).permute(0, 3, 1, 2)  # (T', 3, 96, 96)

        # pad 到 window_frames（当 clip 比窗口短时）
        if self.clip_mode == "window" and lip_t.shape[0] < self.window_frames:
            pad = self.window_frames - lip_t.shape[0]
            lip_t = torch.cat([lip_t, lip_t.new_zeros(pad, *lip_t.shape[1:])], dim=0)

        face_t = None
        if face is not None:
            face_f = face.astype(np.float32) / 255.0
            if self.normalize:
                face_f = (face_f - IMAGENET_MEAN) / IMAGENET_STD
            face_t = torch.from_numpy(face_f).permute(0, 3, 1, 2)
            if self.clip_mode == "window" and face_t.shape[0] < self.window_frames:
                pad = self.window_frames - face_t.shape[0]
                face_t = torch.cat([face_t, face_t.new_zeros(pad, *face_t.shape[1:])], dim=0)

        latent_t = torch.from_numpy(latent.astype(np.float32))
        labels_t = torch.from_numpy(frame_labels)
        valid_T  = min(T, self.window_frames) if self.clip_mode == "window" else T
        mask_t   = torch.zeros(T_out, dtype=torch.bool)
        mask_t[:valid_T] = True

        # pad latent 到 window_frames // 2
        if self.clip_mode == "window":
            T_a_out = self.window_frames // 2
            if latent_t.shape[0] < T_a_out:
                pad = T_a_out - latent_t.shape[0]
                latent_t = torch.cat([latent_t, latent_t.new_zeros(pad, latent_t.shape[1])], dim=0)
            else:
                latent_t = latent_t[:T_a_out]

        result = {
            "lip":          lip_t,        # (T_out, 3, 96, 96)
            "latent":       latent_t,     # (T_a_out, 512)
            "frame_labels": labels_t,     # (T_out,)
            "mask":         mask_t,       # (T_out,) True=有效帧
            "clip_id":      meta.get("clip_id", ""),
            "n_frames":     valid_T,
        }
        if face_t is not None:
            result["face"] = face_t  # (T, 3, 256, 256)

        return result

    def _make_frame_labels(self, words: list, T_full: int,
                            T_out: int, start: int) -> np.ndarray:
        """词级时间戳 → 帧级标签，窗口内有效段。"""
        labels = np.full(T_out, self.sil_id, dtype=np.int64)
        if not words or self.tokenizer is None:
            return labels

        for w in words:
            f_s = int(w["start"] * 25) - start
            f_e = int(w["end"]   * 25) - start
            f_s = max(0, f_s)
            f_e = min(T_out, f_e)
            if f_s >= f_e:
                continue
            ids = self.tokenizer.encode(w["word"], add_special_tokens=False)
            if ids:
                labels[f_s:f_e] = ids[0]

        return labels


# ── collate_fn（clip_mode="window" 时直接用默认 collate）────────────────────

def collate_variable_length(batch: list[dict]) -> dict:
    """clip_mode="full" 时：pad 到批内最长，返回带 mask 的 batch。"""
    T_max = max(b["n_frames"] for b in batch)

    def pad_to(arr: torch.Tensor, target_T: int) -> torch.Tensor:
        pad = target_T - arr.shape[0]
        if pad == 0:
            return arr
        return torch.cat([arr, arr.new_zeros(pad, *arr.shape[1:])], dim=0)

    result = {}
    for key in ["lip", "latent", "frame_labels"]:
        if key in batch[0]:
            result[key] = torch.stack([pad_to(b[key], T_max) for b in batch])

    if "face" in batch[0]:
        result["face"] = torch.stack([pad_to(b["face"], T_max) for b in batch])

    result["mask"] = torch.stack([
        torch.cat([b["mask"], torch.zeros(T_max - b["n_frames"], dtype=torch.bool)])
        for b in batch
    ])
    return result
