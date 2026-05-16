"""
视频处理 worker（单进程，无 transformers 依赖）。
由 preprocess_lrs3.py 通过 subprocess 并行调用。

存储格式：
  lip.npy    (T, 96, 96, 3) uint8  — 无压缩，mmap 读取极快
  face.npz   data=(总JPEG字节) uint8 + offsets=(T+1,) int64
             — JPEG 压缩（~26x），单文件，支持按帧随机读取
  audio.wav  24kHz mono PCM
  latent.npz (T_a, 512) float16
  text.json  词+时间戳
"""

import argparse
import json
import io
import subprocess
import warnings
from pathlib import Path

import cv2
import face_alignment
import kornia
import numpy as np
import torch
from decord import VideoReader, cpu as decord_cpu
from face_alignment.utils import crop as fa_crop, get_preds_fromhm
from PIL import Image

warnings.filterwarnings("ignore")

LIP_SIZE   = 96
FACE_SIZE  = 256
TARGET_FPS = 25
MIMI_SR    = 24000
DET_BATCH  = 512
FAN_BATCH  = 512
JPEG_QUALITY = 85

MEAN_5PT_256 = np.array([
    [0.34, 0.37], [0.66, 0.37], [0.50, 0.55],
    [0.37, 0.72], [0.63, 0.72],
], dtype=np.float32) * FACE_SIZE

LIP_Y1 = int(0.54 * FACE_SIZE)
LIP_Y2 = int(0.94 * FACE_SIZE)
LIP_X1 = int(0.17 * FACE_SIZE)
LIP_X2 = int(0.83 * FACE_SIZE)


# ── FaceProcessor ────────────────────────────────────────────────────────────

class FaceProcessor:
    def __init__(self, fa_device: str = "cuda:0"):
        self.fa_device = fa_device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=fa_device, flip_input=False,
        )
        self._ref_scale = self.fa.face_detector.reference_scale

        dummy = torch.zeros(1, 3, 224, 224)
        try:
            self.fa.face_detector.detect_from_batch(dummy)
        except Exception:
            pass
        with torch.no_grad():
            try:
                self.fa.face_alignment_net(torch.zeros(1, 3, 256, 256).to(fa_device))
            except Exception:
                pass

    def detect_faces_batch(self, frames_rgb: np.ndarray) -> list:
        tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float()
        all_bboxes = []
        for start in range(0, len(tensor), DET_BATCH):
            results = self.fa.face_detector.detect_from_batch(tensor[start:start+DET_BATCH])
            for dets in results:
                if not dets:
                    all_bboxes.append(None)
                else:
                    best = max(dets, key=lambda d: d[4])
                    all_bboxes.append([float(v) for v in best[:4]])
        return all_bboxes

    def warp_frames_gpu(self, frames_rgb: np.ndarray, bboxes: list):
        N = len(frames_rgb)
        Ms = np.zeros((N, 2, 3), dtype=np.float32)
        invalid = np.zeros(N, dtype=bool)

        for i, bbox in enumerate(bboxes):
            if bbox is None:
                Ms[i] = np.eye(2, 3, dtype=np.float32)
                invalid[i] = True
                continue
            x1, y1, x2, y2 = bbox
            w, h = x2-x1, y2-y1
            kps = np.array([
                [x1+w*0.30, y1+h*0.38], [x1+w*0.70, y1+h*0.38],
                [x1+w*0.50, y1+h*0.55], [x1+w*0.35, y1+h*0.72],
                [x1+w*0.65, y1+h*0.72],
            ], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(kps, MEAN_5PT_256, method=cv2.LMEDS)
            if M is None:
                Ms[i] = np.eye(2, 3, dtype=np.float32)
                invalid[i] = True
            else:
                Ms[i] = M

        frames_gpu = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().to(self.fa_device)
        M_gpu = torch.from_numpy(Ms).float().to(self.fa_device)
        aligned_gpu = kornia.geometry.transform.warp_affine(
            frames_gpu, M_gpu, dsize=(FACE_SIZE, FACE_SIZE),
            mode='bilinear', align_corners=False,
        )
        return aligned_gpu, invalid

    def landmarks_on_aligned(self, aligned_gpu, invalid: np.ndarray) -> list:
        N = aligned_gpu.shape[0]
        landmarks = [None] * N
        valid = [i for i in range(N) if not invalid[i]]
        if not valid:
            return landmarks

        inp_all = (aligned_gpu / 255.0).clamp(0, 1)
        center = torch.FloatTensor([FACE_SIZE/2, FACE_SIZE/2])
        scale  = float(FACE_SIZE / self._ref_scale)

        for start in range(0, len(valid), FAN_BATCH):
            idx_batch = valid[start:start+FAN_BATCH]
            with torch.no_grad():
                out = self.fa.face_alignment_net(inp_all[idx_batch]).cpu()
            for j, i in enumerate(idx_batch):
                pts, pts_img, _ = get_preds_fromhm(
                    out[j:j+1].numpy(), center.numpy(), scale)
                landmarks[i] = pts_img.squeeze()
        return landmarks

    def crop_lips_gpu(self, aligned_gpu, landmarks: list, invalid: np.ndarray) -> np.ndarray:
        roi = aligned_gpu[:, :, LIP_Y1:LIP_Y2, LIP_X1:LIP_X2]
        lip_gpu = kornia.geometry.transform.resize(roi, (LIP_SIZE, LIP_SIZE))

        valid_lm = [(i, lm) for i, lm in enumerate(landmarks)
                    if lm is not None and not invalid[i]]
        if valid_lm:
            mouths, idxs = [], []
            for i, lm in valid_lm:
                mouth = lm[48:68]
                cx, cy = mouth.mean(0)
                size = max(mouth[:,0].max()-mouth[:,0].min(),
                           mouth[:,1].max()-mouth[:,1].min()) * 1.8
                y1m = max(0, int(cy - size/2))
                y2m = min(FACE_SIZE, int(cy + size/2))
                x1m = max(0, int(cx - size/2))
                x2m = min(FACE_SIZE, int(cx + size/2))
                crop = aligned_gpu[i:i+1, :, y1m:y2m, x1m:x2m]
                if crop.shape[2] > 0 and crop.shape[3] > 0:
                    mouths.append(kornia.geometry.transform.resize(crop, (LIP_SIZE, LIP_SIZE)))
                    idxs.append(i)
            if mouths:
                stacked = torch.cat(mouths, dim=0)
                for k, i in enumerate(idxs):
                    lip_gpu[i] = stacked[k]

        return lip_gpu.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


# ── 文本解析 ─────────────────────────────────────────────────────────────────

def parse_txt(txt_path):
    lines = Path(txt_path).read_text().strip().splitlines()
    transcript, words, in_words = "", [], False
    for line in lines:
        line = line.strip()
        if line.startswith("Text:"):
            transcript = line[len("Text:"):].strip()
        elif line.startswith("WORD"):
            in_words = True
        elif in_words and line:
            parts = line.split()
            if len(parts) >= 3:
                words.append({"word": parts[0],
                               "start": float(parts[1]), "end": float(parts[2])})
    return {"transcript": transcript, "words": words}


# ── face JPEG 存储工具 ────────────────────────────────────────────────────────

def save_face_jpeg(face_np: np.ndarray, path: str):
    """
    将 (T, H, W, 3) uint8 face 帧编码为 JPEG 并存入单个 npz。
    内部结构：data (总字节数,) uint8 + offsets (T+1,) int64
    读取任意帧 i：data[offsets[i]:offsets[i+1]] → cv2.imdecode
    """
    jpeg_bufs = []
    for frame in face_np:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=JPEG_QUALITY)
        jpeg_bufs.append(buf.getvalue())

    lengths = np.array([len(b) for b in jpeg_bufs], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    data    = np.frombuffer(b"".join(jpeg_bufs), dtype=np.uint8)
    np.savez(path, data=data, offsets=offsets)


def load_face_jpeg(path: str) -> np.ndarray:
    """
    读取 face.npz，返回 (T, H, W, 3) uint8 RGB。
    训练时通常只需要部分帧，可按 offsets 随机读取。
    """
    f = np.load(path)
    data, offsets = f["data"], f["offsets"]
    T = len(offsets) - 1
    frames = []
    for i in range(T):
        buf = data[offsets[i]:offsets[i+1]].tobytes()
        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


# ── 单 clip ───────────────────────────────────────────────────────────────────

def process_clip(mp4_path, txt_path, out_dir, fp: FaceProcessor):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "lip.npy").exists() and (out_dir / "text.json").exists():
        n = np.load(str(out_dir / "lip.npy"), mmap_mode="r").shape[0]
        return {"ok": True, "path": str(out_dir), "n_frames": n, "n_failed": 0, "skipped": True}

    vr = VideoReader(str(mp4_path), ctx=decord_cpu(0))
    frames_rgb = np.stack([vr[i].asnumpy() for i in range(len(vr))])
    total = len(frames_rgb)
    if total == 0:
        return {"ok": False, "path": str(out_dir), "error": "no frames"}

    bboxes = fp.detect_faces_batch(frames_rgb)
    aligned_gpu, invalid = fp.warp_frames_gpu(frames_rgb, bboxes)
    landmarks = fp.landmarks_on_aligned(aligned_gpu, invalid)
    lip_np  = fp.crop_lips_gpu(aligned_gpu, landmarks, invalid)
    face_np = aligned_gpu.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    n_failed = int(invalid.sum())

    # lip：np.save（无压缩，mmap 读取极快）
    np.save(str(out_dir / "lip.npy"), lip_np)

    # face：JPEG 压缩单文件（~26x 压缩，单文件无小文件问题）
    save_face_jpeg(face_np, str(out_dir / "face.npz"))

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-ar", str(MIMI_SR), "-ac", "1",
         str(out_dir / "audio.wav")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    text_data = parse_txt(txt_path)
    text_data["n_frames"] = total
    text_data["fps"] = TARGET_FPS
    (out_dir / "text.json").write_text(json.dumps(text_data, ensure_ascii=False, indent=2))

    return {"ok": True, "path": str(out_dir), "n_frames": total,
            "n_failed": n_failed, "skipped": False}


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file",  required=True)
    parser.add_argument("--fa_device", default="cuda:0")
    args = parser.parse_args()

    jobs = json.loads(Path(args.job_file).read_text())
    fp = FaceProcessor(fa_device=args.fa_device)
    results = []
    for job in jobs:
        res = process_clip(job["mp4"], job["txt"], job["out"], fp)
        results.append(res)
        status = "skip" if res.get("skipped") else ("ok" if res["ok"] else "err")
        print(json.dumps({"status": status, **res}), flush=True)

    Path(args.job_file.replace(".json", "_result.json")).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
