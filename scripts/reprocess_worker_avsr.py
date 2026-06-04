"""
重处理 worker（正确版）。

Pipeline（与 reprocess_lip_avsr.py 等价，但 FAN 批量推理）：
  1. 读原始 mp4
  2. FAN 批量：batch 人脸检测 → batch 对齐网络 → get_preds_fromhm → 原始帧坐标 68 点
  3. 68 点 → 4 点（右眼/左眼/鼻/嘴中心）
  4. Auto-AVSR VideoProcess（原始帧 + 原始帧坐标关键点）→ (T, 96, 96) uint8 灰度
  5. 模糊过滤，保存 lip_avsr.npy

与 reprocess_lip_avsr.py 的区别：FAN 改为整 clip 一次批量推理，约快 10x。
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu as decord_cpu
from face_alignment.utils import get_preds_fromhm

warnings.filterwarnings("ignore")

sys.path.insert(0, "third_party/auto_avsr/preparation/detectors/mediapipe")
from video_process import VideoProcess  # noqa: E402

DET_BATCH  = 512
FAN_BATCH  = 512
LAP_THRESH = 20.0


class FaceProcessor:
    def __init__(self, fa_device: str = "cuda:0"):
        self.fa_device = fa_device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=fa_device, flip_input=False,
        )
        self._ref_scale = self.fa.face_detector.reference_scale
        # warmup 用真实帧尺寸，避免第一次检测触发 cuDNN 编译
        dummy = torch.zeros(4, 3, 224, 224).to(fa_device)
        try:
            self.fa.face_detector.detect_from_batch(dummy.cpu())
        except Exception:
            pass
        with torch.no_grad():
            try:
                self.fa.face_alignment_net(torch.zeros(1, 3, 256, 256).to(fa_device))
            except Exception:
                pass

    def landmarks_batch(self, frames_rgb: np.ndarray, det_stride: int = 5) -> list:
        """
        批量 FAN：在原始帧上检测人脸、推理关键点，返回原始帧坐标的 4 点列表。
        det_stride: 每隔 N 帧做一次人脸检测，中间帧用最近检测结果填充（LRS3 单人说话头，脸不动）。
        """
        N = len(frames_rgb)

        # ── 1. 跨帧采样检测（只检测 1/det_stride 的帧）────────────────────
        sample_idxs = list(range(0, N, det_stride))
        sample_frames = frames_rgb[sample_idxs]
        tensor = torch.from_numpy(sample_frames).permute(0, 3, 1, 2).float()
        sample_bboxes = []
        for s in range(0, len(tensor), DET_BATCH):
            results = self.fa.face_detector.detect_from_batch(tensor[s:s + DET_BATCH])
            for dets in results:
                if not dets:
                    sample_bboxes.append(None)
                else:
                    best = max(dets, key=lambda d: d[4])
                    sample_bboxes.append([float(v) for v in best[:5]])

        # 把采样 bbox 填充到全部帧（前向填充）
        raw_bboxes = [None] * N
        last = None
        si = 0
        for i in range(N):
            if si < len(sample_idxs) and sample_idxs[si] == i:
                last = sample_bboxes[si]
                si += 1
            raw_bboxes[i] = last
        # 有时前几帧没检测到，用后向填充补上
        last = None
        for i in range(N - 1, -1, -1):
            if raw_bboxes[i] is not None:
                last = raw_bboxes[i]
            elif last is not None:
                raw_bboxes[i] = last

        # ── 2. GPU 批量 fa_crop（affine_grid + grid_sample 替代串行 CPU crop）──
        H_orig, W_orig = frames_rgb.shape[1], frames_rgb.shape[2]
        centers, scales, valid_idxs = [], [], []
        for i, bbox in enumerate(raw_bboxes):
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0 - (y2 - y1) * 0.12
            sc = ((x2 - x1) + (y2 - y1)) / self._ref_scale
            centers.append((cx, cy))
            scales.append(sc)
            valid_idxs.append(i)

        landmarks_68 = [None] * N
        if valid_idxs:
            # 构造仿射矩阵：fa_crop 等价于 scale+translate，映射 256×256 输出 → 原始帧像素
            # grid_sample 期望归一化坐标 [-1,1]
            theta = torch.zeros(len(valid_idxs), 2, 3)
            for k, (i, (cx, cy), sc) in enumerate(zip(valid_idxs, centers, scales)):
                h = 200.0 * sc
                theta[k, 0, 0] = h / W_orig
                theta[k, 0, 2] = 2.0 * cx / W_orig - 1.0
                theta[k, 1, 1] = h / H_orig
                theta[k, 1, 2] = 2.0 * cy / H_orig - 1.0

            frames_t = torch.from_numpy(
                frames_rgb[valid_idxs].transpose(0, 3, 1, 2)
            ).float().to(self.fa_device)                           # (M, 3, H, W)
            grid = F.affine_grid(theta.to(self.fa_device),
                                 (len(valid_idxs), 3, 256, 256),
                                 align_corners=False)
            crops_gpu = F.grid_sample(frames_t, grid,
                                      mode='bilinear', padding_mode='zeros',
                                      align_corners=False)         # (M, 3, 256, 256)

            # ── 3. 批量对齐网络推理 ──────────────────────────────────────────
            inp = crops_gpu.div(255.0).clamp(0, 1)
            for s in range(0, len(valid_idxs), FAN_BATCH):
                batch = inp[s:s + FAN_BATCH]
                with torch.no_grad():
                    out = self.fa.face_alignment_net(batch).cpu().numpy()
                for j, gi in enumerate(valid_idxs[s:s + FAN_BATCH]):
                    cx, cy = centers[s + j]
                    sc = scales[s + j]
                    _, pts_img, _ = get_preds_fromhm(
                        out[j:j + 1], np.array([cx, cy]), sc
                    )
                    landmarks_68[gi] = pts_img.squeeze()

        # ── 4. 68 点 → 4 点（VideoProcess 格式） ────────────────────────────
        lms_4pt = []
        for lm in landmarks_68:
            if lm is None:
                lms_4pt.append(None)
            else:
                lms_4pt.append(np.array([
                    lm[36:42].mean(0),  # 右眼
                    lm[42:48].mean(0),  # 左眼
                    lm[31:36].mean(0),  # 鼻
                    lm[48:68].mean(0),  # 嘴
                ]))
        return lms_4pt


def process_clip(mp4_path, out_dir, fp: FaceProcessor, vp: VideoProcess,
                 force: bool = False):
    out_dir  = Path(out_dir)
    out_path = out_dir / "lip_avsr.npy"

    if not force and out_path.exists():
        return {"ok": True, "path": str(out_dir), "skipped": True}

    if not Path(mp4_path).exists():
        return {"ok": False, "path": str(out_dir), "error": "mp4 not found"}

    try:
        vr = VideoReader(str(mp4_path), ctx=decord_cpu(0))
        frames_rgb = np.stack([vr[i].asnumpy() for i in range(len(vr))])
        if len(frames_rgb) == 0:
            return {"ok": False, "path": str(out_dir), "error": "no frames"}

        lms_4pt = fp.landmarks_batch(frames_rgb)
        cropped = vp(frames_rgb, lms_4pt)       # (T, 96, 96) uint8 灰度
        if cropped is None or len(cropped) == 0:
            return {"ok": False, "path": str(out_dir), "error": "VideoProcess failed"}

        # 模糊过滤
        mid    = len(cropped) // 2
        sample = cropped[max(0, mid - 3): mid + 3]
        lv     = float(np.mean([cv2.Laplacian(f, cv2.CV_64F).var() for f in sample]))
        if lv < LAP_THRESH:
            return {"ok": False, "path": str(out_dir), "error": f"blurry lap={lv:.1f}"}

        np.save(str(out_path), cropped)
        return {"ok": True, "path": str(out_dir), "n_frames": len(cropped), "skipped": False}

    except Exception as e:
        return {"ok": False, "path": str(out_dir), "error": f"{type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file",  required=True)
    parser.add_argument("--fa_device", default="cuda:0")
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    fp = FaceProcessor(fa_device=args.fa_device)
    vp = VideoProcess(convert_gray=True)

    jobs = json.loads(Path(args.job_file).read_text())
    results = []
    for job in jobs:
        res = process_clip(job["mp4"], job["out"], fp, vp, force=args.force)
        results.append(res)
        status = "skip" if res.get("skipped") else ("ok" if res["ok"] else "err")
        print(json.dumps({"status": status, **res}), flush=True)

    Path(args.job_file.replace(".json", "_result.json")).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
