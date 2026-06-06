"""Auto-AVSR-compatible lip crop reprocess worker.

This batch-FAN worker reads source MP4 files, estimates 68-point landmarks,
converts them to the 4-point format expected by Auto-AVSR's ``VideoProcess``,
and writes ``lip_avsr.npy`` as ``(T, 96, 96)`` uint8 grayscale frames.
"""

from __future__ import annotations

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
from decord import VideoReader
from decord import cpu as decord_cpu
from face_alignment.utils import get_preds_fromhm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIAPIPE_DIR = REPO_ROOT / "third_party/auto_avsr/preparation/detectors/mediapipe"
if (MEDIAPIPE_DIR / "video_process.py").exists():
    sys.path.insert(0, str(MEDIAPIPE_DIR))
else:
    raise FileNotFoundError(
        "Auto-AVSR mediapipe video_process.py not found. Expected it under "
        f"{MEDIAPIPE_DIR}."
    )
from video_process import VideoProcess  # noqa: E402


DET_BATCH = 512
FAN_BATCH = 512
LAP_THRESH = 20.0


class FaceProcessor:
    def __init__(self, fa_device: str = "cuda:0"):
        self.fa_device = fa_device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=fa_device,
            flip_input=False,
        )
        self._ref_scale = self.fa.face_detector.reference_scale

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

    def landmarks_batch(self, frames_rgb: np.ndarray, det_stride: int = 5) -> list[np.ndarray | None]:
        """Return Auto-AVSR 4-point landmarks in original-frame coordinates."""
        n_frames = len(frames_rgb)

        sample_idxs = list(range(0, n_frames, det_stride))
        sample_frames = frames_rgb[sample_idxs]
        tensor = torch.from_numpy(sample_frames).permute(0, 3, 1, 2).float()
        sample_bboxes = []
        for start in range(0, len(tensor), DET_BATCH):
            results = self.fa.face_detector.detect_from_batch(tensor[start : start + DET_BATCH])
            for dets in results:
                if not dets:
                    sample_bboxes.append(None)
                else:
                    best = max(dets, key=lambda det: det[4])
                    sample_bboxes.append([float(v) for v in best[:5]])

        raw_bboxes = [None] * n_frames
        last = None
        sample_pos = 0
        for frame_idx in range(n_frames):
            if sample_pos < len(sample_idxs) and sample_idxs[sample_pos] == frame_idx:
                last = sample_bboxes[sample_pos]
                sample_pos += 1
            raw_bboxes[frame_idx] = last

        last = None
        for frame_idx in range(n_frames - 1, -1, -1):
            if raw_bboxes[frame_idx] is not None:
                last = raw_bboxes[frame_idx]
            elif last is not None:
                raw_bboxes[frame_idx] = last

        h_orig, w_orig = frames_rgb.shape[1], frames_rgb.shape[2]
        centers, scales, valid_idxs = [], [], []
        for frame_idx, bbox in enumerate(raw_bboxes):
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0 - (y2 - y1) * 0.12
            scale = ((x2 - x1) + (y2 - y1)) / self._ref_scale
            centers.append((cx, cy))
            scales.append(scale)
            valid_idxs.append(frame_idx)

        landmarks_68 = [None] * n_frames
        if valid_idxs:
            theta = torch.zeros(len(valid_idxs), 2, 3)
            for k, (_frame_idx, (cx, cy), scale) in enumerate(zip(valid_idxs, centers, scales)):
                height = 200.0 * scale
                theta[k, 0, 0] = height / w_orig
                theta[k, 0, 2] = 2.0 * cx / w_orig - 1.0
                theta[k, 1, 1] = height / h_orig
                theta[k, 1, 2] = 2.0 * cy / h_orig - 1.0

            frames_t = torch.from_numpy(frames_rgb[valid_idxs].transpose(0, 3, 1, 2)).float().to(
                self.fa_device
            )
            grid = F.affine_grid(
                theta.to(self.fa_device),
                (len(valid_idxs), 3, 256, 256),
                align_corners=False,
            )
            crops_gpu = F.grid_sample(
                frames_t,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            inp = crops_gpu.div(255.0).clamp(0, 1)
            for start in range(0, len(valid_idxs), FAN_BATCH):
                batch = inp[start : start + FAN_BATCH]
                with torch.no_grad():
                    out = self.fa.face_alignment_net(batch).cpu().numpy()
                for offset, frame_idx in enumerate(valid_idxs[start : start + FAN_BATCH]):
                    cx, cy = centers[start + offset]
                    scale = scales[start + offset]
                    _, pts_img, _ = get_preds_fromhm(out[offset : offset + 1], np.array([cx, cy]), scale)
                    landmarks_68[frame_idx] = pts_img.squeeze()

        lms_4pt = []
        for lm in landmarks_68:
            if lm is None:
                lms_4pt.append(None)
            else:
                lms_4pt.append(
                    np.array(
                        [
                            lm[36:42].mean(0),
                            lm[42:48].mean(0),
                            lm[31:36].mean(0),
                            lm[48:68].mean(0),
                        ]
                    )
                )
        return lms_4pt


def process_clip(
    mp4_path: str | Path,
    out_dir: str | Path,
    fp: FaceProcessor,
    vp: VideoProcess,
    force: bool = False,
):
    out_dir = Path(out_dir)
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
        cropped = vp(frames_rgb, lms_4pt)
        if cropped is None or len(cropped) == 0:
            return {"ok": False, "path": str(out_dir), "error": "VideoProcess failed"}

        mid = len(cropped) // 2
        sample = cropped[max(0, mid - 3) : mid + 3]
        lap_var = float(np.mean([cv2.Laplacian(frame, cv2.CV_64F).var() for frame in sample]))
        if lap_var < LAP_THRESH:
            return {"ok": False, "path": str(out_dir), "error": f"blurry lap={lap_var:.1f}"}

        np.save(str(out_path), cropped)
        return {"ok": True, "path": str(out_dir), "n_frames": len(cropped), "skipped": False}

    except Exception as exc:
        return {"ok": False, "path": str(out_dir), "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file", required=True)
    parser.add_argument("--fa_device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
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
