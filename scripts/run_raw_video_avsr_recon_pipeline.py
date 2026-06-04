"""One-command raw video -> AVSR-conditioned audio reconstruction pipeline.

This script is intentionally small and explicit. It reproduces the raw-video
flow used for the external trump/hrx checks:

1. standardize input video to 224x224, 25 fps, 24 kHz mono audio
2. run the existing face/audio preprocessing for face.npz, audio.wav, lip.npy
3. run the new Auto-AVSR-compatible reprocess_worker_avsr.py for lip_avsr.npy
4. extract Mimi latent, Auto-AVSR encoder/text, SmolLM2 hidden, speaker/timbre
5. run the current best recon checkpoint with 3s audio prompt
6. mux post-3s predicted audio back to the standardized video
7. export face/lip_avsr visualization videos

Run with the repo .venv, for example:

  /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \
    scripts/run_raw_video_avsr_recon_pipeline.py \
    --input data/hrx.mov --exp hrx_reprocess_avsr
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path("/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A")
DEFAULT_CONFIG = (
    "configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_"
    "residual_samplecorr02_from1000_recon_textjson_wordts.yaml"
)
DEFAULT_CKPT = (
    "runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_"
    "residual_samplecorr02_from1000_recon_textjson_wordts_v1/step_001500.pt"
)


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def standardize_video(input_video: Path, out_mp4: Path, size: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_video),
        "-vf", f"fps=25,scale={size}:{size}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-ar", "24000", "-ac", "1", "-c:a", "aac",
        str(out_mp4),
    ])


def run_face_audio_preprocess(std_mp4: Path, clip_dir: Path, out_dir: Path, fa_device: str) -> None:
    empty_txt = out_dir / "empty.txt"
    empty_txt.write_text("Text: \nWORD START END ASDSCORE\n")
    job_file = out_dir / "preprocess_job.json"
    write_json(job_file, [{"mp4": str(std_mp4), "txt": str(empty_txt), "out": str(clip_dir)}])
    run([
        sys.executable,
        str(REPO_ROOT / "scripts/run_preprocess_worker_no_flash_attn.py"),
        "--job_file", str(job_file),
        "--fa_device", fa_device,
    ])


def run_avsr_lip_reprocess(
    std_mp4: Path,
    clip_dir: Path,
    out_dir: Path,
    fa_device: str,
    avsr_worker: Path,
) -> None:
    require_file(avsr_worker, "reprocess_worker_avsr.py")
    job_file = out_dir / "reprocess_avsr_job.json"
    write_json(job_file, [{"mp4": str(std_mp4), "out": str(clip_dir)}])
    run([
        sys.executable,
        str(avsr_worker),
        "--job_file", str(job_file),
        "--fa_device", fa_device,
        "--force",
    ])


def extract_mimi_latent(clip_dir: Path, mimi_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.preprocess_lrs3 import build_mimi, extract_latent
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mimi = build_mimi(device, mimi_path)
    lat = extract_latent(mimi, clip_dir / "audio.wav", device)
    np.savez(str(clip_dir / "latent.npz"), latent=lat)
    print(f"latent: {lat.shape} {lat.dtype}", flush=True)


def extract_speaker_and_timbre(clip_dir: Path, processed_root: Path, resnet50_weights: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from scripts.extract_speaker_emb import extract_clip
    from scripts.extract_timbre_cond import build_timbre_condition
    from streaminlip.fm_avsr_dataset import normalize_latent, set_norm_stats_path
    from streaminlip.v2.speaker_encoder import SpeakerEncoder

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = SpeakerEncoder(weights_path=str(resnet50_weights)).to(device).eval()
    extract_clip(encoder, clip_dir, device, force=True)
    set_norm_stats_path(processed_root / "latent_norm_stats.npz")
    latent = np.load(str(clip_dir / "latent.npz"))["latent"].astype("float32")
    timbre = build_timbre_condition(normalize_latent(latent), prompt_frames=38)
    np.save(str(clip_dir / "timbre_cond.npy"), timbre.astype("float16"))
    print(
        f"speaker: {np.load(str(clip_dir / 'speaker_emb.npy')).shape} "
        f"timbre: {timbre.shape}",
        flush=True,
    )


def extract_avsr_and_text(
    processed_root: Path,
    clip_list: Path,
    auto_avsr_ckpt: Path,
    smollm2_path: Path,
) -> None:
    run([
        sys.executable,
        str(REPO_ROOT / "scripts/extract_avsr_enc.py"),
        "--data_root", str(processed_root),
        "--clip_list", str(clip_list),
        "--input_name", "lip_avsr.npy",
        "--output_name", "avsr_enc_lipavsr.npy",
        "--text_output_name", "avsr_text_lipavsr.txt",
        "--avsr_ckpt", str(auto_avsr_ckpt),
        "--gpu", "0",
        "--force",
        "--batch_size", "1",
    ])
    rel = clip_list.read_text().strip()
    clip_dir = processed_root / rel
    shutil.copyfile(clip_dir / "avsr_text_lipavsr.txt", clip_dir / "avsr_text.txt")
    run([
        sys.executable,
        str(REPO_ROOT / "scripts/extract_smollm2_h.py"),
        "--data_root", str(processed_root),
        "--clip_list", str(clip_list),
        "--text_source", "lipavsr",
        "--smollm2_path", str(smollm2_path),
        "--batch_size", "1",
        "--overwrite",
    ])


def run_recon(
    processed_root: Path,
    clip_list: Path,
    output_dir: Path,
    config: Path,
    ckpt: Path,
) -> None:
    run([
        sys.executable,
        str(REPO_ROOT / "scripts/eval_fm_avsr.py"),
        "--config", str(config),
        "--ckpt", str(ckpt),
        "--data_root", str(processed_root),
        "--clip_list", str(clip_list),
        "--text_source", "lipavsr",
        "--text_alignment_mode", "uniform",
        "--output_dir", str(output_dir),
        "--n", "1",
        "--use_recon",
        "--wav_start_frame", "38",
        "--metric_start_frame", "38",
        "--save_gt",
        "--metrics_json", str(output_dir / "metrics.json"),
    ])


def mux_outputs(exp_dir: Path, exp: str, recon_dir: Path, std_mp4: Path) -> None:
    pred_mp4 = exp_dir / f"{exp}_pred_prompt3s_post3s.mp4"
    gt_mp4 = exp_dir / f"{exp}_gt_mimi_post3s.mp4"
    for wav_name, out_mp4 in [("0000_pred.wav", pred_mp4), ("0000_gt.wav", gt_mp4)]:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", "3.04", "-i", str(std_mp4),
            "-i", str(recon_dir / wav_name),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-shortest",
            str(out_mp4),
        ])


def write_video(frames: np.ndarray, path: Path, fps: int = 25) -> None:
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_visualization(exp_dir: Path, clip_dir: Path) -> None:
    vis_dir = exp_dir / "vis_reprocess_avsr"
    vis_dir.mkdir(parents=True, exist_ok=True)

    face_npz = np.load(str(clip_dir / "face.npz"))
    data, offsets = face_npz["data"], face_npz["offsets"]
    faces = []
    for i in range(len(offsets) - 1):
        bgr = cv2.imdecode(
            np.frombuffer(data[offsets[i] : offsets[i + 1]].tobytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        faces.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    faces = np.stack(faces)

    lip = np.load(str(clip_dir / "lip_avsr.npy"))
    lip_rgb = np.repeat(lip[..., None], 3, axis=-1)
    lip_big = np.stack([
        cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
        for frame in lip_rgb
    ])
    n = min(len(faces), len(lip_big))
    combo = np.concatenate([faces[:n], lip_big[:n]], axis=2)

    write_video(lip_big[:n], vis_dir / "lip_avsr_crop_silent.mp4")
    write_video(combo, vis_dir / "face_lip_avsr_side_by_side_silent.mp4")
    for silent, with_audio in [
        ("lip_avsr_crop_silent.mp4", "lip_avsr_crop_with_audio.mp4"),
        ("face_lip_avsr_side_by_side_silent.mp4", "face_lip_avsr_side_by_side_with_audio.mp4"),
    ]:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(vis_dir / silent),
            "-i", str(clip_dir / "audio.wav"),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-shortest",
            str(vis_dir / with_audio),
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .mov/.mp4 with audio.")
    parser.add_argument("--exp", required=True, help="Output experiment name under eval_out/<exp>.")
    parser.add_argument("--fa_device", default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--force", action="store_true", help="Delete eval_out/<exp> before running.")
    parser.add_argument("--mimi_path", default=str(MAIN_ROOT / "pretrained/mimi"))
    parser.add_argument("--smollm2_path", default=str(MAIN_ROOT / "pretrained/smollm2-360m"))
    parser.add_argument(
        "--auto_avsr_ckpt",
        default=str(MAIN_ROOT / "pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth"),
    )
    parser.add_argument("--resnet50_weights", default=str(MAIN_ROOT / "pretrained/resnet50-11ad3fa6.pth"))
    parser.add_argument("--norm_stats", default=str(MAIN_ROOT / "data/processed/latent_norm_stats.npz"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument(
        "--avsr_worker",
        default=str(REPO_ROOT / "scripts/reprocess_worker_avsr.py"),
        help="Worker used to generate lip_avsr.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = Path(args.input).resolve()
    require_file(input_video, "input video")

    exp = args.exp.strip().replace("/", "_")
    exp_dir = REPO_ROOT / "eval_out" / exp
    if args.force and exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    processed_root = exp_dir / "processed"
    clip_dir = processed_root / "custom" / exp / "00001"
    clip_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.norm_stats, processed_root / "latent_norm_stats.npz")

    std_mp4 = exp_dir / f"{exp}_224_25fps.mp4"
    clip_list = exp_dir / "clip_list.txt"
    clip_list.write_text(f"custom/{exp}/00001\n")

    standardize_video(input_video, std_mp4, args.size)
    run_face_audio_preprocess(std_mp4, clip_dir, exp_dir, args.fa_device)
    run_avsr_lip_reprocess(std_mp4, clip_dir, exp_dir, args.fa_device, Path(args.avsr_worker))
    extract_mimi_latent(clip_dir, Path(args.mimi_path))
    extract_avsr_and_text(processed_root, clip_list, Path(args.auto_avsr_ckpt), Path(args.smollm2_path))
    extract_speaker_and_timbre(clip_dir, processed_root, Path(args.resnet50_weights))

    recon_dir = exp_dir / "recon_lipavsr_prompt3s"
    run_recon(processed_root, clip_list, recon_dir, Path(args.config), Path(args.ckpt))
    mux_outputs(exp_dir, exp, recon_dir, std_mp4)
    make_visualization(exp_dir, clip_dir)

    print("\nArtifacts:")
    print(f"  pred mp4: {exp_dir / f'{exp}_pred_prompt3s_post3s.mp4'}")
    print(f"  gt mp4:   {exp_dir / f'{exp}_gt_mimi_post3s.mp4'}")
    print(f"  vis mp4:  {exp_dir / 'vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4'}")
    print(f"  metrics:  {recon_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
