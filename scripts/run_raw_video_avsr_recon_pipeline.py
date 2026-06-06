"""One-command raw video -> StreamLip-conditioned audio reconstruction pipeline.

This script is intentionally small and explicit. It reproduces the raw-video
flow used for the external trump/hrx checks:

1. standardize input video to 224x224, 25 fps, 24 kHz mono audio
   - for silent video with --ref_audio, prepend a black 3.04s prompt segment
     carrying the first 3.04s of reference audio
2. run the existing face/audio preprocessing for face.npz, audio.wav, lip.npy
3. run the StreamLip visual preprocessing worker for lip_avsr.npy
4. extract Mimi latent, visual encoder latent, StreamLip V5 text, SmolLM2
   hidden states, speaker/timbre conditions
5. run the current best recon checkpoint with 3s audio prompt
6. mux post-3.04s predicted audio back to the standardized video, cropping the
   prompt segment away
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
DEFAULT_CONFIG = (
    "configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_"
    "residual_samplecorr02_lossstart38_from1500_recon_textjson_wordts.yaml"
)
DEFAULT_CKPT = (
    "ckpt/recon/streamlip_recon_timbrefix_step_002000.pt"
)
DEFAULT_V5_CKPT = "ckpt/v5/streamlip_v5_olmo_step_001500_infer.pt"
DEFAULT_V5_LM_PATH = "ckpt/streamlip-v5-lm"
PROMPT_FRAMES = 38
LATENT_DIM = 512
VIDEO_FPS = 25.0
LATENT_HZ = 12.5
PROMPT_SECONDS = PROMPT_FRAMES / LATENT_HZ


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _path_str(value: str | Path) -> str:
    return str(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def has_audio_stream(video_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def effective_silent_input(requested_silent: bool, input_video: Path) -> bool:
    return bool(requested_silent or not has_audio_stream(input_video))


def standardize_video(input_video: Path, out_mp4: Path, size: int, *, silent_input: bool = False) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_video),
        "-vf", f"fps=25,scale={size}:{size}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
    ]
    if silent_input:
        cmd += ["-an"]
    else:
        cmd += ["-ar", "24000", "-ac", "1", "-c:a", "aac"]
    cmd.append(str(out_mp4))
    run(cmd)


def add_silent_audio(video_mp4: Path, out_mp4: Path) -> None:
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_mp4),
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_mp4),
    ])


def should_concat_ref_prompt_prefix(*, silent_input: bool, has_ref_audio: bool) -> bool:
    return bool(silent_input and has_ref_audio)


def make_black_ref_prompt_prefix(ref_audio: Path, out_mp4: Path, size: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-t", f"{PROMPT_SECONDS:.6f}",
        "-i", f"color=c=black:s={size}x{size}:r=25",
        "-i", str(ref_audio),
        "-filter_complex",
        (
            f"[1:a]atrim=0:{PROMPT_SECONDS:.6f},asetpts=PTS-STARTPTS,"
            "aresample=24000,aformat=channel_layouts=mono,"
            f"apad=whole_dur={PROMPT_SECONDS:.6f}[a]"
        ),
        "-map", "0:v:0", "-map", "[a]",
        "-t", f"{PROMPT_SECONDS:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out_mp4),
    ])


def concat_prompt_prefix_video(prompt_mp4: Path, body_mp4: Path, out_mp4: Path) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(prompt_mp4),
        "-i", str(body_mp4),
        "-filter_complex",
        (
            "[0:v]setsar=1,format=yuv420p[v0];"
            "[1:v]setsar=1,format=yuv420p[v1];"
            "[0:a]aresample=24000,aformat=channel_layouts=mono[a0];"
            "[1:a]aresample=24000,aformat=channel_layouts=mono[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        ),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out_mp4),
    ])


def build_silent_ref_prompt_video(body_silent_mp4: Path, ref_audio: Path, out_mp4: Path, size: int) -> None:
    prompt_mp4 = out_mp4.with_name(out_mp4.stem + "_black_ref_prompt3s.mp4")
    body_audio_mp4 = out_mp4.with_name(out_mp4.stem + "_body_silent_audio.mp4")
    make_black_ref_prompt_prefix(ref_audio, prompt_mp4, size)
    add_silent_audio(body_silent_mp4, body_audio_mp4)
    concat_prompt_prefix_video(prompt_mp4, body_audio_mp4, out_mp4)


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


def encode_audio_latent(audio_path: Path, mimi_path: Path) -> np.ndarray:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.preprocess_lrs3 import build_mimi, extract_latent
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mimi = build_mimi(device, mimi_path)
    return extract_latent(mimi, audio_path, device).astype("float32")


def standardize_ref_audio(ref_audio: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(ref_audio),
        "-vn", "-ar", "24000", "-ac", "1",
        str(out_wav),
    ])


def build_audio_prompt_condition(latent: np.ndarray | None, prompt_frames: int = PROMPT_FRAMES) -> np.ndarray:
    prompt = np.zeros((prompt_frames, LATENT_DIM), dtype=np.float32)
    if latent is None:
        return prompt
    n_prompt = min(prompt_frames, latent.shape[0])
    prompt[:n_prompt] = latent[:n_prompt, :LATENT_DIM]
    return prompt


def build_timbre_condition_for_pipeline(latent: np.ndarray | None) -> np.ndarray:
    if latent is None:
        return np.zeros((LATENT_DIM * 2,), dtype=np.float32)
    from scripts.extract_timbre_cond import build_timbre_condition

    return build_timbre_condition(latent, prompt_frames=PROMPT_FRAMES).astype("float32")


def write_dummy_target_latent(clip_dir: Path, n_video_frames: int) -> np.ndarray:
    n_latent = max(1, int(np.ceil(float(n_video_frames) * LATENT_HZ / VIDEO_FPS)))
    latent = np.zeros((n_latent, LATENT_DIM), dtype=np.float16)
    np.savez(str(clip_dir / "latent.npz"), latent=latent)
    return latent


def extract_speaker_and_timbre(
    clip_dir: Path,
    processed_root: Path,
    resnet50_weights: Path,
    *,
    ref_latent: np.ndarray | None = None,
    zero_condition: bool = False,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from scripts.extract_speaker_emb import extract_clip
    from streaminlip.fm_avsr_dataset import normalize_latent, set_norm_stats_path
    from streaminlip.v2.speaker_encoder import SpeakerEncoder

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = SpeakerEncoder(weights_path=str(resnet50_weights)).to(device).eval()
    extract_clip(encoder, clip_dir, device, force=True)
    set_norm_stats_path(processed_root / "latent_norm_stats.npz")
    latent = None
    if not zero_condition:
        latent = ref_latent
    if latent is None and not zero_condition:
        latent = np.load(str(clip_dir / "latent.npz"))["latent"].astype("float32")
    norm_latent = normalize_latent(latent) if latent is not None else None
    timbre = build_timbre_condition_for_pipeline(norm_latent)
    np.save(str(clip_dir / "timbre_cond.npy"), timbre.astype("float16"))
    prompt = build_audio_prompt_condition(norm_latent)
    np.save(str(clip_dir / "audio_prompt.npy"), prompt.astype("float16"))
    print(
        f"speaker: {np.load(str(clip_dir / 'speaker_emb.npy')).shape} "
        f"timbre: {timbre.shape} audio_prompt: {prompt.shape}",
        flush=True,
    )


def build_extract_avsr_enc_command(
    processed_root: str | Path,
    clip_list: str | Path,
    auto_avsr_ckpt: str | Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/extract_avsr_enc.py"),
        "--data_root", _path_str(processed_root),
        "--clip_list", _path_str(clip_list),
        "--input_name", "lip_avsr.npy",
        "--output_name", "avsr_enc_lipavsr.npy",
        "--text_output_name", "avsr_text_lipavsr.txt",
        "--avsr_ckpt", _path_str(auto_avsr_ckpt),
        "--gpu", "0",
        "--force",
        "--batch_size", "1",
    ]


def build_extract_v5_text_command(
    processed_root: str | Path,
    clip_list: str | Path,
    avsr_ckpt: str | Path,
    v5_ckpt: str | Path,
    v5_lm_path: str | Path,
    *,
    beam: int = 3,
    cross_attn_every_n: int = 4,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/extract_v5_text.py"),
        "--data_root", _path_str(processed_root),
        "--clip_list", _path_str(clip_list),
        "--input_name", "avsr_enc_lipavsr.npy",
        "--output_name", "streamlip_v5_text.txt",
        "--avsr_ckpt", _path_str(avsr_ckpt),
        "--v5_ckpt", _path_str(v5_ckpt),
        "--v5_lm_path", _path_str(v5_lm_path),
        "--cross_attn_every_n", str(cross_attn_every_n),
        "--beam", str(beam),
        "--overwrite",
    ]


def build_extract_smollm2_command(
    processed_root: str | Path,
    clip_list: str | Path,
    smollm2_path: str | Path,
    text_source: str,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/extract_smollm2_h.py"),
        "--data_root", _path_str(processed_root),
        "--clip_list", _path_str(clip_list),
        "--text_source", text_source,
        "--smollm2_path", _path_str(smollm2_path),
        "--batch_size", "1",
        "--overwrite",
    ]


def extract_avsr_and_text(
    processed_root: Path,
    clip_list: Path,
    auto_avsr_ckpt: Path,
    smollm2_path: Path,
    *,
    text_model: str,
    v5_ckpt: Path,
    v5_lm_path: Path,
    v5_beam: int,
) -> str:
    run(build_extract_avsr_enc_command(processed_root, clip_list, auto_avsr_ckpt))
    rel = clip_list.read_text().strip()
    clip_dir = processed_root / rel
    shutil.copyfile(clip_dir / "avsr_text_lipavsr.txt", clip_dir / "avsr_text.txt")
    if text_model == "v5":
        run(build_extract_v5_text_command(
            processed_root,
            clip_list,
            auto_avsr_ckpt,
            v5_ckpt,
            v5_lm_path,
            beam=v5_beam,
        ))
        text_source = "v5"
    elif text_model == "avsr":
        text_source = "lipavsr"
    else:
        raise ValueError(f"unknown text_model={text_model!r}")
    run(build_extract_smollm2_command(processed_root, clip_list, smollm2_path, text_source))
    return text_source


def build_eval_recon_command(
    processed_root: str | Path,
    clip_list: str | Path,
    output_dir: str | Path,
    config: str | Path,
    ckpt: str | Path,
    text_source: str,
    *,
    silent_input: bool = False,
) -> list[str]:
    output_dir = Path(output_dir)
    wav_start = recon_wav_start_frame(silent_input=silent_input)
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/eval_fm_avsr.py"),
        "--config", _path_str(config),
        "--ckpt", _path_str(ckpt),
        "--data_root", _path_str(processed_root),
        "--clip_list", _path_str(clip_list),
        "--text_source", text_source,
        "--text_alignment_mode", "uniform",
        "--output_dir", _path_str(output_dir),
        "--n", "1",
        "--use_recon",
        "--audio_prompt_name", "audio_prompt.npy",
        "--wav_start_frame", str(wav_start),
        "--metric_start_frame", str(wav_start),
        "--metrics_json", str(output_dir / "metrics.json"),
    ] + ([] if silent_input else ["--save_gt"])


def run_recon(
    processed_root: Path,
    clip_list: Path,
    output_dir: Path,
    config: Path,
    ckpt: Path,
    text_source: str,
    *,
    silent_input: bool = False,
) -> None:
    run(build_eval_recon_command(
        processed_root,
        clip_list,
        output_dir,
        config,
        ckpt,
        text_source,
        silent_input=silent_input,
    ))


def recon_wav_start_frame(*, silent_input: bool) -> int:
    return PROMPT_FRAMES


def result_video_names(exp: str, *, silent_input: bool) -> tuple[str, str | None]:
    if silent_input:
        return f"{exp}_pred_post3s.mp4", None
    return f"{exp}_pred_prompt3s_post3s.mp4", f"{exp}_gt_mimi_post3s.mp4"


def mux_outputs(exp_dir: Path, exp: str, recon_dir: Path, std_mp4: Path, *, silent_input: bool) -> None:
    pred_name, gt_name = result_video_names(exp, silent_input=silent_input)
    jobs = [("0000_pred.wav", exp_dir / pred_name)]
    if gt_name is not None:
        jobs.append(("0000_gt.wav", exp_dir / gt_name))
    for wav_name, out_mp4 in jobs:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        ]
        cmd += ["-ss", "3.04"]
        cmd += [
            "-i", str(std_mp4),
            "-i", str(recon_dir / wav_name),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-shortest",
            str(out_mp4),
        ]
        run([
            *cmd,
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
    parser.add_argument("--input", required=True, help="Input .mov/.mp4 video.")
    parser.add_argument("--exp", required=True, help="Output experiment name under eval_out/<exp>.")
    parser.add_argument("--silent_input", action="store_true",
                        help="Treat input as silent video and keep full output length.")
    parser.add_argument("--ref_audio", default=None,
                        help="Optional reference audio/video used for timbre and audio prompt in silent_input mode.")
    parser.add_argument("--fa_device", default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--force", action="store_true", help="Delete eval_out/<exp> before running.")
    parser.add_argument("--mimi_path", default=str(REPO_ROOT / "ckpt/mimi"))
    parser.add_argument("--smollm2_path", default=str(REPO_ROOT / "ckpt/smollm2-360m"))
    parser.add_argument("--text_model", choices=["v5", "avsr"], default="v5",
                        help="Visual-to-text model used for the semantic condition. Default: StreamLip V5.")
    parser.add_argument("--v5_ckpt", default=str(REPO_ROOT / DEFAULT_V5_CKPT),
                        help="StreamLip V5 checkpoint used when --text_model v5.")
    parser.add_argument("--v5_lm_path", default=str(REPO_ROOT / DEFAULT_V5_LM_PATH),
                        help="LM/tokenizer path for StreamLip V5 decoding.")
    parser.add_argument("--v5_beam", type=int, default=3,
                        help="Beam size for StreamLip V5 offline decoding.")
    parser.add_argument(
        "--auto_avsr_ckpt",
        default=str(REPO_ROOT / "ckpt/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth"),
    )
    parser.add_argument("--resnet50_weights", default=str(REPO_ROOT / "ckpt/speaker/resnet50-11ad3fa6.pth"))
    parser.add_argument("--norm_stats", default=str(REPO_ROOT / "ckpt/norm/latent_norm_stats.npz"))
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
    ref_audio = Path(args.ref_audio).resolve() if args.ref_audio else None
    if ref_audio is not None:
        require_file(ref_audio, "reference audio")

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
    preprocess_mp4 = std_mp4
    clip_list = exp_dir / "clip_list.txt"
    clip_list.write_text(f"custom/{exp}/00001\n")

    silent_input = effective_silent_input(args.silent_input, input_video)
    if silent_input and not args.silent_input:
        print("Input has no audio stream; enabling --silent_input automatically.", flush=True)

    standardize_video(input_video, std_mp4, args.size, silent_input=silent_input)
    concat_ref_prompt = should_concat_ref_prompt_prefix(
        silent_input=silent_input,
        has_ref_audio=ref_audio is not None,
    )
    if concat_ref_prompt:
        ref_prompt_mp4 = exp_dir / f"{exp}_224_25fps_refprompt_concat.mp4"
        build_silent_ref_prompt_video(std_mp4, ref_audio, ref_prompt_mp4, args.size)
        std_mp4 = ref_prompt_mp4
        preprocess_mp4 = ref_prompt_mp4
    elif silent_input:
        preprocess_mp4 = exp_dir / f"{exp}_224_25fps_silent_audio_for_preprocess.mp4"
        add_silent_audio(std_mp4, preprocess_mp4)
    run_face_audio_preprocess(preprocess_mp4, clip_dir, exp_dir, args.fa_device)
    run_avsr_lip_reprocess(std_mp4, clip_dir, exp_dir, args.fa_device, Path(args.avsr_worker))
    ref_latent = None
    if silent_input and not concat_ref_prompt:
        n_video_frames = int(np.load(str(clip_dir / "lip_avsr.npy"), mmap_mode="r").shape[0])
        write_dummy_target_latent(clip_dir, n_video_frames)
    else:
        extract_mimi_latent(clip_dir, Path(args.mimi_path))
    text_source = extract_avsr_and_text(
        processed_root,
        clip_list,
        Path(args.auto_avsr_ckpt),
        Path(args.smollm2_path),
        text_model=args.text_model,
        v5_ckpt=Path(args.v5_ckpt),
        v5_lm_path=Path(args.v5_lm_path),
        v5_beam=args.v5_beam,
    )
    extract_speaker_and_timbre(
        clip_dir,
        processed_root,
        Path(args.resnet50_weights),
        ref_latent=ref_latent,
        zero_condition=silent_input and ref_latent is None,
    )

    recon_dir = exp_dir / "recon_lipavsr_prompt3s"
    run_recon(
        processed_root,
        clip_list,
        recon_dir,
        Path(args.config),
        Path(args.ckpt),
        text_source,
        silent_input=silent_input,
    )
    mux_outputs(exp_dir, exp, recon_dir, std_mp4, silent_input=silent_input)
    make_visualization(exp_dir, clip_dir)

    pred_name, gt_name = result_video_names(exp, silent_input=silent_input)
    print("\nArtifacts:")
    print(f"  pred mp4: {exp_dir / pred_name}")
    if gt_name:
        print(f"  gt mp4:   {exp_dir / gt_name}")
    print(f"  vis mp4:  {exp_dir / 'vis_reprocess_avsr/face_lip_avsr_side_by_side_with_audio.mp4'}")
    print(f"  metrics:  {recon_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
