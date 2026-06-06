#!/usr/bin/env python3
"""Check the runtime environment for StreamLip inference.

This script intentionally avoids loading model weights. It verifies the pieces
that usually fail on a fresh machine: system binaries, Python imports, CUDA
visibility, third-party source folders, and the local ckpt/ model bundle.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def required_checkpoint_paths() -> list[Path]:
    return [
        Path("ckpt/mimi/config.json"),
        Path("ckpt/mimi/model.safetensors"),
        Path("ckpt/mimi/preprocessor_config.json"),
        Path("ckpt/smollm2-360m/config.json"),
        Path("ckpt/smollm2-360m/model.safetensors"),
        Path("ckpt/smollm2-360m/tokenizer.json"),
        Path("ckpt/streamlip-v5-lm/config.json"),
        Path("ckpt/streamlip-v5-lm/model.safetensors"),
        Path("ckpt/streamlip-v5-lm/tokenizer.json"),
        Path("ckpt/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth"),
        Path("ckpt/speaker/resnet50-11ad3fa6.pth"),
        Path("ckpt/norm/latent_norm_stats.npz"),
        Path("ckpt/v5/streamlip_v5_olmo_step_002000_infer.pt"),
        Path("ckpt/recon/streamlip_recon_timbrefix_step_002000.pt"),
        Path("ckpt/recon/streamlip_recon_residual_base_step_005000.pt"),
    ]


def missing_checkpoint_paths(repo_root: Path = REPO_ROOT) -> list[Path]:
    return [path for path in required_checkpoint_paths() if not (repo_root / path).exists()]


def collect_binary_status() -> list[CheckResult]:
    results = []
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        results.append(CheckResult(binary, bool(found), found or "not found on PATH"))
    return results


def collect_python_status() -> list[CheckResult]:
    version = sys.version_info
    ok = version.major == 3 and version.minor == 10
    detail = f"{version.major}.{version.minor}.{version.micro}"
    return [CheckResult("python==3.10", ok, detail)]


def collect_cuda_status() -> list[CheckResult]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - covered by import check in practice
        return [CheckResult("torch import", False, repr(exc))]
    available = torch.cuda.is_available()
    detail = f"torch={torch.__version__}, cuda_available={available}"
    if available:
        detail += f", device={torch.cuda.get_device_name(0)}"
    return [CheckResult("CUDA visible", available, detail)]


def collect_import_status(args: argparse.Namespace) -> list[CheckResult]:
    if getattr(args, "skip_imports", False):
        return []
    modules = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("safetensors", "safetensors"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("cv2", "opencv-python-headless"),
        ("decord", "decord"),
        ("face_alignment", "face-alignment"),
        ("kornia", "kornia"),
        ("skimage", "scikit-image"),
        ("PIL", "pillow"),
        ("gradio", "gradio"),
        ("jiwer", "jiwer"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("einops", "einops"),
        ("sentencepiece", "sentencepiece"),
        ("pytorch_lightning", "pytorch-lightning"),
    ]
    results = []
    for module_name, package_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            results.append(CheckResult(package_name, True, str(version)))
        except Exception as exc:
            results.append(CheckResult(package_name, False, f"{type(exc).__name__}: {exc}"))
    return results


def install_flash_attn_stub() -> None:
    flash_attn = types.ModuleType("flash_attn")
    modules = types.ModuleType("flash_attn.modules")
    mha = types.ModuleType("flash_attn.modules.mha")

    class FlashCrossAttention:  # pragma: no cover - only used to satisfy optional imports
        pass

    mha.FlashCrossAttention = FlashCrossAttention
    sys.modules.setdefault("flash_attn", flash_attn)
    sys.modules.setdefault("flash_attn.modules", modules)
    sys.modules.setdefault("flash_attn.modules.mha", mha)


def collect_third_party_status(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    paths = [
        Path("third_party/auto_avsr/espnet/nets/pytorch_backend/e2e_asr_conformer.py"),
        Path("third_party/auto_avsr/preparation/detectors/mediapipe/video_process.py"),
    ]
    return [
        CheckResult(str(path), (repo_root / path).exists(), "present" if (repo_root / path).exists() else "missing")
        for path in paths
    ]


def collect_checkpoint_status(repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    missing = missing_checkpoint_paths(repo_root)
    if not missing:
        return [CheckResult("ckpt bundle", True, "all required files present")]
    return [CheckResult("ckpt bundle", False, "missing:\n  " + "\n  ".join(map(str, missing)))]


def print_results(results: list[CheckResult]) -> bool:
    all_ok = True
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        all_ok = all_ok and result.ok
    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-imports", action="store_true", help="Only check binaries, Python, CUDA, third_party, and ckpt files.")
    parser.add_argument("--skip-cuda", action="store_true", help="Do not require CUDA visibility.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    install_flash_attn_stub()
    results = []
    results.extend(collect_python_status())
    results.extend(collect_binary_status())
    results.extend(collect_third_party_status())
    results.extend(collect_checkpoint_status())
    results.extend(collect_import_status(args))
    if not args.skip_cuda:
        results.extend(collect_cuda_status())
    return 0 if print_results(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
