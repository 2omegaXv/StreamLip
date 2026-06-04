"""Run preprocess_worker with flash_attn stubbed out.

Kornia imports its optional LightGlue/flash-attn path at package import time.
In the local .venv, flash_attn is present but compiled against an incompatible
torch ABI, so importing it raises ImportError instead of ModuleNotFoundError.
preprocess_worker only needs kornia geometric transforms, not flash attention.
"""
from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


flash_attn = types.ModuleType("flash_attn")
modules = types.ModuleType("flash_attn.modules")
mha = types.ModuleType("flash_attn.modules.mha")


class FlashCrossAttention:  # pragma: no cover - never used by preprocessing
    pass


mha.FlashCrossAttention = FlashCrossAttention
sys.modules.setdefault("flash_attn", flash_attn)
sys.modules.setdefault("flash_attn.modules", modules)
sys.modules.setdefault("flash_attn.modules.mha", mha)

runpy.run_path(str(Path(__file__).with_name("preprocess_worker.py")), run_name="__main__")
