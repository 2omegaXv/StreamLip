"""
AV-HuBERT feature extractor.

Loads the fairseq checkpoint by directly instantiating AVHubertModel
from the checkpoint's w2v_args config, bypassing fairseq task setup.
Patches the 3D stem for RGB input (1-ch → 3-ch).
"""
import sys
import torch
import torch.nn as nn
from argparse import Namespace
from pathlib import Path

_AVHUBERT_REGISTERED = False


def _ensure_avhubert():
    """Register av_hubert fairseq models/tasks (once only)."""
    global _AVHUBERT_REGISTERED
    if _AVHUBERT_REGISTERED:
        return
    _AVHUBERT_REGISTERED = True

    # Add av_hubert to path
    _root = str(Path(__file__).parent.parent.parent / "third_party/av_hubert")
    for _p in [_root, _root + "/avhubert"]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    # Patch registries to ignore duplicates
    import fairseq.models as _fm
    import fairseq.tasks as _ft

    _orig_rm = _fm.register_model
    def _safe_rm(name, dataclass=None):
        def wrap(cls):
            return cls if name in _fm.MODEL_REGISTRY else (
                _orig_rm(name, dataclass=dataclass)(cls) if dataclass else _orig_rm(name)(cls))
        return wrap
    _fm.register_model = _safe_rm

    _orig_rt = _ft.register_task
    def _safe_rt(name, dataclass=None):
        def wrap(cls):
            return cls if name in _ft.TASK_REGISTRY else (
                _orig_rt(name, dataclass=dataclass)(cls) if dataclass else _orig_rt(name)(cls))
        return wrap
    _ft.register_task = _safe_rt

    import avhubert  # noqa


def load_avhubert(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load AVHubertModel (encoder only) from checkpoint.
    Patches frontend3D stem for RGB (1-ch → 3-ch).
    Only the stem is left trainable; rest is frozen.
    """
    _ensure_avhubert()

    # Patch torch.load for torch 2.6+ weights_only compat
    import functools
    _orig_load = torch.load
    @functools.wraps(_orig_load)
    def _load_unsafe(f, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(f, map_location=map_location, **kwargs)

    # Load checkpoint
    ckpt = _load_unsafe(checkpoint_path, map_location="cpu")

    # Build model from w2v_args stored in checkpoint
    from avhubert.hubert import AVHubertModel
    from omegaconf import OmegaConf, DictConfig

    w2v_args = ckpt["cfg"]["model"]["w2v_args"]
    model_cfg_dict = dict(w2v_args["model"])
    task_cfg_dict  = dict(w2v_args["task"])

    # Add any missing fields with safe defaults
    model_cfg_dict.setdefault("required_seq_len_multiple", 1)
    model_cfg_dict.setdefault("resnet_relu_type", "relu")
    model_cfg_dict.setdefault("resnet_weights", None)
    model_cfg_dict.setdefault("sub_encoder_layers", 0)
    model_cfg_dict.setdefault("pos_conv_depth", 1)
    model_cfg_dict.setdefault("layer_type", "transformer")
    task_cfg_dict.setdefault("input_modality", "video")

    # Infer None fields from state_dict
    _state_tmp = {k[len("encoder.w2v_model."):]: v
                  for k, v in ckpt["model"].items()
                  if k.startswith("encoder.w2v_model.")}

    def _infer(key, state_key_substr, shape_idx, default):
        if model_cfg_dict.get(key):
            return
        for _k, _v in _state_tmp.items():
            if state_key_substr in _k:
                model_cfg_dict[key] = _v.shape[shape_idx]
                return
        model_cfg_dict[key] = default

    _infer("encoder_embed_dim",     "encoder.layers.0.self_attn.q_proj.weight", 1, 768)
    _infer("encoder_ffn_embed_dim", "encoder.layers.0.fc1.weight",              0, 3072)
    _infer("encoder_attention_heads", "encoder.layers.0.self_attn.q_proj.weight", 1, 768)
    model_cfg_dict["encoder_attention_heads"] = (
        model_cfg_dict["encoder_attention_heads"] // 64
    )

    if not model_cfg_dict.get("encoder_layers"):
        n = max((int(_k.split("encoder.layers.")[1].split(".")[0]) + 1
                 for _k in _state_tmp if "encoder.layers." in _k), default=12)
        model_cfg_dict["encoder_layers"] = n

    # Build config: start from dataclass defaults, then override with checkpoint values
    from avhubert.hubert import AVHubertConfig
    from avhubert.hubert_pretraining import AVHubertPretrainingConfig
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.structured(AVHubertConfig)
    task_cfg  = OmegaConf.structured(AVHubertPretrainingConfig)

    # Override with checkpoint values (skip None)
    OmegaConf.set_struct(model_cfg, False)
    OmegaConf.set_struct(task_cfg, False)
    for k, v in model_cfg_dict.items():
        if v is not None and k != "_name":
            try:
                OmegaConf.update(model_cfg, k, v)
            except Exception:
                pass
    for k, v in task_cfg_dict.items():
        if v is not None and k != "_name":
            try:
                OmegaConf.update(task_cfg, k, v)
            except Exception:
                pass

    # Instantiate model
    from fairseq.data.dictionary import Dictionary
    dummy_dict = Dictionary()
    model = AVHubertModel.build_model(
        model_cfg,
        task=Namespace(cfg=task_cfg, dictionaries=[dummy_dict]),
    )

    # Load weights (encoder.w2v_model.* → stripped prefix)
    state = {
        k[len("encoder.w2v_model."):]: v
        for k, v in ckpt["model"].items()
        if k.startswith("encoder.w2v_model.")
    }
    # Remove label embedding (depends on vocab size; not needed for feature extraction)
    state.pop("label_embs_concat", None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[av_hubert] missing keys: {len(missing)}")
    if unexpected:
        print(f"[av_hubert] unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()

    # ── RGB patch: stem (64,1,5,7,7) → (64,3,5,7,7) ──────────────────────
    for module in model.modules():
        if (hasattr(module, "weight") and module.weight is not None
                and tuple(module.weight.shape) == (64, 1, 5, 7, 7)):
            w = module.weight.data.clone()
            module.weight = nn.Parameter(
                w.repeat(1, 3, 1, 1, 1) / 3.0
            )
            # stem is trainable; everything else frozen
            module.weight.requires_grad_(True)
            break

    for name, p in model.named_parameters():
        if "frontend3D.0" not in name:
            p.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[av_hubert] loaded. trainable params: {n_train/1e6:.2f}M (stem only)")

    return model


class AVHuBERTExtractor(nn.Module):
    """AV-HuBERT encoder: lip frames → 1024-dim features."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.model = load_avhubert(checkpoint_path, device)

    def forward(self, lip_frames: torch.Tensor) -> torch.Tensor:
        """
        lip_frames: (B, T, C, H, W)  float32, C=3, H=W=96, values ∈ [-1,1]
        returns:    (B, T, D)  D=768 for this model
        """
        B, T, C, H, W = lip_frames.shape
        x = lip_frames.permute(0, 2, 1, 3, 4)   # (B, C, T, H, W)
        # Cast input to match model weight dtype (float32 normally, bf16 after .bfloat16())
        x = x.to(next(self.model.parameters()).dtype)
        feats, _ = self.model.extract_finetune(
            source={"video": x, "audio": None},
            padding_mask=None,
            mask=False,
        )
        # Return in same dtype as input
        return feats.to(lip_frames.dtype)  # (B, T, D)
