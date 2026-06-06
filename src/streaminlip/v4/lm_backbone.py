"""
LM Backbone V4: SmolLM2-360M with per-layer Gated Cross-Attention.

Each of the 8 cross-attn layers attends to a different AV-HuBERT Transformer
layer output, linearly interpolated across the 12 available layers:

  LM cross-attn i  →  AV-HuBERT layer  round(2 + i * (n_av-2) / (n_ca-1))
  (n_ca=8, n_av=12 → layers 2,3,5,6,8,9,11,12)

This gives shallow LM layers access to early visual features and deep LM
layers access to fully-processed audiovisual representations.
"""
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..cross_attention import GatedCrossAttentionLayer

CHUNK_SIZE   = 6
AVHUBERT_DIM = 1024


def _av_layer_indices(n_ca: int, n_av: int) -> list[int]:
    """
    Return n_ca 0-based AV-HuBERT layer indices spread across n_av layers.
    e.g. n_ca=8, n_av=12 → [1, 2, 4, 5, 7, 8, 10, 11]
    """
    if n_ca == 1:
        return [n_av - 1]
    return [round(i * (n_av - 1) / (n_ca - 1)) for i in range(n_ca)]


def _build_causal_vis_mask(
    lm_idx_fm:  torch.Tensor,
    L:          int,
    T_vis:      int,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    B      = lm_idx_fm.shape[0]
    device = lm_idx_fm.device
    lm_pos = lm_idx_fm.long()
    frame_idx = torch.arange(T_vis, device=device).unsqueeze(0).expand(B, -1)
    max_frame = torch.zeros(B, L, dtype=torch.long, device=device)
    max_frame.scatter_reduce_(1, lm_pos, frame_idx, reduce="amax", include_self=True)
    text_chunk = max_frame // chunk_size
    vis_chunk  = torch.arange(T_vis, device=device) // chunk_size
    return text_chunk.unsqueeze(2) >= vis_chunk.unsqueeze(0).unsqueeze(0)


class LMBackboneV4(nn.Module):

    def __init__(
        self,
        model:              nn.Module,
        cross_attn_layers:  nn.ModuleList,
        cross_attn_every_n: int,
        av_layer_indices:   list[int],   # which AV-HuBERT layer each CA uses
        chunk_size:         int = CHUNK_SIZE,
    ):
        super().__init__()
        self.model              = model
        self.cross_attn_layers  = cross_attn_layers
        self.cross_attn_every_n = cross_attn_every_n
        self.av_layer_indices   = av_layer_indices
        self.chunk_size         = chunk_size
        # layer_feats[i] will be set before each forward
        self._layer_feats: list[torch.Tensor] | None = None
        self._cross_mask:  torch.Tensor | None       = None
        self._register_hooks()

    def _register_hooks(self):
        layers = self.model.model.layers
        ca_idx = 0
        for i, layer in enumerate(layers):
            if (i + 1) % self.cross_attn_every_n == 0 and ca_idx < len(self.cross_attn_layers):
                def _make_hook(ca, av_idx):
                    def hook(_, _in, output):
                        if self._layer_feats is None:
                            return output
                        vis = self._layer_feats[av_idx]
                        hs  = output[0] if isinstance(output, tuple) else output
                        hs  = ca(hs, vis, self._cross_mask)
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook
                layer.register_forward_hook(
                    _make_hook(self.cross_attn_layers[ca_idx],
                               self.av_layer_indices[ca_idx])
                )
                ca_idx += 1

    def forward(
        self,
        input_ids:    torch.Tensor,                  # (B, L)
        attn_mask:    torch.Tensor,                  # (B, L)
        layer_feats:  list[torch.Tensor],            # 12 × (B, T, 768)
        lm_idx_fm:    torch.Tensor | None = None,    # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = input_ids.shape[1]
        T = layer_feats[0].shape[1]
        self._layer_feats = layer_feats
        self._cross_mask  = (
            _build_causal_vis_mask(lm_idx_fm, L, T, self.chunk_size)
            if lm_idx_fm is not None else None
        )
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        finally:
            self._layer_feats = None
            self._cross_mask  = None
        return out.logits, out.hidden_states[-1]


def build_lm_backbone_v4(
    pretrained_path:    str,
    cross_attn_every_n: int  = 4,
    n_av_layers:        int  = 12,
    chunk_size:         int  = CHUNK_SIZE,
    vis_dim:            int  = AVHUBERT_DIM,
) -> LMBackboneV4:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.bfloat16,
        )

    n_lm_layers = len(model.model.layers)
    n_ca        = n_lm_layers // cross_attn_every_n
    hidden_dim  = model.config.hidden_size   # 960

    av_indices = _av_layer_indices(n_ca, n_av_layers)
    print(f"[LMBackboneV4] {n_ca} cross-attn layers → AV-HuBERT layers "
          f"{[i+1 for i in av_indices]}")

    cross_attn_layers = nn.ModuleList([
        GatedCrossAttentionLayer(hidden_dim=hidden_dim, vis_dim=vis_dim)
        for _ in range(n_ca)
    ])
    for layer in cross_attn_layers:
        nn.init.constant_(layer.gate, 0.5)

    return LMBackboneV4(model, cross_attn_layers, cross_attn_every_n,
                        av_indices, chunk_size)
