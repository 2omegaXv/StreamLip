"""
LM Backbone V3: SmolLM2-360M with Gated Cross-Attention to visual features.

GatedCrossAttentionLayer (Q=text, K/V=vis_feat) injected via forward hooks
every `cross_attn_every_n` transformer layers.

SmolLM2-360M: hidden_size=960 = vis_feat dim → no projection needed.
32 layers, cross_attn_every_n=4 → 8 cross-attention layers.

Chunk-causal masking (streaming consistency):
  Text token l may only attend to visual frames up to (and including) the
  chunk that contains frame lm_idx_fm[l].  This matches inference, where
  the LM has only seen frames up to the current chunk boundary.
  mask[b, l, t] = True  iff  chunk(lm_frame[b,l]) >= chunk(t)
  where lm_frame[b,l] = lm_idx_fm[b, :].argmax over positions that equal l
  (approximated cheaply as: the latest vis frame whose lm_idx_fm == l).

Returns:
  logits (B, L, 49152)   — visually-conditioned next-token predictions
  h_lm   (B, L, 960)     — last hidden states for FM conditioning
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ..cross_attention import GatedCrossAttentionLayer

CHUNK_SIZE = 6  # must match ConformerAdapter / dataset


def _build_causal_vis_mask(
    lm_idx_fm:  torch.Tensor,   # (B, T_vis)  — per-frame LM position after commit
    L:          int,             # number of text tokens
    T_vis:      int,             # number of visual frames
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """
    Returns bool mask (B, L, T_vis): True = text token l may attend to vis frame t.

    For each text position l, find the latest visual frame f whose lm_idx_fm == l,
    then allow attending to all frames in chunk(f) and earlier chunks.
    Falls back to the very first frame (chunk 0) for positions not found.
    """
    B = lm_idx_fm.shape[0]
    device = lm_idx_fm.device

    # lm_idx_fm: (B, T_vis)  values in [0, L)
    # For each (b, l), find max t s.t. lm_idx_fm[b,t] == l (or 0 if none).
    # Efficient: scatter max over the T_vis dimension.
    lm_pos = lm_idx_fm.long()                              # (B, T_vis)
    frame_idx = torch.arange(T_vis, device=device).unsqueeze(0).expand(B, -1)  # (B, T_vis)

    # max_frame[b, l] = largest t where lm_idx_fm[b,t] == l; defaults to 0
    max_frame = torch.zeros(B, L, dtype=torch.long, device=device)
    max_frame.scatter_reduce_(1, lm_pos, frame_idx, reduce="amax", include_self=True)

    # chunk index for each text position
    text_chunk = max_frame // chunk_size                   # (B, L)

    # chunk index for each visual frame
    vis_chunk = torch.arange(T_vis, device=device) // chunk_size  # (T_vis,)

    # mask[b, l, t] = True iff text_chunk[b,l] >= vis_chunk[t]
    mask = text_chunk.unsqueeze(2) >= vis_chunk.unsqueeze(0).unsqueeze(0)  # (B, L, T_vis)
    return mask


class LMBackboneV3(nn.Module):

    def __init__(
        self,
        model:              nn.Module,
        cross_attn_layers:  nn.ModuleList,
        cross_attn_every_n: int,
        chunk_size:         int = CHUNK_SIZE,
    ):
        super().__init__()
        self.model              = model
        self.cross_attn_layers  = cross_attn_layers
        self.cross_attn_every_n = cross_attn_every_n
        self.chunk_size         = chunk_size
        self._vis_feat:    torch.Tensor | None = None
        self._cross_mask:  torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self):
        # SmolLM2 is LlamaForCausalLM: model.model.layers
        layers = self.model.model.layers
        ca_idx = 0
        for i, layer in enumerate(layers):
            if (i + 1) % self.cross_attn_every_n == 0 and ca_idx < len(self.cross_attn_layers):
                def _make_hook(ca):
                    def hook(_, _in, output):
                        if self._vis_feat is None:
                            return output
                        hs = output[0] if isinstance(output, tuple) else output
                        hs = ca(hs, self._vis_feat, self._cross_mask)
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook
                layer.register_forward_hook(_make_hook(self.cross_attn_layers[ca_idx]))
                ca_idx += 1

    def forward(
        self,
        input_ids:      torch.Tensor,               # (B, L)
        attention_mask: torch.Tensor,               # (B, L)
        vis_feat:       torch.Tensor,               # (B, T, 960)
        lm_idx_fm:      torch.Tensor | None = None, # (B, T) for chunk-causal mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B,L,vocab), h_lm (B,L,960))."""
        L = input_ids.shape[1]
        T = vis_feat.shape[1]
        self._vis_feat   = vis_feat
        self._cross_mask = (
            _build_causal_vis_mask(lm_idx_fm, L, T, self.chunk_size)
            if lm_idx_fm is not None else None
        )
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        finally:
            self._vis_feat   = None
            self._cross_mask = None
        return out.logits, out.hidden_states[-1]


def build_lm_backbone_v3(
    pretrained_path:    str,
    cross_attn_every_n: int = 4,
    chunk_size:         int = CHUNK_SIZE,
) -> LMBackboneV3:
    """
    Load SmolLM2-360M (fully frozen) and inject GatedCrossAttentionLayers.
    Only cross_attn_layers are trainable in phase 1.
    """
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

    n_layers   = len(model.model.layers)
    n_ca       = n_layers // cross_attn_every_n
    hidden_dim = model.config.hidden_size   # 960

    cross_attn_layers = nn.ModuleList([
        GatedCrossAttentionLayer(hidden_dim=hidden_dim)
        for _ in range(n_ca)
    ])
    for layer in cross_attn_layers:
        nn.init.constant_(layer.gate, 0.1)

    return LMBackboneV3(model, cross_attn_layers, cross_attn_every_n, chunk_size)
