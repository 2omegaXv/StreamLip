"""
StreamLip Phase 1 model.

Architecture:
  lip.npy → AVHuBERTExtractor (stem trainable, rest frozen) → 768-dim
           → MLP adapter → visual buffer (1152-dim)
  text tokens → Gemma-3-1B (frozen)
               every 2 layers: cross-attn(Q=text, K/V=visual buffer)
  → text logits (CE loss)
"""
import torch
import torch.nn as nn

from .visual_encoder import VisualEncoder
from .av_hubert import AVHuBERTExtractor
from .backbone import build_backbone, GEMMA3_1B_CONFIG
from .cross_attention import GatedCrossAttentionLayer


class StreamLip(nn.Module):

    D     = GEMMA3_1B_CONFIG["hidden_size"]     # 1152
    VOCAB = GEMMA3_1B_CONFIG["vocab_size"]      # 262144
    N_LAYERS = GEMMA3_1B_CONFIG["num_hidden_layers"]  # 26

    def __init__(
        self,
        avhubert_ckpt: str | None = None,
        pretrained_backbone: str | None = None,
        random_init: bool = False,
        cross_attn_every_n: int = 2,
    ):
        super().__init__()

        # 1. AV-HuBERT  (optional: only needed for raw-frame input)
        self.av_hubert: AVHuBERTExtractor | None = None
        if avhubert_ckpt:
            self.av_hubert = AVHuBERTExtractor(avhubert_ckpt)

        # 2. MLP visual adapter  (trainable): 768 → 1152
        self.visual_encoder = VisualEncoder(backbone_dim=self.D)

        # 3. Gemma-3-1B base  (frozen)
        self.backbone = build_backbone(
            pretrained_path=pretrained_backbone,
            lora_rank=0,
            random_init=random_init,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # 4. Cross-attention layers  (trainable)
        self.cross_attn_every_n = cross_attn_every_n
        n_ca = self.N_LAYERS // cross_attn_every_n
        self.cross_attn_layers = nn.ModuleList([
            GatedCrossAttentionLayer(self.D) for _ in range(n_ca)
        ])

        self._vis_buffer: torch.Tensor | None = None
        self._register_hooks()

    # ── hook injection ────────────────────────────────────────────────────────

    def _register_hooks(self):
        layers = self.backbone.model.layers   # Gemma3TextModel decoder layers
        ca_idx = 0
        for i, layer in enumerate(layers):
            if (i + 1) % self.cross_attn_every_n == 0:
                def _make_hook(ca):
                    def hook(_, _in, output):
                        if self._vis_buffer is None:
                            return output
                        # output is a Tensor (B, T, D)
                        hs = output if isinstance(output, torch.Tensor) else output[0]
                        hs = ca(hs, self._vis_buffer)
                        return hs if isinstance(output, torch.Tensor) else (hs,) + output[1:]
                    return hook
                layer.register_forward_hook(_make_hook(self.cross_attn_layers[ca_idx]))
                ca_idx += 1

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        av_feats: torch.Tensor,    # (B, T_vis, 768) OR (B, T, C, H, W) raw frames
        input_ids: torch.Tensor,   # (B, T_text)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits (B, T_text, vocab_size)."""
        # Run AV-HuBERT if raw frames provided
        if av_feats.dim() == 5:
            assert self.av_hubert is not None
            av_feats = self.av_hubert(av_feats)    # (B, T, 768)

        self._vis_buffer = self.visual_encoder(av_feats)   # (B, T_vis, 1152)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self._vis_buffer = None
        return outputs.logits   # (B, T_text, vocab_size)

    # ── utils ─────────────────────────────────────────────────────────────────

    def param_counts(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_M": total / 1e6, "trainable_M": trainable / 1e6,
                "frozen_M": (total - trainable) / 1e6}
