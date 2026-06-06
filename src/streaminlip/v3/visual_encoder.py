"""
Visual Encoder V3: AV-HuBERT (frozen, stem trainable) + Conformer Adapter.

Identical to V2 but without the visual classification head.
Output vis_feat (B, T, 960) is used as K/V in the LM cross-attention layers.
"""
import torch
import torch.nn as nn

from ..av_hubert import AVHuBERTExtractor
from ..v2.visual_encoder import ConformerAdapter, BACKBONE_DIM, CHUNK_SIZE


class VisualEncoderV3(nn.Module):

    def __init__(
        self,
        avhubert_ckpt:      str,
        n_conformer_layers: int = 4,
        chunk_size:         int = CHUNK_SIZE,
    ):
        super().__init__()
        self.av_hubert = AVHuBERTExtractor(avhubert_ckpt, device="cpu")
        self.conformer = ConformerAdapter(n_conformer_layers, chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3, 96, 96) raw lip frames  — runs AV-HuBERT
           OR (B, T, 768)     pre-extracted   — fast path
        returns: vis_feat (B, T, 960)
        """
        if x.dim() == 5:
            feats = self.av_hubert(x)
        else:
            feats = x
        return self.conformer(feats)   # (B, T, 960)
