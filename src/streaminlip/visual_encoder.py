"""
AV-HuBERT visual encoder + MLP adapter.

AV-HuBERT Large:
  - frontend3D stem: (64, 1, 5, 7, 7)  ← expects grayscale (1-ch)
  - LRS3 data is RGB (3-ch)  → patch stem in_channels 1→3 at load time
  - output: 1024-dim per frame (frozen)

MLP adapter projects 1024 → backbone_dim (trainable).
"""
import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    AV-HuBERT Large (frozen) + 2-layer MLP adapter (trainable).

    Accepts either:
      - Pre-extracted AV-HuBERT features: (B, T, 1024)   [Phase 1 default]
      - Raw lip frames: (B, T, C, H, W)                  [requires loaded av_hubert]
    """

    AVHUBERT_DIM = 768  # this checkpoint outputs 768-dim (encoder_embed_dim)

    def __init__(self, backbone_dim: int, av_hubert=None):
        super().__init__()
        self.av_hubert = av_hubert  # None → expects pre-extracted features

        # 2-layer MLP: 1024 → backbone_dim
        hidden = (self.AVHUBERT_DIM + backbone_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(self.AVHUBERT_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, backbone_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 768) pre-extracted features  OR  (B, T, C, H, W) raw frames
        returns: (B, T, backbone_dim)
        """
        if x.dim() == 5:
            assert self.av_hubert is not None
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            with torch.no_grad():
                x = self.av_hubert(x)
            x = x.view(B, T, self.AVHUBERT_DIM)
        # Run MLP in its own dtype, then cast back
        dtype = x.dtype
        return self.mlp(x.to(next(self.mlp.parameters()).dtype)).to(dtype)


def patch_avhubert_stem_for_rgb(state_dict: dict) -> dict:
    """
    AV-HuBERT's frontend3D stem expects 1-channel (grayscale) input.
    LRS3 videos are RGB (3-channel).

    Patch: average the single-channel weight across 3 channels so that
    the mean-grayscale of RGB ≈ the original grayscale response.

    Key: encoder.w2v_model.feature_extractor_video.resnet.frontend3D.0.weight
    Shape: (64, 1, 5, 7, 7) → (64, 3, 5, 7, 7)
    """
    key = "encoder.w2v_model.feature_extractor_video.resnet.frontend3D.0.weight"
    if key in state_dict:
        w = state_dict[key]          # (64, 1, 5, 7, 7)
        state_dict[key] = w.repeat(1, 3, 1, 1, 1) / 3.0   # (64, 3, 5, 7, 7)
    return state_dict
