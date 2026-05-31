"""
FM Head V4: same as V2 but COND_DIM uses AV-HuBERT dim (1024) instead of Conformer dim (960).

Condition: cond_proj( sg(last_feat↓2x) ∥ sg(h̃_lm↓2x) ∥ id̂ )
  last_feat: (B, T_a, 1024)  AV-HuBERT final layer, downsampled 2×
  h̃_lm:     (B, T_a, 960)   LM hidden state, downsampled 2×
  id̂:        (B, 256)        speaker identity
"""
from ..v2.fm_head import FMHead as _FMHeadBase, SinusoidalTimeEmb, DiTBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 512
COND_DIM   = 1024 + 960 + 256   # avhubert_last + lm_hidden + speaker = 2240


class FMHeadV4(_FMHeadBase):
    """FM Head with updated COND_DIM for V4 (1024 vis instead of 960)."""

    DIM = LATENT_DIM

    def __init__(self, n_layers: int = 6, n_heads: int = 8):
        # bypass parent __init__ to set different COND_DIM
        nn.Module.__init__(self)
        self.cond_proj  = nn.Linear(COND_DIM, self.DIM)
        self.cond_token_proj = nn.Linear(self.DIM, self.DIM)
        self.time_emb   = SinusoidalTimeEmb(self.DIM)
        self.blocks     = nn.ModuleList([DiTBlock(self.DIM, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(self.DIM)
        self.final_proj = nn.Linear(self.DIM, self.DIM)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
