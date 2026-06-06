"""
Flamingo-style gated cross-attention layer.

Inserted after every N self-attention layers in the LM backbone.
Text tokens (Q) attend to visual buffer (K, V).
tanh gate initializes near zero → stable fine-tuning.

Supports an optional chunk-causal attn_mask (B, L, T_vis) to enforce
streaming causality: text position l may only attend to visual frames
whose chunk has already been "committed" by the time l is processed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, vis_dim: int | None = None):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        vis_dim = vis_dim or hidden_dim  # K/V input dim, defaults to Q dim

        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(vis_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(vis_dim,    hidden_dim, bias=False)
        self.v = nn.Linear(vis_dim,    hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh gate, starts ~0

    def forward(
        self,
        x:         torch.Tensor,               # (B, T_text, D)
        vis:       torch.Tensor,               # (B, T_vis,  D)
        attn_mask: torch.Tensor | None = None, # (B, T_text, T_vis) bool, True=attend
    ) -> torch.Tensor:
        B, T, D = x.shape
        T_v = vis.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q   = self.q(self.norm_x(x)).view(B, T,   H, Dh).transpose(1, 2)
        vis_n = self.norm_v(vis)
        k   = self.k(vis_n).view(B, T_v, H, Dh).transpose(1, 2)
        v   = self.v(vis_n).view(B, T_v, H, Dh).transpose(1, 2)

        # attn_mask (B, T, T_v) → (B, 1, T, T_v) for head broadcast
        mask = attn_mask.unsqueeze(1) if attn_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o(out)

        return x + torch.tanh(self.gate).to(x.dtype) * out
