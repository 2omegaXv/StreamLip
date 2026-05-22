"""
Flamingo-style gated cross-attention layer.

Inserted after every N self-attention layers in Gemma backbone.
Text tokens (Q) attend to visual buffer (K, V).
tanh gate initializes near zero → stable fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh gate, starts ~0

    def forward(self, x: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, T_text, D)
        vis: (B, T_vis,  D)   visual buffer from MLP adapter
        """
        B, T, D = x.shape
        T_v = vis.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.q(self.norm_x(x)).view(B, T,   H, Dh).transpose(1, 2)
        k = self.k(self.norm_v(vis)).view(B, T_v, H, Dh).transpose(1, 2)
        v = self.v(vis).view(B, T_v, H, Dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)   # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o(out)

        return x + torch.tanh(self.gate) * out
