"""
Visual Encoder V2: AV-HuBERT (frozen, stem trainable) + Conformer Adapter + Visual Head.

AV-HuBERT Large outputs 768-dim per-frame features.
Conformer Adapter: N-layer chunk-causal Transformer, 768 → 960.
Visual Head: Linear(960, 49152) → visual logits s_vis = log p(x_t | v_t).

chunk_size=6 (240ms @ 25fps). Within-chunk: bidirectional attention.
Cross-chunk: causal (can attend to all past chunks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..av_hubert import AVHuBERTExtractor

AVHUBERT_DIM = 768
BACKBONE_DIM = 960    # SmolLM2-360M hidden_size
VOCAB_SIZE   = 49152  # SmolLM2-360M vocab_size
CHUNK_SIZE   = 6      # 240ms @ 25fps


def make_chunk_causal_mask(T: int, C: int, device: torch.device) -> torch.BoolTensor:
    """
    Returns (T, T) bool mask. True = position k is visible to query q.
    Rule: chunk(k) <= chunk(q)  →  past/current chunks are visible, future chunks masked.
    """
    chunk_idx = torch.arange(T, device=device) // C
    return chunk_idx.unsqueeze(1) >= chunk_idx.unsqueeze(0)  # (T, T)


class ConformerLayer(nn.Module):
    """Pre-norm Transformer layer with chunk-causal self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_ratio: float = 4.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.BoolTensor) -> torch.Tensor:
        """
        x:         (B, T, dim)
        attn_mask: (T, T) bool, True = allowed to attend (chunk-causal)
        """
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # Self-attention (pre-norm)
        normed = self.norm1(x)
        qkv = self.qkv(normed).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, Dh)

        # F.sdpa bool mask: True = attend, False = ignore
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, T, T) → broadcast
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        x = x + self.out_proj(out)

        # FFN (pre-norm)
        x = x + self.ffn(self.norm2(x))
        return x


class ConformerAdapter(nn.Module):
    """
    Projects AV-HuBERT features (768) to backbone dim (960),
    then applies N chunk-causal Conformer layers.
    """

    def __init__(self, n_layers: int = 4, chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.chunk_size = chunk_size
        self.input_proj = nn.Linear(AVHUBERT_DIM, BACKBONE_DIM)
        self.layers     = nn.ModuleList([ConformerLayer(BACKBONE_DIM) for _ in range(n_layers)])
        self.norm       = nn.LayerNorm(BACKBONE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 768)  →  (B, T, 960)"""
        x = self.input_proj(x)  # (B, T, 960)
        mask = make_chunk_causal_mask(x.shape[1], self.chunk_size, x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class VisualEncoderV2(nn.Module):
    """
    Full visual stack:
      lip (B, T, 3, 96, 96)
        → AV-HuBERT (frozen, stem trainable) → (B, T, 768)
        → ConformerAdapter (trainable)        → vis_feat (B, T, 960)
        → Visual Head (trainable)             → s_vis (B, T, 49152)
    """

    def __init__(
        self,
        avhubert_ckpt: str,
        n_conformer_layers: int = 4,
        chunk_size: int = CHUNK_SIZE,
    ):
        super().__init__()
        self.av_hubert   = AVHuBERTExtractor(avhubert_ckpt, device="cpu")
        self.conformer   = ConformerAdapter(n_conformer_layers, chunk_size)
        self.visual_head = nn.Linear(BACKBONE_DIM, VOCAB_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, 3, 96, 96) raw lip frames  — runs AV-HuBERT internally
           OR (B, T, 768)     pre-extracted   — skips AV-HuBERT (fast path)
        returns:
          vis_feat (B, T, 960)   — continuous representation ṽ_t, used as FM condition
          s_vis    (B, T, 49152) — visual logits log p(x_t | v_t)
        """
        if x.dim() == 5:
            feats = self.av_hubert(x)        # raw frames → (B, T, 768)
        else:
            feats = x                        # pre-extracted, already (B, T, 768)
        vis_feat = self.conformer(feats)     # (B, T, 960)
        s_vis    = self.visual_head(vis_feat)  # (B, T, 49152)
        return vis_feat, s_vis
