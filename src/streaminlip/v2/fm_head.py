"""
FM Head V2: DiT-based Optimal Transport Conditional Flow Matching.

Condition: c_t = cond_proj( sg(ṽ_t) ∥ sg(h̃_t^LM) ∥ id̂ )
  - sg(·): stop-gradient, applied by caller before passing here
  - ṽ_t:   (B, T_a, 960) visual features, downsampled 2× from vis_feat
  - h̃_t^LM:(B, T_a, 960) LM hidden states, downsampled 2× from h_lm
  - id̂:    (B, 256)       speaker identity

Training:  OT-CFM straight-line path, x_t = (1-t)*x_0 + t*x_1, target = x_1 - x_0
Inference: Euler solver, NFE=10

Each token receives its aligned per-frame condition and a sinusoidal sequence
position embedding. DiT block adaLN still uses global time/sample conditioning.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 512
COND_DIM   = 960 + 960 + 256   # vis + lm + id = 2176


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional encoding for diffusion time t ∈ [0, 1], followed by MLP."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0, 1]  →  (B, dim)"""
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)    # (B, dim)
        return self.mlp(emb.to(t.dtype))


def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Returns (1, length, dim) sinusoidal sequence position embeddings."""
    assert dim % 2 == 0
    half = dim // 2
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(
        -math.log(10000) *
        torch.arange(half, device=device, dtype=torch.float32) / half
    ).unsqueeze(0)
    emb = torch.cat([(pos * freqs).sin(), (pos * freqs).cos()], dim=-1)
    return emb.to(dtype).unsqueeze(0)


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning.

    cond_vec = t_emb + mean_pool(cond)  — (B, dim) global modulation vector
    adaLN: scale/shift/gate derived from cond_vec via a zero-initialized linear.
    Attention: bidirectional (no causal mask; FM generation is non-causal).
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        # 6 parameters per block: (shift1, scale1, gate1, shift2, scale2, gate2)
        # Zero-init → all gates=0 at start → pure residual → training stability
        self.adaLN_proj = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, T_a, dim)
        cond_vec: (B, dim) = t_emb + mean_pool(cond)
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.adaLN_proj(cond_vec).chunk(6, dim=-1)  # each (B, dim)
        )
        # unsqueeze for broadcasting over T_a
        s1, sh1, g1 = scale1[:, None], shift1[:, None], gate1[:, None]
        s2, sh2, g2 = scale2[:, None], shift2[:, None], gate2[:, None]

        # Attention branch
        B, T, D = x.shape
        normed = self.norm1(x) * (1 + s1) + sh1
        qkv = self.normed_qkv(normed, B, T, D)
        q, k, v = qkv
        attn_out = F.scaled_dot_product_attention(q, k, v)   # bidirectional
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        x = x + g1 * self.out_proj(attn_out)

        # FFN branch
        x = x + g2 * self.ffn(self.norm2(x) * (1 + s2) + sh2)
        return x

    def normed_qkv(self, normed, B, T, D):
        H, Dh = self.num_heads, self.head_dim
        qkv = self.qkv(normed).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        return qkv.unbind(0)  # q, k, v each (B, H, T, Dh)


class FMHead(nn.Module):
    """
    FM Head: cond_proj → N DiT blocks → output proj.

    Caller must apply stop-gradient before passing vis_down and h_down.
    """

    DIM = LATENT_DIM  # 512

    def __init__(self, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        self.cond_proj  = nn.Linear(COND_DIM, self.DIM)         # 2176 → 512
        self.cond_token_proj = nn.Linear(self.DIM, self.DIM)
        self.time_emb   = SinusoidalTimeEmb(self.DIM)
        self.blocks     = nn.ModuleList([DiTBlock(self.DIM, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(self.DIM)
        # Zero-init final proj for stable startup
        self.final_proj = nn.Linear(self.DIM, self.DIM)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def _build_cond(
        self,
        vis_down: torch.Tensor,   # (B, T_a, 960)
        h_down:   torch.Tensor,   # (B, T_a, 960)
        id_vec:   torch.Tensor,   # (B, 256)
    ) -> torch.Tensor:
        """Concatenate and project to (B, T_a, 512)."""
        T_a    = vis_down.shape[1]
        id_exp = id_vec.unsqueeze(1).expand(-1, T_a, -1)           # (B, T_a, 256)
        cat    = torch.cat([vis_down, h_down, id_exp], dim=-1)      # (B, T_a, 2176)
        return self.cond_proj(cat)                                  # (B, T_a, 512)

    def _forward_dit(
        self,
        x:    torch.Tensor,   # (B, T_a, 512)
        cond: torch.Tensor,   # (B, T_a, 512)
        t:    torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        t_emb    = self.time_emb(t)          # (B, 512)
        cond_vec = t_emb + cond.mean(dim=1)  # (B, 512): t_emb + mean_pool(cond)
        x = x + self.cond_token_proj(cond)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cond_vec)
        return self.final_proj(self.final_norm(x))

    def forward_train(
        self,
        vis_down: torch.Tensor,   # (B, T_a, 960) — sg already applied by caller
        h_down:   torch.Tensor,   # (B, T_a, 960) — sg already applied by caller
        id_vec:   torch.Tensor,   # (B, 256)
        x_1:      torch.Tensor,   # (B, T_a, 512) target latent
    ) -> torch.Tensor:
        """OT-CFM loss: E||v_θ(x_t, c, t) - (x_1 - x_0)||²"""
        # Unify dtype to match model weights (handles float32/bfloat16 mixed input)
        dtype = next(self.parameters()).dtype
        x_1   = x_1.to(dtype)
        cond  = self._build_cond(vis_down, h_down, id_vec)   # (B, T_a, 512)

        B   = x_1.shape[0]
        t   = torch.rand(B, device=x_1.device, dtype=dtype)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1

        pred = self._forward_dit(x_t, cond, t)
        return F.mse_loss(pred.float(), (x_1 - x_0).float())

    def reconstruct_from_cond(
        self,
        vis_down: torch.Tensor,
        h_down: torch.Tensor,
        id_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic probe: predict x_1 directly from aligned conditions."""
        cond = self._build_cond(vis_down, h_down, id_vec)
        x = torch.zeros_like(cond)
        t = torch.ones(cond.shape[0], device=cond.device, dtype=cond.dtype)
        return self._forward_dit(x, cond, t)

    @torch.no_grad()
    def forward_inference(
        self,
        vis_down: torch.Tensor,   # (B, T_a, 960)
        h_down:   torch.Tensor,   # (B, T_a, 960)
        id_vec:   torch.Tensor,   # (B, 256)
        nfe:      int = 10,
    ) -> torch.Tensor:
        """Euler solver. Returns predicted Mimi latent (B, T_a, 512)."""
        cond = self._build_cond(vis_down, h_down, id_vec)
        B, T_a, _ = cond.shape

        x  = torch.randn(B, T_a, self.DIM, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / nfe
        for step in range(nfe):
            t = torch.full((B,), step / nfe, device=cond.device, dtype=cond.dtype)
            x = x + dt * self._forward_dit(x, cond, t)
        return x
