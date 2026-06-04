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


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
    start_frame: int = 0,
) -> torch.Tensor:
    """MSE over valid sequence frames only; falls back to full MSE without lengths."""
    start_frame = max(int(start_frame), 0)
    if start_frame > 0:
        pred = pred[:, start_frame:]
        target = target[:, start_frame:]
        if lengths is not None:
            lengths = (lengths - start_frame).clamp_min(0)
    if lengths is None:
        return F.mse_loss(pred.float(), target.float())
    lengths = lengths.to(device=pred.device, dtype=torch.long).clamp(min=0, max=pred.shape[1])
    mask = torch.arange(pred.shape[1], device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)
    diff2 = (pred.float() - target.float()).pow(2) * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1)
    return diff2.sum() / denom


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

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.use_cross_attn = use_cross_attn

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        if self.use_cross_attn:
            self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_q = nn.Linear(dim, dim, bias=False)
            self.cross_kv = nn.Linear(dim, 2 * dim, bias=False)
            self.cross_out_proj = nn.Linear(dim, dim, bias=False)
            self.cross_gate = nn.Parameter(torch.zeros(dim))

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

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: torch.Tensor,
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:           (B, T_a, dim)
        cond_vec:    (B, dim) = t_emb + mean_pool(cond)
        cond_tokens: (B, T_k, dim), used as cross-attention key/value when enabled.
        cond_token_mask: optional bool mask (B, T_k), True for valid tokens.
        """
        if self.use_cross_attn and cond_tokens is None:
            raise ValueError("cond_tokens is required when use_cross_attn=True")
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

        if self.use_cross_attn:
            cross_q = self.cross_q(self.cross_norm(x)).reshape(B, T, self.num_heads, self.head_dim)
            cross_q = cross_q.transpose(1, 2)
            Tk = cond_tokens.shape[1]
            cond_tokens = cond_tokens + sinusoidal_positions(
                Tk, D, cond_tokens.device, cond_tokens.dtype
            )
            cross_kv = self.cross_kv(cond_tokens).reshape(B, Tk, 2, self.num_heads, self.head_dim)
            cross_kv = cross_kv.permute(2, 0, 3, 1, 4)
            cross_k, cross_v = cross_kv.unbind(0)
            attn_mask = None
            if cond_token_mask is not None:
                valid = cond_token_mask.to(device=x.device, dtype=torch.bool)
                if valid.shape != (B, Tk):
                    raise ValueError(
                        f"cond_token_mask shape {tuple(valid.shape)} does not match {(B, Tk)}"
                    )
                attn_mask = valid[:, None, None, :]
            cross_out = F.scaled_dot_product_attention(
                cross_q, cross_k, cross_v, attn_mask=attn_mask
            )
            cross_out = cross_out.transpose(1, 2).reshape(B, T, D)
            x = x + self.cross_gate[None, None, :] * self.cross_out_proj(cross_out)

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

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 8,
        use_cross_attn: bool = False,
        use_text_token_cross_attn: bool = False,
        extra_cond_dim: int = 0,
        timbre_condition_dim: int = 0,
        audio_prompt_dim: int = 0,
        audio_prompt_pool_cond: bool = False,
        audio_prompt_stat_pool_cond: bool = False,
        audio_prompt_learned_pool_cond: bool = False,
        audio_prompt_cross_attn: bool = True,
        ctc_vocab_size: int = 0,
        ctc_topk: int = 0,
        ctc_token_emb_dim: int = 0,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_text_token_cross_attn = use_text_token_cross_attn
        self.extra_cond_dim = extra_cond_dim
        self.timbre_condition_dim = timbre_condition_dim
        self.audio_prompt_dim = audio_prompt_dim
        self.audio_prompt_pool_cond = audio_prompt_pool_cond
        self.audio_prompt_stat_pool_cond = audio_prompt_stat_pool_cond
        self.audio_prompt_learned_pool_cond = audio_prompt_learned_pool_cond
        self.audio_prompt_cross_attn = audio_prompt_cross_attn
        self.ctc_topk = ctc_topk
        self.ctc_token_emb_dim = ctc_token_emb_dim
        ctc_cond_dim = ctc_token_emb_dim + ctc_topk if ctc_topk > 0 else 0
        self.ctc_token_emb = None
        if ctc_topk > 0:
            if ctc_vocab_size <= 0 or ctc_token_emb_dim <= 0:
                raise ValueError("ctc top-k condition requires positive vocab and embedding dims")
            self.ctc_token_emb = nn.Embedding(ctc_vocab_size, ctc_token_emb_dim)
        self.cond_proj  = nn.Linear(
            COND_DIM + extra_cond_dim + timbre_condition_dim + ctc_cond_dim,
            self.DIM,
        )  # condition → 512
        self.cond_token_proj = nn.Linear(self.DIM, self.DIM)
        self.text_token_proj = nn.Linear(960, self.DIM)
        self.audio_prompt_proj = (
            nn.Linear(audio_prompt_dim, self.DIM) if audio_prompt_dim > 0 else None
        )
        self.audio_prompt_stat_pool_proj = None
        if audio_prompt_dim > 0 and audio_prompt_stat_pool_cond:
            self.audio_prompt_stat_pool_proj = nn.Linear(self.DIM * 2, self.DIM)
            nn.init.zeros_(self.audio_prompt_stat_pool_proj.weight)
            nn.init.zeros_(self.audio_prompt_stat_pool_proj.bias)
        self.audio_prompt_learned_pool_score = None
        self.audio_prompt_learned_pool_proj = None
        if audio_prompt_dim > 0 and audio_prompt_learned_pool_cond:
            self.audio_prompt_learned_pool_score = nn.Linear(self.DIM, 1)
            self.audio_prompt_learned_pool_proj = nn.Linear(self.DIM, self.DIM)
            nn.init.zeros_(self.audio_prompt_learned_pool_proj.weight)
            nn.init.zeros_(self.audio_prompt_learned_pool_proj.bias)
        self.extra_pred_head = None
        if extra_cond_dim > 0:
            self.extra_pred_head = nn.Sequential(
                nn.LayerNorm(self.DIM),
                nn.Linear(self.DIM, self.DIM),
                nn.SiLU(),
                nn.Linear(self.DIM, extra_cond_dim),
            )
        self.time_emb   = SinusoidalTimeEmb(self.DIM)
        self.blocks     = nn.ModuleList([
            DiTBlock(self.DIM, n_heads, use_cross_attn=use_cross_attn)
            for _ in range(n_layers)
        ])
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
        timbre_cond: torch.Tensor | None = None,  # (B, timbre_condition_dim)
        text_tokens: torch.Tensor | None = None,  # (B, L, 960)
        audio_prompt: torch.Tensor | None = None,  # (B, T_prompt, audio_prompt_dim)
        extra_cond: torch.Tensor | None = None,  # (B, T_a, extra_cond_dim)
        ctc_topk_ids: torch.Tensor | None = None,  # (B, T_a, K)
        ctc_topk_probs: torch.Tensor | None = None,  # (B, T_a, K)
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Concatenate and project to (B, T_a, 512)."""
        T_a    = vis_down.shape[1]
        id_exp = id_vec.unsqueeze(1).expand(-1, T_a, -1)           # (B, T_a, 256)
        parts = [vis_down, h_down, id_exp]
        if self.timbre_condition_dim > 0:
            if timbre_cond is None:
                timbre_cond = torch.zeros(
                    vis_down.shape[0],
                    self.timbre_condition_dim,
                    device=vis_down.device,
                    dtype=vis_down.dtype,
                )
            if timbre_cond.shape != (vis_down.shape[0], self.timbre_condition_dim):
                raise ValueError(
                    f"timbre_cond shape must be {(vis_down.shape[0], self.timbre_condition_dim)}, "
                    f"got {tuple(timbre_cond.shape)}"
                )
            timbre_exp = timbre_cond.to(
                device=vis_down.device, dtype=vis_down.dtype
            ).unsqueeze(1).expand(-1, T_a, -1)
            parts.append(timbre_exp)
        if self.extra_cond_dim > 0:
            if extra_cond is None:
                extra_cond = torch.zeros(
                    *vis_down.shape[:2],
                    self.extra_cond_dim,
                    device=vis_down.device,
                    dtype=vis_down.dtype,
                )
            parts.append(extra_cond.to(device=vis_down.device, dtype=vis_down.dtype))
        if self.ctc_topk > 0:
            if ctc_topk_ids is None:
                ctc_topk_ids = torch.zeros(
                    *vis_down.shape[:2],
                    self.ctc_topk,
                    device=vis_down.device,
                    dtype=torch.long,
                )
            if ctc_topk_probs is None:
                ctc_topk_probs = torch.zeros(
                    *vis_down.shape[:2],
                    self.ctc_topk,
                    device=vis_down.device,
                    dtype=vis_down.dtype,
                )
            ctc_topk_ids = ctc_topk_ids.to(device=vis_down.device, dtype=torch.long)
            ctc_topk_probs = ctc_topk_probs.to(device=vis_down.device, dtype=vis_down.dtype)
            if ctc_topk_ids.shape[:2] != vis_down.shape[:2] or ctc_topk_ids.shape[-1] != self.ctc_topk:
                raise ValueError("ctc_topk_ids shape must be (B, T_a, ctc_topk)")
            if ctc_topk_probs.shape != ctc_topk_ids.shape:
                raise ValueError("ctc_topk_probs shape must match ctc_topk_ids")
            emb = self.ctc_token_emb(ctc_topk_ids).to(dtype=vis_down.dtype)
            weighted_emb = (emb * ctc_topk_probs.unsqueeze(-1)).sum(dim=2)
            parts.extend([weighted_emb, ctc_topk_probs])
        cat = torch.cat(parts, dim=-1)
        cond = self.cond_proj(cat)
        cond_tokens = None
        if self.use_text_token_cross_attn and text_tokens is not None:
            cond_tokens = self.text_token_proj(text_tokens.to(dtype=cond.dtype))
        if self.audio_prompt_dim > 0 and audio_prompt is not None:
            if audio_prompt.ndim != 3:
                raise ValueError("audio_prompt shape must be (B, T_prompt, audio_prompt_dim)")
            if audio_prompt.shape[0] != vis_down.shape[0] or audio_prompt.shape[-1] != self.audio_prompt_dim:
                raise ValueError(
                    f"audio_prompt shape must be (B, T_prompt, {self.audio_prompt_dim}), "
                    f"got {tuple(audio_prompt.shape)}"
                )
            prompt_tokens = self.audio_prompt_proj(
                audio_prompt.to(device=cond.device, dtype=cond.dtype)
            )
            if self.audio_prompt_pool_cond:
                cond = cond + prompt_tokens.mean(dim=1, keepdim=True)
            if self.audio_prompt_stat_pool_proj is not None:
                prompt_mean = prompt_tokens.mean(dim=1)
                prompt_std = prompt_tokens.float().std(dim=1, unbiased=False).to(
                    dtype=prompt_tokens.dtype
                )
                prompt_stats = torch.cat([prompt_mean, prompt_std], dim=-1)
                cond = cond + self.audio_prompt_stat_pool_proj(prompt_stats).unsqueeze(1)
            if self.audio_prompt_learned_pool_proj is not None:
                score = self.audio_prompt_learned_pool_score(prompt_tokens).float()
                weights = torch.softmax(score, dim=1).to(dtype=prompt_tokens.dtype)
                pooled = (prompt_tokens * weights).sum(dim=1)
                cond = cond + self.audio_prompt_learned_pool_proj(pooled).unsqueeze(1)
            if self.audio_prompt_cross_attn:
                cond_tokens = (
                    prompt_tokens if cond_tokens is None
                    else torch.cat([cond_tokens, prompt_tokens], dim=1)
                )
        if cond_tokens is None:
            return cond
        return cond, cond_tokens

    def _forward_dit(
        self,
        x:    torch.Tensor,   # (B, T_a, 512)
        cond: torch.Tensor,   # (B, T_a, 512)
        t:    torch.Tensor,   # (B,)
        cond_tokens: torch.Tensor | None = None,
        cond_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb    = self.time_emb(t)          # (B, 512)
        cond_vec = t_emb + cond.mean(dim=1)  # (B, 512): t_emb + mean_pool(cond)
        x = x + self.cond_token_proj(cond)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)
        if cond_tokens is None:
            cond_tokens = cond
        if cond_token_mask is None and cond_tokens is cond:
            cond_token_mask = torch.ones(
                cond.shape[:2], device=cond.device, dtype=torch.bool
            )
        for block in self.blocks:
            x = block(
                x,
                cond_vec,
                cond_tokens if self.use_cross_attn else None,
                cond_token_mask if self.use_cross_attn else None,
            )
        return self.final_proj(self.final_norm(x))

    def _build_cond_parts(
        self,
        vis_down: torch.Tensor,
        h_down: torch.Tensor,
        id_vec: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        extra_cond: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        built = self._build_cond(
            vis_down, h_down, id_vec,
            timbre_cond=timbre_cond,
            text_tokens=text_tokens,
            audio_prompt=audio_prompt,
            extra_cond=extra_cond,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )
        if isinstance(built, tuple):
            cond, cond_tokens = built
        else:
            cond, cond_tokens = built, None
        cond_token_mask = None
        if (
            cond_tokens is not None
            and text_token_mask is not None
            and self.use_text_token_cross_attn
        ):
            prompt_len = 0
            if (
                self.audio_prompt_cross_attn
                and self.audio_prompt_dim > 0
                and audio_prompt is not None
            ):
                prompt_len = audio_prompt.shape[1]
            if prompt_len > 0:
                prompt_mask = torch.ones(
                    text_token_mask.shape[0],
                    prompt_len,
                    device=text_token_mask.device,
                    dtype=torch.bool,
                )
                cond_token_mask = torch.cat([text_token_mask, prompt_mask], dim=1)
            else:
                cond_token_mask = text_token_mask
        return cond, cond_tokens, cond_token_mask

    def predict_extra_condition(
        self,
        vis_down: torch.Tensor,
        h_down: torch.Tensor,
        id_vec: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict frame-level extra condition from vision/text/speaker only."""
        if self.extra_pred_head is None:
            raise ValueError("predict_extra_condition requires extra_cond_dim > 0")
        dtype = next(self.parameters()).dtype
        zeros = torch.zeros(
            *vis_down.shape[:2],
            self.extra_cond_dim,
            device=vis_down.device,
            dtype=dtype,
        )
        built = self._build_cond(
            vis_down,
            h_down,
            id_vec,
            text_tokens=text_tokens,
            timbre_cond=timbre_cond,
            audio_prompt=audio_prompt,
            extra_cond=zeros,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )
        cond = built[0] if isinstance(built, tuple) else built
        return self.extra_pred_head(cond)

    def forward_train(
        self,
        vis_down: torch.Tensor,   # (B, T_a, 960) — sg already applied by caller
        h_down:   torch.Tensor,   # (B, T_a, 960) — sg already applied by caller
        id_vec:   torch.Tensor,   # (B, 256)
        x_1:      torch.Tensor,   # (B, T_a, 512) target latent
        lengths:  torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        extra_cond: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OT-CFM loss: E||v_θ(x_t, c, t) - (x_1 - x_0)||²"""
        # Unify dtype to match model weights (handles float32/bfloat16 mixed input)
        dtype = next(self.parameters()).dtype
        x_1   = x_1.to(dtype)
        cond, cond_tokens, cond_token_mask = self._build_cond_parts(
            vis_down, h_down, id_vec,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            timbre_cond=timbre_cond,
            audio_prompt=audio_prompt,
            extra_cond=extra_cond,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )

        B   = x_1.shape[0]
        t   = torch.rand(B, device=x_1.device, dtype=dtype)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1

        pred = self._forward_dit(x_t, cond, t, cond_tokens, cond_token_mask)
        return masked_mse_loss(pred, x_1 - x_0, lengths)

    def reconstruct_from_cond(
        self,
        vis_down: torch.Tensor,
        h_down: torch.Tensor,
        id_vec: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        extra_cond: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic probe: predict x_1 directly from aligned conditions."""
        cond, cond_tokens, cond_token_mask = self._build_cond_parts(
            vis_down, h_down, id_vec,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            timbre_cond=timbre_cond,
            audio_prompt=audio_prompt,
            extra_cond=extra_cond,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )
        x = torch.zeros_like(cond)
        t = torch.ones(cond.shape[0], device=cond.device, dtype=cond.dtype)
        return self._forward_dit(x, cond, t, cond_tokens, cond_token_mask)

    def denoise_from_noise(
        self,
        vis_down: torch.Tensor,
        h_down: torch.Tensor,
        id_vec: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        extra_cond: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Endpoint denoiser: predict x_1 directly from noisy latent tokens."""
        dtype = next(self.parameters()).dtype
        cond, cond_tokens, cond_token_mask = self._build_cond_parts(
            vis_down, h_down, id_vec,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            timbre_cond=timbre_cond,
            audio_prompt=audio_prompt,
            extra_cond=extra_cond,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )
        noise = noise.to(device=cond.device, dtype=dtype)
        if t is None:
            t = torch.zeros(cond.shape[0], device=cond.device, dtype=dtype)
        else:
            t = t.to(device=cond.device, dtype=dtype)
        return self._forward_dit(noise, cond, t, cond_tokens, cond_token_mask)

    def sample(
        self,
        vis_down: torch.Tensor,   # (B, T_a, 960)
        h_down:   torch.Tensor,   # (B, T_a, 960)
        id_vec:   torch.Tensor,   # (B, 256)
        nfe:      int = 10,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        extra_cond: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler solver. Keeps gradients so endpoint losses can train sampling."""
        cond, cond_tokens, cond_token_mask = self._build_cond_parts(
            vis_down, h_down, id_vec,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            timbre_cond=timbre_cond,
            audio_prompt=audio_prompt,
            extra_cond=extra_cond,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )
        B, T_a, _ = cond.shape

        x  = torch.randn(B, T_a, self.DIM, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / nfe
        for step in range(nfe):
            t = torch.full((B,), step / nfe, device=cond.device, dtype=cond.dtype)
            x = x + dt * self._forward_dit(
                x, cond, t, cond_tokens, cond_token_mask
            )
        return x

    @torch.no_grad()
    def forward_inference(
        self,
        vis_down: torch.Tensor,   # (B, T_a, 960)
        h_down:   torch.Tensor,   # (B, T_a, 960)
        id_vec:   torch.Tensor,   # (B, 256)
        nfe:      int = 10,
        text_tokens: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        timbre_cond: torch.Tensor | None = None,
        audio_prompt: torch.Tensor | None = None,
        extra_cond: torch.Tensor | None = None,
        ctc_topk_ids: torch.Tensor | None = None,
        ctc_topk_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """No-grad Euler solver for evaluation/inference."""
        return self.sample(
            vis_down, h_down, id_vec, nfe=nfe,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            timbre_cond=timbre_cond,
            audio_prompt=audio_prompt,
            extra_cond=extra_cond,
            ctc_topk_ids=ctc_topk_ids,
            ctc_topk_probs=ctc_topk_probs,
        )
