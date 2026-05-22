"""
StreamLip V2: Dual-path architecture with Product of Experts decoding.

Theory: see theory.md (causal graph, PoE derivation, FM conditioning).
Design: see MODEL_DESIGN.md §4–6.

Text path:
  lip → VisualEncoderV2 → (vis_feat, s_vis)
  text_ids → LMBackbone   → (h_lm, s_lm)
  posterior = s_vis + α·s_lm  →  CE loss  →  argmax → x̂_t

Audio path (Phase 2, latent is not None):
  sg(vis_feat)[:, ::2] ∥ sg(h_lm)[:, ::2] ∥ id̂ → FMHead → pred_latent

Loss:
  L = L_fm + λ·L_CE(posterior, frame_labels)    λ=0.005
  FM loss does NOT update VisualEncoder or LM (stop-gradient applied here).
  CE loss drives both VisualEncoder (via s_vis) and LM (via s_lm).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import VisualEncoderV2
from .lm_backbone    import LMBackbone, build_lm_backbone
from .speaker_encoder import SpeakerEncoder
from .fm_head        import FMHead


class StreamLipV2(nn.Module):

    def __init__(
        self,
        avhubert_ckpt:        str,
        smollm2_path:         str,
        lambda_text:          float       = 0.005,
        alpha:                float       = 1.0,
        lora_rank:            int         = 16,
        n_conformer_layers:   int         = 4,
        n_dit_layers:         int         = 6,
        chunk_size:           int         = 6,
        resnet50_weights:     str | None  = None,  # e.g. 'pretrained/resnet50-11ad3fa6.pth'
    ):
        super().__init__()
        self.lambda_text = lambda_text
        self.alpha       = alpha

        self.visual_encoder  = VisualEncoderV2(avhubert_ckpt, n_conformer_layers, chunk_size)
        self.lm              = build_lm_backbone(smollm2_path, lora_rank=lora_rank)
        self.speaker_encoder = SpeakerEncoder(weights_path=resnet50_weights)
        self.fm_head         = FMHead(n_layers=n_dit_layers)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        lip:          torch.Tensor,              # (B, T, 3, 96, 96)
        text_ids:     torch.Tensor,              # (B, T) int64, GT shifted right (teacher forcing)
        face:         torch.Tensor,              # (B, 3, 256, 256) or (B, C, 3, 256, 256)
        frame_labels: torch.Tensor,              # (B, T) int64, MFA frame-level labels
        mask:         torch.Tensor,              # (B, T) bool, True = valid frame
        latent:       torch.Tensor | None = None,  # (B, T_a, 512); None → Phase 1 (text-only)
    ) -> dict:
        # ── Text path ─────────────────────────────────────────────────────────
        vis_feat, s_vis = self.visual_encoder(lip)          # (B,T,960), (B,T,49152)
        h_lm, s_lm      = self.lm(text_ids, mask.long())   # (B,T,960), (B,T,49152)

        # Product of Experts: log p(x_t | v_{1:t}) ≈ s_vis + α·s_lm
        posterior = s_vis + self.alpha * s_lm               # (B, T, 49152)

        valid     = mask.reshape(-1)                        # (B·T,) bool
        loss_text = F.cross_entropy(
            posterior.reshape(-1, posterior.shape[-1])[valid],
            frame_labels.reshape(-1)[valid],
        )

        # ── Audio path (Phase 2 only) ─────────────────────────────────────────
        loss_fm = posterior.new_zeros(())
        if latent is not None:
            id_vec = self.speaker_encoder(face)
            v_down = vis_feat.detach()[:, ::2, :]   # (B, T_a, 960)
            h_down = h_lm.detach()[:, ::2, :]       # (B, T_a, 960)
            loss_fm = self.fm_head.forward_train(v_down, h_down, id_vec, latent.float())

        loss = loss_fm + self.lambda_text * loss_text

        return {
            "loss":      loss,
            "loss_fm":   loss_fm.detach(),
            "loss_text": loss_text.detach(),
            "posterior": posterior.detach(),   # (B,T,49152) for x̂_t decoding if needed
        }

    # ── Inference (full-sequence, non-streaming) ──────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        lip:  torch.Tensor,    # (B, T, 3, 96, 96)
        face: torch.Tensor,    # (B, 3, 256, 256)
        nfe:  int = 10,
    ) -> dict:
        """
        Two-pass greedy decode + FM latent generation.

        Pass 1: visual-only argmax for initial token sequence.
        Pass 2: re-run LM conditioned on pass-1 tokens → PoE posterior → final x̂_t.

        For proper autoregressive decoding (e.g., streaming), use the streaming
        pipeline in inference/streaming.py instead.
        """
        B, T = lip.shape[:2]
        device = lip.device

        vis_feat, s_vis = self.visual_encoder(lip)

        # Pass 1: visual-only token sequence
        x_vis = s_vis.argmax(dim=-1)                                     # (B, T)

        # Pass 2: LM conditioned on pass-1 tokens → refined posterior
        attn_mask = torch.ones(B, T, dtype=torch.long, device=device)
        h_lm, s_lm = self.lm(x_vis, attn_mask)
        posterior   = s_vis + self.alpha * s_lm
        x_hat       = posterior.argmax(dim=-1)                           # (B, T)

        # Audio generation (requires FM head)
        v_down      = vis_feat[:, ::2, :]                                # (B, T_a, 960)
        h_down      = h_lm[:, ::2, :]                                    # (B, T_a, 960)
        id_vec      = self.speaker_encoder(face)                         # (B, 256)
        pred_latent = self.fm_head.forward_inference(v_down, h_down, id_vec, nfe=nfe)

        return {"x_hat": x_hat, "latent": pred_latent}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def param_counts(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M":     round(total     / 1e6, 1),
            "trainable_M": round(trainable / 1e6, 1),
            "frozen_M":    round((total - trainable) / 1e6, 1),
        }

    def phase1_mode(self):
        """Freeze FM head + speaker encoder for Phase 1 (text-path only training)."""
        for p in self.fm_head.parameters():
            p.requires_grad_(False)
        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)

    def phase2_mode(self):
        """Freeze visual encoder + LM; unfreeze FM head for Phase 2 training."""
        for p in self.visual_encoder.conformer.parameters():
            p.requires_grad_(False)
        for p in self.visual_encoder.visual_head.parameters():
            p.requires_grad_(False)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        for p in self.fm_head.parameters():
            p.requires_grad_(True)
        # AV-HuBERT stem stays as-is (trainable in phase1, frozen in phase2)
        for p in self.visual_encoder.av_hubert.parameters():
            p.requires_grad_(False)
