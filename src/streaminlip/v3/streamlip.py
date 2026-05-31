"""
StreamLip V3: Cross-Attention fusion (Flamingo-style).

V2 PoE replaced by GatedCrossAttention inside the LM.
Alignment (frame_labels, clean_ids, lm_idx_text, lm_idx_fm) unchanged from V2.

Text path:
  lip → VisualEncoderV3 → vis_feat (B, T, 960)   [K/V for cross-attn]
  clean_ids → LMBackboneV3 [cross-attn to vis_feat every 4 layers]
            → logits (B, L, vocab), h_lm (B, L, 960)
  gather(logits, lm_idx_text) → posterior (B, T, vocab)
  CE loss on non-SIL frames only

SIL path:   vis_feat → sil_head → binary BCE (all valid frames)
Audio path: vis_feat ∥ h_lm_gathered → FMHead (Phase 2)

Removed vs V2: visual_head (49K MLP), alpha parameter, PoE logit addition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder  import VisualEncoderV3, BACKBONE_DIM
from .lm_backbone     import LMBackboneV3, build_lm_backbone_v3
from ..v2.speaker_encoder import SpeakerEncoder
from ..v2.fm_head         import FMHead
from ..v2.data.dataset    import SIL_ID


class StreamLipV3(nn.Module):

    def __init__(
        self,
        avhubert_ckpt:      str,
        smollm2_path:       str,
        lambda_text:        float      = 1.0,
        lambda_sil:         float      = 0.3,
        cross_attn_every_n: int        = 4,
        n_conformer_layers: int        = 4,
        n_dit_layers:       int        = 6,
        chunk_size:         int        = 6,
        resnet50_weights:   str | None = None,
    ):
        super().__init__()
        self.lambda_text = lambda_text
        self.lambda_sil  = lambda_sil

        self.visual_encoder  = VisualEncoderV3(avhubert_ckpt, n_conformer_layers, chunk_size)
        self.sil_head        = nn.Linear(BACKBONE_DIM, 1)
        self.lm              = build_lm_backbone_v3(smollm2_path, cross_attn_every_n)
        self.speaker_encoder = SpeakerEncoder(weights_path=resnet50_weights)
        self.fm_head         = FMHead(n_layers=n_dit_layers)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        visual:       torch.Tensor,              # (B, T, 768) or (B, T, 3, 96, 96)
        clean_ids:    torch.Tensor,              # (B, L)
        clean_mask:   torch.Tensor,              # (B, L)
        lm_idx_text:  torch.Tensor,              # (B, T)
        lm_idx_fm:    torch.Tensor,              # (B, T)
        face:         torch.Tensor,              # (B, 3, 256, 256)
        frame_labels: torch.Tensor,              # (B, T)
        mask:         torch.Tensor,              # (B, T) bool
        latent:       torch.Tensor | None = None,
    ) -> dict:
        B, T = mask.shape

        # ── Visual encoding ───────────────────────────────────────────────────
        vis_feat = self.visual_encoder(visual)         # (B, T, 960)

        # ── SIL detection (all valid frames) ─────────────────────────────────
        sil_logit  = self.sil_head(vis_feat).squeeze(-1)   # (B, T)
        sil_target = (frame_labels == SIL_ID).float()
        loss_sil   = F.binary_cross_entropy_with_logits(sil_logit[mask], sil_target[mask])

        # ── LM with chunk-causal cross-attention to vis_feat ─────────────────
        logits, h_lm = self.lm(clean_ids, clean_mask, vis_feat, lm_idx_fm)  # (B,L,vocab), (B,L,960)

        # Gather per-frame logits using the same alignment index as V2
        V   = logits.shape[-1]
        idx = lm_idx_text.unsqueeze(-1).expand(B, T, V)
        posterior = logits.gather(1, idx)                  # (B, T, vocab)

        # CE loss on non-SIL frames only
        non_sil = mask & (frame_labels != SIL_ID)
        if non_sil.any():
            loss_text = F.cross_entropy(
                posterior.reshape(-1, V)[non_sil.reshape(-1)],
                frame_labels.reshape(-1)[non_sil.reshape(-1)],
            )
        else:
            loss_text = posterior.new_zeros(())

        # ── Audio path (Phase 2) ──────────────────────────────────────────────
        loss_fm = posterior.new_zeros(())
        if latent is not None:
            id_vec  = self.speaker_encoder(face)
            v_down  = vis_feat.detach()[:, ::2, :]           # (B, T_a, 960)
            idx_f   = lm_idx_fm.unsqueeze(-1)
            h_lm_fm = h_lm.gather(1, idx_f.expand(B, T, h_lm.shape[-1]))
            h_down  = h_lm_fm.detach()[:, ::2, :]            # (B, T_a, 960)
            loss_fm = self.fm_head.forward_train(v_down, h_down, id_vec, latent.float())

        loss = loss_fm + self.lambda_text * loss_text + self.lambda_sil * loss_sil

        return {
            "loss":      loss,
            "loss_fm":   loss_fm.detach(),
            "loss_text": loss_text.detach(),
            "loss_sil":  loss_sil.detach(),
            "sil_logit": sil_logit.detach(),
            "posterior": posterior.detach(),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        visual: torch.Tensor,
        face:   torch.Tensor,
        nfe:    int = 10,
    ) -> dict:
        """
        Teacher-forced eval: same interface as V2 for decode_v2.py compatibility.
        Real streaming generation (autoregressive with pacing head) is a future step.
        """
        B, T = visual.shape[:2]
        device = visual.device

        vis_feat = self.visual_encoder(visual)    # (B, T, 960)

        # Build rough clean_ids from SIL detection
        sil_pred = self.sil_head(vis_feat).squeeze(-1) > 0   # (B, T) True=SIL
        bos_id   = self.lm.model.config.bos_token_id or SIL_ID

        from ..v2.streamlip import _compute_lm_indices_np
        import numpy as np

        clean_ids_list, lm_idx_text_list = [], []
        for b in range(B):
            # Treat non-SIL frames as a single dummy token for index building
            labels = np.where(sil_pred[b].cpu().numpy(), SIL_ID,
                              bos_id + 1).astype("int64")
            cids, idx_t, _ = _compute_lm_indices_np(labels, T, bos_id, SIL_ID)
            clean_ids_list.append(torch.from_numpy(cids).to(device))
            lm_idx_text_list.append(torch.from_numpy(idx_t).to(device))

        max_L = max(c.shape[0] for c in clean_ids_list)
        clean_ids  = torch.full((B, max_L), SIL_ID, dtype=torch.long, device=device)
        clean_mask = torch.zeros(B, max_L, dtype=torch.long, device=device)
        for b, cids in enumerate(clean_ids_list):
            L = cids.shape[0]
            clean_ids[b, :L]  = cids
            clean_mask[b, :L] = 1
        lm_idx_text = torch.stack(lm_idx_text_list)

        logits, h_lm = self.lm(clean_ids, clean_mask, vis_feat)
        idx      = lm_idx_text.unsqueeze(-1).expand(B, T, logits.shape[-1])
        posterior = logits.gather(1, idx)
        x_hat    = posterior.argmax(-1)

        v_down      = vis_feat[:, ::2, :]
        h_down      = h_lm.gather(1, lm_idx_text.unsqueeze(-1).expand(B, T, h_lm.shape[-1]))[:, ::2, :]
        id_vec      = self.speaker_encoder(face)
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
        """Phase 1: train visual encoder (Conformer) + cross-attn layers + sil_head.
        LM base (SmolLM2 weights) frozen — only GatedCrossAttentionLayers train."""
        for p in self.fm_head.parameters():
            p.requires_grad_(False)
        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)
        # Freeze LM base weights
        for p in self.lm.model.parameters():
            p.requires_grad_(False)
        # Cross-attention layers stay trainable (they're separate from lm.model)
        for p in self.lm.cross_attn_layers.parameters():
            p.requires_grad_(True)

    def phase2_mode(self):
        """Phase 2: freeze visual + LM, train FM head only."""
        for p in self.visual_encoder.parameters():
            p.requires_grad_(False)
        for p in self.sil_head.parameters():
            p.requires_grad_(False)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        for p in self.fm_head.parameters():
            p.requires_grad_(True)
        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)
