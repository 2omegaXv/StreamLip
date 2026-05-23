"""
StreamLip V2: Dual-path architecture with Product of Experts decoding.

Theory: see theory.md (causal graph, PoE derivation, FM conditioning).
Design: see MODEL_DESIGN.md §4–6.

Text path:
  lip → VisualEncoderV2 → (vis_feat, s_vis)
  clean_ids → LMBackbone → (h_lm, s_lm)   shape: (B, L, 960/49152), L = # tokens

  Per-frame PoE via gather (log-prob space):
    s_lm_text = s_lm.gather(lm_idx_text)
    posterior  = log_softmax(s_vis) + α·log_softmax(s_lm_text)  → CE loss (label_smooth=0.1)

Audio path (Phase 2, latent is not None):
  h_lm_fm = h_lm.gather(lm_idx_fm)         # LM state AFTER current token
  sg(vis_feat)[:, ::2] ∥ sg(h_lm_fm)[:, ::2] ∥ id̂ → FMHead → pred_latent

Loss:
  L = L_fm + λ·L_CE(posterior, frame_labels)    λ=0.005
  FM loss does NOT update VisualEncoder or LM (stop-gradient applied here).
  CE loss drives both VisualEncoder (via s_vis) and LM (via s_lm).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder  import VisualEncoderV2, BACKBONE_DIM
from .lm_backbone     import LMBackbone, build_lm_backbone
from .speaker_encoder import SpeakerEncoder
from .fm_head         import FMHead
from .data.dataset    import SIL_ID


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
        lambda_sil:           float       = 0.3,
        resnet50_weights:     str | None  = None,  # e.g. 'pretrained/resnet50-11ad3fa6.pth'
    ):
        super().__init__()
        self.lambda_text = lambda_text
        self.lambda_sil  = lambda_sil
        # Learnable balance between visual and LM experts in log-prob PoE.
        # s_vis is unit-RMS normalized; alpha scales the LM contribution.
        # Initialized to s_lm's typical RMS (~2.0) so both experts start equal-weight.
        self.alpha = nn.Parameter(torch.tensor(alpha))

        self.visual_encoder  = VisualEncoderV2(avhubert_ckpt, n_conformer_layers, chunk_size)
        self.sil_head        = nn.Linear(BACKBONE_DIM, 1)   # vis_feat → P(SIL), binary BCE
        self.lm              = build_lm_backbone(smollm2_path, lora_rank=lora_rank)
        self.speaker_encoder = SpeakerEncoder(weights_path=resnet50_weights)
        self.fm_head         = FMHead(n_layers=n_dit_layers)

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        visual:       torch.Tensor,              # (B, T, 768) pre-extracted or (B, T, 3, 96, 96)
        clean_ids:    torch.Tensor,              # (B, L) int64, collapsed SIL-free token seq
        clean_mask:   torch.Tensor,              # (B, L) long, 1 = valid token
        lm_idx_text:  torch.Tensor,              # (B, T) int64, gather index for text CE
        lm_idx_fm:    torch.Tensor,              # (B, T) int64, gather index for FM cond
        face:         torch.Tensor,              # (B, 3, 256, 256)
        frame_labels: torch.Tensor,              # (B, T) int64, CE targets
        mask:         torch.Tensor,              # (B, T) bool, True = valid frame
        latent:       torch.Tensor | None = None,  # (B, T_a, 512); None → Phase 1
    ) -> dict:
        B, T = mask.shape

        # ── Text path ─────────────────────────────────────────────────────────
        vis_feat, s_vis = self.visual_encoder(visual)       # (B,T,960), (B,T,49152)
        h_lm, s_lm      = self.lm(clean_ids, clean_mask)   # (B,L,960), (B,L,49152)

        # Gather per-frame LM logits: LM prior BEFORE current token (teacher-forcing)
        idx_t     = lm_idx_text.unsqueeze(-1)                              # (B,T,1)
        s_lm_text = s_lm.gather(1, idx_t.expand(B, T, s_lm.shape[-1]))    # (B,T,49152)

        # PoE in log-prob space. Clamp prevents -inf (bfloat16 underflow for low-prob tokens)
        # from combining with label_smoothing's ε/V > 0 weight → +inf loss → NaN gradients.
        LOG_CLAMP = -100.0
        lp_vis = F.log_softmax(s_vis,      dim=-1).clamp(min=LOG_CLAMP)
        lp_lm  = F.log_softmax(s_lm_text,  dim=-1).clamp(min=LOG_CLAMP)
        posterior = lp_vis + self.alpha * lp_lm

        # SIL detection: binary BCE on all valid frames (vis_feat → P(SIL)).
        # Separates speech/silence detection from token identity learning.
        sil_logit  = self.sil_head(vis_feat).squeeze(-1)         # (B, T)
        sil_target = (frame_labels == SIL_ID).float()
        loss_sil   = F.binary_cross_entropy_with_logits(
            sil_logit[mask], sil_target[mask],
        )

        # Token CE loss: non-SIL frames only, so SIL class cannot dominate.
        non_sil = mask & (frame_labels != SIL_ID)
        if non_sil.any():
            loss_text = F.cross_entropy(
                posterior.reshape(-1, posterior.shape[-1])[non_sil.reshape(-1)],
                frame_labels.reshape(-1)[non_sil.reshape(-1)],
            )
        else:
            loss_text = posterior.new_zeros(())

        # ── Audio path (Phase 2 only) ─────────────────────────────────────────
        loss_fm = posterior.new_zeros(())
        if latent is not None:
            id_vec  = self.speaker_encoder(face)
            v_down  = vis_feat.detach()[:, ::2, :]             # (B,T_a,960)
            # Gather FM conditioning: LM state AFTER current token is committed
            idx_f   = lm_idx_fm.unsqueeze(-1)                  # (B,T,1)
            h_lm_fm = h_lm.gather(1, idx_f.expand(B, T, h_lm.shape[-1]))  # (B,T,960)
            h_down  = h_lm_fm.detach()[:, ::2, :]              # (B,T_a,960)
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

    # ── Inference (full-sequence, non-streaming) ──────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        visual: torch.Tensor,    # (B, T, 768) or (B, T, 3, 96, 96)
        face:   torch.Tensor,    # (B, 3, 256, 256)
        nfe:    int = 10,
    ) -> dict:
        """
        Two-pass greedy decode + FM latent generation.

        Pass 1: visual-only CTC greedy collapse → clean token sequence x_clean.
        Pass 2: LM teacher-forced on x_clean → PoE posterior → final per-frame x̂_t.

        The LM runs on the collapsed sequence (length L << T), keeping KV clean.
        """
        B, T = visual.shape[:2]
        device = visual.device

        vis_feat, s_vis = self.visual_encoder(visual)   # (B,T,960), (B,T,49152)

        # Pass 1: CTC greedy collapse on visual-only logits
        vis_ids = s_vis.argmax(dim=-1)   # (B, T) frame-level argmax
        bos_id  = self.lm.model.config.bos_token_id or SIL_ID
        clean_ids_list, lm_idx_text_list, lm_idx_fm_list = [], [], []
        for b in range(B):
            labels_np = vis_ids[b].cpu().numpy().astype("int64")
            cids, idx_t, idx_f = _compute_lm_indices_np(labels_np, T, bos_id, _SIL)
            clean_ids_list.append(torch.from_numpy(cids).to(device))
            lm_idx_text_list.append(torch.from_numpy(idx_t).to(device))
            lm_idx_fm_list.append(torch.from_numpy(idx_f).to(device))

        # Pad clean_ids across batch
        max_L = max(c.shape[0] for c in clean_ids_list)
        clean_ids  = torch.full((B, max_L), SIL_ID, dtype=torch.long, device=device)
        clean_mask = torch.zeros(B, max_L, dtype=torch.long, device=device)
        lm_idx_text = torch.stack(lm_idx_text_list)   # (B, T)
        lm_idx_fm   = torch.stack(lm_idx_fm_list)     # (B, T)
        for b, cids in enumerate(clean_ids_list):
            L = cids.shape[0]
            clean_ids[b, :L]  = cids
            clean_mask[b, :L] = 1

        # Pass 2: LM on clean sequence → PoE posterior → per-frame x̂_t
        h_lm, s_lm = self.lm(clean_ids, clean_mask)   # (B, L, 960/49152)

        idx_t     = lm_idx_text.unsqueeze(-1)
        s_lm_text = s_lm.gather(1, idx_t.expand(B, T, s_lm.shape[-1]))
        posterior = (F.log_softmax(s_vis, dim=-1).clamp(min=-100.0) +
                     self.alpha * F.log_softmax(s_lm_text, dim=-1).clamp(min=-100.0))
        x_hat     = posterior.argmax(dim=-1)           # (B, T) frame-level prediction

        # Audio generation
        v_down    = vis_feat[:, ::2, :]
        idx_f     = lm_idx_fm.unsqueeze(-1)
        h_lm_fm   = h_lm.gather(1, idx_f.expand(B, T, h_lm.shape[-1]))
        h_down    = h_lm_fm[:, ::2, :]
        id_vec    = self.speaker_encoder(face)
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
        """Phase 1: train visual encoder + sil_head. LM provides fixed language prior (frozen).
        Freezing LM prevents credit-assignment collapse where LM dominates PoE training."""
        for p in self.fm_head.parameters():
            p.requires_grad_(False)
        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)
        for p in self.lm.parameters():
            p.requires_grad_(False)

    def phase2_mode(self):
        """Freeze visual encoder + LM + sil_head; unfreeze FM head for Phase 2 training."""
        for p in self.visual_encoder.conformer.parameters():
            p.requires_grad_(False)
        for p in self.visual_encoder.visual_head.parameters():
            p.requires_grad_(False)
        for p in self.sil_head.parameters():
            p.requires_grad_(False)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        for p in self.fm_head.parameters():
            p.requires_grad_(True)
        for p in self.visual_encoder.av_hubert.parameters():
            p.requires_grad_(False)


def _compute_lm_indices_np(
    frame_labels: "np.ndarray", T: int, bos_id: int, sil_id: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Standalone version of LRS3DatasetV2._compute_lm_indices for inference use."""
    import numpy as np
    clean_ids   = [bos_id]
    lm_idx_text = np.zeros(T, dtype=np.int64)
    lm_idx_fm   = np.zeros(T, dtype=np.int64)

    t = 0
    while t < T:
        tok = int(frame_labels[t])
        end = t + 1
        while end < T and frame_labels[end] == tok:
            end += 1
        if tok == sil_id:
            last = len(clean_ids) - 1
            lm_idx_text[t:end] = last
            lm_idx_fm[t:end]   = last
        else:
            pos_before = len(clean_ids) - 1
            pos_after  = len(clean_ids)
            clean_ids.append(tok)
            lm_idx_text[t:end] = pos_before
            lm_idx_fm[t:end]   = pos_after
        t = end

    return np.array(clean_ids, dtype=np.int64), lm_idx_text, lm_idx_fm