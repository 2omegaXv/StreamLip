"""
StreamLip V4: AV-HuBERT online + all-layer extraction, no Conformer.

All 12 AV-HuBERT Transformer layer outputs are linearly mapped to
8 LM cross-attn layers, giving shallow LM layers access to early
visual features and deep LM layers access to fully-processed
audiovisual representations.

Trainable in phase1:
  - AV-HuBERT stem (always)
  - AV-HuBERT LoRA layers (if lora_rank > 0)
  - LM cross-attn layers
  - sil_head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import AVHuBERTMultiLayerExtractor, AVHUBERT_DIM, AVHUBERT_NLAYERS
from .lm_backbone    import LMBackboneV4, build_lm_backbone_v4
from .fm_head        import FMHeadV4
from ..v2.speaker_encoder import SpeakerEncoder
from ..v2.data.dataset    import SIL_ID
from .data.dataset        import BLANK_ID

CHUNK_SIZE = 6   # matches dataset's chunk-aligned prefix
N_CHARS    = BLANK_ID + 1   # 28: a-z(0-25) + space(26) + blank(27)


class StreamLipV4(nn.Module):

    def __init__(
        self,
        avhubert_ckpt:      str,
        smollm2_path:       str,
        lambda_text:        float      = 1.0,
        lambda_sil:         float      = 0.3,
        lambda_visual:      float      = 0.5,    # auxiliary visual-only CE loss
        cross_attn_every_n: int        = 4,
        lora_rank:          int        = 16,
        n_dit_layers:       int        = 6,
        mask_prob:          float      = 0.2,    # clean_ids masking probability
        no_text_cond:       bool       = False,  # ablation: zero out h_lm in FM
        resnet50_weights:   str | None = None,
    ):
        super().__init__()
        self.lambda_text   = lambda_text
        self.lambda_sil    = lambda_sil
        self.lambda_visual = lambda_visual
        self.mask_prob     = mask_prob
        self.no_text_cond  = no_text_cond

        self.visual_encoder = AVHuBERTMultiLayerExtractor(
            avhubert_ckpt, device="cpu",
            lora_rank=lora_rank,
        )
        self.sil_head        = nn.Linear(AVHUBERT_DIM, 1)
        self.lm              = build_lm_backbone_v4(
            smollm2_path, cross_attn_every_n,
            n_av_layers=AVHUBERT_NLAYERS,
            vis_dim=AVHUBERT_DIM,
        )
        self.vocab_size      = self.lm.model.config.vocab_size
        self.bos_id          = self.lm.model.config.bos_token_id or SIL_ID
        # CTC auxiliary head: forces AV-HuBERT features to be character-discriminative
        self.visual_head     = nn.Linear(AVHUBERT_DIM, N_CHARS)
        self.speaker_encoder = SpeakerEncoder(weights_path=resnet50_weights)
        self.fm_head         = FMHeadV4(n_layers=n_dit_layers)

    # ---- Training forward ----------------------------------------------------

    def forward(
        self,
        visual:       torch.Tensor,              # (B, T, 3, 96, 96)
        clean_ids:    torch.Tensor,              # (B, L)
        clean_mask:   torch.Tensor,              # (B, L)
        lm_idx_text:  torch.Tensor,              # (B, T)
        lm_idx_fm:    torch.Tensor,              # (B, T)
        face:         torch.Tensor,              # (B, 3, 256, 256)
        frame_labels: torch.Tensor,              # (B, T)
        mask:         torch.Tensor,              # (B, T) bool
        latent:       torch.Tensor | None = None,
        ctc_ids:      torch.Tensor | None = None,  # (B, max_C) char IDs
        ctc_lens:     torch.Tensor | None = None,  # (B,) actual char count per sample
    ) -> dict:
        B, T = mask.shape

        # AV-HuBERT online: last_feat (B,T,768), layer_feats list[12x(B,T,768)]
        last_feat, layer_feats = self.visual_encoder(visual)

        # SIL head: all valid frames
        sil_logit  = self.sil_head(last_feat).squeeze(-1)
        sil_target = (frame_labels == SIL_ID).float()
        loss_sil   = F.binary_cross_entropy_with_logits(
            sil_logit[mask], sil_target[mask]
        )

        # clean_ids masking: force LM to rely on visual cross-attn
        if self.training and self.mask_prob > 0:
            rand_mask = (torch.rand_like(clean_ids, dtype=torch.float) < self.mask_prob)
            rand_mask = rand_mask & (clean_ids != self.bos_id) & (clean_mask.bool())
            rand_toks = torch.randint(0, self.vocab_size, clean_ids.shape, device=clean_ids.device)
            clean_ids_in = torch.where(rand_mask, rand_toks, clean_ids)
        else:
            clean_ids_in = clean_ids

        logits, h_lm = self.lm(clean_ids_in, clean_mask, layer_feats, lm_idx_fm)

        V         = logits.shape[-1]
        idx       = lm_idx_text.unsqueeze(-1).expand(B, T, V)
        posterior = logits.gather(1, idx)               # (B, T, vocab)

        # CE loss on all non-SIL frames (full T)
        non_sil = mask & (frame_labels != SIL_ID)
        if non_sil.any():
            loss_text = F.cross_entropy(
                posterior.reshape(-1, V)[non_sil.reshape(-1)],
                frame_labels.reshape(-1)[non_sil.reshape(-1)],
            )
        else:
            loss_text = posterior.new_zeros(())

        # CTC auxiliary loss: character-level supervision on visual features
        if ctc_ids is not None and ctc_lens is not None and ctc_lens.sum() > 0:
            input_lengths = mask.sum(dim=1)                          # (B,) actual T per sample
            log_probs     = F.log_softmax(
                self.visual_head(last_feat).float(), dim=-1
            ).transpose(0, 1)                                        # (T, B, N_CHARS)
            loss_visual = F.ctc_loss(
                log_probs, ctc_ids, input_lengths, ctc_lens,
                blank=BLANK_ID, zero_infinity=True,
            )
        else:
            loss_visual = posterior.new_zeros(())

        # Audio path (Phase 2)
        loss_fm = posterior.new_zeros(())
        if latent is not None:
            id_vec  = self.speaker_encoder(face)
            v_down  = last_feat.detach()[:, ::2, :]
            idx_f   = lm_idx_fm.unsqueeze(-1)
            h_lm_fm = h_lm.gather(1, idx_f.expand(B, T, h_lm.shape[-1]))
            h_down  = h_lm_fm.detach()[:, ::2, :]
            if self.no_text_cond:
                h_down = torch.zeros_like(h_down)
            loss_fm = self.fm_head.forward_train(v_down, h_down, id_vec, latent.float())

        loss = loss_fm + self.lambda_text * loss_text + self.lambda_sil * loss_sil \
                       + self.lambda_visual * loss_visual

        return {
            "loss":         loss,
            "loss_fm":      loss_fm,          # not detached — needed for phase2 backward
            "loss_text":    loss_text.detach(),
            "loss_sil":     loss_sil.detach(),
            "loss_visual":  loss_visual.detach(),
            "sil_logit": sil_logit.detach(),
            "posterior": posterior.detach(),
        }

    # ---- Generation (inference) ---------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        visual:       torch.Tensor,   # (B, T, 768) or (B, T, 3, 96, 96)
        clean_ids:    torch.Tensor,   # (B, L)
        clean_mask:   torch.Tensor,   # (B, L)
        lm_idx_fm:    torch.Tensor,   # (B, T)
        face:         torch.Tensor,   # (B, 256) pre-extracted speaker emb
        mask:         torch.Tensor,   # (B, T)
        nfe:          int = 10,
    ) -> torch.Tensor:
        """Returns predicted Mimi latent (B, T_a, 512)."""
        B, T = mask.shape

        last_feat, layer_feats = self.visual_encoder(visual)

        clean_ids_in = clean_ids  # no masking at inference
        logits, h_lm = self.lm(clean_ids_in, clean_mask, layer_feats, lm_idx_fm)

        id_vec  = self.speaker_encoder(face)
        v_down  = last_feat[:, ::2, :]
        idx_f   = lm_idx_fm.unsqueeze(-1)
        h_lm_fm = h_lm.gather(1, idx_f.expand(B, T, h_lm.shape[-1]))
        h_down  = h_lm_fm[:, ::2, :]
        if self.no_text_cond:
            h_down = torch.zeros_like(h_down)

        return self.fm_head.forward_inference(v_down, h_down, id_vec, nfe=nfe)

    def param_counts(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M":     round(total     / 1e6, 1),
            "trainable_M": round(trainable / 1e6, 1),
            "frozen_M":    round((total - trainable) / 1e6, 1),
        }

    def phase1_mode(self):
        """Phase 1: text path only. Trainable: AV-HuBERT stem+LoRA, cross-attn, sil_head."""
        for p in self.fm_head.parameters():
            p.requires_grad_(False)
        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)
        for p in self.lm.model.parameters():
            p.requires_grad_(False)
        for p in self.lm.cross_attn_layers.parameters():
            p.requires_grad_(True)

    def phase2_mode(self):
        """Phase 2: train FM head only."""
        for p in self.visual_encoder.parameters():
            p.requires_grad_(False)
        for p in self.sil_head.parameters():
            p.requires_grad_(False)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)
        for p in self.fm_head.parameters():
            p.requires_grad_(True)
