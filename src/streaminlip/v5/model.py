"""
StreamLip V5: Visual Prefix LM  or  V4-style Multi-Layer Cross-Attention.

cross_attn_every_n=0  → visual prefix
    - 输入: avsr_enc.npy (T, 768) — Auto-AVSR Conformer 最终输出
    - 直接 proj → LM，Conformer 不参与训练

cross_attn_every_n>0  → Flamingo-style gated cross-attention
    - 输入: avsr_enc.npy (T, 768)
    - n_ca cross-attn 层均匀分布在 LM 层间
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "third_party/auto_avsr"))

from ..cross_attention import GatedCrossAttentionLayer

AVSR_DIM     = 768   # Auto-AVSR Conformer 维度
AVSR_NLAYERS = 12    # Conformer 层数


def _layer_indices(n_ca: int, n_enc: int) -> list[int]:
    if n_ca == 1:
        return [n_enc - 1]
    return [round(i * (n_enc - 1) / (n_ca - 1)) for i in range(n_ca)]


class StreamLipV5(nn.Module):

    def __init__(
        self,
        avsr_ckpt:          str,
        smollm2_path:       str,
        lora_rank:          int   = 0,
        cfg_drop_prob:      float = 0.1,
        cross_attn_every_n: int   = 0,
        # 兼容旧参数名
        avhubert_ckpt:      str   = "",
        lambda_sil:         float = 0.0,   # 已废弃，保留兼容
    ):
        super().__init__()
        if not avsr_ckpt and avhubert_ckpt:
            avsr_ckpt = avhubert_ckpt

        self.cfg_drop_prob   = cfg_drop_prob
        self.cross_attn_mode = cross_attn_every_n > 0

        # ── Auto-AVSR 视觉编码器（frozen） ────────────────────────────────────
        from streaminlip.auto_avsr import AutoAVSRInferencer
        _asr = AutoAVSRInferencer(avsr_ckpt, device="cpu")
        for p in _asr.parameters():
            p.requires_grad_(False)
        self.frontend     = _asr.model.frontend
        self.proj_encoder = _asr.model.proj_encoder
        # ── CTC head（预训练权重，frozen，用于推理时估算生成长度） ──────────────
        self.ctc_lo = nn.Linear(AVSR_DIM, _asr.model.ctc.ctc_lo.out_features)
        self.ctc_lo.load_state_dict(_asr.model.ctc.ctc_lo.state_dict())
        for p in self.ctc_lo.parameters():
            p.requires_grad_(False)
        del _asr

        # Multi-layer buffer for cross-attention
        self._layer_bufs: list[torch.Tensor | None] = [None] * AVSR_NLAYERS

        # ── LM ────────────────────────────────────────────────────────────────
        lm = AutoModelForCausalLM.from_pretrained(smollm2_path)
        lm_dim = lm.config.hidden_size
        self._embed_tokens = lm.model.embed_tokens
        if lora_rank > 0:
            for p in lm.parameters():
                p.requires_grad_(False)
            lm = self._apply_lora(lm, lora_rank)
        else:
            n = sum(p.numel() for p in lm.parameters())
            print(f"[StreamLipV5] LM full fine-tune ({lm.config.model_type}): {n/1e6:.1f}M")
        self.lm = lm
        self.vocab_size = self.lm.config.vocab_size

        # ── Projection + null embedding ───────────────────────────────────────
        self.proj     = nn.Linear(AVSR_DIM, lm_dim)
        self.null_vis = nn.Parameter(torch.randn(lm_dim))

        # ── Cross-attention ───────────────────────────────────────────────────
        self._vis_feats_buf: list[torch.Tensor] | None = None
        if self.cross_attn_mode:
            lm_layers   = self.lm.model.layers
            n_lm        = len(lm_layers)
            n_ca        = sum(1 for i in range(n_lm) if (i + 1) % cross_attn_every_n == 0)
            enc_indices = _layer_indices(n_ca, AVSR_NLAYERS)
            self.ca_enc_indices = enc_indices
            self.ca_layers = nn.ModuleList([
                GatedCrossAttentionLayer(lm_dim, num_heads=8, vis_dim=AVSR_DIM)
                for _ in range(n_ca)
            ])
            self._register_ca_hooks(lm_layers, cross_attn_every_n)
            print(f"[StreamLipV5] Cross-attn: {n_ca} CA layers every {cross_attn_every_n} LM layers, "
                  f"encoder layers: {enc_indices}")

    # ── Hooks ──────────────────────────────────────────────────────────────────

    def _register_ca_hooks(self, lm_layers, every_n: int):
        ca_idx = 0
        for i, layer in enumerate(lm_layers):
            if (i + 1) % every_n == 0 and ca_idx < len(self.ca_layers):
                def _make_hook(ca_layer, enc_idx):
                    def hook(_, _in, output):
                        if self._vis_feats_buf is None:
                            return output
                        vis = self._vis_feats_buf[enc_idx]
                        if vis is None:
                            return output
                        hs = output[0] if isinstance(output, tuple) else output
                        # beam search 时 hs.shape[0] = num_beams，vis 需要扩展
                        if vis.shape[0] != hs.shape[0]:
                            vis = vis.expand(hs.shape[0], -1, -1)
                        hs = ca_layer(hs, vis)
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook
                layer.register_forward_hook(
                    _make_hook(self.ca_layers[ca_idx], self.ca_enc_indices[ca_idx])
                )
                ca_idx += 1

    @staticmethod
    def _apply_lora(lm, rank: int):
        from peft import get_peft_model, LoraConfig
        cfg = LoraConfig(
            r=rank, lora_alpha=rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none",
        )
        lm = get_peft_model(lm, cfg)
        n = sum(p.numel() for p in lm.parameters() if p.requires_grad)
        print(f"[StreamLipV5] LoRA rank={rank}, trainable: {n/1e6:.2f}M")
        return lm

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode(self, visual: torch.Tensor) -> torch.Tensor:
        if self.cross_attn_mode:
            for i in range(AVSR_NLAYERS):
                self._layer_bufs[i] = visual
        return visual

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        visual:          torch.Tensor,
        input_ids:       torch.Tensor,
        target_ids:      torch.Tensor,
        video_pos:       torch.Tensor,
        text_pos:        torch.Tensor,
        video_mask:      torch.Tensor,
        text_mask:       torch.Tensor,
        last_chunk_mask: torch.Tensor | None = None,
        # 兼容旧接口（已废弃）
        sil_labels:      torch.Tensor | None = None,
    ) -> dict:
        B, T_max = video_mask.shape
        dtype = self.proj.weight.dtype

        feat       = self._encode(visual).to(dtype)
        vis_tokens = self.proj(feat)

        if self.training and self.cfg_drop_prob > 0 and not self.cross_attn_mode:
            drop_mask  = torch.rand(B, 1, 1, device=vis_tokens.device) < self.cfg_drop_prob
            null       = self.null_vis.to(vis_tokens.dtype).view(1, 1, -1).expand_as(vis_tokens)
            vis_tokens = torch.where(drop_mask, null, vis_tokens)

        text_emb = self._embed_tokens(input_ids)

        if self.cross_attn_mode:
            def _safe(buf):
                if buf is None:
                    return None
                while isinstance(buf, (tuple, list)):
                    buf = buf[0]
                return buf.detach().to(dtype) if isinstance(buf, torch.Tensor) else None
            self._vis_feats_buf = [_safe(buf) for buf in self._layer_bufs]
            self._layer_bufs = [None] * AVSR_NLAYERS
            try:
                out = self.lm(inputs_embeds=text_emb, attention_mask=text_mask.long())
            finally:
                self._vis_feats_buf = None
            text_logits = out.logits
        else:
            seq       = torch.cat([vis_tokens, text_emb], dim=1)
            pos_ids   = torch.cat([video_pos, text_pos],  dim=1)
            attn_mask = torch.cat([video_mask.long(), text_mask.long()], dim=1)
            out = self.lm(inputs_embeds=seq, position_ids=pos_ids, attention_mask=attn_mask)
            text_logits = out.logits[:, T_max:, :]

        valid = (last_chunk_mask & (target_ids != -100)) if (
            last_chunk_mask is not None and last_chunk_mask.any()
        ) else (target_ids != -100)

        loss_ce = F.cross_entropy(
            text_logits.reshape(-1, self.vocab_size)[valid.reshape(-1)],
            target_ids.reshape(-1)[valid.reshape(-1)],
        )

        return {
            "loss":        loss_ce,
            "loss_ce":     loss_ce.detach(),
            "text_logits": text_logits.detach(),
        }

    @torch.no_grad()
    def predict(
        self,
        visual:     torch.Tensor,
        input_ids:  torch.Tensor,
        video_pos:  torch.Tensor,
        text_pos:   torch.Tensor,
        video_mask: torch.Tensor,
        text_mask:  torch.Tensor,
    ) -> torch.Tensor:
        dtype      = self.proj.weight.dtype
        feat       = self._encode(visual).to(dtype)
        vis_tokens = self.proj(feat)
        text_emb   = self._embed_tokens(input_ids)
        T          = visual.shape[1]

        if self.cross_attn_mode:
            def _safe_p(buf):
                if buf is None: return None
                while isinstance(buf, (tuple, list)): buf = buf[0]
                return buf.to(dtype) if isinstance(buf, torch.Tensor) else None
            self._vis_feats_buf = [_safe_p(buf) for buf in self._layer_bufs]
            try:
                out = self.lm(inputs_embeds=text_emb, attention_mask=text_mask.long())
            finally:
                self._vis_feats_buf = None
            return out.logits
        else:
            seq       = torch.cat([vis_tokens, text_emb], dim=1)
            pos_ids   = torch.cat([video_pos, text_pos],  dim=1)
            attn_mask = torch.cat([video_mask.long(), text_mask.long()], dim=1)
            out = self.lm(inputs_embeds=seq, position_ids=pos_ids, attention_mask=attn_mask)
            return out.logits[:, T:, :]

    @torch.no_grad()
    def ctc_len_estimate(
        self,
        feat: torch.Tensor,
        bpe_spm_ratio: float = 1.25,
        margin: float = 1.3,
    ) -> int:
        """用 CTC greedy collapse 估算应生成的 BPE token 数上界。

        Args:
            feat:          (1, T, 768) encoder 输出（float，任意 dtype）
            bpe_spm_ratio: BPE token 数 / SPM token 数的经验比值（英文约 1.1~1.4）
            margin:        在估算值上乘的安全系数，避免截断过早

        Returns:
            max_new_tokens 建议值（至少 8）
        """
        logits = self.ctc_lo(feat[0].to(self.ctc_lo.weight.dtype))  # (T, vocab_spm)
        ids    = logits.argmax(-1).tolist()       # greedy
        blank  = 0
        n_spm  = sum(
            1 for i, t in enumerate(ids)
            if t != blank and (i == 0 or t != ids[i - 1])
        )
        return max(8, int(n_spm * bpe_spm_ratio * margin))

    def param_counts(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M":     round(total     / 1e6, 1),
            "trainable_M": round(trainable / 1e6, 1),
            "frozen_M":    round((total - trainable) / 1e6, 1),
        }
