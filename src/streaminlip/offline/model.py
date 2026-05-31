"""
StreamLipOffline Phase 1: lip-reading text prediction.

Gemma-3-1B + LoRA (trainable) + 13 GatedCrossAttentionLayers (trainable).
AV-HuBERT 12-layer outputs as K/V for each cross-attn layer.
Standard next-token prediction loss on transcript.

No SIL tokens, no MFA alignment, no frame-level supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ..av_hubert import load_avhubert
from ..cross_attention import GatedCrossAttentionLayer

AVHUBERT_DIM = 768
GEMMA_DIM    = 1152
N_AV_LAYERS  = 12


def _av_layer_indices(n_ca: int, n_av: int) -> list[int]:
    if n_ca == 1:
        return [n_av - 1]
    return [round(i * (n_av - 1) / (n_ca - 1)) for i in range(n_ca)]


class AVHuBERTOnlineExtractor(nn.Module):
    """AV-HuBERT Transformer (frozen) with hooks on all 12 layers."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.model = load_avhubert(checkpoint_path, device="cpu")
        self._layer_bufs: list = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.model.encoder.layers):
            def _make_hook(idx):
                def hook(_, _in, output):
                    x = output[0] if isinstance(output, tuple) else output
                    self._layer_bufs[idx] = x.transpose(0, 1)
                return hook
            layer.register_forward_hook(_make_hook(i))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, T, 768)  →  list of 12 × (B, T, 768)"""
        dtype  = x.dtype
        x_enc  = self.model.dropout_input(x.to(next(self.model.parameters()).dtype))
        n      = len(self.model.encoder.layers)
        self._layer_bufs = [None] * n
        last, _ = self.model.encoder(x_enc)
        return [(buf if buf is not None else last).to(dtype)
                for buf in self._layer_bufs]


class StreamLipOffline(nn.Module):

    def __init__(
        self,
        avhubert_ckpt:       str,
        gemma_path:          str,
        cross_attn_every_n:  int = 2,
        lora_rank:           int = 16,
    ):
        super().__init__()

        self.visual_enc = AVHuBERTOnlineExtractor(avhubert_ckpt)

        self.lm = AutoModelForCausalLM.from_pretrained(
            gemma_path, torch_dtype=torch.bfloat16
        )

        # Build cross-attn layers and register hooks BEFORE LoRA.
        # peft only wraps linear layers inside each transformer layer;
        # the transformer layer objects themselves stay the same, so hooks survive.
        n_lm = len(self.lm.model.layers)
        n_ca = n_lm // cross_attn_every_n
        self.cross_attn_every_n = cross_attn_every_n
        self.av_indices         = _av_layer_indices(n_ca, N_AV_LAYERS)
        self.cross_attn_layers  = nn.ModuleList([
            GatedCrossAttentionLayer(hidden_dim=GEMMA_DIM, vis_dim=AVHUBERT_DIM)
            for _ in range(n_ca)
        ])
        for layer in self.cross_attn_layers:
            nn.init.constant_(layer.gate, 0.1)

        print(f"[StreamLipOffline] {n_ca} cross-attn → AV-HuBERT layers "
              f"{[i+1 for i in self.av_indices]}")

        self._layer_feats = None
        self._register_lm_hooks()   # must be before get_peft_model

        # Apply LoRA AFTER hooks are registered
        if lora_rank > 0:
            from peft import get_peft_model, LoraConfig
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, lora_cfg)
            n_lora = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
            print(f"[StreamLipOffline] Gemma LoRA rank={lora_rank}, trainable: {n_lora/1e6:.2f}M")
        else:
            for p in self.lm.parameters():
                p.requires_grad_(False)

    def _register_lm_hooks(self):
        ca_idx = 0
        for i, layer in enumerate(self.lm.model.layers):
            if (i + 1) % self.cross_attn_every_n == 0 and ca_idx < len(self.cross_attn_layers):
                def _make_hook(ca, av_idx):
                    def hook(_, _in, output):
                        if self._layer_feats is None:
                            return output
                        hs  = output[0] if isinstance(output, tuple) else output
                        vis = self._layer_feats[av_idx]
                        if vis.shape[0] != hs.shape[0]:
                            vis = vis.expand(hs.shape[0], -1, -1)
                        hs  = ca(hs, vis)
                        return (hs,) + output[1:] if isinstance(output, tuple) else hs
                    return hook
                layer.register_forward_hook(
                    _make_hook(self.cross_attn_layers[ca_idx],
                               self.av_indices[ca_idx])
                )
                ca_idx += 1

    def forward(
        self,
        visual:         torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels:         torch.Tensor | None = None,
    ) -> dict:
        layer_feats = self.visual_enc(visual.to(next(self.visual_enc.parameters()).dtype))
        layer_feats = [f.to(visual.dtype) for f in layer_feats]

        self._layer_feats = layer_feats
        try:
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self._layer_feats = None

        return {"loss": out.loss, "logits": out.logits}

    @torch.no_grad()
    def generate_text(
        self,
        visual:         torch.Tensor,
        max_new_tokens: int = 64,
        num_beams:      int = 5,
        **gen_kwargs,
    ) -> torch.Tensor:
        layer_feats = self.visual_enc(visual.to(next(self.visual_enc.parameters()).dtype))
        layer_feats = [f.to(visual.dtype) for f in layer_feats]

        bos  = torch.tensor([[self.lm.config.bos_token_id]],
                             dtype=torch.long, device=visual.device)
        amsk = torch.ones_like(bos)
        self._layer_feats = layer_feats
        try:
            ids = self.lm.generate(
                bos,
                attention_mask=amsk,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                length_penalty=1.0,
                eos_token_id=self.lm.config.eos_token_id,
                pad_token_id=self.lm.config.eos_token_id,
                **gen_kwargs,
            )
        finally:
            self._layer_feats = None

        return ids[:, 1:]

    def param_counts(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_M":     round(total / 1e6, 1),
                "trainable_M": round(trainable / 1e6, 1),
                "frozen_M":    round((total - trainable) / 1e6, 1)}
