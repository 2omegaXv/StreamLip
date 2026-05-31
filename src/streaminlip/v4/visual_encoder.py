"""
Visual Encoder V4.

AV-HuBERT runs online (no cached features).  Outputs:
  last_feat   (B, T, 768)       — final layer, for FM conditioning & sil_head
  layer_feats list of (B,T,768) — all 12 Transformer layer outputs, for per-layer cross-attn

No Conformer.  The chunk-aligned prefix training strategy ensures
train/inference consistency without any causal masking.

LoRA on AV-HuBERT attention layers is supported via peft.
"""
import torch
import torch.nn as nn
from ..av_hubert import load_avhubert

AVHUBERT_DIM  = 1024
AVHUBERT_NLAYERS = 24


class AVHuBERTMultiLayerExtractor(nn.Module):
    """
    Runs AV-HuBERT and returns (last_feat, layer_feats).

    layer_feats: list of n_av_layers tensors (B, T, 768), one per Transformer layer.
    lora_rank:   if > 0, apply LoRA to q_proj/v_proj of all Transformer layers.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device:          str = "cpu",
        lora_rank:       int = 0,
    ):
        super().__init__()
        self.model       = load_avhubert(checkpoint_path, device)
        self._layer_bufs: list[torch.Tensor | None] = []

        if lora_rank > 0:
            self._apply_lora(lora_rank)

        self._register_all_hooks()  # register after LoRA so hooks attach to final model

    def _register_all_hooks(self):
        enc_layers = self.model.encoder.layers
        n = len(enc_layers)
        self._layer_bufs = [None] * n

        for i, layer in enumerate(enc_layers):
            def _make_hook(idx):
                def hook(_, _in, output):
                    x = output[0] if isinstance(output, tuple) else output
                    # x is (T, B, D) — transpose to (B, T, D)
                    self._layer_bufs[idx] = x.transpose(0, 1)
                return hook
            layer.register_forward_hook(_make_hook(i))

    def _apply_lora(self, rank: int):
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft is required for LoRA: uv add peft")

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[AVHuBERTMultiLayerExtractor] LoRA rank={rank}, trainable: {n_train/1e6:.2f}M")

    def forward(
        self, lip_frames: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        lip_frames: (B, T, 3, 96, 96)  raw frames  — runs full AV-HuBERT
                 OR (B, T, 768)         pre-extracted CNN frontend features
                                        — skips CNN, runs only Transformer + LoRA
        returns:
          last_feat   (B, T, 768)        — final Transformer layer output
          layer_feats list[(B, T, 768)]  — all 12 layer outputs (index 0 = layer 1)
        """
        if lip_frames.dim() == 3:
            # Fast path: pre-extracted (B, T, 768) CNN frontend features
            # Skip CNN, run only Transformer + LoRA
            x = lip_frames.to(next(self.model.parameters()).dtype)  # (B, T, 768)
            self._layer_bufs = [None] * len(self.model.encoder.layers)
            x_enc = self.model.dropout_input(x)          # (B, T, 768)
            last_enc, _ = self.model.encoder(x_enc, padding_mask=None)
            last = last_enc.to(lip_frames.dtype)          # (B, T, 768) — encoder already returns (B,T,D)
        else:
            # Slow path: raw lip frames
            x = lip_frames.permute(0, 2, 1, 3, 4)  # (B, 3, T, 96, 96)
            x = x.to(next(self.model.parameters()).dtype)
            self._layer_bufs = [None] * len(self.model.encoder.layers)
            last, _ = self.model.extract_finetune(
                source={"video": x, "audio": None},
                padding_mask=None,
                mask=False,
            )
            last = last.to(lip_frames.dtype)

        layer_feats = [
            (buf.to(lip_frames.dtype) if buf is not None else last)
            for buf in self._layer_bufs
        ]
        return last, layer_feats
