"""
Auto-AVSR wrapper for lip-to-text inference.

Takes preprocessed lip frames (96×96 RGB, ImageNet-normalized) and outputs
predicted text using the Auto-AVSR model (WER ~20.3% on LRS3).

Usage:
    asr = AutoAVSRInferencer('pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth')
    text = asr.infer(lip_frames)  # lip_frames: (T, 3, 96, 96) float32 or (T, 96, 96, 3) uint8
"""
import sys
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from pathlib import Path

_AUTOAVSR_ROOT = Path(__file__).parent.parent.parent / "third_party" / "auto_avsr"


class AutoAVSRInferencer(nn.Module):
    """
    Wraps Auto-AVSR for inference.
    Input: lip frames in our project's format (96×96 RGB, ImageNet-norm OR uint8)
    Output: predicted text string
    """

    AVSR_MEAN = 0.421
    AVSR_STD  = 0.165

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        beam_size: int = 40,
    ):
        super().__init__()
        # Add auto_avsr to path temporarily
        avsr_root = str(_AUTOAVSR_ROOT)
        if avsr_root not in sys.path:
            sys.path.insert(0, avsr_root)

        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
        from datamodule.transforms import TextTransform

        self.text_transform = TextTransform()
        self.token_list     = self.text_transform.token_list
        self.beam_size      = beam_size

        # Build model
        self.model = E2E(len(self.token_list), modality="video", ctc_weight=0.1)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        print(f"[AutoAVSR] Loaded: {ckpt_path}")

        self.model = self.model.to(device).eval()
        self._device = device
        self._beam_search = None  # lazily initialized

    def _get_beam_search(self):
        if self._beam_search is None:
            avsr_root = str(_AUTOAVSR_ROOT)
            if avsr_root not in sys.path:
                sys.path.insert(0, avsr_root)
            from lightning import get_beam_search_decoder
            self._beam_search = get_beam_search_decoder(
                self.model, self.token_list, beam_size=self.beam_size
            ).to(self._device)
        return self._beam_search

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert from our format to Auto-AVSR format.

        Accepts:
          (T, 3, 96, 96) float32  — ImageNet-normalized RGB
          (T, 96, 96, 3) uint8    — raw RGB from lip.npy

        Returns:
          (T, 1, 88, 88) float32  — grayscale, Auto-AVSR normalized
        """
        if frames.dtype == torch.uint8:
            # raw uint8 (T, H, W, C) → (T, C, H, W) float [0,1]
            frames = frames.float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)
        else:
            # ImageNet-normalized (T, 3, 96, 96) → undo normalization → [0,1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1)
            frames = frames * std + mean  # (T, 3, 96, 96) in [0,1]

        # CenterCrop 88×88
        frames = TF.center_crop(frames, [88, 88])  # (T, 3, 88, 88)

        # Grayscale: weighted average
        frames = 0.2126 * frames[:, 0] + 0.7152 * frames[:, 1] + 0.0722 * frames[:, 2]
        frames = frames.unsqueeze(1)  # (T, 1, 88, 88)

        # Auto-AVSR normalization
        frames = (frames - self.AVSR_MEAN) / self.AVSR_STD

        return frames

    @torch.no_grad()
    def _encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-clip encode. Returns (enc_feat (T', 768), log_probs (T', vocab)).
        """
        x = self.preprocess_frames(frames).to(self._device)  # (T, 1, 88, 88)
        x = x.unsqueeze(0)                                    # (1, T, 1, 88, 88)
        x = self.model.frontend(x)                            # (1, T', 512)
        x = self.model.proj_encoder(x)                        # (1, T', 768)
        enc_feat, _ = self.model.encoder(x, None)             # (1, T', 768)
        log_probs = self.model.ctc.ctc_lo(enc_feat)           # (1, T', vocab)
        return enc_feat.squeeze(0), log_probs.squeeze(0)

    @torch.no_grad()
    def encode_batch(self, frames_list: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch encode for fast extraction. Pads to max length.
        Returns list of (enc_feat (T_i, 768), log_probs (T_i, vocab)).
        """
        preprocessed = [self.preprocess_frames(f).to(self._device) for f in frames_list]
        lens = [x.shape[0] for x in preprocessed]
        max_T = max(lens)
        B = len(lens)
        # Pad to (B, max_T, 1, 88, 88)
        padded = torch.zeros(B, max_T, 1, 88, 88,
                             device=self._device, dtype=preprocessed[0].dtype)
        for i, (x, l) in enumerate(zip(preprocessed, lens)):
            padded[i, :l] = x
        x = self.model.frontend(padded)         # (B, T', 512)
        x = self.model.proj_encoder(x)          # (B, T', 768)
        enc_feat, _ = self.model.encoder(x, None)  # (B, T', 768)
        log_probs = self.model.ctc.ctc_lo(enc_feat)  # (B, T', vocab)
        # Unpad results (use original lens for enc_feat — frontend may change T)
        results = []
        T_out = enc_feat.shape[1]
        for i, l in enumerate(lens):
            # enc_feat has same T as input (no temporal downsampling in this model)
            t_i = min(l, T_out)
            results.append((enc_feat[i, :t_i], log_probs[i, :t_i]))
        return results

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Public alias for single clip. Returns (enc_feat (T', 768), log_probs (T', vocab))."""
        return self._encode(frames)

    @torch.no_grad()
    def infer(self, frames: torch.Tensor) -> str:
        """Beam-search decode (~20% WER)."""
        enc_feat, _ = self._encode(frames)
        beam_search = self._get_beam_search()
        nbest = beam_search(enc_feat)
        nbest = [h.asdict() for h in nbest[:1]]
        pred_ids = torch.tensor(list(map(int, nbest[0]["yseq"][1:])))
        return self.text_transform.post_process(pred_ids).replace("<eos>", "").strip()

    @torch.no_grad()
    def infer_ctc(self, frames: torch.Tensor) -> str:
        """CTC greedy decode (~24% WER, streaming-stable prefix)."""
        _, log_probs = self._encode(frames)
        ids = log_probs.argmax(-1).tolist()
        blank, prev, collapsed = 0, 0, []
        for t in ids:
            if t != blank and t != prev:
                collapsed.append(t)
            prev = t
        pred_ids = torch.tensor(collapsed, dtype=torch.long)
        return self.text_transform.post_process(pred_ids).replace("<eos>", "").strip()
