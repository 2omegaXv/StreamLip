"""
Speaker Encoder: ImageNet-pretrained ResNet50 (frozen) + Linear(2048, 256).

Input:  first-chunk face crop, (B, 3, 256, 256), ImageNet normalized
        OR stacked frames (B, C, 3, 256, 256) — mean-pooled internally
Output: id̂ (B, 256) — speaker identity vector for FM conditioning

Frozen throughout training; no audio required at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class SpeakerEncoder(nn.Module):
    """ResNet50 backbone (frozen) + linear projection → 256-dim speaker embedding."""

    RESNET_DIM = 2048
    ID_DIM     = 256

    def __init__(self, weights_path: str | None = None):
        """
        weights_path: local .pth file (e.g. 'pretrained/resnet50-11ad3fa6.pth').
                      If None, attempts auto-download (requires internet).
        Download: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
        """
        super().__init__()
        if weights_path is not None:
            backbone = tvm.resnet50(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)
        else:
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

        # Drop the final FC; keep everything up to and including avgpool.
        # Output shape after avgpool: (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(self.RESNET_DIM, self.ID_DIM)

        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, face: torch.Tensor) -> torch.Tensor:
        """
        face: (B, 3, 256, 256)     — single face image, ImageNet normalized
              OR (B, C, 3, 256, 256) — C face frames, mean-pooled before encoding
              OR (B, 256)           — pre-extracted speaker embedding (fast path)
        returns: id̂ (B, 256)
        """
        if face.dim() == 2:
            # Fast path: already a 256-d speaker embedding, cast dtype only
            return face.to(self.proj.weight.dtype)

        if face.dim() == 5:
            face = face.mean(dim=1)

        face = F.interpolate(face, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.backbone(face)
        feat = feat.flatten(1)
        return self.proj(feat.to(self.proj.weight.dtype))
