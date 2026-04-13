"""
CompactBilinearHead: second-order pooling head for texture classification.

Pipeline:
  (B, C_in, H, W)
    → 1×1 Conv → (B, D, H, W)          # channel reduction
    → GAP       → (B, D)               # spatial averaging
    → outer product → (B, D, D)        # second-order statistics
    → signed sqrt + L2 norm → (B, D²)  # normalisation for linear separability
    → Linear(D², num_classes)

Why second-order pooling for textures:
  Standard GAP captures first-order (mean) activations only. Textures are defined
  by co-activation patterns between channels (e.g. channels responding to horizontal
  and vertical edges fire together in a grid texture). The outer product captures
  these channel covariances. Signed sqrt + L2 norm is the B-CNN normalisation from
  Lin et al. (2015) that improves linear separability of the resulting descriptor.

Memory note:
  D=256 → 65,536-dim descriptor → ~10MB for the Linear weight at float32.
  D=128 → 16,384-dim → ~2.5MB. Default is 256; set bilinear_dim=128 in TrainConfig if OOM.

Quantisation note:
  The outer product and signed sqrt operations are kept in FP32 even during QAT.
  Set qconfig=None on this module in quantise.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactBilinearHead(nn.Module):
    def __init__(self, in_channels: int, d: int, num_classes: int, dropout: float = 0.3):
        """
        Args:
            in_channels: number of channels from the backbone feature map
            d:           reduced channel dimension (bilinear_dim in TrainConfig)
            num_classes: number of output classes (47 for DTD)
            dropout:     dropout probability before the final linear layer
        """
        super().__init__()
        self.d = d

        # Channel reduction
        self.reduce = nn.Conv2d(in_channels, d, kernel_size=1, bias=False)
        self.bn     = nn.BatchNorm2d(d)

        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(d * d, num_classes)

        # Initialise fc with small weights to avoid exploding logits early in training
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)  — backbone feature map
        returns: (B, num_classes) logits
        """
        # Channel reduction: (B, C_in, H, W) → (B, D, H, W)
        x = F.relu(self.bn(self.reduce(x)))

        # Global average pooling: (B, D, H, W) → (B, D)
        # Cast to fp32 before the bilinear ops: 1e-10 underflows to 0 in fp16,
        # making sqrt(0) produce inf gradients that GradScaler silently drops.
        x = x.mean(dim=[2, 3]).float()

        # Outer product: (B, D) × (B, D)^T → (B, D, D)
        # Equivalent to bmm of column and row vectors
        x = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))  # (B, D, D)

        # Signed square root (improves linear separability of texture descriptors)
        x = torch.sign(x) * torch.sqrt(x.abs() + 1e-6)

        # L2 normalise
        x = F.normalize(x.view(x.size(0), -1), p=2, dim=1)  # (B, D*D)

        x = self.dropout(x)
        return self.fc(x)

    def set_dropout(self, p: float):
        """Allow updating dropout rate between training stages."""
        self.dropout.p = p


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    head = CompactBilinearHead(in_channels=768, d=256, num_classes=47)
    dummy = torch.randn(2, 768, 7, 7)
    out = head(dummy)
    assert out.shape == (2, 47), f"Expected (2, 47), got {out.shape}"
    print(f"CompactBilinearHead output shape: {out.shape}  ✓")

    # Reduced-dim variant
    head_small = CompactBilinearHead(in_channels=768, d=128, num_classes=47)
    out_small = head_small(dummy)
    assert out_small.shape == (2, 47)
    print(f"CompactBilinearHead (d=128) output shape: {out_small.shape}  ✓")
