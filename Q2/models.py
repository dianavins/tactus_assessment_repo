"""
Model builders for the DTD texture classifier.

Three variants:
  1. build_convnext()     — ConvNeXt-Tiny (ImageNet-22k pretrained) + CompactBilinearHead
  2. build_mobilenetv3()  — MobileNetV3-Large (ImageNet-1k pretrained) + GAP + Linear head
  3. build_dinov2_probe() — DINOv2 ViT-S/14 (frozen) + single Linear head

Each builder returns (model, stage_groups) where stage_groups is a list of
parameter group dicts used by get_layer_lr_groups() in utils.py.
"""

import torch
import torch.nn as nn

import timm

from heads import CompactBilinearHead
from config import TrainConfig


# ── ConvNeXt-Tiny ──────────────────────────────────────────────────────────

def build_convnext(cfg: TrainConfig) -> tuple[nn.Module, list[nn.Module]]:
    """
    Load ConvNeXt-Tiny pretrained on ImageNet-22k via timm.
    Replace the classifier head with CompactBilinearHead.

    Returns:
        model:        the full model
        stage_groups: [stem, stage0, stage1, stage2, stage3, head]
                      ordered from input → output for layer-wise LR assignment
    """
    model = timm.create_model(
        "convnext_tiny.fb_in22k",
        pretrained=cfg.pretrained,
        num_classes=0,          # remove timm's classifier; we attach our own
        drop_path_rate=cfg.drop_path_rate,
    )

    # ConvNeXt-Tiny feature map channels at the final stage: 768
    in_channels = model.num_features  # 768 for convnext_tiny

    head = CompactBilinearHead(
        in_channels=in_channels,
        d=cfg.bilinear_dim,
        num_classes=47,
        dropout=cfg.head_dropout,
    )
    model.head = head

    # Override timm's forward_features + head with a unified forward
    # timm's ConvNeXt with num_classes=0 returns the feature map from forward_features()
    # We need to route it through our head instead of the identity.
    class ConvNeXtWithBilinearHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            # forward_features() returns (B, C, H, W) in this version of timm.
            features = self.backbone.forward_features(x)
            # Dequantize if backbone was run in INT8 mode (no-op otherwise).
            if features.is_quantized:
                features = features.dequantize()
            return self.head(features)

    wrapped = ConvNeXtWithBilinearHead(model, head)

    # Stage groups ordered input → output (for layer-wise LR decay from output)
    stage_groups = [
        model.stem,
        model.stages[0],
        model.stages[1],
        model.stages[2],
        model.stages[3],
        head,
    ]

    return wrapped, stage_groups


# ── MobileNetV3-Large ──────────────────────────────────────────────────────

def build_mobilenetv3(cfg: TrainConfig) -> tuple[nn.Module, list[nn.Module]]:
    """
    Load MobileNetV3-Large pretrained on ImageNet-1k via torchvision.
    Replace the classifier with GAP → Dropout → Linear(960, 47).

    Uses standard GAP (not bilinear pooling): at 960 channels,
    bilinear pooling would produce a 921,600-dim descriptor — unacceptable.
    """
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if cfg.pretrained else None
    )

    # Replace the full classifier (original: Linear→HSwish→Linear 1000)
    in_features = 960  # output of MobileNetV3-Large's avgpool
    model.classifier = nn.Sequential(
        nn.Dropout(p=cfg.head_dropout),
        nn.Linear(in_features, 47),
    )
    nn.init.normal_(model.classifier[1].weight, std=0.01)
    nn.init.zeros_(model.classifier[1].bias)

    # Stage groups: features is a Sequential of 16 InvertedResidual blocks + misc layers
    # Group into: early (first half), late (second half), head
    features = list(model.features.children())
    mid = len(features) // 2
    stage_groups = [
        nn.Sequential(*features[:mid]),   # early layers
        nn.Sequential(*features[mid:]),   # late layers
        model.classifier,                 # head
    ]

    return model, stage_groups


# ── DINOv2 Linear Probe ────────────────────────────────────────────────────

def build_dinov2_probe(cfg: TrainConfig, local_path: str = None) -> tuple[nn.Module, list[nn.Module]]:
    """
    Load DINOv2 ViT-S/14 as a frozen feature extractor.
    Attach a single Linear(384, 47) probe head.

    Args:
        local_path: path to a local dinov2_vits14.pth if torch.hub is unavailable

    Why DINOv2 for textures:
        DINOv2's self-supervised objective (DINO + iBOT) produces features that are
        strongly texture-aware, unlike standard supervised ImageNet CNNs which are
        shape-biased (Geirhos et al., 2019). Even a linear probe on frozen DINOv2
        features is competitive with fully fine-tuned CNNs on DTD.
    """
    if local_path is not None:
        # Load architecture from hub (no weights) then load local checkpoint
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=False
        )
        state = torch.load(local_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=False)
        print(f"[model] Loaded DINOv2 weights from local path: {local_path}")
    else:
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        )
        print("[model] Loaded DINOv2 ViT-S/14 from torch.hub")

    # Freeze all backbone parameters
    for p in backbone.parameters():
        p.requires_grad = False

    embed_dim = backbone.embed_dim  # 384 for ViT-S

    class DINOv2Probe(nn.Module):
        def __init__(self, backbone, embed_dim, num_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(embed_dim, num_classes)
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):
            with torch.no_grad():
                features = self.backbone(x)  # (B, embed_dim) CLS token
            return self.head(features)

    model = DINOv2Probe(backbone, embed_dim, num_classes=47)

    # Only the head has trainable parameters
    stage_groups = [model.head]

    return model, stage_groups


# ── DenseNet-201 ───────────────────────────────────────────────────────────

def build_densenet201(cfg: TrainConfig) -> tuple[nn.Module, list[nn.Module]]:
    """
    Load DenseNet-201 pretrained on ImageNet-1k via torchvision.
    Replace the classifier with Dropout → Linear(1920, 47).

    DenseNet-201 has 1920 output channels after the final dense block.
    Dense connections reuse features from all preceding layers, which is
    beneficial for texture recognition where fine-grained channel statistics
    matter across all scales.

    Stage groups ordered input → output (for layer-wise LR assignment):
        [stem+denseblock1+transition1, denseblock2+transition2+denseblock3+transition3,
         denseblock4+norm5, classifier]

    Quantisation note:
        DenseNet uses pre-activation order (BN→ReLU→Conv). Because Conv layers
        come *after* BN in each block, the standard Conv+BN absorption pattern
        does not apply. fuse_modules() is skipped for DenseNet-201 (see quantise.py).
    """
    from torchvision.models import densenet201, DenseNet201_Weights

    model = densenet201(
        weights=DenseNet201_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    )

    in_features = model.classifier.in_features  # 1920
    model.classifier = nn.Sequential(
        nn.Dropout(p=cfg.head_dropout),
        nn.Linear(in_features, 47),
    )
    nn.init.normal_(model.classifier[1].weight, std=0.01)
    nn.init.zeros_(model.classifier[1].bias)

    # features children (12 total):
    #   [0] Conv2d  [1] BN  [2] ReLU  [3] MaxPool
    #   [4] DenseBlock1  [5] Transition1
    #   [6] DenseBlock2  [7] Transition2
    #   [8] DenseBlock3  [9] Transition3
    #   [10] DenseBlock4  [11] BN (norm5)
    features = list(model.features.children())
    stage_groups = [
        nn.Sequential(*features[:6]),    # stem + denseblock1 + transition1
        nn.Sequential(*features[6:10]),  # denseblock2 + transition2 + denseblock3 + transition3
        nn.Sequential(*features[10:]),   # denseblock4 + norm5
        model.classifier,
    ]

    return model, stage_groups


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = TrainConfig(pretrained=False)  # skip download for smoke test

    print("Testing ConvNeXt-Tiny...")
    m_cx, groups_cx = build_convnext(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = m_cx(x)
    assert out.shape == (2, 47), f"ConvNeXt output shape: {out.shape}"
    total = sum(p.numel() for p in m_cx.parameters())
    print(f"  Output: {out.shape}  Params: {total/1e6:.1f}M  ✓")

    print("Testing MobileNetV3-Large...")
    m_mv, groups_mv = build_mobilenetv3(cfg)
    out_mv = m_mv(x)
    assert out_mv.shape == (2, 47), f"MobileNetV3 output shape: {out_mv.shape}"
    total_mv = sum(p.numel() for p in m_mv.parameters())
    print(f"  Output: {out_mv.shape}  Params: {total_mv/1e6:.1f}M  ✓")

    print("Testing DenseNet-201...")
    m_dn, groups_dn = build_densenet201(cfg)
    out_dn = m_dn(x)
    assert out_dn.shape == (2, 47), f"DenseNet-201 output shape: {out_dn.shape}"
    total_dn = sum(p.numel() for p in m_dn.parameters())
    print(f"  Output: {out_dn.shape}  Params: {total_dn/1e6:.1f}M  ✓")

    print("All model smoke tests passed.")
