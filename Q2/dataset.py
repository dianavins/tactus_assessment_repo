"""
DTD dataset loading and augmentation pipelines.

Supports two training data sources:
  - torchvision.datasets.DTD (original, 1,880 train images)
  - data/dtd_expanded/ (offline-expanded, ~8k–11k train images; see expand_dataset.py)

Val and test splits are always loaded from the original torchvision DTD.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import DTD, ImageFolder

from config import DataConfig


# ── ImageNet normalisation (used by all backbones) ─────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(stage: str, image_size: int = 224) -> dict:
    """
    Return {'train': transform, 'val': transform} for the given training stage.

    stage: 'stage1' | 'stage2' | 'stage3' | 'qat' | 'dinov2_probe' | 'eval'
      - stage1/2/3: full augmentation (RandAugment + rotation + Mixup-ready)
      - qat: minimal augmentation (crop + flip only — Mixup and RandAugment disabled)
      - dinov2_probe: simple augmentation (no RandAugment, no Mixup)
      - eval: deterministic centre-crop for validation/test
    """
    normalise = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalise,
    ])

    if stage in ("stage1", "stage2", "stage3"):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Rotation 0–360°: textures are rotation-invariant
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalise,
        ])

    elif stage == "qat":
        # Minimal augmentation during QAT — RandAugment and Mixup disabled
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalise,
        ])

    elif stage == "dinov2_probe":
        # No RandAugment; DINOv2 uses its own pretraining normalisation but
        # ImageNet stats work fine for the linear probe head
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalise,
        ])

    elif stage == "eval":
        train_transform = val_transform  # no training; reuse val for consistency

    else:
        raise ValueError(f"Unknown stage '{stage}'. "
                         "Choose: stage1 | stage2 | stage3 | qat | dinov2_probe | eval")

    return {"train": train_transform, "val": val_transform}


def _expanded_root_valid(expanded_root: str) -> bool:
    """Return True if dtd_expanded/ exists and has at least 47 class subdirectories."""
    p = Path(expanded_root) / "train"
    if not p.exists():
        return False
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    return len(subdirs) == 47


def get_dataloaders(
    cfg: DataConfig,
    stage: str,
    batch_size: int,
    mixup_alpha: float = 0.0,
) -> dict:
    """
    Build and return {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.

    Training data source:
      - If cfg.use_expanded and dtd_expanded/ is valid → ImageFolder on dtd_expanded/train/
      - Otherwise → torchvision DTD split cfg.split_id

    Val and test always come from torchvision DTD.
    """
    tfms = get_transforms(stage, cfg.image_size)

    # ── Training set ──────────────────────────────────────────────────────
    if cfg.use_expanded and _expanded_root_valid(cfg.expanded_root):
        train_dataset = ImageFolder(
            root=os.path.join(cfg.expanded_root, "train"),
            transform=tfms["train"],
        )
        print(f"[dataset] Using expanded training set: {len(train_dataset)} images "
              f"({cfg.expanded_root})")
    else:
        if cfg.use_expanded:
            print(f"[dataset] WARNING: expanded root '{cfg.expanded_root}' not found or "
                  "incomplete. Falling back to original torchvision DTD "
                  f"(split {cfg.split_id}).")
        train_dataset = DTD(
            root=cfg.data_root,
            split="train",
            partition=cfg.split_id,
            transform=tfms["train"],
            download=True,
        )
        print(f"[dataset] Using original DTD training set: {len(train_dataset)} images")

    # ── Val / test sets (always original DTD) ─────────────────────────────
    val_dataset = DTD(
        root=cfg.data_root,
        split="val",
        partition=cfg.split_id,
        transform=tfms["val"],
        download=True,
    )
    test_dataset = DTD(
        root=cfg.data_root,
        split="test",
        partition=cfg.split_id,
        transform=tfms["val"],
        download=True,
    )

    print(f"[dataset] Val:  {len(val_dataset)} images")
    print(f"[dataset] Test: {len(test_dataset)} images")
    _verify_classes(train_dataset, val_dataset, cfg.num_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def _verify_classes(train_ds, val_ds, expected: int):
    """Sanity-check that both datasets expose the expected number of classes."""
    def _n_classes(ds):
        if hasattr(ds, "classes"):
            return len(ds.classes)
        # torchvision DTD stores classes differently across versions
        if hasattr(ds, "_labels"):
            return len(set(ds._labels))
        return None

    n_train = _n_classes(train_ds)
    n_val   = _n_classes(val_ds)

    if n_train is not None and n_train != expected:
        raise RuntimeError(f"Training set has {n_train} classes, expected {expected}.")
    if n_val is not None and n_val != expected:
        raise RuntimeError(f"Val set has {n_val} classes, expected {expected}.")

    print(f"[dataset] Classes: {n_train or '?'} train / {n_val or '?'} val "
          f"(expected {expected})")


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mixup to a batch. Returns (mixed_images, mixed_labels_onehot).

    mixed_labels are soft targets suitable for cross-entropy:
        loss = -(mixed_labels * log_softmax(logits)).sum(dim=1).mean()

    Note: label smoothing is NOT applied here — apply it separately on the
    CE component only (see train.py) to avoid double-softening.
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    bsz = images.size(0)
    idx = torch.randperm(bsz, device=images.device)

    mixed = lam * images + (1 - lam) * images[idx]

    # One-hot encode then mix
    y_a = torch.zeros(bsz, num_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1)
    y_b = torch.zeros(bsz, num_classes, device=labels.device).scatter_(1, labels[idx].unsqueeze(1), 1)
    mixed_labels = lam * y_a + (1 - lam) * y_b

    return mixed, mixed_labels


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = DataConfig()
    loaders = get_dataloaders(cfg, stage="stage1", batch_size=8)
    imgs, labels = next(iter(loaders["train"]))
    print(f"Batch shape: {imgs.shape}, labels: {labels[:8].tolist()}")
    print("Smoke test passed.")
