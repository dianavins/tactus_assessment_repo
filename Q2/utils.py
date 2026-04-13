"""
Shared utilities: checkpointing, seeding, metrics, logging, layer-wise LR groups.
"""

import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Checkpointing ──────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    ema,              # EMAModel instance or None
    metric: float,
    extra: dict = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict() if scaler is not None else None,
        "ema":       ema.shadow if ema is not None else None,
        "metric":    metric,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scaler=None, ema=None):
    """
    Load checkpoint. Returns (epoch, metric).
    Model weights and EMA are always restored. Optimizer/scaler are skipped
    silently when the saved state is incompatible (e.g. resuming across stages
    where the number of parameter groups changes).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        saved_groups  = len(ckpt["optimizer"]["param_groups"])
        current_groups = len(optimizer.param_groups)
        if saved_groups == current_groups:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(
                f"[ckpt] Skipping optimizer state: saved {saved_groups} param group(s), "
                f"current optimizer has {current_groups}. "
                f"This is normal when resuming across training stages."
            )
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema"):
        device = next(model.parameters()).device
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
    print(f"[ckpt] Loaded epoch {ckpt['epoch']} | metric {ckpt['metric']:.4f} from '{path}'")
    return ckpt["epoch"], ckpt["metric"]


def load_ema_weights(path: str, model: nn.Module):
    """Load EMA shadow weights into model (used for evaluation).

    Two-step load to handle two edge cases:
      1. Buffers (e.g. BatchNorm running_mean/var) are never tracked by EMA
         (named_parameters() skips them). Loading full model state first fills them.
      2. When the same sub-module is reachable via two attribute paths
         (e.g. ConvNeXtWithBilinearHead where .backbone.head and .head are the
         same object), state_dict() stores keys for both paths but named_parameters()
         deduplicates, so the EMA shadow only has one path. strict=False on the
         second load silently skips the already-correct duplicate.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    shadow = ckpt.get("ema")
    if shadow is None:
        print("[ckpt] WARNING: no EMA weights found; loading live model weights.")
        model.load_state_dict(ckpt["model"])
    else:
        # Step 1: full checkpoint → correct buffers + all module paths
        model.load_state_dict(ckpt["model"])
        # Step 2: overlay EMA-tracked params (strict=False skips missing duplicate paths)
        model.load_state_dict(shadow, strict=False)
    print(f"[ckpt] EMA weights loaded from '{path}' (epoch {ckpt['epoch']})")


# ── Metrics ────────────────────────────────────────────────────────────────

class AverageMeter:
    """Running mean and count."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a float in [0, 1]."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ── Layer-wise LR groups ───────────────────────────────────────────────────

def get_layer_lr_groups(
    stage_groups: list[nn.Module],
    base_lr: float,
    decay: float = 0.8,
) -> list[dict]:
    """
    Assign learning rates to parameter groups using layer-wise LR decay.

    stage_groups is ordered input → output (index 0 = earliest / deepest layers).
    The output group (last element) gets base_lr; each preceding group is scaled
    by decay^k where k is the distance from the output.

    Example with decay=0.8, base_lr=1e-3 and 6 groups:
        group[5] (head)    → 1e-3
        group[4] (stage3)  → 8e-4
        group[3] (stage2)  → 6.4e-4
        ...

    Returns a list of {'params': [...], 'lr': float} dicts for AdamW.
    """
    n = len(stage_groups)
    param_groups = []
    for i, group in enumerate(stage_groups):
        depth_from_output = n - 1 - i
        lr = base_lr * (decay ** depth_from_output)
        params = [p for p in group.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr})
    return param_groups


# ── TensorBoard logger ─────────────────────────────────────────────────────

class TBLogger:
    """Thin wrapper around SummaryWriter; no-ops if tensorboard not installed."""
    def __init__(self, log_dir: str, comment: str = ""):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir, comment=comment)
            self._active = True
        except ImportError:
            self._active = False
            print("[logger] tensorboard not found; logging disabled.")

    def scalar(self, tag: str, value: float, step: int):
        if self._active:
            self._writer.add_scalar(tag, value, step)

    def close(self):
        if self._active:
            self._writer.close()


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Checkpoint round-trip
    import tempfile
    model = torch.nn.Linear(10, 5)
    opt   = torch.optim.Adam(model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name

    save_checkpoint(path, epoch=3, model=model, optimizer=opt,
                    scaler=None, ema=None, metric=0.72)

    model2 = torch.nn.Linear(10, 5)
    epoch, metric = load_checkpoint(path, model2, optimizer=opt)
    assert epoch == 3 and abs(metric - 0.72) < 1e-6, "Checkpoint round-trip failed"
    print("Checkpoint round-trip ✓")

    # AverageMeter
    meter = AverageMeter("loss")
    for v in [1.0, 2.0, 3.0]:
        meter.update(v)
    assert abs(meter.avg - 2.0) < 1e-6
    print("AverageMeter ✓")

    # Layer LR groups
    groups = [torch.nn.Linear(4, 4) for _ in range(4)]
    lr_groups = get_layer_lr_groups(groups, base_lr=1e-3, decay=0.8)
    lrs = [g["lr"] for g in lr_groups]
    assert lrs[-1] == 1e-3, f"Head LR should be base_lr, got {lrs[-1]}"
    assert lrs[0] < lrs[-1], "Early layer should have lower LR"
    print(f"Layer LRs: {[f'{lr:.2e}' for lr in lrs]}  ✓")

    os.unlink(path)
    print("All utils smoke tests passed.")
