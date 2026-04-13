"""
Three-stage fine-tuning loop for DTD texture classifier.

Stages:
  1  Freeze backbone, train head only            (head_lr=1e-3, 15 epochs)
  2  Unfreeze last 2 blocks + layer-wise LR      (30-50 epochs)
  3  Full network, very low LR, dropout→0.5      (30-50 epochs)

Usage:
    # Stage 1 (fresh run)
    python train.py --model convnext --stage 1

    # Stage 2 (resume from stage 1 best checkpoint)
    python train.py --model convnext --stage 2 --resume checkpoints/convnext_stage1_best.pth

    # Stage 3
    python train.py --model convnext --stage 3 --resume checkpoints/convnext_stage2_best.pth

    # DINOv2 linear probe (runs stage 1 only by design)
    python train.py --model dinov2_probe --stage 1
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from tqdm import tqdm

from config import DataConfig, TrainConfig
from dataset import get_dataloaders, mixup_batch
from ema import EMAModel
from models import build_convnext, build_mobilenetv3, build_dinov2_probe, build_densenet201
from utils import (
    AverageMeter, TBLogger, get_layer_lr_groups,
    load_checkpoint, save_checkpoint, set_seed, top1_accuracy,
)


# ── Early stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when val accuracy has not improved by at least `min_delta`
    for `patience` consecutive epochs.

    Also tracks whether training loss has stagnated (std of recent losses < 1e-4)
    and includes that in the stop message for diagnosability.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_acc  = 0.0
        self.counter   = 0
        self._recent_losses: list[float] = []

    def step(self, val_acc: float, train_loss: float) -> bool:
        """Call once per epoch. Returns True when training should stop."""
        self._recent_losses.append(train_loss)
        if len(self._recent_losses) > self.patience:
            self._recent_losses.pop(0)

        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def stop_reason(self) -> str:
        msg = (f"val_acc has not improved by >{self.min_delta:.0e} "
               f"for {self.patience} consecutive epochs (best={self.best_acc:.4f})")
        if len(self._recent_losses) >= 3:
            import statistics
            std = statistics.stdev(self._recent_losses[-self.patience:])
            if std < 1e-4:
                msg += f"; train loss also stagnant (std={std:.2e})"
        return msg


# ── Freeze / unfreeze helpers ──────────────────────────────────────────────

def freeze_backbone(model: nn.Module, model_name: str):
    """Freeze all parameters except the head."""
    if model_name == "convnext":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif model_name == "mobilenetv3":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name == "dinov2_probe":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif model_name == "densenet201":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    _print_trainable(model)


def unfreeze_last_n_blocks(model: nn.Module, model_name: str, n: int = 2):
    """Unfreeze the last n ConvNeXt stages or last n MobileNetV3 feature groups."""
    if model_name == "convnext":
        stages = list(model.backbone.stages)
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
        # Also unfreeze the final norm layer
        if hasattr(model.backbone, "norm_pre"):
            for p in model.backbone.norm_pre.parameters():
                p.requires_grad = True
    elif model_name == "mobilenetv3":
        features = list(model.features.children())
        for layer in features[-n * 3:]:    # approx last 6 layers
            for p in layer.parameters():
                p.requires_grad = True
    elif model_name == "densenet201":
        features = list(model.features.children())
        for layer in features[-n * 2:]:    # n=2 → denseblock3,transition3,denseblock4,norm5
            for p in layer.parameters():
                p.requires_grad = True
    _print_trainable(model)


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True
    _print_trainable(model)


def _print_trainable(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")


# ── Optimiser / scheduler builders ────────────────────────────────────────

def build_optimizer_stage1(model: nn.Module, cfg: TrainConfig):
    """Stage 1: head parameters only, uniform LR."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)


def build_optimizer_stage2(model: nn.Module, stage_groups: list, cfg: TrainConfig):
    """Stage 2: layer-wise LR decay, backbone LR < head LR."""
    return torch.optim.AdamW(
        get_layer_lr_groups(stage_groups, base_lr=cfg.head_lr, decay=cfg.lr_layer_decay),
        weight_decay=cfg.weight_decay,
    )


def build_optimizer_stage3(model: nn.Module, stage_groups: list, cfg: TrainConfig):
    """Stage 3: very low LR throughout."""
    groups = get_layer_lr_groups(
        stage_groups, base_lr=cfg.stage3_backbone_lr, decay=cfg.lr_layer_decay
    )
    # Override head group with a slightly higher LR
    if groups:
        groups[-1]["lr"] = cfg.head_lr * 0.1
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg: TrainConfig, num_epochs: int, steps_per_epoch: int):
    """Linear warmup for warmup_epochs then cosine annealing to 0.

    Uses a single LambdaLR instead of SequentialLR to avoid PyTorch's
    spurious 'scheduler.step() before optimizer.step()' warning, which
    SequentialLR triggers by calling step() on sub-schedulers during __init__.
    """
    import math
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps  = num_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear ramp from 1% to 100% of base LR
            return 0.01 + 0.99 * current_step / max(1, warmup_steps)
        # Cosine decay from 100% to 0%
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
            category=UserWarning,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ── Training / validation loops ────────────────────────────────────────────

def train_epoch(
    model, loader, criterion, optimizer, scheduler,
    scaler, ema, cfg, device, desc="Train"
) -> dict:
    model.train()
    loss_m = AverageMeter("loss")
    acc_m  = AverageMeter("acc")

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg.use_mixup and cfg.mixup_alpha > 0:
            images, soft_labels = mixup_batch(
                images, labels, cfg.mixup_alpha, cfg.data.num_classes
            )
            with autocast("cuda", enabled=cfg.amp):
                logits = model(images)
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                loss = -(soft_labels * log_prob).sum(dim=1).mean()
        else:
            with autocast("cuda", enabled=cfg.amp):
                logits = model(images)
                loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        loss_m.update(loss.item(), images.size(0))
        if not cfg.use_mixup:
            acc_m.update(top1_accuracy(logits.detach(), labels), images.size(0))

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.3f}")

    return {"loss": loss_m.avg, "acc": acc_m.avg}


@torch.no_grad()
def validate(model, loader, criterion, ema, device) -> dict:
    loss_m = AverageMeter("val_loss")
    acc_m  = AverageMeter("val_acc")

    with ema.apply_shadow(model):
        model.eval()
        pbar = tqdm(loader, desc="Val", leave=False, unit="batch", dynamic_ncols=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            loss_m.update(loss.item(), images.size(0))
            acc_m.update(top1_accuracy(logits, labels), images.size(0))
            pbar.set_postfix(acc=f"{acc_m.avg:.3f}", loss=f"{loss_m.avg:.4f}")

    return {"loss": loss_m.avg, "acc": acc_m.avg}


# ── Stage runner ───────────────────────────────────────────────────────────

def run_stage(
    stage: int,
    model: nn.Module,
    stage_groups: list,
    model_name: str,
    cfg: TrainConfig,
    data_cfg: DataConfig,
    device: torch.device,
):
    # Determine stage-specific parameters
    aug_stage_map = {1: "stage1", 2: "stage2", 3: "stage3"}
    if model_name == "dinov2_probe":
        aug_stage = "dinov2_probe"
    else:
        aug_stage = aug_stage_map[stage]

    num_epochs = {1: cfg.stage1_epochs, 2: cfg.stage2_epochs, 3: cfg.stage3_epochs}[stage]
    ckpt_prefix = f"{cfg.checkpoint_dir}/{model_name}_stage{stage}"

    # Freeze / unfreeze
    if stage == 1:
        freeze_backbone(model, model_name)
        optimizer = build_optimizer_stage1(model, cfg)
    elif stage == 2:
        unfreeze_last_n_blocks(model, model_name, n=2)
        optimizer = build_optimizer_stage2(model, stage_groups, cfg)
    else:  # stage 3
        unfreeze_all(model)
        optimizer = build_optimizer_stage3(model, stage_groups, cfg)
        # Increase dropout for stage 3
        if hasattr(model, "head") and hasattr(model.head, "set_dropout"):
            model.head.set_dropout(0.5)

    # Data
    loaders = get_dataloaders(data_cfg, stage=aug_stage, batch_size=cfg.batch_size,
                              mixup_alpha=cfg.mixup_alpha)

    # Attach data config to cfg for mixup_batch access in train_epoch
    cfg.data = data_cfg

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg, num_epochs, len(loaders["train"]))

    # Loss (label smoothing for non-Mixup batches)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing).to(device)

    # EMA + AMP
    ema    = EMAModel(model, decay=cfg.ema_decay)
    scaler = GradScaler("cuda", enabled=cfg.amp)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if cfg.resume:
        start_epoch, best_val_acc = load_checkpoint(
            cfg.resume, model, optimizer, scaler, ema
        )
        start_epoch += 1

    # Logger
    logger = TBLogger(log_dir=f"{cfg.log_dir}/{model_name}_stage{stage}")

    early_stop = EarlyStopping(
        patience=cfg.early_stop_patience,
        min_delta=cfg.early_stop_min_delta,
    )

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Stage {stage} | {model_name} | {num_epochs} epochs | device: {device}")
    tqdm.write(f"Early stopping: patience={cfg.early_stop_patience}, "
               f"min_delta={cfg.early_stop_min_delta:.0e}")
    tqdm.write(f"{'='*60}")

    global_step = start_epoch * len(loaders["train"])
    epoch_bar = tqdm(
        range(start_epoch, num_epochs),
        desc=f"{model_name} S{stage}",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        train_stats = train_epoch(
            model, loaders["train"], criterion, optimizer, scheduler,
            scaler, ema, cfg, device,
            desc=f"E{epoch+1:02d}/{num_epochs} Train",
        )
        val_stats = validate(model, loaders["val"], criterion, ema, device)

        val_acc = val_stats["acc"]
        logger.scalar("train/loss", train_stats["loss"], epoch)
        logger.scalar("val/acc",    val_acc,             epoch)
        logger.scalar("val/loss",   val_stats["loss"],   epoch)

        epoch_bar.set_postfix(
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}",
            tr_loss=f"{train_stats['loss']:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_loss={val_stats['loss']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                f"{ckpt_prefix}_best.pth",
                epoch, model, optimizer, scaler, ema, val_acc,
                extra={"stage": stage, "model_name": model_name},
            )
            tqdm.write(f"  ** New best: {best_val_acc:.4f}  (saved)")

        if early_stop.step(val_acc, train_stats["loss"]):
            tqdm.write(f"\n[early stop] Triggered at epoch {epoch+1}: {early_stop.stop_reason()}")
            break

        global_step += len(loaders["train"])

    # Save EMA weights separately for downstream use
    ema_path = f"{ckpt_prefix}_ema_final.pth"
    save_checkpoint(ema_path, num_epochs - 1, model, optimizer, scaler, ema, best_val_acc,
                    extra={"stage": stage, "model_name": model_name, "is_ema_final": True})
    print(f"\nStage {stage} complete. Best val acc: {best_val_acc:.4f}")
    print(f"EMA final saved to: {ema_path}")

    logger.close()
    return best_val_acc


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DTD texture classifier training")
    p.add_argument("--model",   default="convnext",
                   choices=["convnext", "mobilenetv3", "dinov2_probe", "densenet201"],
                   help="Model architecture")
    p.add_argument("--stage",   type=int, default=1, choices=[1, 2, 3],
                   help="Training stage (1=head only, 2=last 2 blocks, 3=full)")
    p.add_argument("--resume",  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--data_root",      default="./data")
    p.add_argument("--expanded_root",  default="./data/dtd_expanded")
    p.add_argument("--no_expanded",    action="store_true",
                   help="Disable offline-expanded dataset; use original DTD only")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--log_dir",        default="./runs")
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--bilinear_dim",   type=int, default=256,
                   help="Channel reduction dim for bilinear head (use 128 if OOM)")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--dinov2_local",   default=None,
                   help="Local path to dinov2_vits14.pth (if torch.hub unavailable)")
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early stopping: epochs without val-acc improvement (default 5)")
    p.add_argument("--min_delta",   type=float, default=0.1,
                   help="Early stopping: minimum val-acc gain that counts (default 0.1)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    data_cfg = DataConfig(
        data_root=args.data_root,
        expanded_root=args.expanded_root,
        use_expanded=not args.no_expanded,
    )

    cfg = TrainConfig(
        model=args.model,
        stage=args.stage,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        bilinear_dim=args.bilinear_dim,
        seed=args.seed,
        early_stop_patience=args.patience,
        early_stop_min_delta=args.min_delta,
    )

    # Build model
    if args.model == "convnext":
        model, stage_groups = build_convnext(cfg)
    elif args.model == "mobilenetv3":
        model, stage_groups = build_mobilenetv3(cfg)
    elif args.model == "dinov2_probe":
        model, stage_groups = build_dinov2_probe(cfg, local_path=args.dinov2_local)
    elif args.model == "densenet201":
        model, stage_groups = build_densenet201(cfg)

    model = model.to(device)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    run_stage(
        stage=args.stage,
        model=model,
        stage_groups=stage_groups,
        model_name=args.model,
        cfg=cfg,
        data_cfg=data_cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
