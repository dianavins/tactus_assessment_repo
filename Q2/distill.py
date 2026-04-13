"""
Knowledge distillation: ConvNeXt-Tiny teacher → MobileNetV3-Large student.

Loss = 0.7 * KL_div(student/T, teacher/T) * T² + 0.3 * CE(student, labels)

Temperature T=4.0 produces soft probability distributions that encode
which texture classes look similar to each other — information that
one-hot labels discard.

Optional --feat_match: adds an L2 loss between a learned projection of the
student's penultimate features and the teacher's penultimate features.
This transfers intermediate representations, not just output logits.

Usage:
    # Stage 1: freeze student backbone, distil into head only
    python distill.py --stage 1 --teacher checkpoints/convnext_stage3_ema_final.pth

    # Stage 2: unfreeze last 2 student blocks
    python distill.py --stage 2 --resume checkpoints/mobilenetv3_distil_stage1_best.pth \\
                      --teacher checkpoints/convnext_stage3_ema_final.pth

    # With optional feature matching
    python distill.py --stage 2 --feat_match \\
                      --teacher checkpoints/convnext_stage3_ema_final.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from tqdm import tqdm

from config import DataConfig, TrainConfig
from dataset import get_dataloaders, mixup_batch
from ema import EMAModel
from models import build_convnext, build_mobilenetv3, build_dinov2_probe, build_densenet201
from utils import (
    AverageMeter, TBLogger, get_layer_lr_groups,
    load_checkpoint, load_ema_weights, save_checkpoint,
    set_seed, top1_accuracy,
)
from train import (
    freeze_backbone, unfreeze_last_n_blocks, unfreeze_all,
    build_optimizer_stage1, build_optimizer_stage2, build_optimizer_stage3,
    build_scheduler, validate,
)


# ── Feature projection (optional feat_match) ──────────────────────────────

TEACHER_FEAT_DIM = {
    "convnext":     768,
    "dinov2_probe": 384,
    "densenet201":  1920,
}


class FeatureProjection(nn.Module):
    """
    Projects student's penultimate features (960-d from MobileNetV3 avgpool)
    to match teacher's penultimate features. teacher_dim varies by architecture:
      convnext: 768  |  dinov2_probe: 384  |  densenet201: 1920
    Used only when --feat_match is enabled.
    """
    def __init__(self, student_dim: int = 960, teacher_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=1)


# ── Teacher: extract penultimate features ─────────────────────────────────

def get_teacher_features(teacher, teacher_model: str, x, device):
    """Extract (logits, penultimate_feat_vec) from any supported teacher architecture."""
    x = x.to(device)
    with torch.no_grad():
        if teacher_model == "convnext":
            # forward_features() → (B, 768, 7, 7), channel-first
            features = teacher.backbone.forward_features(x)
            feat_vec = features.mean(dim=[2, 3])       # (B, 768)
            logits   = teacher.head(features)
        elif teacher_model == "dinov2_probe":
            feat_vec = teacher.backbone(x)             # (B, 384) CLS token
            logits   = teacher.head(feat_vec)
        elif teacher_model == "densenet201":
            features = teacher.features(x)             # (B, 1920, 7, 7)
            feat_vec = F.relu(features, inplace=False).mean(dim=[2, 3])  # (B, 1920)
            logits   = teacher.classifier(feat_vec)
        else:
            raise ValueError(f"Unsupported teacher model: {teacher_model}")
    return logits, feat_vec


def get_student_features(student, x, device):
    """Extract (logits, penultimate_features) from MobileNetV3 student."""
    x = x.to(device)
    # Intercept after features + avgpool, before classifier
    feat = student.features(x)           # (B, 960, 7, 7) approx
    feat = student.avgpool(feat)         # (B, 960, 1, 1)
    feat = feat.flatten(1)              # (B, 960)
    logits = student.classifier(feat)   # (B, 47)
    return logits, feat


# ── Distillation loss ──────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Hinton et al. KD loss.

    loss = alpha * KL(student/T || teacher/T) * T²
         + (1-alpha) * CE(student, labels, label_smoothing)

    Label smoothing is applied only to the CE component to avoid
    double-softening when Mixup is also active.
    """
    T = temperature

    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits  / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss


# ── Training loop ──────────────────────────────────────────────────────────

def train_epoch_distill(
    student, teacher, teacher_model, loader,
    optimizer, scheduler, scaler, ema,
    projector, cfg, data_cfg, device,
    temperature=4.0, alpha=0.7, feat_match_weight=0.1,
    use_feat_match=False,
) -> dict:
    student.train()
    teacher.eval()

    loss_m = AverageMeter("loss")

    pbar = tqdm(loader, desc="Distil", leave=False, unit="batch", dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=cfg.amp):
            teacher_logits, teacher_feat = get_teacher_features(teacher, teacher_model, images, device)

            if use_feat_match:
                student_logits, student_feat = get_student_features(student, images, device)
                proj_feat = projector(student_feat)
                feat_loss = F.mse_loss(proj_feat, F.normalize(teacher_feat.float(), dim=1))
            else:
                student_logits, _ = get_student_features(student, images, device)
                feat_loss = torch.tensor(0.0, device=device)

            kd_ce_loss = distillation_loss(
                student_logits, teacher_logits.float(), labels,
                temperature=temperature, alpha=alpha,
                label_smoothing=cfg.label_smoothing,
            )
            loss = kd_ce_loss + feat_match_weight * feat_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(student)

        loss_m.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

    return {"loss": loss_m.avg}


# ── Stage runner ───────────────────────────────────────────────────────────

def run_distil_stage(
    stage, student, student_groups, teacher, teacher_model,
    cfg, data_cfg, device,
    use_feat_match=False, temperature=4.0, alpha=0.7,
):
    num_epochs = {1: cfg.stage1_epochs, 2: cfg.stage2_epochs, 3: cfg.stage3_epochs}[stage]
    ckpt_prefix = f"{cfg.checkpoint_dir}/mobilenetv3_distil_stage{stage}"

    # Freeze/unfreeze student
    if stage == 1:
        freeze_backbone(student, "mobilenetv3")
        optimizer = build_optimizer_stage1(student, cfg)
    elif stage == 2:
        unfreeze_last_n_blocks(student, "mobilenetv3", n=2)
        optimizer = build_optimizer_stage2(student, student_groups, cfg)
    else:
        unfreeze_all(student)
        optimizer = build_optimizer_stage3(student, student_groups, cfg)

    loaders   = get_dataloaders(data_cfg, stage=f"stage{stage}", batch_size=cfg.batch_size)
    scheduler = build_scheduler(optimizer, cfg, num_epochs, len(loaders["train"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing).to(device)
    ema       = EMAModel(student, decay=cfg.ema_decay)
    scaler    = GradScaler("cuda", enabled=cfg.amp)
    cfg.data  = data_cfg

    projector = None
    if use_feat_match:
        teacher_dim = TEACHER_FEAT_DIM[teacher_model]
        projector = FeatureProjection(student_dim=960, teacher_dim=teacher_dim).to(device)
        # Add projector params to optimizer
        optimizer.add_param_group({"params": projector.parameters(), "lr": cfg.head_lr})

    start_epoch  = 0
    best_val_acc = 0.0
    if cfg.resume:
        start_epoch, best_val_acc = load_checkpoint(cfg.resume, student, optimizer, scaler, ema)
        start_epoch += 1

    logger = TBLogger(log_dir=f"{cfg.log_dir}/mobilenetv3_distil_stage{stage}")

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Distil Stage {stage} | {num_epochs} epochs | T={temperature} | alpha={alpha}")
    tqdm.write(f"{'='*60}")

    epoch_bar = tqdm(
        range(start_epoch, num_epochs),
        desc=f"Distil S{stage}",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        train_stats = train_epoch_distill(
            student, teacher, teacher_model, loaders["train"],
            optimizer, scheduler, scaler, ema,
            projector, cfg, data_cfg, device,
            temperature=temperature, alpha=alpha,
            use_feat_match=use_feat_match,
        )
        val_stats = validate(student, loaders["val"], criterion, ema, device)

        val_acc = val_stats["acc"]
        logger.scalar("train/loss", train_stats["loss"], epoch)
        logger.scalar("val/acc",    val_acc,             epoch)

        epoch_bar.set_postfix(
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}",
            tr_loss=f"{train_stats['loss']:.4f}",
        )
        tqdm.write(f"Epoch {epoch+1:3d}/{num_epochs} | "
                   f"train_loss={train_stats['loss']:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(f"{ckpt_prefix}_best.pth", epoch, student,
                            optimizer, scaler, ema, val_acc)
            tqdm.write(f"  ** New best: {best_val_acc:.4f}")

    ema_path = f"{ckpt_prefix}_ema_final.pth"
    save_checkpoint(ema_path, num_epochs - 1, student, optimizer, scaler, ema, best_val_acc)
    tqdm.write(f"\nDistil Stage {stage} complete. Best val acc: {best_val_acc:.4f}")

    logger.close()
    return best_val_acc


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Knowledge distillation: ConvNeXt → MobileNetV3")
    p.add_argument("--teacher",       required=True, help="Path to teacher checkpoint")
    p.add_argument("--teacher_model", required=True,
                   choices=["convnext", "dinov2_probe", "densenet201"],
                   help="Architecture of the teacher model")
    p.add_argument("--stage",      type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--resume",     default=None, help="Student checkpoint to resume from")
    p.add_argument("--feat_match", action="store_true",
                   help="Add intermediate feature matching loss (weight=0.1)")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha",       type=float, default=0.7,
                   help="Weight of KD loss (1-alpha = CE weight)")
    p.add_argument("--data_root",      default="./data")
    p.add_argument("--expanded_root",  default="./data/dtd_expanded")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--log_dir",        default="./runs")
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[distill] Device: {device}")

    data_cfg = DataConfig(data_root=args.data_root, expanded_root=args.expanded_root)
    cfg      = TrainConfig(model="mobilenetv3", resume=args.resume,
                           checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
                           batch_size=args.batch_size, seed=args.seed)

    # Load teacher (architecture determined by --teacher_model)
    teacher_cfg = TrainConfig(model=args.teacher_model, pretrained=False)
    if args.teacher_model == "convnext":
        teacher, _ = build_convnext(teacher_cfg)
    elif args.teacher_model == "dinov2_probe":
        teacher, _ = build_dinov2_probe(teacher_cfg)
    elif args.teacher_model == "densenet201":
        teacher, _ = build_densenet201(teacher_cfg)

    load_ema_weights(args.teacher, teacher)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[distill] Teacher ({args.teacher_model}) loaded and frozen from '{args.teacher}'")

    # Build student
    student, student_groups = build_mobilenetv3(cfg)
    student = student.to(device)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    run_distil_stage(
        stage=args.stage,
        student=student,
        student_groups=student_groups,
        teacher=teacher,
        teacher_model=args.teacher_model,
        cfg=cfg,
        data_cfg=data_cfg,
        device=device,
        use_feat_match=args.feat_match,
        temperature=args.temperature,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
