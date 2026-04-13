"""
Evaluation: top-1 accuracy, 8-view TTA, latency benchmark, model size.

Usage:
    # Evaluate EMA weights on test split
    python evaluate.py --checkpoint checkpoints/convnext_stage3_ema_final.pth --model convnext --ema

    # Evaluate without TTA
    python evaluate.py --checkpoint checkpoints/convnext_stage3_best.pth --model convnext --no_tta
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from config import DataConfig, TrainConfig
from dataset import get_dataloaders
from models import build_convnext, build_mobilenetv3, build_dinov2_probe, build_densenet201
from utils import AverageMeter, load_ema_weights, set_seed, top1_accuracy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── TTA: 8 views = 4 rotations × 2 flips ──────────────────────────────────

def tta_predict(model: torch.nn.Module, images: torch.Tensor, device) -> torch.Tensor:
    """
    Average softmax predictions over 8 augmented views per image.

    Views: 4 rotations (0, 90, 180, 270 degrees) × original + horizontal flip.
    Returns averaged softmax probabilities: (B, num_classes).

    Why 8-view TTA works for textures:
        Textures are rotation-invariant — "braided" looks the same at any angle.
        Averaging predictions over rotations reduces variance and consistently
        improves accuracy by ~1–2% on DTD.
    """
    angles  = [0, 90, 180, 270]
    probs_sum = None

    for angle in angles:
        for flip in [False, True]:
            view = images.clone()
            if angle != 0:
                view = torch.rot90(view, k=angle // 90, dims=[2, 3])
            if flip:
                view = torch.flip(view, dims=[3])
            with torch.no_grad():
                logits = model(view.to(device))
            probs = F.softmax(logits, dim=1).cpu()
            probs_sum = probs if probs_sum is None else probs_sum + probs

    return probs_sum / 8.0


# ── Latency benchmark ──────────────────────────────────────────────────────

def benchmark_latency(model: torch.nn.Module, image_size: int = 224,
                       warmup: int = 100, trials: int = 500) -> dict:
    """
    Measure mean and std of single-image inference latency on CPU (batch_size=1).
    Returns {'mean_ms': float, 'std_ms': float}.
    """
    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 3, image_size, image_size)

    times = []
    with torch.no_grad():
        for _ in tqdm(range(warmup), desc="Latency warmup", unit="step",
                      dynamic_ncols=True, leave=False):
            _ = model_cpu(dummy)
        for _ in tqdm(range(trials), desc="Latency trials", unit="step",
                      dynamic_ncols=True, leave=False):
            t0 = time.perf_counter()
            _ = model_cpu(dummy)
            times.append((time.perf_counter() - t0) * 1000)  # ms

    import statistics
    return {
        "mean_ms": round(statistics.mean(times), 2),
        "std_ms":  round(statistics.stdev(times), 2),
    }


# ── Model size ─────────────────────────────────────────────────────────────

def model_stats(model: torch.nn.Module, checkpoint_path: str) -> dict:
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2) if checkpoint_path else None
    return {
        "param_count": param_count,
        "param_count_M": round(param_count / 1e6, 2),
        "checkpoint_size_mb": round(size_mb, 2) if size_mb else None,
    }


# ── Main evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    data_cfg: DataConfig,
    device: torch.device,
    use_tta: bool = True,
    batch_size: int = 32,
) -> dict:
    loaders = get_dataloaders(data_cfg, stage="eval", batch_size=batch_size)
    test_loader = loaders["test"]

    model.eval()

    # Single-crop accuracy
    acc_m = AverageMeter("top1")
    pbar = tqdm(test_loader, desc="FP32 eval", unit="batch",
                dynamic_ncols=True, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
        acc_m.update(top1_accuracy(logits, labels), images.size(0))
        pbar.set_postfix(top1=f"{acc_m.avg:.4f}")
    top1 = acc_m.avg

    # TTA accuracy
    top1_tta = None
    if use_tta:
        correct = 0
        total   = 0
        pbar = tqdm(test_loader, desc="TTA eval ", unit="batch",
                    dynamic_ncols=True, leave=False)
        for images, labels in pbar:
            probs = tta_predict(model, images, device)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            pbar.set_postfix(top1=f"{correct/total:.4f}")
        top1_tta = correct / total

    return {"top1_fp32": round(top1, 4), "top1_tta": round(top1_tta, 4) if top1_tta else None}


def main():
    p = argparse.ArgumentParser(description="DTD texture classifier evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--model",      required=True,
                   choices=["convnext", "mobilenetv3", "dinov2_probe", "densenet201"])
    p.add_argument("--ema",        action="store_true",
                   help="Load EMA shadow weights from checkpoint")
    p.add_argument("--no_tta",     action="store_true", help="Skip TTA")
    p.add_argument("--latency",    action="store_true", help="Run CPU latency benchmark")
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output",     default="results.json", help="Where to write results")
    p.add_argument("--dinov2_local", default=None)
    args = p.parse_args()

    set_seed(42)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = DataConfig(data_root=args.data_root, use_expanded=False)
    cfg      = TrainConfig(model=args.model, pretrained=False)

    # Build model (no pretrained weights — we load from checkpoint)
    if args.model == "convnext":
        model, _ = build_convnext(cfg)
    elif args.model == "mobilenetv3":
        model, _ = build_mobilenetv3(cfg)
    elif args.model == "dinov2_probe":
        model, _ = build_dinov2_probe(cfg, local_path=args.dinov2_local)
    elif args.model == "densenet201":
        model, _ = build_densenet201(cfg)

    # Load weights
    if args.ema:
        load_ema_weights(args.checkpoint, model)
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])

    model = model.to(device)

    # Accuracy
    results = run_evaluation(
        model, data_cfg, device,
        use_tta=not args.no_tta,
        batch_size=args.batch_size,
    )

    # Model stats
    stats = model_stats(model, args.checkpoint)
    results.update(stats)

    results["fp32_latency_ms"] = None

    if args.latency:
        print("[eval] Running CPU latency benchmark (batch_size=1)...")
        lat = benchmark_latency(model, image_size=224)
        results["fp32_latency_ms"] = lat["mean_ms"]
        results["fp32_latency_std_ms"] = lat["std_ms"]
        print(f"[eval] FP32 latency: {lat['mean_ms']:.1f} ± {lat['std_ms']:.1f} ms")

    # Print
    print(f"\n{'='*50}")
    print(f"Model:          {args.model}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Top-1 (FP32):   {results['top1_fp32']:.4f}")
    if results['top1_tta'] is not None:
        print(f"Top-1 (TTA):    {results['top1_tta']:.4f}")
    print(f"Params:         {results['param_count_M']:.2f}M")
    if results['checkpoint_size_mb']:
        print(f"Checkpoint:     {results['checkpoint_size_mb']:.1f} MB")
    print(f"{'='*50}")

    # Write results.json (merge with existing if present)
    existing = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
    existing[args.model] = results
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[eval] Results written to '{args.output}'")


if __name__ == "__main__":
    main()
