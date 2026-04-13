from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    data_root: str = "./data"
    expanded_root: str = "./data/dtd_expanded"
    use_expanded: bool = True          # use offline-expanded training set if available
    use_freq_mix: bool = False         # enable frequency-domain amplitude mixing in expand_dataset.py
    split_id: int = 1                  # DTD official split 1–10; 1 is most cited in literature
    image_size: int = 224
    num_classes: int = 47
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    # Model
    model: str = "convnext"            # convnext | mobilenetv3 | dinov2_probe
    pretrained: bool = True

    # Batch / epochs
    batch_size: int = 32
    stage1_epochs: int = 15           # head only
    stage2_epochs: int = 40           # last 2 blocks
    stage3_epochs: int = 40           # full network

    # Learning rates
    head_lr: float = 1e-3
    stage2_backbone_lr: float = 1e-4
    stage3_backbone_lr: float = 1e-5
    lr_layer_decay: float = 0.8       # per-block decay factor from output toward input

    # Schedule
    warmup_epochs: int = 5

    # Regularisation
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    head_dropout: float = 0.3         # increased to 0.5 at Stage 3
    drop_path_rate: float = 0.1       # stochastic depth for ConvNeXt

    # Mixup
    mixup_alpha: float = 0.2
    use_mixup: bool = True            # disabled automatically during QAT

    # EMA
    ema_decay: float = 0.9998

    # Bilinear head channel reduction
    bilinear_dim: int = 256           # reduced to 128 if OOM

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume: Optional[str] = None
    stage: int = 1                    # 1 | 2 | 3

    # Early stopping
    early_stop_patience: int = 5      # epochs without improvement before stopping
    early_stop_min_delta: float = 0.1   # minimum val-acc gain that counts as improvement

    # Misc
    seed: int = 42
    log_dir: str = "./runs"
    amp: bool = True                  # automatic mixed precision


@dataclass
class QuantConfig:
    fp32_checkpoint: str = "./checkpoints/ema_final.pth"
    output_dir: str = "./checkpoints"
    backend: str = "x86"              # x86 | qnnpack
    calibration_samples: int = 200

    # Decision thresholds
    ptq_drop_threshold: float = 1.5   # if PTQ drop <= this, ship PTQ
    severe_drop_threshold: float = 5.0  # if PTQ drop > this, do sensitivity analysis

    # QAT
    qat_epochs: int = 15
    qat_lr_fraction: float = 0.01     # QAT LR = this * Stage 3 backbone LR

    # Acceptance targets (documented in evaluate.py output)
    target_size_mb: float = 15.0      # ConvNeXt INT8
    target_latency_ms: float = 50.0   # batch_size=1 on CPU
