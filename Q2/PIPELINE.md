# DTD Texture Classifier — Pipeline Guide

## How to run

All commands are run from the `Q2/` directory.

```bash
cd Q2
bash run_all.sh
```

The script logs everything to a timestamped file (`run_all_YYYYMMDD_HHMMSS.log`) and writes all accuracy/latency/size results to `results.json` as each step completes.

### Common variants

```bash
# Skip dependency install and dataset re-expansion (fastest re-run)
bash run_all.sh --skip_install --skip_expand

# Train ConvNeXt teacher only (no MobileNetV3 or distillation)
bash run_all.sh --large_only

# Train MobileNetV3 only (no ConvNeXt)
bash run_all.sh --small_only --skip_distill

# No DINOv2 benchmark (if no internet access)
bash run_all.sh --skip_dinov2

# Use local DINOv2 weights instead of torch.hub
bash run_all.sh --dinov2_local /path/to/dinov2_vits14.pth

# Enable frequency-domain mixing during dataset expansion
bash run_all.sh --freq_mix

# Smaller batch size for limited VRAM
bash run_all.sh --batch_size 16
```

### All flags

| Flag | Default | Effect |
|---|---|---|
| `--skip_install` | off | Skip `pip install -r requirements.txt` |
| `--skip_expand` | off | Skip dataset expansion (if `data/dtd_expanded/` already exists) |
| `--skip_dinov2` | off | Skip DINOv2 linear probe |
| `--skip_distill` | off | Skip knowledge distillation |
| `--large_only` | off | Train ConvNeXt only; implies `--skip_distill` |
| `--small_only` | off | Train MobileNetV3 only |
| `--freq_mix` | off | Add FFT amplitude mixing to dataset expansion |
| `--dinov2_local PATH` | — | Load DINOv2 weights from a local `.pth` file |
| `--batch_size N` | 32 | Override batch size for all training steps |

---

## What each step does

### Step 0 — Install dependencies

```
pip install -r requirements.txt
```

Installs PyTorch, torchvision, timm (for ConvNeXt weights), and supporting libraries. Skippable with `--skip_install` if already installed.

---

### Step 1 — Expand the DTD training set

```
python expand_dataset.py
```

DTD has only 40 training images per class (1,880 total). This is far too few to fine-tune even a small network without heavy regularisation. This step multiplies the training set 4–6× before any training begins by exploiting a property unique to textures: they are **spatially homogeneous** — a "braided" texture is braided everywhere in the image, so every crop is a valid independent sample with the same label.

**Multi-crop extraction:** Each DTD image is typically 640×480 or larger. The script slides a 224×224 window across each image at stride 224 and saves every non-overlapping crop as a separate JPEG under `data/dtd_expanded/train/<class>/`. A crop is discarded only if its dimensions fall below 224px. This expands 1,880 images to roughly 8,000–11,000.

**Frequency-domain mixing (`--freq_mix`):** For each training image, a random same-class partner is chosen. Their FFT amplitude spectra are blended (up to 50% swap) while preserving the original phase. The result looks like a texture with the frequency fingerprint of one sample and the spatial structure of another — a semantically valid operation because texture identity lives in the frequency domain. This adds another ~2× expansion on top of multi-crop.

Val and test splits are never touched. They are always loaded from the original torchvision DTD to keep evaluation clean.

The script is idempotent: if `data/dtd_expanded/` already exists it exits immediately. Use `--skip_expand` on subsequent runs.

---

### Step 2 — Train ConvNeXt-Tiny (three stages)

ConvNeXt-Tiny is the accuracy-oriented backbone. Pretrained on ImageNet-22k (14 million images, 22,000 classes), it has strong generalisation priors before seeing a single DTD image.

The head is a **compact bilinear pooling head** rather than a standard linear classifier. Standard global average pooling discards co-activation patterns between feature channels. For textures, those co-activation patterns *are* the signal — a "grid" texture fires both horizontal- and vertical-edge detectors simultaneously. The head captures this by computing the outer product of the pooled feature vector with itself, applying a signed square root (which improves linear separability of texture descriptors), and passing the result through a linear layer.

Training is split into three stages to prevent noisy gradients from corrupting the pretrained features:

**Stage 2a — Head only (15 epochs):** The backbone is completely frozen. Only the bilinear head parameters receive gradient updates. This is essentially linear probing: it lets the classifier find a sensible initial position before the backbone is exposed to the task. Without this warmup, random head weights would generate large noisy gradients that propagate into and damage the pretrained features immediately.

**Stage 2b — Last 2 blocks unfrozen (40 epochs):** The final two ConvNeXt stages are unfrozen. A layer-wise learning rate schedule is applied: the head receives the base learning rate, and each earlier block receives a learning rate multiplied by 0.8 per step away from the output. This means the deeper (more generic) features are updated very conservatively while the task-specific later features adapt more aggressively.

**Stage 2c — Full network (40 epochs):** All parameters are unfrozen with very low backbone learning rates. Dropout in the classifier head is increased to 0.5 and stochastic depth remains active. An exponential moving average (EMA) of the weights is maintained throughout all stages and used for evaluation — it acts as an implicit ensemble and consistently outperforms the live weights by 0.5–1%.

---

### Step 3 — Train MobileNetV3-Large (three stages)

MobileNetV3-Large is the deployment-oriented backbone. At ~5.4M parameters it is ~5× smaller than ConvNeXt-Tiny. It uses depthwise separable convolutions and a squeeze-and-excitation attention mechanism to stay accurate despite its small size.

The head is a standard global average pooling → dropout → linear classifier. Unlike ConvNeXt, MobileNetV3 does not use a bilinear pooling head — at 960 channels, the outer product would produce a 921,600-dimensional descriptor, which is unworkable. GAP is the right choice here.

The same three-stage cascade applies with the same rationale. This standalone MobileNetV3 provides the baseline for comparison against the distilled version in Step 6.

---

### Step 4 — DINOv2 ViT-S/14 linear probe

```
python train.py --model dinov2_probe --stage 1
```

This is a benchmark, not a deployment target. DINOv2 is a self-supervised Vision Transformer trained with a DINO+iBOT objective on a large curated dataset. Because its training never used class labels, it was forced to learn representations that capture the intrinsic structure of images rather than fitting to ImageNet object categories.

Standard supervised CNNs trained on ImageNet are **shape-biased**: they recognise objects primarily by their global contours (Geirhos et al., 2019). DINOv2's features are far more texture-aware. Even a single linear layer on top of frozen DINOv2 features — with zero fine-tuning of the backbone — is competitive with fully fine-tuned CNNs on DTD.

The backbone is completely frozen. Only the 384→47 linear head is trained for 50 epochs. This isolates the quality of the feature representation from any fine-tuning advantage: if DINOv2 beats a fully fine-tuned MobileNetV3 with just a linear probe, it tells us that feature quality matters more than model capacity for this task.

The DINOv2 model requires an internet connection to download from `torch.hub`. Use `--dinov2_local` to point to a locally downloaded checkpoint if running offline.

---

### Step 5 — Evaluate all models

```
python evaluate.py --ema --latency
```

Each model is evaluated on the official DTD test split (split 1, never seen during training). Results are written to `results.json`.

**Single-crop top-1:** Standard accuracy — one centre-cropped view per test image.

**8-view TTA (test-time augmentation):** Each test image is passed through 8 augmented views: 4 rotations (0°, 90°, 180°, 270°) × original + horizontal flip. The softmax outputs are averaged before taking the argmax. This works especially well for textures because they are genuinely rotation-invariant — averaging over rotations reduces variance without introducing any label ambiguity. Expect ~1–2% improvement over single-crop.

**Latency benchmark:** 100 warm-up passes then 500 timed forward passes at batch size 1 on CPU. This is the deployment-relevant metric for edge devices, not GPU throughput.

**Model size:** Checkpoint file size in MB and parameter count.

---

### Step 6 — Knowledge distillation (ConvNeXt → MobileNetV3)

```
python distill.py --teacher checkpoints/convnext_stage3_ema_final.pth
```

The distilled MobileNetV3 is trained using the converged ConvNeXt as a teacher. Instead of learning from hard one-hot labels (class 23 = 1, everything else = 0), the student learns from the teacher's soft probability distributions.

Those soft probabilities carry information that hard labels discard: the teacher might assign 72% probability to "braided", 15% to "woven", and 5% to "knitted", reflecting the genuine visual similarity between these classes. The student learns not just the right answer but *how similar* the wrong answers are to each other. This regularises the student and typically recovers 3–5% accuracy compared to training it without distillation.

**Loss:** `0.7 × KL(student/T ∥ teacher/T) × T²  +  0.3 × CE(student, labels)` where T=4.0. The temperature T softens both distributions before computing the KL divergence, amplifying the relative differences between similar classes.

**Optional feature matching (`--feat_match`):** Adds an L2 loss between a learned projection of the student's intermediate features and the teacher's intermediate features. This transfers representational structure, not just output behaviour.

The same three-stage freeze/unfreeze cascade is applied to the student. The teacher is always frozen in eval mode.

---

## Output files

After a full run:

```
checkpoints/
├── convnext_stage1_best.pth
├── convnext_stage2_best.pth
├── convnext_stage3_best.pth
├── convnext_stage3_ema_final.pth       ← used for evaluation
├── mobilenetv3_stage3_ema_final.pth
└── mobilenetv3_distil_stage3_ema_final.pth

results.json                            ← all accuracy/latency/size numbers
run_all_YYYYMMDD_HHMMSS.log            ← full console output
```
