# Tactus Embodied AI Assessment — Diana Vins

## Q2: DTD Texture Classifier

### Test accuracy (DTD split 1, 47 classes)

| Model | Top-1 | Top-1 + TTA | Params | Latency (CPU) |
|---|---|---|---|---|
| DINOv2 ViT-S/14 linear probe | **75.64%** | **76.81%** | 22.1M | — |
| ConvNeXt-Tiny (fine-tuned) | 70.16% | 70.64% | 31.1M | 61 ms |
| MobileNetV3-Large (fine-tuned) | 67.71% | 67.71% | 3.0M | 37 ms |
| DenseNet-201 (fine-tuned) | 54.68% | 56.60% | 18.2M | 193 ms |

TTA = 8-view test-time augmentation (4 rotations × flip). All models evaluated on the official DTD test split, never seen during training.

---

### Approach

**Dataset expansion.** DTD provides only 40 training images per class (1,880 total) — too few to fine-tune any network reliably. Before training I expanded the training set 4–6× by exploiting texture homogeneity: a "braided" texture is braided everywhere, so every non-overlapping 224×224 crop of a larger image is a valid independent training sample. This takes 1,880 images to ~9,000 without touching val or test.

**ConvNeXt-Tiny with bilinear pooling.** Standard global average pooling throws away co-activation patterns between feature channels. For textures, those co-activations *are* the signal — a "grid" texture fires both horizontal- and vertical-edge detectors simultaneously. I replaced the ConvNeXt classifier with a compact bilinear head: outer product of the pooled feature vector with itself, signed square root, then a linear layer. Training uses a 3-stage freeze/unfreeze cascade (head only → last 2 blocks → full network) with layer-wise learning rate decay and EMA weight averaging.

**MobileNetV3-Large.** Trained with the same 3-stage cascade as a deployment-oriented alternative (~10× smaller checkpoint, 37 ms CPU latency). Uses standard GAP → linear head since the 960-channel bilinear descriptor would be intractable.

**DINOv2 linear probe (benchmark).** A frozen DINOv2 ViT-S/14 backbone with a single linear head — no backbone fine-tuning — outperforms all fully fine-tuned CNNs by 5+ points. This confirms the well-known result that self-supervised ViT features are substantially more texture-aware than supervised CNN features trained on ImageNet (which are shape-biased). It also isolates representational quality from fine-tuning: DINOv2 wins on feature quality alone.

**Knowledge distillation.** The best-performing teacher (DINOv2 or ConvNeXt, selected automatically from `results.json`) distills into MobileNetV3 using Hinton soft-label KD (KL divergence at temperature T=4, mixed with cross-entropy). Soft targets carry inter-class similarity information that hard labels discard — e.g. "braided" and "woven" are genuinely similar, and the teacher's probability mass over these classes teaches the student richer structure than a one-hot label can.

---

### Surprising finding

A **frozen** DINOv2 backbone with a single linear layer beats every **fully fine-tuned** CNN by a wide margin (75.6% vs 70.2% for ConvNeXt). This is striking because fine-tuning should give CNNs a significant advantage — yet feature quality matters more than adaptation for texture recognition. DINOv2 was never trained with texture labels; its self-supervised objective on diverse imagery forced it to encode the intrinsic visual structure of patches rather than fitting ImageNet object categories. The result directly demonstrates the shape bias of supervised ImageNet CNNs (Geirhos et al., 2019) — they never learned to care about texture at the representational level, and 95 epochs of fine-tuning on 9k images cannot fully undo that.
