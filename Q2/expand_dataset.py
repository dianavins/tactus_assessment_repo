"""
Offline dataset expansion for DTD training split.

Run once before training:
    python expand_dataset.py [--freq_mix] [--data_root ./data] [--split_id 1]

Produces data/dtd_expanded/train/<class>/<image>.jpg
Val and test splits are never touched.

Two strategies (training images only):
  1. Multi-crop extraction (always): stride-224 non-overlapping crops from each image.
     DTD images are typically >=640x480; a 640x480 image yields up to 6 valid 224x224 crops.
     Each crop is a genuinely independent texture sample.

  2. Frequency-domain amplitude mixing (--freq_mix): for each training image, pick a
     random same-class partner, swap FFT amplitude spectra (keep original phase), save both.
     Semantically valid: texture identity lives in the frequency domain.
     Capped at alpha=0.5 (swap at most 50% of amplitude spectrum).
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import DTD
from tqdm import tqdm


CROP_SIZE = 224
FREQ_MIX_ALPHA = 0.5   # max fraction of amplitude spectrum swapped


def extract_crops(img: Image.Image, crop_size: int) -> list[Image.Image]:
    """
    Extract all non-overlapping crop_size x crop_size patches from img using stride=crop_size.
    Discard any patch where either dimension is < crop_size (avoids padding-dominated edges).
    """
    w, h = img.size
    crops = []
    for top in range(0, h - crop_size + 1, crop_size):
        for left in range(0, w - crop_size + 1, crop_size):
            box = (left, top, left + crop_size, top + crop_size)
            crops.append(img.crop(box))
    return crops


def freq_mix(img_a: Image.Image, img_b: Image.Image, alpha: float) -> tuple[Image.Image, Image.Image]:
    """
    Swap a fraction alpha of the FFT amplitude spectra between img_a and img_b.
    Both images are resized to CROP_SIZE x CROP_SIZE before mixing.
    Returns two new PIL images.
    """
    size = (CROP_SIZE, CROP_SIZE)
    a = np.array(img_a.resize(size).convert("RGB"), dtype=np.float32)
    b = np.array(img_b.resize(size).convert("RGB"), dtype=np.float32)

    out_a = np.zeros_like(a)
    out_b = np.zeros_like(b)

    for c in range(3):
        fa = np.fft.fft2(a[:, :, c])
        fb = np.fft.fft2(b[:, :, c])

        amp_a, phase_a = np.abs(fa), np.angle(fa)
        amp_b, phase_b = np.abs(fb), np.angle(fb)

        # Blend amplitude spectra
        mixed_amp_a = (1 - alpha) * amp_a + alpha * amp_b
        mixed_amp_b = (1 - alpha) * amp_b + alpha * amp_a

        # Reconstruct with original phase
        out_a[:, :, c] = np.fft.ifft2(mixed_amp_a * np.exp(1j * phase_a)).real
        out_b[:, :, c] = np.fft.ifft2(mixed_amp_b * np.exp(1j * phase_b)).real

    def _to_pil(arr):
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return _to_pil(out_a), _to_pil(out_b)


def expand(data_root: str, expanded_root: str, split_id: int, use_freq_mix: bool):
    out_train = Path(expanded_root) / "train"

    if out_train.exists():
        existing = sum(1 for _ in out_train.rglob("*.jpg"))
        print(f"[expand] '{out_train}' already exists with {existing} images. Skipping.")
        print("         Delete the directory to re-expand.")
        return

    print(f"[expand] Loading DTD split {split_id} from '{data_root}'...")
    dtd = DTD(root=data_root, split="train", partition=split_id, download=True)

    # Group image paths by class label
    # torchvision DTD stores (path, label) pairs in dtd._image_files / dtd._labels
    if hasattr(dtd, "_image_files"):
        pairs = list(zip(dtd._image_files, dtd._labels))
    else:
        # Fallback for older torchvision versions
        pairs = list(zip(dtd.samples, dtd.targets)) if hasattr(dtd, "samples") else []
        if not pairs:
            raise RuntimeError("Cannot read DTD image paths. Update torchvision >= 0.17.")

    class_names = dtd.classes
    by_class: dict[int, list] = defaultdict(list)
    for path, label in pairs:
        by_class[label].append(path)

    total_saved = 0
    few_crop_warnings = 0

    for label, paths in tqdm(by_class.items(), desc="Expanding classes"):
        class_name = class_names[label]
        out_dir = out_train / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, img_path in enumerate(paths):
            img = Image.open(img_path).convert("RGB")
            crops = extract_crops(img, CROP_SIZE)

            if len(crops) < 2:
                few_crop_warnings += 1

            for crop_idx, crop in enumerate(crops):
                fname = f"{img_idx:04d}_crop{crop_idx:02d}.jpg"
                crop.save(out_dir / fname, quality=95)
                total_saved += 1

        # Frequency mixing: pair each image with a random same-class partner
        if use_freq_mix:
            random.shuffle(paths)
            # Pair up; unpaired last image is skipped
            for i in range(0, len(paths) - 1, 2):
                img_a = Image.open(paths[i]).convert("RGB")
                img_b = Image.open(paths[i + 1]).convert("RGB")
                mixed_a, mixed_b = freq_mix(img_a, img_b, FREQ_MIX_ALPHA)
                mixed_a.save(out_dir / f"{i:04d}_freqmix_a.jpg", quality=95)
                mixed_b.save(out_dir / f"{i:04d}_freqmix_b.jpg", quality=95)
                total_saved += 2

    print(f"\n[expand] Done. Saved {total_saved} images to '{out_train}'.")
    if few_crop_warnings:
        print(f"[expand] WARNING: {few_crop_warnings} images produced fewer than 2 crops "
              "(image too small for stride-224 extraction).")

    # Verify
    class_dirs = [d for d in out_train.iterdir() if d.is_dir()]
    print(f"[expand] Classes in output: {len(class_dirs)} (expected 47)")
    if len(class_dirs) != 47:
        print("[expand] ERROR: class count mismatch — check DTD download.")


def main():
    parser = argparse.ArgumentParser(description="Offline DTD dataset expansion")
    parser.add_argument("--data_root",     default="./data",              help="Where DTD is downloaded")
    parser.add_argument("--expanded_root", default="./data/dtd_expanded", help="Output directory")
    parser.add_argument("--split_id",      type=int, default=1,           help="DTD split 1–10")
    parser.add_argument("--freq_mix",      action="store_true",           help="Also apply frequency-domain mixing")
    args = parser.parse_args()

    expand(
        data_root=args.data_root,
        expanded_root=args.expanded_root,
        split_id=args.split_id,
        use_freq_mix=args.freq_mix,
    )


if __name__ == "__main__":
    main()
