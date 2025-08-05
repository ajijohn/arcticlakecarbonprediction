#!/usr/bin/env python3
"""
Simple script to delete image-mask pairs where mask is 99% positive
"""

import cv2
import numpy as np
from pathlib import Path

images_dir = Path("../temp/Water Bodies Dataset/Images")
masks_dir = Path("../temp/Water Bodies Dataset/Masks")
threshold = 0.95

deleted_count = 0

print(f"Analyzing masks with {threshold:.0%} threshold...")

# Get all image files
image_files = sorted(images_dir.glob("*.jpg"))

for i, image_file in enumerate(image_files):
    mask_file = masks_dir / image_file.name

    if not mask_file.exists():
        continue

    # Load mask and calculate positive ratio
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    positive_pixels = np.sum(mask > 127)
    total_pixels = mask.shape[0] * mask.shape[1]
    positive_ratio = positive_pixels / total_pixels

    # Delete if above threshold
    if positive_ratio >= threshold:
        print(f"Deleting {image_file.name} ({positive_ratio:.1%} positive)")
        image_file.unlink()
        mask_file.unlink()
        deleted_count += 1

    # Progress
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1}/{len(image_files)}, deleted: {deleted_count}")

print(f"\nDone! Deleted {deleted_count} high-positive pairs")
