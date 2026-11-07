# -*- coding: utf-8 -*-
"""Generate realistic night-time augmented stair dataset without manual labeling."""

import os
import glob
import shutil
import random
import cv2
import numpy as np

SRC_DIR = r"c:\project4\stairs.v3i.yolo26"
DST_DIR = r"c:\project4\stairs_night_augmented"

def convert_to_night(img_bgr: np.ndarray) -> np.ndarray:
    """Transforms daytime/indoor image into realistic low-light night-time image."""
    h, w = img_bgr.shape[:2]

    # 1. Extreme Brightness Drop (Random 0.15 ~ 0.40 scaling)
    dark_factor = random.uniform(0.18, 0.42)
    img_dark = img_bgr.astype(np.float32) * dark_factor

    # 2. Non-linear Gamma compression (Make shadows even darker)
    gamma = random.uniform(1.4, 2.2)
    norm = img_dark / 255.0
    img_gamma = np.power(norm, gamma) * 255.0

    # 3. Night Color Cast (Sodium streetlight yellow-orange OR cold blue night)
    night_tone = random.choice(["sodium", "cold_blue", "neutral"])
    if night_tone == "sodium":
        # Boost Red & Green slightly, suppress Blue
        img_gamma[:, :, 0] *= random.uniform(0.55, 0.75)  # B
        img_gamma[:, :, 1] *= random.uniform(0.85, 1.05)  # G
        img_gamma[:, :, 2] *= random.uniform(1.05, 1.25)  # R
    elif night_tone == "cold_blue":
        # Boost Blue, suppress Red
        img_gamma[:, :, 0] *= random.uniform(1.10, 1.30)  # B
        img_gamma[:, :, 1] *= random.uniform(0.85, 0.95)  # G
        img_gamma[:, :, 2] *= random.uniform(0.55, 0.75)  # R

    # 4. Vignetting effect (Corners are darker than center)
    kernel_x = cv2.getGaussianKernel(w, w * 0.75)
    kernel_y = cv2.getGaussianKernel(h, h * 0.75)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.dstack([mask] * 3)
    img_vignette = img_gamma * (0.45 + 0.55 * mask)

    # 5. Low-light ISO Sensor Noise (Shot noise)
    noise_sigma = random.uniform(4.0, 10.0)
    noise = np.random.normal(0, noise_sigma, img_vignette.shape)
    noisy_img = np.clip(img_vignette + noise, 0, 255).astype(np.uint8)

    return noisy_img

def main():
    print("[*] Generating Night-Augmented Stair Dataset...")
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)

    for split in ["train", "valid", "test"]:
        src_img_dir = os.path.join(SRC_DIR, split, "images")
        src_lbl_dir = os.path.join(SRC_DIR, split, "labels")
        dst_img_dir = os.path.join(DST_DIR, split, "images")
        dst_lbl_dir = os.path.join(DST_DIR, split, "labels")

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        img_files = glob.glob(os.path.join(src_img_dir, "*.*"))
        print(f"Processing {split}: {len(img_files)} images...")

        for idx, img_path in enumerate(img_files):
            basename = os.path.basename(img_path)
            stem, ext = os.path.splitext(basename)
            lbl_name = stem + ".txt"
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)

            if not os.path.exists(src_lbl_path):
                continue

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Save 1: Original image & label
            shutil.copy2(img_path, os.path.join(dst_img_dir, basename))
            shutil.copy2(src_lbl_path, os.path.join(dst_lbl_dir, lbl_name))

            # Save 2: Synthetic Night Image & Same Label! (1:1 identical coordinates)
            night_img = convert_to_night(img)
            night_base = f"{stem}_night{ext}"
            night_lbl = f"{stem}_night.txt"

            cv2.imwrite(os.path.join(dst_img_dir, night_base), night_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            shutil.copy2(src_lbl_path, os.path.join(dst_lbl_dir, night_lbl))

    # Write data.yaml
    dst_path_clean = DST_DIR.replace("\\", "/")
    yaml_content = f"""path: {dst_path_clean}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['stair']
"""
    with open(os.path.join(DST_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    train_count = len(glob.glob(os.path.join(DST_DIR, "train", "images", "*.*")))
    valid_count = len(glob.glob(os.path.join(DST_DIR, "valid", "images", "*.*")))
    print(f"\n[OK] Night dataset ready! Total Train: {train_count}, Valid: {valid_count}")
    print(f"Dataset path: {DST_DIR}")

if __name__ == "__main__":
    main()
