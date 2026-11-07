# -*- coding: utf-8 -*-
"""Test Retinexformer inference on user dark stair photos."""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, r"c:\project4\backend\services")
from retinexformer_arch import RetinexFormer

WEIGHT_PATH = r"c:\project4\backend\weights\Retinexformer_LOL_v1.pth"
TEST_IMG = r"C:\Users\lgiht\.gemini\antigravity\brain\c40bd5bf-95e3-47ee-9d96-009991637bf0\.user_uploaded\media_1793773897361_8deaf121.jpg"
OUT_IMG = r"c:\project4\scratch\retinexformer_enhanced_stair.jpg"

def enhance_with_retinexformer(img_bgr: np.ndarray, model, device="cpu") -> np.ndarray:
    h, w = img_bgr.shape[:2]
    # Retinexformer requires dimensions divisible by 8 or 16 due to multi-scale transformer
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        out_tensor = model(img_tensor)

    # unpad
    out_tensor = out_tensor[:, :, :h, :w]
    out_tensor = torch.clamp(out_tensor, 0, 1.0)
    out_rgb = (out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def main():
    print("Loading Retinexformer model...")
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    weights = torch.load(WEIGHT_PATH, map_location="cpu")
    state_dict = weights["params"] if "params" in weights else weights
    clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=True)
    model.eval()

    print(f"Reading test image: {TEST_IMG}")
    img_bgr = cv2.imread(TEST_IMG)
    if img_bgr is None:
        print("Failed to read image")
        return

    # Resize slightly for fast CPU testing (e.g. width=480)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    import time
    st = time.time()
    enhanced = enhance_with_retinexformer(img_bgr, model)
    elapsed = round(time.time() - st, 2)
    print(f"Enhancement finished in {elapsed}s! Saving to {OUT_IMG}")

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    cv2.imwrite(OUT_IMG, enhanced)
    print("Saved successfully!")

if __name__ == "__main__":
    main()
