# -*- coding: utf-8 -*-
"""Fine-tune YOLO26 Instance Segmentation on 3,200 Night-Augmented Stair Dataset."""

import os
from ultralytics import YOLO

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_dir, "stairs_night_augmented", "data.yaml")
    base_model = os.path.join(project_dir, "yolo26n-seg.pt")
    output_dir = os.path.join(project_dir, "runs", "stairs_night_yolo26")

    print("[*] Training YOLO26 on Night-Augmented Stair Dataset (3,216 images)...")
    print(f"[*] Dataset: {yaml_path}")
    print(f"[*] Model:   {base_model}")

    model = YOLO(base_model)
    results = model.train(
        data=yaml_path,
        epochs=10,
        batch=16,
        imgsz=480,
        device="cpu",
        project=output_dir,
        name="exp",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )

    best_weight = os.path.join(output_dir, "exp", "weights", "best.pt")
    print(f"\n[OK] Night Training completed! Best weight: {best_weight}")

if __name__ == "__main__":
    main()
