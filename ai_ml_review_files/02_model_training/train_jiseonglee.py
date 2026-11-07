# -*- coding: utf-8 -*-
"""Fine-tune YOLO26 Instance Segmentation on jiseonglee (ramp & stair) dataset."""

import os
from ultralytics import YOLO

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_dir, "jiseonglee.v1i.yolo26", "data.yaml")
    base_model = os.path.join(project_dir, "yolo26n-seg.pt")
    output_dir = os.path.join(project_dir, "runs", "jiseonglee_yolo26")

    print("=" * 60)
    print("🚀 Training YOLO26 on jiseonglee dataset (ramp & stair)")
    print(f"• Dataset YAML: {yaml_path}")
    print(f"• Base Model:   {base_model}")
    print(f"• Output Dir:   {output_dir}")
    print("=" * 60)

    model = YOLO(base_model)
    results = model.train(
        data=yaml_path,
        epochs=15,
        batch=8,
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
    print(f"\n✅ Training completed! Best weight: {best_weight}")

if __name__ == "__main__":
    main()
