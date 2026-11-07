# -*- coding: utf-8 -*-
"""Train script for YOLO26 Instance Segmentation on Stairs Dataset."""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 on Stairs Dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default 8 for CPU/fast RAM)")
    parser.add_argument("--imgsz", type=int, default=480, help="Image size for training (480 or 640)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or 0 for cuda)")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_dir, "stairs.v3i.yolo26", "data.yaml")
    model_weight = os.path.join(project_dir, "yolo26n-seg.pt")

    print("=" * 60)
    print("🚀 YOLO26 Stairs Instance Segmentation Training")
    print(f"• Dataset YAML: {yaml_path}")
    print(f"• Base Model:   {model_weight}")
    print(f"• Epochs:       {args.epochs}")
    print(f"• Batch Size:   {args.batch}")
    print(f"• Image Size:   {args.imgsz}")
    print(f"• Device:       {args.device}")
    print("=" * 60)

    # Load YOLO26 Segmentation base model
    model = YOLO(model_weight)

    # Train
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=os.path.join(project_dir, "runs", "stairs_yolo26"),
        name="exp",
        exist_ok=True,
        # Augmentation for night & low-light robustness
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,    # High value/brightness variation for night simulation
        flipud=0.0,   # Stairs don't flip upside down
        fliplr=0.5,   # Left-right mirror allowed
    )

    print("\n✅ Training finished successfully!")
    print(f"Best weights saved at: {os.path.join(project_dir, 'runs', 'stairs_yolo26', 'exp', 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()
