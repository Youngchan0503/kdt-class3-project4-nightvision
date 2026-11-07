# -*- coding: utf-8 -*-
"""Custom Night Stair Detection Training Script (Dual Classes)

Dataset: dataset_custom_stairs/data.yaml (총 123장: Train 97, Val 26)
Classes:
  0: ascending_stairs (오르막 계단)
  1: descending_stairs (내리막 계단)
Base Model: yolo26n-seg.pt
"""

import os
import sys
import torch
from ultralytics import YOLO

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_dir, "dataset_custom_stairs", "data.yaml")
    # 1,600장 대규모 계단 지식(대리석/사선 앵글 85.3% 신뢰도)을 가진 사전학습 모델을 베이스로 전이학습
    base_model = os.path.join(project_dir, "backup", "legacy_models_and_runs", "stairs_yolo26", "exp", "weights", "best.pt")
    if not os.path.exists(base_model):
        base_model = os.path.join(project_dir, "yolo26n-seg.pt")
    output_dir = os.path.join(project_dir, "runs", "custom_stairs_seg")

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 68)
    print("🚀 [SafeStep AI] 1,600장 계단 전문 모델 기반 고도화 전이학습 시작")
    print(f"[*] 데이터셋 설정: {yaml_path} (총 123장, Train: 97, Val: 26)")
    print(f"[*] 베이스 모델:   {base_model} (1,600-image Stairs Pretrained)")
    print(f"[*] 실행 디바이스: {device} ({'GPU' if device == 0 else 'CPU'})")
    print(f"[*] 클래스 정의:   [0: ascending_stairs (오르막), 1: descending_stairs (내리막)]")
    print("=" * 68)

    model = YOLO(base_model)
    results = model.train(
        data=yaml_path,
        epochs=25,
        batch=8,
        imgsz=640,
        device=device,
        project=output_dir,
        name="exp",
        exist_ok=True,
        lr0=0.002,        # 기존 계단 지식(대리석/다각도)을 보존하는 부드러운 전이학습률
        lrf=0.01,
        degrees=15.0,     # 대각선/사선 앵글 계단 대응 회전 증강
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,        # 조명 및 야간 밝기 변화 증강
        fliplr=0.5,       # 좌우 반전 증강
        verbose=True,
    )

    best_weight = os.path.join(output_dir, "exp", "weights", "best.pt")
    print("=" * 68)
    print(f"🎉 [성공] 학습 완료! 최종 가중치: {best_weight}")
    print("=" * 68)

if __name__ == "__main__":
    main()
