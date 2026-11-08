# -*- coding: utf-8 -*-
"""SafeStep 학습 스크립트.

학습은 한 번 돌리고 끝나는 별도 작업이라 메인 파일(stair_vision.py)과 분리한다.
결과는 realtime/experiments/<name>/ 에 저장되고, best.pt 를 stair_vision.py --model 로 쓴다.

실행 예시 (프로젝트 루트에서):
  .venv/Scripts/python.exe realtime/train.py --data realtime/datasets/stairs_v3.yaml --name step2_stairs_v3
"""

import argparse
import os

from ultralytics import YOLO

REALTIME_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(REALTIME_DIR, ".."))


def main():
    p = argparse.ArgumentParser(description="SafeStep YOLO 세그멘테이션 학습")
    p.add_argument("--data", required=True, help="데이터셋 yaml 경로")
    p.add_argument("--name", required=True, help="실험 이름 (experiments/ 아래 폴더명)")
    p.add_argument("--model", default=os.path.join(ROOT_DIR, "yolo26n-seg.pt"), help="시작 가중치")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", action="store_true", help="이미지를 메모리에 미리 올림 (원본이 큰 소규모 데이터에 유리)")
    p.add_argument("--no-test", action="store_true", help="test 분할이 없는 데이터셋이면 마지막 test 평가를 건너뜀")
    args = p.parse_args()

    # COCO 사전학습 가중치에서 시작 -> 계단 데이터로 전이학습
    model = YOLO(args.model)
    model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache="ram" if args.cache else False,
        project=os.path.join(REALTIME_DIR, "experiments"),
        name=args.name,
        exist_ok=False,   # 같은 이름이 있으면 덮어쓰지 않고 멈춘다 (기록 보호)
        seed=0,
        deterministic=True,
        plots=True,
    )

    # 학습이 끝나면 best.pt 로 test 분할 성능을 따로 잰다 (학습/검증에 안 쓴 데이터)
    if args.no_test:
        return
    best = os.path.join(REALTIME_DIR, "experiments", args.name, "weights", "best.pt")
    metrics = YOLO(best).val(data=os.path.abspath(args.data), split="test", imgsz=args.imgsz,
                             project=os.path.join(REALTIME_DIR, "experiments", args.name), name="test_eval")
    print(f"[test] box mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  "
          f"mask mAP50={metrics.seg.map50:.3f}  mAP50-95={metrics.seg.map:.3f}")


if __name__ == "__main__":  # 윈도우에서 데이터로더 워커를 쓰려면 꼭 필요
    main()
