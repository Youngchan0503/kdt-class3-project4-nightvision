# -*- coding: utf-8 -*-
"""야간 계단 시연 영상 만들기: 원본 영상 위에 탐지 결과만 덧그린다.

최종 방식 = 공개 데이터 학습 모델(step2_stairs_v3) + 어두울 때만 CLAHE -> 감마.
전처리는 모델 입력에만 쓰고, 화면은 원본 그대로 보여준다.
(전처리한 화면을 보여주면 처리 시간만큼 화면이 늦게 따라와 멀미를 유발할 수 있어서)

프레임별 탐지 여부, 신뢰도, 마스크 면적, 앞 프레임과의 마스크 IoU(흔들림), 처리 시간을 기록한다.
Windows Media Foundation(H.264)으로 저장해서 PowerPoint 에서 재생할 수 있게 한다.

실행 (프로젝트 루트에서):
  .venv/Scripts/python.exe realtime/experiments/step3_enhance/demo/make_demo.py --src realtime/testmp4.mp4
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "realtime"))
import stair_vision as sv  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sv.MODEL_PATH = os.path.join(ROOT, "realtime", "experiments", "step2_stairs_v3", "weights", "best.pt")
sv.ENHANCE_STEPS = ["clahe", "gamma"]
sv.ENHANCE_DARK_THRES = 70
COLOR = (60, 200, 255)   # 주황 (BGR): stair / ascending_stairs
CLASS_COLOR = {"descending_stairs": (255, 140, 60)}   # 파랑 (BGR): 내리막 첫 단차


def overlay_on_original(frame, r, thick):
    """원본 프레임 위에 탐지 영역을 반투명하게 칠하고 테두리를 그린다."""
    if not r["detections"]:
        return frame
    ov = frame.copy()
    for d in r["detections"]:
        cv2.fillPoly(ov, [np.array(d["polygon"], np.int32)], CLASS_COLOR.get(d["class"], COLOR))
    vis = cv2.addWeighted(ov, 0.35, frame, 0.65, 0)
    for d in r["detections"]:
        cv2.polylines(vis, [np.array(d["polygon"], np.int32)], True, CLASS_COLOR.get(d["class"], COLOR), thick)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "realtime", "testmp4.mp4"))
    ap.add_argument("--scale", type=float, default=0.5, help="저장 크기 배율 (분석은 원본 크기로)")
    ap.add_argument("--model", default=None, help="가중치 경로 (기본: step2_stairs_v3)")
    ap.add_argument("--tag", default="", help="출력 파일 이름에 붙일 표시 (예: first_step)")
    ap.add_argument("--no-enhance", action="store_true", help="전처리 없이 원본으로 판단")
    ap.add_argument("--conf", type=float, default=None, help="신뢰도 기준 (기본 0.25)")
    args = ap.parse_args()
    if args.conf is not None:
        sv.CONF_THRES = args.conf
    if args.model:
        sv.MODEL_PATH = os.path.abspath(args.model)
    if args.no_enhance:
        sv.ENHANCE_STEPS = []

    name = os.path.splitext(os.path.basename(args.src))[0] + (f"_{args.tag}" if args.tag else "")
    out_path = os.path.join(OUT_DIR, f"demo_{name}_overlay.mp4")
    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    vw, vh = int(w * args.scale) // 2 * 2, int(h * args.scale) // 2 * 2
    writer = cv2.VideoWriter(out_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, (vw, vh))
    assert writer.isOpened(), "H.264 writer 를 열 수 없음"

    sv.load_model()
    ok, frame = cap.read()
    sv.analyze(frame)  # 워밍업
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    thick = max(3, w // 200)
    rows, prev_mask, poster = [], None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = sv.analyze(frame)
        total_ms = (time.perf_counter() - t0) * 1000
        m = np.zeros((h, w), np.uint8)
        for d in r["detections"]:
            cv2.fillPoly(m, [np.array(d["polygon"], np.int32)], 1)
        iou = None
        if prev_mask is not None and (m.any() or prev_mask.any()):
            iou = float((m & prev_mask).sum() / max((m | prev_mask).sum(), 1))
        prev_mask = m
        rows.append(dict(frame=len(rows), brightness=r["brightness"], enhanced=r["enhanced"],
                         hit=bool(r["detections"]), classes=sorted({d["class"] for d in r["detections"]}), conf=max([d["conf"] for d in r["detections"]], default=0.0),
                         area=float(m.mean()), iou_prev=iou, pre_ms=r["pre_ms"], infer_ms=r["infer_ms"], total_ms=total_ms))
        vis = overlay_on_original(frame, r, thick)
        small = cv2.resize(vis, (vw, vh), interpolation=cv2.INTER_AREA)
        writer.write(small)
        if poster is None and r["detections"]:
            poster = small
    cap.release()
    writer.release()
    cv2.imwrite(os.path.join(OUT_DIR, f"demo_{name}_poster.png"), poster)

    n = len(rows)
    hit = [x for x in rows if x["hit"]]
    ious = [x["iou_prev"] for x in rows if x["iou_prev"] is not None]
    summary = dict(
        src=os.path.relpath(args.src, ROOT), out=os.path.relpath(out_path, ROOT), size=[vw, vh],
        model=os.path.relpath(sv.MODEL_PATH, ROOT), enhance="+".join(sv.ENHANCE_STEPS) or "none", conf=sv.CONF_THRES,
        class_frames={c: sum(c in x["classes"] for x in rows) for c in sorted({c for x in rows for c in x["classes"]})},
        frames=n, fps=round(fps, 2), seconds=round(n / fps, 1),
        mean_brightness=round(float(np.mean([x["brightness"] for x in rows])), 1),
        enhanced_frames=sum(x["enhanced"] for x in rows),
        detected_frames=len(hit), detect_rate=round(len(hit) / n, 3),
        mean_conf_when_detected=round(float(np.mean([x["conf"] for x in hit])), 3) if hit else 0,
        detect_flips=sum(1 for a, b in zip(rows, rows[1:]) if a["hit"] != b["hit"]),
        mean_iou_prev=round(float(np.mean(ious)), 3) if ious else None,
        frames_iou_prev_below_05=int(sum(i < 0.5 for i in ious)),
        mean_total_ms=round(float(np.mean([x["total_ms"] for x in rows])), 1),
        mean_pre_ms=round(float(np.mean([x["pre_ms"] for x in rows])), 1),
        mean_infer_ms=round(float(np.mean([x["infer_ms"] for x in rows])), 1),
    )
    summary["processing_fps"] = round(1000 / summary["mean_total_ms"], 1)
    json.dump(dict(summary=summary, frames=rows), open(os.path.join(OUT_DIR, f"demo_{name}_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
