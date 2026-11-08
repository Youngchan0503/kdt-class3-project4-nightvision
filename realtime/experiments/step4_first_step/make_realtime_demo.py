# -*- coding: utf-8 -*-
"""mp4 영상을 '실시간 카메라'라고 치고 시연 영상을 만든다.

실시간이라면:
  - 카메라 프레임은 1/fps 초마다 들어오고, 화면에는 원본이 바로 나간다 (화면은 늦지 않음).
  - 모델은 한 번에 한 프레임만 분석한다. 분석하는 동안 들어온 프레임은 건너뛰고,
    분석이 끝나면 '그 시점의 가장 최신 프레임'을 다시 분석한다.
  - 오버레이는 '가장 최근에 끝난 분석 결과'를 그린다 → 몇 프레임 전 결과일 수 있다.

이 스크립트는 모든 프레임을 미리 실제로 분석해서 프레임별 처리 시간을 잰 뒤,
위 규칙대로 가상 시계를 돌려서 '실시간이었다면 화면에 무엇이 보였을지'를 영상으로 만든다.
(실제로 재지 않은 지연 — 카메라→PC 전송, 화면 표시 — 은 포함하지 않는다)

실행 (프로젝트 루트에서):
  .venv/Scripts/python.exe realtime/experiments/step4_first_step/make_realtime_demo.py
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "realtime"))
import stair_vision as sv  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = {"descending_stairs": (255, 140, 60)}   # 파랑 (BGR)
DEFAULT_COLOR = (60, 200, 255)                   # 주황: 오르막 / stair


def overlay(frame, dets, thick):
    if not dets:
        return frame
    ov = frame.copy()
    for d in dets:
        cv2.fillPoly(ov, [np.array(d["polygon"], np.int32)], COLORS.get(d["class"], DEFAULT_COLOR))
    vis = cv2.addWeighted(ov, 0.35, frame, 0.65, 0)
    for d in dets:
        cv2.polylines(vis, [np.array(d["polygon"], np.int32)], True, COLORS.get(d["class"], DEFAULT_COLOR), thick)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "realtime", "testmp4.mp4"))
    ap.add_argument("--model", default=os.path.join(ROOT, "realtime", "experiments", "step4_first_step", "weights", "best.pt"))
    ap.add_argument("--enhance", default="", help="전처리 (예: clahe,gamma). 기본은 없음")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--scale", type=float, default=0.5, help="저장 크기 배율")
    ap.add_argument("--tag", default="first_step_rt")
    args = ap.parse_args()

    sv.MODEL_PATH = os.path.abspath(args.model)
    sv.CONF_THRES = args.conf
    sv.ENHANCE_STEPS = [s for s in args.enhance.split(",") if s]

    # 1) 프레임을 하나씩 읽으며 실제로 분석해서 프레임별 처리 시간을 잰다
    #    (프레임을 메모리에 쌓아 두면 1080x1920 x 995장 = 약 6GB라 메모리가 부족해져 처리 시간이 왜곡된다)
    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sv.load_model()
    ok, f0 = cap.read()
    for _ in range(3):
        sv.analyze(f0)  # 워밍업
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results, cost_ms = [], []
    h, w = f0.shape[:2]
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = sv.analyze(f)
        cost_ms.append((time.perf_counter() - t0) * 1000)
        results.append(r["detections"])
    cap.release()
    n = len(results)

    # 2) 가상 시계: 프레임 i 는 i/fps 초에 도착. 모델이 쉬고 있으면 그 시점의 최신 프레임을 잡아 분석한다
    period = 1000.0 / fps
    done = []           # (끝난 시각 ms, 분석한 프레임 번호)
    t_free, i = 0.0, 0
    while i < n:
        start = max(t_free, i * period)
        # 시작 시점에 이미 도착한 가장 최신 프레임 (소수점 오차로 이미 분석한 프레임을 다시 고르지 않게 i 이상으로)
        latest = max(i, min(n - 1, int(start / period + 1e-6)))
        t_free = start + cost_ms[latest]
        done.append((t_free, latest))
        i = latest + 1

    # 3) 화면 프레임 j (j/fps 초): 그때까지 끝난 결과 중 가장 최근 것을 원본 위에 그린다
    cap = cv2.VideoCapture(args.src)
    vw, vh = int(w * args.scale) // 2 * 2, int(h * args.scale) // 2 * 2
    out_path = os.path.join(OUT_DIR, f"demo_testmp4_{args.tag}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), fps, (vw, vh))
    assert writer.isOpened()
    thick = max(3, w // 200)
    k, shown, lags = -1, None, []
    for j in range(n):
        now = j * period
        while k + 1 < len(done) and done[k + 1][0] <= now:
            k += 1
        shown = done[k][1] if k >= 0 else None
        dets = results[shown] if shown is not None else []
        lag = (j - shown) if shown is not None else None
        lags.append(lag)
        ok, frame = cap.read()
        vis = overlay(frame, dets, thick)
        small = cv2.resize(vis, (vw, vh), interpolation=cv2.INTER_AREA)
        txt = f"result: {lag} frame(s) old" if lag is not None else "result: -"
        cv2.putText(small, txt, (10, vh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(small, txt, (10, vh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(small)
    writer.release()
    cap.release()

    analyzed = len(done)
    valid = [l for l in lags if l is not None]
    hit_frames = sum(1 for j in range(n) if lags[j] is not None and results[j - lags[j]])
    summary = dict(
        src=os.path.relpath(args.src, ROOT), model=os.path.relpath(sv.MODEL_PATH, ROOT),
        enhance="+".join(sv.ENHANCE_STEPS) or "none", conf=args.conf, out=os.path.relpath(out_path, ROOT),
        frames=n, fps=round(fps, 2), frame_period_ms=round(period, 1),
        mean_cost_ms=round(float(np.mean(cost_ms)), 1), p95_cost_ms=round(float(np.percentile(cost_ms, 95)), 1),
        analyzed_frames=analyzed, skipped_frames=n - analyzed, analyzed_ratio=round(analyzed / n, 3),
        mean_result_age_frames=round(float(np.mean(valid)), 2), max_result_age_frames=int(max(valid)),
        mean_result_age_ms=round(float(np.mean(valid)) * period, 1),
        frames_showing_detection=hit_frames, showing_detection_ratio=round(hit_frames / n, 3),
    )
    json.dump(summary, open(os.path.join(OUT_DIR, f"demo_testmp4_{args.tag}_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
