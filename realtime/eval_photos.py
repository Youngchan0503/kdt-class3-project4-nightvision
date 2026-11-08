# -*- coding: utf-8 -*-
"""팀이 촬영한 계단 사진으로 모델을 평가한다 (단계별 비교용).

같은 사진, 같은 기준으로 여러 설정(전처리 조합)을 돌려서 표로 비교한다.
  - 밝기 = 흑백 평균 밝기 (0~255). 30 미만 = 매우 어두움, 70 미만 = 어두움
  - "계단 탐지" = stair 탐지가 1개 이상 있는 사진
  - 마스크 면적 = 화면에서 계단으로 칠해진 비율 (탐지 못한 사진은 0으로 평균)
  - 쌍 IoU = 같은 계단의 어두운 사진(_0) 마스크와 밝은 사진(_1) 마스크가 겹치는 정도
            (밝은 쪽 결과를 정답 대용으로 쓴다. 구도가 완전히 같지는 않아서 대략적인 비교)

실행 예시 (프로젝트 루트에서):
  .venv/Scripts/python.exe realtime/eval_photos.py --model realtime/experiments/step2_stairs_v3/weights/best.pt \
      --configs "none;bilateral;bilateral,gamma;bilateral,gamma,clahe" --out realtime/experiments/step3_enhance
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stair_vision as sv  # noqa: E402

PHOTO_DIRS = [
    "backup/legacy_datasets/raw_photos/계단",
    "backup/legacy_datasets/raw_photos/drive-download-20261105T045852Z-1-001",
]
SAVE_EXAMPLES = ("seo06_0", "seo15_0", "seo14_0", "han02_0")  # 설정별 결과 이미지를 저장할 사진


def read_image(path, max_dim=1280):
    """한글 경로도 읽을 수 있게 imdecode 사용, 긴 변을 max_dim 으로 줄인다."""
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    s = max_dim / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def mask_of(result, shape):
    m = np.zeros(shape[:2], np.uint8)
    for d in result["detections"]:
        if d["class"] == "stair" and d["polygon"]:
            cv2.fillPoly(m, [np.array(d["polygon"], np.int32)], 1)
    return m


def evaluate(images, steps, always=False, out_dir=None, tag=""):
    """images: {이름: 이미지}. 설정 하나로 전체를 돌려 지표 dict 를 돌려준다."""
    sv.ENHANCE_STEPS = steps
    sv.ENHANCE_DARK_THRES = None if always else 70
    rows = {}
    for name, im in images.items():
        r = sv.analyze(im)
        confs = [d["conf"] for d in r["detections"] if d["class"] == "stair"]
        m = mask_of(r, im.shape)
        rows[name] = dict(b=r["brightness"], hit=bool(confs), conf=max(confs, default=0.0),
                          area=float(m.mean()), mask=m, pre_ms=r["pre_ms"], infer_ms=r["infer_ms"])
        if out_dir and name in SAVE_EXAMPLES:
            cv2.imwrite(os.path.join(out_dir, f"{name}__{tag}.jpg"), sv.draw(im, r))

    def group(cond):
        g = [v for v in rows.values() if cond(v["b"])]
        return dict(n=len(g), hit=sum(v["hit"] for v in g),
                    rate=sum(v["hit"] for v in g) / max(len(g), 1),
                    area=float(np.mean([v["area"] for v in g])) if g else 0.0,
                    conf=float(np.mean([v["conf"] for v in g if v["hit"]])) if any(v["hit"] for v in g) else 0.0)

    # 같은 계단 쌍: _0 이 실제로 더 어둡고, 밝은 쪽(_1)이 계단을 찾은 쌍만
    ious = []
    for k, v in rows.items():
        if k.endswith("_0") and k[:-2] + "_1" in rows:
            w = rows[k[:-2] + "_1"]
            if v["b"] < w["b"] and w["hit"] and v["mask"].shape == w["mask"].shape:
                inter = (v["mask"] & w["mask"]).sum(); uni = (v["mask"] | w["mask"]).sum()
                ious.append(inter / uni if uni else 0.0)
    dark = [v for v in rows.values() if v["b"] < 70]
    return {
        "steps": "+".join(steps) or "none", "always": always,
        "very_dark": group(lambda b: b < 30), "dark": group(lambda b: b < 70), "light": group(lambda b: b >= 70),
        "pairs": len(ious), "pair_iou": float(np.mean(ious)) if ious else 0.0,
        "pair_iou50": int(sum(i >= 0.5 for i in ious)),
        "pre_ms_dark": float(np.mean([v["pre_ms"] for v in dark])) if dark else 0.0,
        "infer_ms": float(np.mean([v["infer_ms"] for v in rows.values()])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--configs", default="none", help='세미콜론으로 설정 구분, 설정 안은 쉼표 (예: "none;bilateral,gamma")')
    p.add_argument("--always", action="store_true", help="밝은 사진에도 전처리 적용")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--clahe-clip", type=float, default=None)
    p.add_argument("--out", default=None, help="예시 이미지와 결과 json 을 저장할 폴더")
    args = p.parse_args()

    sv.MODEL_PATH = args.model
    if args.gamma: sv.GAMMA = args.gamma
    if args.clahe_clip: sv.CLAHE_CLIP = args.clahe_clip
    if args.out: os.makedirs(args.out, exist_ok=True)

    files = [f for d in PHOTO_DIRS for f in sorted(glob.glob(os.path.join(d, "*.jpg")))]
    images = {os.path.splitext(os.path.basename(f))[0]: read_image(f) for f in files}
    sv.load_model(); sv.analyze(next(iter(images.values())))  # GPU 워밍업

    print(f"[모델] {args.model} / conf {sv.CONF_THRES} / gamma {sv.GAMMA} / clahe clip {sv.CLAHE_CLIP} / "
          f"{'모든 사진' if args.always else '어두운 사진(밝기<70)만'} 전처리 / 사진 {len(images)}장")
    print(f"{'설정':<28}{'매우어두움<30':>14}{'면적':>7}{'어두움<70':>11}{'밝음':>9}{'쌍IoU':>8}{'IoU≥.5':>8}{'전처리ms':>9}")
    results = []
    for cfg in args.configs.split(";"):
        steps = [] if cfg.strip() in ("", "none") else [s.strip() for s in cfg.split(",")]
        tag = ("+".join(steps) or "none") + ("_always" if args.always else "")
        r = evaluate(images, steps, args.always, args.out, tag)
        results.append(r)
        vd, d, l = r["very_dark"], r["dark"], r["light"]
        print(f"{r['steps']:<28}{vd['hit']:>5}/{vd['n']} ({vd['rate']*100:>3.0f}%){vd['area']*100:>6.0f}%"
              f"{d['rate']*100:>10.0f}%{l['rate']*100:>8.0f}%{r['pair_iou']:>8.2f}{r['pair_iou50']:>5}/{r['pairs']}"
              f"{r['pre_ms_dark']:>9.1f}")
    if args.out:
        name = f"results_{int(time.time())}.json"
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(dict(model=args.model, gamma=sv.GAMMA, clahe_clip=sv.CLAHE_CLIP, results=results),
                      f, ensure_ascii=False, indent=1)
        print(f"[저장] {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
