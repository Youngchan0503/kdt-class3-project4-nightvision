# -*- coding: utf-8 -*-
"""SafeStep 메인 비전 코드.

사진 / 영상 / 실시간 카메라, 세 가지 입력을 같은 방식으로 분석한다.
  - analyze(frame) : 프레임 1장 -> 결과 dict (데이터만)
  - draw(frame, result) : 결과를 프레임 위에 그려서 반환 (그림만)

실행 예시 (프로젝트 루트에서):
  .venv/Scripts/python.exe realtime/stair_vision.py --source photo.jpg   # 사진
  .venv/Scripts/python.exe realtime/stair_vision.py --source night.mp4   # 영상
  .venv/Scripts/python.exe realtime/stair_vision.py --source 0           # 카메라 0번

조작키: q 또는 ESC = 종료, 스페이스 = 일시정지(영상/카메라)
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 설정 (단계가 바뀌면 여기 값을 바꾼다)
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1단계: COCO 사전학습 모델 그대로 사용 (계단 클래스는 없다!)
MODEL_PATH = os.path.join(ROOT_DIR, "yolo26n-seg.pt")
CONF_THRES = 0.25   # 이 신뢰도 미만인 탐지는 버린다
IMG_SIZE = 640      # YOLO 입력 크기

# 3단계: 모델에 넣기 전 전처리 (적힌 순서대로 적용, 빈 리스트 = 전처리 없음)
#   "bilateral" : 노이즈 제거 (경계는 살리고 자글거림만 줄임)
#   "gamma"     : 감마 보정 (어두운 부분을 비선형으로 끌어올림)
#   "clahe"     : 구역별 대비 향상 (계단 단의 경계를 살림)
#   "sharpen"   : 언샤프 마스킹 (경계 선명화)
ENHANCE_STEPS = []
ENHANCE_DARK_THRES = 70   # 평균 밝기가 이 값보다 낮을 때만 전처리 (None = 항상)
BILATERAL = dict(d=7, sigmaColor=35, sigmaSpace=35)
GAMMA = 1.6               # 클수록 어두운 부분을 더 밝힘 (출력 = (입력/255)^(1/GAMMA) * 255)
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 모델 로딩
# ============================================================
_model = None


def load_model(path: str = None) -> YOLO:
    """모델을 한 번만 불러와서 재사용한다 (매 프레임마다 로딩하면 매우 느림).

    주의: 기본값을 MODEL_PATH 로 쓰면 '함수 정의 시점'의 값으로 고정돼서
    --model 로 바꾼 경로가 무시된다. 그래서 None 으로 두고 호출 시점에 읽는다.
    """
    global _model
    if _model is None:
        path = path or MODEL_PATH
        _model = YOLO(path)
        print(f"[모델] {path} 로딩 완료 / 클래스 수: {len(_model.names)}")
    return _model


# ============================================================
# 전처리: 어두운 프레임을 모델이 보기 좋게 보정
# ============================================================
def brightness(frame: np.ndarray) -> float:
    """흑백으로 바꾼 평균 밝기 (0~255)."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def preprocess(frame: np.ndarray, steps=None) -> np.ndarray:
    """ENHANCE_STEPS 순서대로 보정한 새 이미지를 돌려준다 (원본은 그대로).

    감마와 CLAHE는 LAB 색공간의 밝기(L) 채널에만 적용해서 색이 틀어지지 않게 한다.
    """
    steps = ENHANCE_STEPS if steps is None else steps
    out = frame
    for step in steps:
        if step == "bilateral":
            out = cv2.bilateralFilter(out, **BILATERAL)
        elif step in ("gamma", "clahe"):
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            if step == "gamma":
                lut = np.array([((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)], np.uint8)
                l = cv2.LUT(l, lut)
            else:
                l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(l)
            out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        elif step == "sharpen":
            blur = cv2.GaussianBlur(out, (0, 0), 2.0)
            out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
        else:
            raise ValueError(f"알 수 없는 전처리: {step}")
    return out


# ============================================================
# 분석: 프레임 1장 -> 결과 dict
# ============================================================
def analyze(frame: np.ndarray) -> dict:
    """프레임 한 장을 분석해서 '데이터만' 돌려준다. 그림은 그리지 않는다.

    result["model_input"] 에 모델이 실제로 본 이미지(전처리 후)를 담아 둔다 (디버깅용).
    화면 표시는 draw() 가 원본 프레임 위에 한다.
    """
    model = load_model()

    t0 = time.perf_counter()
    src_brightness = brightness(frame)
    use_enhance = bool(ENHANCE_STEPS) and (ENHANCE_DARK_THRES is None or src_brightness < ENHANCE_DARK_THRES)
    x = preprocess(frame) if use_enhance else frame
    pre_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res = model.predict(x, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=False)[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    detections = []
    if res.boxes is not None:
        boxes = res.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        # 세그멘테이션 모델이면 masks.xy 에 윤곽선 다각형 좌표가 들어 있다
        polys = res.masks.xy if res.masks is not None else [None] * len(boxes)

        for box, conf, cls_id, poly in zip(boxes, confs, clss, polys):
            detections.append({
                "class": model.names[cls_id],
                "class_id": int(cls_id),
                "conf": float(conf),
                "box": [int(v) for v in box],
                "polygon": poly.astype(int).tolist() if poly is not None and len(poly) else [],
            })

    return {
        "detections": detections,
        "infer_ms": round(infer_ms, 1),
        "pre_ms": round(pre_ms, 1),
        "brightness": round(src_brightness, 1),
        "enhanced": use_enhance,
        "model_input": x,   # numpy 이미지 (JSON으로 보낼 때는 빼야 함)
    }


# ============================================================
# 그리기: 결과를 프레임 위에 오버레이
# ============================================================
def _color_for(class_id: int) -> tuple:
    """클래스마다 항상 같은 색이 나오게 한다."""
    rng = np.random.default_rng(class_id)
    return tuple(int(c) for c in rng.integers(80, 255, size=3))


def draw(frame: np.ndarray, result: dict, fps: float = None, on_model_input: bool = False) -> np.ndarray:
    """result 를 '원본 프레임' 위에 그린 새 이미지를 반환한다 (원본은 건드리지 않음).

    화면은 원본 그대로 보여주고 탐지 결과만 덧그린다. 전처리한 이미지를 화면에 보여주면
    처리 시간만큼 화면 자체가 늦게 따라와서 실제 움직임과 어긋나고 멀미를 유발할 수 있기 때문.
    (전처리는 밝기만 바꾸고 좌표는 그대로라서 원본 위에 그대로 그릴 수 있다.)
    on_model_input=True 면 디버깅용으로 모델이 실제로 본 이미지(전처리 후) 위에 그린다.
    """
    out = (result.get("model_input", frame) if on_model_input else frame).copy()

    # 1) 마스크를 반투명하게 칠한다
    overlay = out.copy()
    for det in result["detections"]:
        if det["polygon"]:
            pts = np.array(det["polygon"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], _color_for(det["class_id"]))
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)

    # 2) 박스와 라벨
    for det in result["detections"]:
        x1, y1, x2, y2 = det["box"]
        color = _color_for(det["class_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # 3) 왼쪽 위 개발용 정보 (fps, 추론 시간, 탐지 수)
    info = [f"model: {os.path.basename(MODEL_PATH)}",
            f"infer: {result['infer_ms']:.1f} ms",
            f"objects: {len(result['detections'])}",
            f"enhance: {'+'.join(ENHANCE_STEPS) if result.get('enhanced') else 'off'}"
            f" (bright {result.get('brightness', 0):.0f})"]
    if fps is not None:
        info.insert(1, f"fps: {fps:.1f}")
    cv2.rectangle(out, (0, 0), (360, 22 * len(info) + 10), (0, 0, 0), -1)
    for i, text in enumerate(info):
        cv2.putText(out, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return out


# ============================================================
# 입력 3종 처리
# ============================================================
def run_image(path: str, show: bool, save: str = None):
    """사진: 한 장 분석하고 키를 누를 때까지 보여준다."""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"사진을 열 수 없음: {path}")

    load_model()
    analyze(frame)  # 첫 추론은 GPU 준비(워밍업) 때문에 느리므로 한 번 버린다
    result = analyze(frame)
    vis = draw(frame, result)

    print(f"[사진] 탐지 {len(result['detections'])}개, 추론 {result['infer_ms']} ms")
    for det in result["detections"]:
        print(f"   - {det['class']:<15} conf={det['conf']:.2f}  box={det['box']}")

    if save:
        cv2.imwrite(save, vis)
        print(f"[저장] {save}")
    if show:
        cv2.imshow("SafeStep", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(source, show: bool, save: str = None, max_frames: int = None):
    """영상 파일 / 카메라: 프레임을 하나씩 꺼내서 분석 -> 그리기 -> 보여주기 반복."""
    is_camera = isinstance(source, int)
    # 윈도우에서는 DSHOW 백엔드가 카메라를 훨씬 빨리 연다
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if is_camera else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"입력을 열 수 없음: {source}")

    load_model()
    writer = None
    fps_smooth = None
    prev_t = time.perf_counter()
    frame_idx = 0
    paused = False
    stats = []  # 마지막에 평균을 내기 위한 기록

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:  # 영상이 끝났거나 카메라 연결이 끊김
                break

            result = analyze(frame)

            # fps = 1 / (이번 프레임과 지난 프레임 사이 시간). 숫자가 튀지 않게 부드럽게 평균낸다
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_smooth = inst_fps if fps_smooth is None else fps_smooth * 0.9 + inst_fps * 0.1

            vis = draw(frame, result, fps=fps_smooth)
            if frame_idx > 0:  # 첫 프레임은 워밍업이라 통계에서 뺀다
                stats.append((result["infer_ms"], len(result["detections"])))

            if save:
                if writer is None:
                    h, w = vis.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(vis)

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

        if show:
            cv2.imshow("SafeStep", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q 또는 ESC
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer:
        writer.release()
        print(f"[저장] {save}")
    cv2.destroyAllWindows()

    if stats:
        ms = np.array([s[0] for s in stats])
        print(f"[요약] {frame_idx}프레임 / 평균 추론 {ms.mean():.1f} ms "
              f"(최대 {ms.max():.1f}) / 프레임당 평균 탐지 {np.mean([s[1] for s in stats]):.2f}개")


def main():
    global MODEL_PATH, CONF_THRES, ENHANCE_STEPS, ENHANCE_DARK_THRES

    p = argparse.ArgumentParser(description="SafeStep 메인 비전 코드")
    p.add_argument("--source", required=True, help="사진/영상 파일 경로, 또는 카메라 번호(0, 1, ...)")
    p.add_argument("--model", default=MODEL_PATH, help="YOLO 가중치 경로")
    p.add_argument("--conf", type=float, default=CONF_THRES, help="신뢰도 기준")
    p.add_argument("--save", default=None, help="결과 저장 경로 (사진 .jpg / 영상 .mp4)")
    p.add_argument("--no-show", action="store_true", help="창을 띄우지 않음 (테스트용)")
    p.add_argument("--max-frames", type=int, default=None, help="영상/카메라에서 처리할 최대 프레임 수")
    p.add_argument("--enhance", default="", help="전처리 순서, 쉼표로 구분 (예: bilateral,gamma,clahe)")
    p.add_argument("--enhance-always", action="store_true", help="밝은 프레임에도 전처리 적용")
    args = p.parse_args()

    MODEL_PATH = args.model
    CONF_THRES = args.conf
    ENHANCE_STEPS = [s.strip() for s in args.enhance.split(",") if s.strip()]
    if args.enhance_always:
        ENHANCE_DARK_THRES = None
    show = not args.no_show

    # 입력 종류 판별: 숫자면 카메라, 사진 확장자면 사진, 나머지는 영상
    if args.source.isdigit():
        run_stream(int(args.source), show, args.save, args.max_frames)
    elif os.path.splitext(args.source)[1].lower() in IMAGE_EXTS:
        run_image(args.source, show, args.save)
    else:
        run_stream(args.source, show, args.save, args.max_frames)


if __name__ == "__main__":
    main()
