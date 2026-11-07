"""Posture Service wrapping Project 1 (MediaPipe AI Posture Checker)."""

import os
import sys
import base64
import math
import cv2
import numpy as np

P1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "01_posture_checker"))
if P1_DIR not in sys.path:
    sys.path.insert(0, P1_DIR)

from pose_detector import MediaPipePoseDetector
from posture_analyzer import PostureAnalyzer

_detector = None
_analyzer = None

def get_detector_and_analyzer():
    global _detector, _analyzer
    if _detector is None:
        _detector = MediaPipePoseDetector(static_image_mode=True, model_complexity=0)
    if _analyzer is None:
        _analyzer = PostureAnalyzer()
    return _detector, _analyzer


def ndarray_to_base64_jpeg(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


def base64_or_bytes_to_rgb(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def analyze_posture(img_bytes: bytes, shoulder_thresh: float = 7.0, neck_thresh: float = 70.0):
    detector, analyzer = get_detector_and_analyzer()
    rgb = base64_or_bytes_to_rgb(img_bytes)
    h, w = rgb.shape[:2]

    analyzer.SHOULDER_TILT_THRESHOLD = float(shoulder_thresh)
    analyzer.NECK_SLOUCH_ANGLE_MIN = float(neck_thresh)

    pose_landmarks, keypoints = detector.process(rgb)
    annotated = rgb.copy()

    if pose_landmarks:
        detector.draw_landmarks(annotated, pose_landmarks)

    is_good, warnings, metrics = analyzer.analyze(keypoints)

    status_color = (34, 197, 94) if is_good else (239, 68, 68)  # Green or Red
    status_label = "GOOD POSTURE" if is_good else "POOR POSTURE"

    # Draw HUD Box
    cv2.rectangle(annotated, (15, 15), (280, 75), (20, 24, 33), -1)
    cv2.rectangle(annotated, (15, 15), (280, 75), status_color, 2)
    cv2.putText(annotated, status_label, (30, 52), cv2.FONT_HERSHEY_DUPLEX, 0.85, status_color, 2)

    return {
        "success": True,
        "is_good": is_good,
        "has_body": bool(keypoints),
        "warnings": warnings,
        "metrics": metrics,
        "annotated_image": f"data:image/jpeg;base64,{ndarray_to_base64_jpeg(annotated)}",
    }

