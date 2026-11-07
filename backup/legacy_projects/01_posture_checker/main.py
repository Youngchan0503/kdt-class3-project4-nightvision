import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pose_detector import MediaPipePoseDetector
from posture_analyzer import PostureAnalyzer


def draw_text_korean(image, text, position, font_size=20, color=(255, 255, 255)):
    """OpenCV 이미지(BGR)에 한글 텍스트를 렌더링하는 헬퍼 함수"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Windows 기본 폰트(맑은 고딕) 시도
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/batang.ttc",
    ]
    font = None
    for p in font_paths:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, font_size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # PIL RGB 색상으로 변환
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    print("=" * 60)
    print("실시간 자세 체크 시스템 (MediaPipe Pose)")
    print("종료하려면 영상 창에서 'q' 키를 누르세요.")
    print("=" * 60)

    # 카메라 초기화 (기본 웹캠: 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[에러] 웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인해주세요.")
        return

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = MediaPipePoseDetector()
    analyzer = PostureAnalyzer()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[경고] 카메라 프레임을 읽어올 수 없습니다.")
            break

        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 1. 포즈 감지
        pose_landmarks, keypoints = detector.process(frame)

        # 2. 스켈레톤 시각화
        detector.draw_landmarks(frame, pose_landmarks)

        # 3. 자세 분석
        is_good, warnings, metrics = analyzer.analyze(keypoints)

        # 4. 상단 대시보드 UI 영역 생성 (반투명 오버레이)
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 180), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # FPS 계산
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        # 5. 상태 텍스트 렌더링
        if keypoints:
            if is_good:
                status_text = "● 상태: 바른 자세 (양호)"
                status_color = (0, 255, 100)  # 연초록 (BGR)
            else:
                status_text = "▲ 상태: 불량 자세 주의!"
                status_color = (0, 50, 255)  # 빨강 (BGR)
        else:
            status_text = "○ 사람을 인식하는 중..."
            status_color = (180, 180, 180)

        frame = draw_text_korean(frame, f"FPS: {fps:.1f}", (35, 30), font_size=16, color=(200, 200, 200))
        frame = draw_text_korean(frame, status_text, (35, 55), font_size=22, color=status_color)

        # 수치 정보 표시
        metric_str = f"어깨 기울기: {metrics.get('shoulder_tilt', 0.0)}° | 목 각도: {metrics.get('neck_angle', 0.0)}°"
        frame = draw_text_korean(frame, metric_str, (35, 90), font_size=16, color=(220, 220, 220))

        # 세부 경고 메시지 표시
        y_offset = 120
        if warnings and keypoints:
            for w_msg in warnings[:2]:  # 최대 2줄 표시
                frame = draw_text_korean(frame, f"⚠️ {w_msg}", (35, y_offset), font_size=16, color=(0, 180, 255))
                y_offset += 25
        elif keypoints:
            frame = draw_text_korean(frame, "✔ 허리와 목을 꼿꼿이 유지하고 있습니다.", (35, y_offset), font_size=16, color=(150, 255, 150))

        # 화면 출력
        cv2.imshow("Posture Checker (Press 'q' to exit)", frame)

        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()

