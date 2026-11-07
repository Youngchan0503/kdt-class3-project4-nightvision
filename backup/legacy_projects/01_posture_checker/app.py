# -*- coding: utf-8 -*-
"""Interactive Web Application for AI Posture Checker (High-FPS Optimized).

Performance Optimizations:
1. Downscales frame resolution to 640x360 for zero-latency streaming.
2. Lightweight MediaPipe model (complexity=0) for 60+ FPS inference.
3. Native RGB pipeline (fixes avatar-blue skin color issue).
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import time
import cv2
import numpy as np
from PIL import Image
import gradio as gr

from pose_detector import MediaPipePoseDetector
from posture_analyzer import PostureAnalyzer


# Initialize lightweight, ultra-fast detector (model_complexity=0)
detector = MediaPipePoseDetector(static_image_mode=False, model_complexity=0)
analyzer = PostureAnalyzer()


def process_posture_frame(frame: np.ndarray, shoulder_thresh: float, neck_thresh: float):
    """Processes a video/photo frame with zero-latency optimizations."""
    if frame is None:
        return None, "⚪ 카메라 입력 대기 중", "카메라를 켜거나 프레임을 캡처해주세요.", ""

    # Adjust analyzer thresholds
    analyzer.SHOULDER_TILT_THRESHOLD = float(shoulder_thresh)
    analyzer.NECK_SLOUCH_ANGLE_MIN = float(neck_thresh)

    # 1. Performance Downscaling: Resize to max width 640 for lightning-fast transmission
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640.0 / w
        target_w = 640
        target_h = int(h * scale)
        proc_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        proc_frame = frame.copy()

    # Note: Gradio image input is ALREADY native RGB!
    # 2. Pose detection
    pose_landmarks, keypoints = detector.process(proc_frame)

    # 3. Draw skeleton landmarks in RGB
    if pose_landmarks:
        detector.draw_landmarks(proc_frame, pose_landmarks)

    # 4. Analyze posture metrics
    is_good, warnings, metrics = analyzer.analyze(keypoints)

    # 5. Status determination & formatted markdown
    if not keypoints:
        status_badge = "⚪ 신체 인식 대기 중"
        summary_md = "⚠️ 상체(양쪽 어깨와 귀)가 화면에 잘 보이도록 카메라 정면에 앉아주세요."
        guide_text = "• 어깨와 양쪽 귀가 카메라 화각 내에 포함되어야 정확한 측정이 가능합니다."
    elif is_good:
        status_badge = "🟢 바른 자세 (양호)"
        s_tilt = metrics.get("shoulder_tilt", 0.0)
        n_angle = metrics.get("neck_angle", 90.0)
        summary_md = f"""
        ### 🌟 아주 훌륭하고 바른 자세입니다!
        * **어깨 수평 기울기**: `{s_tilt:.1f}°` (기준: {shoulder_thresh:.1f}° 이내 정상)
        * **목-어깨 직립 각도**: `{n_angle:.1f}°` (기준: {neck_thresh:.1f}° 이상 정상)
        """
        guide_text = "현재 허리와 목이 바르게 펴져 있습니다. 이 자세를 유지해 주세요!"
    else:
        status_badge = "🔴 불량 자세 주의!"
        s_tilt = metrics.get("shoulder_tilt", 0.0)
        n_angle = metrics.get("neck_angle", 90.0)

        warning_list = "\n".join([f"* ⚠️ **{w}**" for w in warnings])
        summary_md = f"""
        ### 🚨 자세 교정이 필요합니다!
        {warning_list}

        ---
        * **현재 어깨 기울기**: `{s_tilt:.1f}°`
        * **현재 목 각도**: `{n_angle:.1f}°`
        """
        guide_text = "• 턱을 살짝 목 쪽으로 당기고, 양쪽 어깨의 힘을 뺀 뒤 수평을 맞춰주세요.\n• 모니터 높이를 눈높이와 맞추면 거북목 예방에 큰 도움이 됩니다."

    # 6. Draw on-image HUD overlay (Colors are in RGB: [R, G, B])
    cv2.rectangle(proc_frame, (12, 12), (360, 72), (20, 20, 20), -1)
    text_color = (100, 240, 100) if is_good else (255, 80, 80)
    clean_status = "Status: GOOD POSTURE" if is_good else "Warning: POOR POSTURE"
    if not keypoints:
        clean_status = "Detecting Human Pose..."
        text_color = (180, 180, 180)

    cv2.putText(proc_frame, clean_status, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, text_color, 2, cv2.LINE_AA)
    if keypoints:
        cv2.putText(
            proc_frame,
            f"Shoulder: {metrics.get('shoulder_tilt', 0):.1f}deg | Neck: {metrics.get('neck_angle', 0):.1f}deg",
            (22, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # Return pure RGB (No BGR color flipping!)
    return proc_frame, status_badge, summary_md, guide_text


CUSTOM_CSS = """
.gradio-container {
    max-width: 1360px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.hero-banner {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
    color: white;
    padding: 24px 30px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -4px rgba(6, 95, 70, 0.25);
}
.hero-title {
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 6px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-desc {
    font-size: 14px;
    color: #a7f3d0;
    line-height: 1.5;
}

.analyze-btn {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
    width: 100% !important;
    margin-top: 8px !important;
}
"""


def build_app():
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        radius_size="md",
    )

    with gr.Blocks(title="AI 실시간 자세 교정 대시보드 (고속 최적화)", theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="hero-banner">
                <div class="hero-title">
                    <span>🧘</span> AI 실시간 자세 교정 헬스케어 스튜디오
                </div>
                <div class="hero-desc">
                    MediaPipe Pose 초경량 모델을 통해 <b>거북목(Forward Head Angle)</b> 및 <b>어깨 비대칭(Shoulder Tilt)</b>을
                    실시간 지연(렉) 없이 정밀 측정하고 피드백을 제공합니다.
                </div>
            </div>
            """
        )

        with gr.Row():
            # LEFT: Input Source & Settings
            with gr.Column(scale=6):
                with gr.Group():
                    gr.Markdown("### 📷 1. 입력 모드 (실시간 웹캠 / 사진 업로드)")

                    with gr.Tabs():
                        with gr.TabItem("🎥 실시간 웹캠 모드"):
                            webcam_input = gr.Image(
                                label="웹캠 화면",
                                sources=["webcam"],
                                streaming=True,
                                type="numpy",
                                height=360,
                            )
                            btn_capture_analyze = gr.Button(
                                "⚡ [원클릭] 현재 자세 분석하기",
                                variant="primary",
                                elem_classes=["analyze-btn"],
                            )
                            gr.Markdown(
                                "💡 **안내**: 하단 **[중지/녹음]**으로 스트리밍을 켜고 끌 수 있으며, "
                                "위의 **`[⚡ 현재 자세 분석하기]`**를 누르면 즉시 프레임이 갱신됩니다."
                            )

                        with gr.TabItem("🖼️ 사진 파일 분석 모드"):
                            photo_input = gr.Image(
                                label="앉은 자세 사진 업로드",
                                sources=["upload", "clipboard"],
                                type="numpy",
                                height=360,
                            )
                            btn_analyze_photo = gr.Button(
                                "🔍 업로드된 사진 정밀 분석",
                                variant="primary",
                                elem_classes=["analyze-btn"],
                            )

                with gr.Accordion("⚙️ 판정 임계치 세부 조절", open=False):
                    shoulder_slider = gr.Slider(
                        3.0, 15.0, value=7.0, step=0.5,
                        label="어깨 비대칭 허용 각도 (°)"
                    )
                    neck_slider = gr.Slider(
                        50.0, 85.0, value=70.0, step=1.0,
                        label="거북목 경고 기준 각도 (°, 낮을수록 고개 숙임)"
                    )

            # RIGHT: Visual Dashboard & Analysis
            with gr.Column(scale=6):
                with gr.Group():
                    gr.Markdown("### 📊 2. 실시간 스켈레톤 및 자세 분석 뷰")
                    out_image = gr.Image(label="관절 랜드마크 시각화", interactive=False, height=360)

                    with gr.Row():
                        out_status = gr.Textbox(label="현재 자세 상태", interactive=False, scale=2)

                    out_summary = gr.Markdown("자세를 감지하면 상세 분석 보고서가 나타납니다.")
                    out_guide = gr.Markdown("")

        # Live stream event with concurrency optimization
        webcam_input.stream(
            fn=process_posture_frame,
            inputs=[webcam_input, shoulder_slider, neck_slider],
            outputs=[out_image, out_status, out_summary, out_guide],
            time_limit=300,
            stream_every=0.08,  # ~12.5 FPS target for buttery smooth web stream
        )

        btn_capture_analyze.click(
            fn=process_posture_frame,
            inputs=[webcam_input, shoulder_slider, neck_slider],
            outputs=[out_image, out_status, out_summary, out_guide],
        )

        btn_analyze_photo.click(
            fn=process_posture_frame,
            inputs=[photo_input, shoulder_slider, neck_slider],
            outputs=[out_image, out_status, out_summary, out_guide],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.queue(max_size=3)  # Drops old queued frames to eliminate stream lag
    print("[Posture Checker] High-FPS Ready on http://127.0.0.1:7862 ...")
    demo.launch(server_name="127.0.0.1", server_port=7862)
