import cv2
import mediapipe as mp


class MediaPipePoseDetector:
    """MediaPipe를 이용한 고속 실시간 신체 랜드마크 검출기 (경량 최적화)"""

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=0,  # 0: 초경량 고속 모드 (실시간 웹캠용), 1: 일반 모드
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame):
        """
        RGB 프레임을 입력받아 랜드마크를 추출합니다.
        (Gradio 및 웹 스트림은 이미 RGB 형식이므로 추가 색변환 없이 즉시 처리)
        """
        h, w, _ = rgb_frame.shape
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None, {}

        landmarks = results.pose_landmarks.landmark
        keypoints = {}

        def to_point(lm):
            return {"x": int(lm.x * w), "y": int(lm.y * h), "vis": lm.visibility}

        mp_idx = self.mp_pose.PoseLandmark
        keypoints["left_ear"] = to_point(landmarks[mp_idx.LEFT_EAR.value])
        keypoints["right_ear"] = to_point(landmarks[mp_idx.RIGHT_EAR.value])
        keypoints["left_shoulder"] = to_point(landmarks[mp_idx.LEFT_SHOULDER.value])
        keypoints["right_shoulder"] = to_point(landmarks[mp_idx.RIGHT_SHOULDER.value])
        keypoints["left_hip"] = to_point(landmarks[mp_idx.LEFT_HIP.value])
        keypoints["right_hip"] = to_point(landmarks[mp_idx.RIGHT_HIP.value])

        return results.pose_landmarks, keypoints

    def draw_landmarks(self, rgb_frame, pose_landmarks):
        """RGB 프레임 위에 관절 랜드마크 및 연결선 시각화"""
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                rgb_frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
