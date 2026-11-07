import math


class PostureAnalyzer:
    """관절 좌표를 기반으로 앉은 자세 및 상태를 판별하는 분석기"""

    def __init__(self):
        # 판별 임계값 (상황에 따라 튜닝 가능)
        self.SHOULDER_TILT_THRESHOLD = 7.0  # 어깨 기울기 각도 (도)
        self.NECK_SLOUCH_ANGLE_MIN = 70.0  # 목-어깨 각도 정상 하한선 (도)
        self.HEAD_DROP_RATIO = 0.5  # 목 움츠러듦 판정 기준

    @staticmethod
    def calculate_angle_2d(p1, p2):
        """두 점 (x, y) 사이의 각도(수평선 기준, 도 단위) 계산"""
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        rad = math.atan2(dy, dx)
        deg = math.degrees(rad)
        return abs(deg)

    def analyze(self, keypoints):
        """
        keypoints 딕셔너리를 받아 자세 상태와 경고 메시지 목록을 반환합니다.
        반환값:
            is_good_posture (bool): 올바른 자세 여부
            warnings (list of str): 감지된 불량 자세 경고 메시지
            metrics (dict): 계산된 주요 수치값
        """
        warnings = []
        metrics = {}

        if not keypoints:
            return False, ["인식된 신체가 없습니다."], metrics

        l_shoulder = keypoints.get("left_shoulder")
        r_shoulder = keypoints.get("right_shoulder")
        l_ear = keypoints.get("left_ear")
        r_ear = keypoints.get("right_ear")

        # 1. 어깨 수평도 (Shoulder Tilt) 체크
        if l_shoulder and r_shoulder and l_shoulder["vis"] > 0.5 and r_shoulder["vis"] > 0.5:
            # 수평선과의 기울기 각도 (0도에 가까울수록 수평)
            dx = r_shoulder["x"] - l_shoulder["x"]
            dy = r_shoulder["y"] - l_shoulder["y"]
            shoulder_angle = abs(math.degrees(math.atan2(dy, dx)))
            # 180도 근처면 수평이므로 오차 계산
            tilt_angle = min(shoulder_angle, abs(180 - shoulder_angle))
            metrics["shoulder_tilt"] = round(tilt_angle, 1)

            if tilt_angle > self.SHOULDER_TILT_THRESHOLD:
                if l_shoulder["y"] < r_shoulder["y"]:
                    warnings.append(f"왼쪽 어깨가 올라감 (기울기: {tilt_angle:.1f}°)")
                else:
                    warnings.append(f"오른쪽 어깨가 올라감 (기울기: {tilt_angle:.1f}°)")

        # 2. 목 각도 / 거북목 (Forward Head Posture) 체크
        # 시야에서 더 잘 보이는 쪽 귀-어깨 쌍 선택
        side = None
        ear, shoulder = None, None
        if l_ear and l_shoulder and l_ear["vis"] > 0.5 and l_shoulder["vis"] > 0.5:
            if not r_ear or not r_shoulder or l_ear["vis"] >= r_ear["vis"]:
                side = "Left"
                ear, shoulder = l_ear, l_shoulder
        if not ear and r_ear and r_shoulder and r_ear["vis"] > 0.5 and r_shoulder["vis"] > 0.5:
            side = "Right"
            ear, shoulder = r_ear, r_shoulder

        if ear and shoulder:
            # 귀와 어깨가 이루는 수직선 기준 각도 (90도가 완벽한 수직 자세)
            # x 차이가 크면 머리가 앞으로 나온 거북목 상태
            dx = abs(ear["x"] - shoulder["x"])
            dy = abs(shoulder["y"] - ear["y"])  # y는 아래로 갈수록 커짐

            if dy > 0:
                neck_angle = math.degrees(math.atan2(dy, dx))
                metrics["neck_angle"] = round(neck_angle, 1)

                if neck_angle < self.NECK_SLOUCH_ANGLE_MIN:
                    warnings.append(f"거북목/고개 숙임 주의 (각도: {neck_angle:.1f}°)")

        # 종합 판정
        is_good_posture = len(warnings) == 0

        return is_good_posture, warnings, metrics

