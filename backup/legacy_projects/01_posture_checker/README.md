# 01. AI 실시간 자세 교정 헬스케어 스튜디오 (Posture Checker)

컴퓨터 웹캠 또는 사진 파일을 활용하여 실시간으로 사용자의 신체 관절을 추적하고, **거북목(Forward Head Posture)** 및 **어깨 비대칭(Shoulder Tilt)** 등을 정밀 분석하여 시각적 피드백을 제공하는 프로젝트입니다.

---

## 📁 디렉터리 구성

```text
01_posture_checker/
├── app.py                # 🌐 [NEW] 인터랙티브 웹 대시보드 (Gradio)
├── main.py               # 데스크톱 실시간 OpenCV 창 실행 스크립트
├── pose_detector.py      # 포즈 추정기 (MediaPipe Pose)
├── posture_analyzer.py   # 신체 각도 계산 및 불량 자세 판정 로직
├── requirements.txt      # 의존 라이브러리 목록
└── README.md             # 사용 가이드
```

---

## 🚀 실행 방법

### 방법 A: 웹 대시보드 실행 (추천)
브라우저 환경에서 **실시간 웹캠 스트리밍** 또는 **사진 업로드 정밀 진단**을 사용할 수 있습니다:

```bash
cd 01_posture_checker
python app.py
```
* 브라우저에서 `http://127.0.0.1:7862` 접속

---

### 방법 B: 데스크톱 OpenCV 창으로 실행
```bash
cd 01_posture_checker
python main.py
```
* 종료: 영상 창 활성화 상태에서 키보드 `q` 키
