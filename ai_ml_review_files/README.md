# 📂 SafeStep AI - 머신러닝/딥러닝 및 컴퓨터 비전 핵심 코드 리뷰 가이드

이 폴더(`c:\project4\ai_ml_review_files`)는 웹(React/CSS/Vite) 및 프론트엔드 UI를 제외하고, **이미지 전처리, 딥러닝 모델 학습, 모델 추론 및 비전 알고리즘**과 관련된 핵심 코드 및 문서를 검토하기 쉽도록 분류하여 모아둔 디렉토리입니다.

---

## 📁 폴더 구조 및 파일별 역할

### 1️⃣ `01_core_inference_and_preprocessing/` (핵심 추론 & 전처리 파이프라인)
* **`stair_service.py` (프로젝트의 핵심 중추)**:
  - **저조도 이미지 전처리**: Adaptive CLAHE (타일 분할 히스토그램 평활화), 비선형 감마 보정 LUT, Bilateral Filtering(경계선 보존 노이즈 제거).
  - **인공지능 모델 3종 융합 추론**:
    1. **Retinexformer**: 트랜스포머 기반 극저조도 조도 복원.
    2. **YOLO26-Seg (Instance Segmentation)**: 계단 표면 폴리곤 마스크 검출 및 비계단 배경 잡음 100% 마스킹.
    3. **Depth Anything V2 (Small-hf)**: 단안 3D 깊이 추정(Monocular Depth Estimation).
  - **기하학적 단차 분석**: Sobel-Y(수평 경사도 가중), Canny 에지, HoughLinesP(확률적 허프 변환), DBSCAN/find_peaks 단차 클러스터링.
  - **1st Step Quad & 착지점 특정**: 보행자 발밑의 첫 번째 단차(First Step Drop-Off) 사각형 영역 특정 및 국토교통부 표준 기준 위험도(SAFE / CAUTION / DANGER) 판정.
  - **AR HUD 오버레이**: PIL(맑은 고딕 한글) 기반 계단 시작 레이저선, 발 디딤 타깃 박스, 보행 유도 화살표 시각화.
* **`retinexformer_arch.py`**:
  - 저조도 향상 SOTA 모델인 Retinexformer의 PyTorch 신경망 아키텍처 (Illumination-Guided Attention, Multi-scale Feature Extraction).
* **`test_retinexformer_infer.py`**:
  - 사용자 어두운 야간 계단 사진을 대상으로 Retinexformer를 단독 구동 및 전처리 패딩(16배수 패딩/언패딩)하는 테스트 스크립트.
* **`main.py`**:
  - FastAPI 기반 백엔드 API 엔드포인트 (`/api/detect-stair`, `/api/health`). 프론트엔드 카메라 프레임 수신 및 비동기 추론 파이프라인 호출.
* **`test_api.py`**:
  - 백엔드 추론 파이프라인 헬스체크 및 더미 프레임/실제 이미지에 대한 단위 테스트 스크립트.

---

### 2️⃣ `02_model_training/` (모델 학습 및 전이학습)
* **`train_custom_stairs.py` (최종 최신 학습 스크립트)**:
  - 대규모 계단 사전학습 가중치를 베이스로 하여 오르막(`ascending_stairs`)과 내리막(`descending_stairs`) 2개 클래스를 전이학습(Transfer Learning).
  - 회전(15°), 원근 왜곡(0.0005), HSV 명도/채도 증강, 좌우 반전 등 야간/사각도 계단 특화 데이터 증강 적용.
* **`train_night_yolo26.py`**:
  - 인위적으로 합성한 3,216장 규모의 야간 증강 데이터셋(`stairs_night_augmented`) 대상 YOLO26 인스턴스 세그멘테이션 학습.
* **`train_yolo26.py`**:
  - 초기 Roboflow 1,608장 계단 실사 데이터셋 기반 1차 파인튜닝 스크립트 (Box mAP50 84.2%, Mask mAP50 82.0% 달성).
* **`train_jiseonglee.py`**:
  - 계단(`stair`) 및 경사로(`ramp`) 다중 클래스 208장 데이터셋 대상 2차 파인튜닝 스크립트.

---

### 3️⃣ `03_data_preprocessing_and_augmentation/` (데이터 전처리 및 증강)
* **`generate_night_dataset.py` (핵심 전처리/증강 스크립트)**:
  - 주간 실사 계단 사진에 물리 기반의 야간 환경 효과를 자동 합성:
    1. 극단적 밝기 저하 (0.18 ~ 0.42배)
    2. 비선형 감마 암부 압축 (Gamma 1.4 ~ 2.2)
    3. 나트륨 가로등(주황빛) 및 암청색 톤 변환
    4. 비네팅(Vignetting) 효과
    5. ISO 고감도 센서 샷 노이즈 합성
* **`prepare_integrated_dataset.py`**:
  - 3곳의 데이터 소스(자체 데이터, 다운로드 데이터 등 176장)를 통합하고 오르막/내리막 라벨을 매핑하여 8:2 Train/Val 분할.
* **`prepare_dataset.py`**:
  - 이미지 파일과 YOLO 형식의 세그멘테이션 폴리곤 라벨(`.txt`)을 1:1 매칭 검증 및 데이터 분할.
* **`data.yaml`**:
  - YOLO 학습용 데이터셋 경로 및 클래스 메타데이터 정의.

---

### 4️⃣ `04_technical_docs/` (프로젝트 기술 문서 및 회고록)
* **`PROJECT_REPORT.md`**:
  - 프로젝트 최종 보고서 (문제 정의, 전체 아키텍처 다이어그램, 4단계 결함 진단 및 극복 과정, 국토부 안전 기준 메트릭).
* **`STAIR_AI_TECH_LOG.md`**:
  - 기술 회고록 (CV vs DeepLearning 선택 이유, 3D 합성 데이터 Sim-to-Real 갭, 4차례에 걸친 데이터셋 튜닝 시행착오, 하이브리드 비전 파이프라인 설계 배경).

---

### 5️⃣ `05_legacy_reference/` (이전 레거시 프로젝트 참고용)
* **`pose_detector.py` / `posture_analyzer.py`**:
  - MediaPipe 기반 자세 추정 및 관절 각도 분석 이전 프로젝트 코드.
