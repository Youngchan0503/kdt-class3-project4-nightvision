# 🌙 SafeStep AI: Night Stair Detection

**React 19 프론트엔드**와 **FastAPI 백엔드**로 구축된 야간 계단 탐지 & 보행 안전 가이드 컴퓨터 비전 대시보드입니다.

---

## 📂 아키텍처 (Architecture)

```
kdt-class3-project4-nightvision/
├── backend/
│   ├── main.py                  # FastAPI REST API 서버 (Port: 8000)
│   ├── test_api.py              # 백엔드 엔드포인트 단위 테스트
│   ├── weights/                 # Retinexformer LOL-v1 가중치
│   └── services/
│       ├── retinexformer_arch.py # Retinexformer 저조도 개선 모델 구조
│       └── stair_service.py     # 야간 계단 탐지 & 보행 안전 가이드 (CLAHE/Retinex/YOLO-Seg/Depth)
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # 쉘 레이아웃
│   │   ├── components/          # Header, FileUploader
│   │   ├── pages/
│   │   │   ├── StairPage.jsx    # 스마트폰/웹캠/사진 업로드 야간 계단 감지기 (내리막 1인칭 특화)
│   │   │   ├── SafetyGuidePage.jsx # 보행 안전 가이드
│   │   │   └── TechSpecsPage.jsx   # 기술 사양
│   │   └── utils/api.js         # REST API 통신 클라이언트
├── runs/custom_stairs_seg/      # 학습된 YOLO26-Seg 계단 모델 (best.pt)
├── dataset_custom_stairs/       # 커스텀 계단 데이터셋
├── ai_ml_review_files/          # AI/ML 리뷰용 코드 정리본
├── requirements.txt             # Python 의존성
├── start_backend.bat            # 백엔드 서버 원클릭 실행 스크립트 (:8000)
└── start_frontend.bat           # 프론트엔드 개발 서버 원클릭 실행 스크립트 (:5173)
```

---

## 🚀 실행 방법 (Execution Guide)

### 0단계: 최초 환경 세팅 (새 PC에서 clone 후 1회)
```powershell
# 프로젝트 루트에서
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend; npm install
```
* RTX 50 시리즈(Blackwell)는 CUDA 12.8(`cu128`) 이상 빌드의 PyTorch가 필요합니다.

### 1단계: 백엔드 API 서버 실행 (FastAPI)
```powershell
# 프로젝트 루트에서
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
* 또는 탐색기에서 **`start_backend.bat`** 더블 클릭
* API 서버 주소: **`http://127.0.0.1:8000`**
* Swagger 자동 문서: **`http://127.0.0.1:8000/docs`**

### 2단계: 프론트엔드 웹 대시보드 실행 (React + Vite)
```powershell
cd frontend
npm run dev
```
* 또는 탐색기에서 **`start_frontend.bat`** 더블 클릭
* 대시보드 브라우저 주소: **`http://localhost:5173`**
*(Vite 프록시가 설정되어 있어 프론트엔드에서 `/api` 호출 시 백엔드로 자동 연동됩니다)*
