# -*- coding: utf-8 -*-
"""SafeStep AI - Night Stair Vision API Backend.

Exposes REST APIs for:
- Low-light Enhancement (Retinexformer / CLAHE)
- Stair Instance Segmentation (YOLO26-Seg)
- Monocular Depth Estimation (Depth Anything V2)
- First-Step Hazard Assessment & Real-time AR Guidance
"""

import os
import sys
import base64
from typing import Optional
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# Add backend directory to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services import stair_service

app = FastAPI(
    title="SafeStep AI - Night Stair Vision API",
    description="REST API server powering SafeStep Night-time Stair Detection & Fall Prevention System",
    version="2.1.0",
)

# Enable CORS for Vite React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "SafeStep AI Night Stair Vision Backend",
        "primary_module": "야간 계단 탐지 & 보행 안전 가이드 (Night Stair Safety AI)",
        "pipeline": [
            "Retinexformer / Adaptive CLAHE Low-Light Enhancement",
            "YOLO26-Seg Stair Treads Multi-Polygon Segmentation",
            "Monocular Metric Depth Anything V2",
            "First Step Quad Geometric Hazard Engine",
        ],
    }


# ==========================================
# Core Module: Night-time Stair Detection & Safety Guide
# ==========================================
@app.post("/api/stair/detect")
async def detect_stair_api(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    gamma: float = Form(1.6),
    stair_type_sample: str = Form("descending"),
    preferred_mode: str = Form("descending"),
    enhancement_mode: str = Form("auto"),
):
    try:
        image_bgr = None
        if file is not None:
            content = await file.read()
            arr = np.frombuffer(content, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_base64)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        res = stair_service.analyze_night_stair(
            image_bgr=image_bgr,
            gamma=float(gamma),
            stair_type_sample=stair_type_sample,
            preferred_mode=preferred_mode,
            enhancement_mode=enhancement_mode,
        )
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ==========================================
# Serve Built React Frontend SPA
# ==========================================
DIST_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend", "dist"))
ASSETS_DIR = os.path.join(DIST_DIR, "assets")

if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    index_file = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SafeStep AI Night Stair Vision Backend Ready. Run frontend with 'npm run dev' or build with 'npm run build'."}


if __name__ == "__main__":
    import uvicorn
    print("[SafeStep AI Server] Launching on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
