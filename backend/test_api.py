"""Unit tests for SafeStep AI Night Stair Vision backend endpoints."""

import os
import sys
import asyncio
import base64
import numpy as np
import cv2

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import (
    health,
    detect_stair_api,
)


async def run_all_tests():
    print("=" * 60)
    print("[SafeStep AI Test 1] Healthcheck...")
    h = health()
    assert h["status"] == "healthy"
    assert "SafeStep" in h["service"]
    print(f"  -> Healthcheck OK! Service: {h['service']}")
    print(f"  -> Active Pipeline Stages: {len(h['pipeline'])}")

    print("\n[SafeStep AI Test 2] Night Stair Detection (Descending Mode)...")
    res_stair = await detect_stair_api(gamma=1.6, preferred_mode="descending", enhancement_mode="clahe")
    assert "metrics" in res_stair
    metrics = res_stair["metrics"]
    assert metrics["hazard_level"] in ["SAFE", "CAUTION", "DANGER"]
    assert "confidence_pct" in metrics
    print(f"  -> Stair API OK! Step count: {metrics.get('step_count')}, Hazard: {metrics.get('hazard_level')}")
    print(f"  -> 1st Step Distance: {metrics.get('distance_to_first_step_m')}m, Drop: {metrics.get('step_drop_cm')}cm")

    print("\n[SafeStep AI Test 3] Night Stair Detection with Dummy Camera Frame...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)
    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
    res_frame = await detect_stair_api(image_base64=b64_str, gamma=1.4, preferred_mode="descending")
    assert "metrics" in res_frame
    print("  -> Frame Processing OK! Time:", res_frame["metrics"].get("processing_time_ms"), "ms")

    print("\n" + "=" * 60)
    print(" [SUCCESS] ALL SAFESTEP AI BACKEND TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
