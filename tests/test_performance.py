"""Measurable inference timing using the local yolov8n weights.

Does not claim real-time performance. Skips if weights are missing.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "yolov8n.pt"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="yolov8n.pt not present")
def test_dummy_frame_inference_completes():
    from detector import VehicleDetector

    os.chdir(ROOT)
    detector = VehicleDetector(model_path=str(WEIGHTS), confidence=0.35, imgsz=480, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Warm-up already ran in constructor; time a few extra frames
    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        detector.detect(frame)
    elapsed = time.perf_counter() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    print(f"\nMeasured dummy-frame detect FPS (cpu, imgsz=480, {n} frames): {fps:.2f}")
    assert elapsed > 0
    detector.reset_tracker()
