"""
utils.py — Shared utility functions for Vehicle Intelligence.
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vehicle_intelligence")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vehicle_intelligence.{name}")


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def now_iso() -> str:
    """Current local datetime as ISO 8601 string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def frame_to_timestamp(frame_index: int, fps: float) -> str:
    """Convert frame index to MM:SS.mmm display string."""
    if fps <= 0:
        return "00:00.000"
    total_ms = (frame_index / fps) * 1000
    total_s = int(total_ms // 1000)
    ms = int(total_ms % 1000)
    minutes = total_s // 60
    seconds = total_s % 60
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS.mmm display string."""
    total_ms = int(seconds * 1000)
    total_s = total_ms // 1000
    ms = total_ms % 1000
    minutes = total_s // 60
    secs = total_s % 60
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


# ── Image helpers ─────────────────────────────────────────────────────────────

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def crop_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              pad: int = 4) -> Optional[np.ndarray]:
    """Safely crop a bounding box from a frame with optional padding."""
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ── Temporary file helpers ────────────────────────────────────────────────────

def save_temp_video(uploaded_bytes: bytes, suffix: str = ".mp4") -> str:
    """Write uploaded bytes to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def cleanup_temp_file(path: str) -> None:
    """Delete a temporary file if it exists."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ── Video metadata ────────────────────────────────────────────────────────────

def get_video_info(path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": 0.0,
    }
    if info["fps"] > 0:
        info["duration_s"] = info["frame_count"] / info["fps"]
    cap.release()
    return info


# ── Drawing helpers ───────────────────────────────────────────────────────────

# Colour palette for bounding boxes — cycles through distinct colours by track ID
_BOX_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]


def get_box_colour(track_id: int) -> tuple:
    return _BOX_PALETTE[track_id % len(_BOX_PALETTE)]


def draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    colour: tuple = (0, 200, 80),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a labelled bounding box on a frame (modifies in place)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Background rectangle for label
    label_y1 = max(y1 - th - baseline - 4, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, y1 - baseline - 2),
        font, font_scale, (255, 255, 255), font_thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_counting_line(
    frame: np.ndarray,
    y_pos: int,
    colour: tuple = (234, 170, 0),  # Marigold
    thickness: int = 2,
) -> np.ndarray:
    """Draw a horizontal counting line on a frame."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, y_pos), (w, y_pos), colour, thickness)
    cv2.putText(
        frame, "COUNT LINE",
        (10, y_pos - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
    )
    return frame
