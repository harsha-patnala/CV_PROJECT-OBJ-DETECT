"""
vehicle_attributes.py — Colour estimation for detected vehicles.

Uses K-Means clustering on HSV values within the bounding box to
estimate the dominant visible body colour. Results are approximate.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("vehicle_intelligence.attributes")

# Named colour lookup table (HSV ranges)
# Each entry: (name, h_lo, h_hi, s_lo, v_lo)
_COLOUR_TABLE = [
    ("red",      0,  10, 80,  60),
    ("red",    170, 180, 80,  60),  # red wraps hue
    ("orange",  10,  25, 100, 80),
    ("yellow",  25,  35, 100, 100),
    ("green",   35,  85, 60,  50),
    ("cyan",    85, 100, 60,  60),
    ("blue",   100, 130, 60,  50),
    ("purple", 130, 160, 50,  40),
    ("pink",   160, 170, 50,  100),
    ("white",    0, 180, 0,   200),   # low saturation, high value
    ("silver",   0, 180, 0,   150),   # low saturation, medium value
    ("grey",     0, 180, 0,   80),    # low saturation, low-medium value
    ("black",    0, 180, 0,   0),     # very low value
]

# Ordered fallback lookup
_ORDERED_NAMES = [
    "red", "orange", "yellow", "green", "cyan",
    "blue", "purple", "pink", "white", "silver", "grey", "black",
]


def _closest_colour_name(h: float, s: float, v: float) -> str:
    """Map an HSV triple (0–180, 0–255, 0–255) to a colour name."""
    # Achromatic checks first (low saturation)
    if s < 30:
        if v > 200:
            return "white"
        if v > 130:
            return "silver"
        if v > 60:
            return "grey"
        return "black"

    # Chromatic — match hue range
    for name, h_lo, h_hi, s_lo, v_lo in _COLOUR_TABLE:
        if h_lo <= h <= h_hi and s >= s_lo and v >= v_lo:
            return name

    return "unknown"


def estimate_vehicle_colour(
    frame_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    n_clusters: int = 3,
) -> Optional[str]:
    """
    Estimate the dominant colour of a vehicle within the bounding box.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full BGR frame.
    x1, y1, x2, y2 : int
        Bounding box coordinates.
    n_clusters : int
        Number of K-Means clusters used to find dominant colour.

    Returns
    -------
    str or None
        Colour name, or None if estimation fails.
    """
    try:
        h_frame, w_frame = frame_bgr.shape[:2]
        # Clip to frame bounds
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w_frame, x2)
        y2c = min(h_frame, y2)

        crop = frame_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None

        # Use centre 60% of the crop to avoid background / shadow
        ch, cw = crop.shape[:2]
        if ch < 10 or cw < 10:
            return None
        margin_y = int(ch * 0.2)
        margin_x = int(cw * 0.2)
        centre = crop[margin_y:ch - margin_y, margin_x:cw - margin_x]
        if centre.size == 0:
            centre = crop

        # Resize for speed
        target = cv2.resize(centre, (32, 32), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(n_clusters, len(pixels))
        if k < 1:
            return None
        _, labels, centres = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        # Find the most populated cluster
        counts = np.bincount(labels.flatten())
        dominant = centres[np.argmax(counts)]
        h, s, v = dominant

        return _closest_colour_name(h, s, v)

    except Exception as exc:
        logger.debug(f"Colour estimation failed: {exc}")
        return None
