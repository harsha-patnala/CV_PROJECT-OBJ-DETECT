"""
plate_recognition.py — Optional number plate detection and OCR.

Uses EasyOCR for text extraction on cropped regions.
A dedicated plate detector (e.g. a YOLO model trained on plates) is
required for reliable detection; this module falls back to attempting
OCR on the full vehicle crop when no plate detector is available.

Status: PARTIALLY IMPLEMENTED — OCR is functional but requires a
dedicated plate-detector model for reliable plate localisation.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("vehicle_intelligence.plate_ocr")

# --------------------------------------------------------------------------- #
# Optional EasyOCR import
# --------------------------------------------------------------------------- #
_easyocr_reader = None
_easyocr_available = False

try:
    import easyocr  # noqa: F401
    _easyocr_available = True
except ImportError:
    logger.info("EasyOCR not installed — plate OCR disabled.")


def is_ocr_available() -> bool:
    return _easyocr_available


def _get_reader():
    """Lazily initialise EasyOCR reader (downloads models on first use)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR reader initialised.")
    return _easyocr_reader


# --------------------------------------------------------------------------- #
# Plate detection helpers
# --------------------------------------------------------------------------- #

def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance a plate crop for better OCR accuracy."""
    # Upscale small crops
    h, w = crop.shape[:2]
    if w < 100:
        scale = 100 / w
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle lighting variation
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    return thresh


_PLATE_PATTERN = re.compile(
    r"[A-Z0-9]{2,4}[\s\-]?[A-Z0-9]{2,4}[\s\-]?[A-Z0-9]{2,4}",
    re.IGNORECASE,
)


def _looks_like_plate(text: str) -> bool:
    """Rough sanity check that extracted text resembles a plate number."""
    cleaned = text.strip().upper().replace(" ", "").replace("-", "")
    if len(cleaned) < 4 or len(cleaned) > 12:
        return False
    alpha = sum(c.isalpha() for c in cleaned)
    digit = sum(c.isdigit() for c in cleaned)
    return alpha >= 1 and digit >= 1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def attempt_plate_ocr(
    frame_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> Optional[str]:
    """
    Attempt to read a number plate from within a vehicle bounding box.

    This function works best with a dedicated plate-detector; without one
    it attempts OCR on the lower portion of the vehicle crop.

    Returns the extracted text or None if recognition fails / is unavailable.

    NOTE: Results are approximate. False positives and misreads are expected,
    especially on small, blurry, or partially visible plates.
    """
    if not _easyocr_available:
        return None

    try:
        h_frame, w_frame = frame_bgr.shape[:2]
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w_frame, x2)
        y2c = min(h_frame, y2)

        crop = frame_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None

        # Use the lower third of the bounding box where plates typically appear
        ch = crop.shape[0]
        plate_region = crop[int(ch * 0.55):, :]
        if plate_region.size == 0:
            plate_region = crop

        processed = _preprocess_plate_crop(plate_region)
        reader = _get_reader()
        results = reader.readtext(processed, detail=0, paragraph=False)

        for text in results:
            cleaned = text.strip()
            if _looks_like_plate(cleaned):
                logger.debug(f"Plate candidate: '{cleaned}'")
                return cleaned.upper()

        return None

    except Exception as exc:
        logger.debug(f"Plate OCR error: {exc}")
        return None
