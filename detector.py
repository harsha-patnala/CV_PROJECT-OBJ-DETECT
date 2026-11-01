"""
detector.py — Vehicle detection using Ultralytics YOLO.

Supported classes (COCO): car(2), motorcycle(3), bus(5), truck(7)

Performance notes
-----------------
* Model is loaded once per process via @st.cache_resource in app.py.
* torch.inference_mode() is used in detect/track to avoid gradient overhead.
* imgsz is configurable to trade accuracy for speed.
* stream=True avoids holding all result tensors in memory simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("vehicle_intelligence.detector")

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Default model weights
DEFAULT_MODEL = "yolov8n.pt"

# Quality preset → inference size, default skip, and capture max-width
QUALITY_PRESETS = {
    "Fast":          {"imgsz": 480, "skip_frames": 2, "max_width": 640},
    "Balanced":      {"imgsz": 640, "skip_frames": 1, "max_width": 960},
    "Higher Detail": {"imgsz": 800, "skip_frames": 0, "max_width": 1280},
}


def preset_config(name: str) -> dict:
    """Return a quality-preset dict; unknown names fall back to Balanced."""
    return QUALITY_PRESETS.get(name, QUALITY_PRESETS["Balanced"])


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class VehicleDetector:
    """
    Wraps Ultralytics YOLO to detect and (optionally) track vehicles.

    Parameters
    ----------
    model_path : str
        Path to YOLO weights file (.pt). Auto-downloaded if not present.
    confidence : float
        Minimum detection confidence threshold (0–1).
    iou : float
        NMS IoU threshold.
    device : str
        'cpu', 'mps' (Apple Silicon), or '0' for CUDA.
    imgsz : int
        Inference image size. Smaller = faster. 480/640/800 are common.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            try:
                self._model.fuse()
            except Exception:
                pass
            # Warm up the model to avoid first-frame latency spike
            _dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model.predict(
                source=_dummy, conf=self.confidence, iou=self.iou,
                classes=list(VEHICLE_CLASS_IDS), device=self.device,
                imgsz=self.imgsz, verbose=False, stream=False, max_det=30,
            )
            logger.info(f"YOLO model loaded and warmed up: {self.model_path} device={self.device} imgsz={self.imgsz}")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            raise RuntimeError(
                f"Could not load YOLO model '{self.model_path}'. "
                "Ensure ultralytics is installed and the model weights are available."
            ) from exc

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame. Returns vehicle detections only."""
        if self._model is None:
            return []
        try:
            import torch
            with torch.inference_mode():
                results = self._model.predict(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou,
                    classes=list(VEHICLE_CLASS_IDS),
                    device=self.device,
                    imgsz=self.imgsz,
                    verbose=False,
                    stream=False,
                    max_det=30,
                )
            return self._parse_results(results)
        except Exception as exc:
            logger.warning(f"Detection error: {exc}")
            return []

    def track(self, frame: np.ndarray, tracker: str = "bytetrack.yaml") -> List[Detection]:
        """Run inference with integrated YOLO tracking on a single BGR frame."""
        if self._model is None:
            return []
        try:
            import torch
            with torch.inference_mode():
                results = self._model.track(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou,
                    classes=list(VEHICLE_CLASS_IDS),
                    device=self.device,
                    imgsz=self.imgsz,
                    tracker=tracker,
                    persist=True,
                    verbose=False,
                    stream=False,
                    max_det=30,
                )
            return self._parse_results(results, with_track=True)
        except Exception as exc:
            logger.warning(f"Tracking error: {exc}")
            return []

    def reset_tracker(self) -> None:
        """Reset tracking state between video sessions without reloading weights."""
        if self._model is None:
            return
        try:
            self._model.reset_tracker()
            logger.debug("Tracker reset.")
        except Exception:
            try:
                predictor = getattr(self._model, "predictor", None)
                if predictor is not None:
                    predictor.trackers = []
                    if hasattr(predictor, "vid_path"):
                        predictor.vid_path = [None]
            except Exception as exc:
                logger.debug(f"Tracker reset skipped: {exc}")

    def _parse_results(self, results, with_track: bool = False) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            try:
                xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
                cls_all  = boxes.cls.cpu().numpy().astype(int)
                conf_all = boxes.conf.cpu().numpy()
                ids_all  = boxes.id.cpu().numpy().astype(int) if (with_track and boxes.id is not None) else None
            except Exception as e:
                logger.debug(f"Batch parse error, falling back to per-box: {e}")
                xyxy_all = cls_all = conf_all = ids_all = None

            if xyxy_all is not None:
                for i in range(len(xyxy_all)):
                    cls_id = int(cls_all[i])
                    if cls_id not in VEHICLE_CLASS_IDS:
                        continue
                    xyxy = xyxy_all[i]
                    conf = float(conf_all[i])
                    track_id = int(ids_all[i]) if ids_all is not None else None
                    detections.append(Detection(
                        x1=int(xyxy[0]), y1=int(xyxy[1]),
                        x2=int(xyxy[2]), y2=int(xyxy[3]),
                        class_id=cls_id,
                        class_name=VEHICLE_CLASS_NAMES.get(cls_id, "vehicle"),
                        confidence=conf,
                        track_id=track_id,
                    ))
            else:
                # Per-box fallback
                for i, box in enumerate(boxes):
                    try:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if cls_id not in VEHICLE_CLASS_IDS:
                            continue
                        track_id = None
                        if with_track and box.id is not None:
                            track_id = int(box.id[0].item())
                        detections.append(Detection(
                            x1=int(xyxy[0]), y1=int(xyxy[1]),
                            x2=int(xyxy[2]), y2=int(xyxy[3]),
                            class_id=cls_id,
                            class_name=VEHICLE_CLASS_NAMES.get(cls_id, "vehicle"),
                            confidence=conf,
                            track_id=track_id,
                        ))
                    except Exception as e:
                        logger.debug(f"Skipping malformed detection box: {e}")
        return detections
