"""
video_processor.py — Frame-by-frame video analysis pipeline.

Orchestrates detection, tracking, colour estimation, optional plate OCR,
counting-line logic, and record management.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np

from analytics import SessionAnalytics
from detector import Detection, VehicleDetector
from utils import (
    bgr_to_rgb,
    draw_counting_line,
    draw_detection,
    frame_to_timestamp,
    get_box_colour,
    get_video_info,
    resize_frame,
)
from vehicle_attributes import estimate_vehicle_colour

logger = logging.getLogger("vehicle_intelligence.video_processor")


@dataclass
class ProcessingConfig:
    """Runtime configuration for a processing session."""
    confidence: float = 0.35
    iou: float = 0.45
    use_tracking: bool = True
    tracker: str = "bytetrack.yaml"
    estimate_colour: bool = True
    run_plate_ocr: bool = False
    use_counting_line: bool = False
    counting_line_y_frac: float = 0.5   # fraction of frame height (0–1)
    counting_direction: str = "both"    # "up", "down", or "both"
    skip_frames: int = 1                # process every (skip+1)-th frame; 1 = every other
    max_width: int = 960                # resize input for speed
    db_path: Optional[Path] = None      # override SQLite path (tests)


@dataclass
class FrameResult:
    """Annotated frame and per-frame stats returned to the caller."""
    frame_rgb: np.ndarray
    detections: List[Detection]
    frame_index: int
    total_frames: int
    fps: float
    unique_vehicles: int
    class_counts: dict
    line_crossings: int
    video_timestamp: str
    processing_fps: float = 0.0


class VideoProcessor:
    """
    Processes an uploaded video file frame by frame.

    Usage
    -----
    processor = VideoProcessor(detector, config)
    for result in processor.process(video_path):
        # result is a FrameResult — render it in the UI
    """

    def __init__(self, detector: VehicleDetector, config: ProcessingConfig):
        self.detector = detector
        self.config = config
        self._session: Optional[SessionAnalytics] = None

    def process(
        self, video_path: str
    ) -> Generator[FrameResult, None, None]:
        """Yield annotated FrameResult objects for each processed frame."""
        info = get_video_info(video_path)
        fps = info.get("fps", 25.0) or 25.0
        total_frames = info.get("frame_count", 0)

        session_id = f"upload_{uuid.uuid4().hex[:8]}"
        session = SessionAnalytics(
            session_id=session_id,
            source="upload",
            persist=True,
            db_path=self.config.db_path,
        )
        self._session = session

        self.detector.reset_tracker()

        # Track previous y-centre positions for counting line direction
        prev_centres: dict[int, int] = {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        frame_index = 0
        processed_count = 0
        proc_start = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1

                # Optional frame skipping — tracker persist=True keeps IDs across gaps
                if self.config.skip_frames > 0 and (frame_index - 1) % (self.config.skip_frames + 1) != 0:
                    continue

                # Resize for speed
                frame = resize_frame(frame, self.config.max_width)

                video_ts = frame_to_timestamp(frame_index, fps)

                # Detection or tracking
                if self.config.use_tracking:
                    detections = self.detector.track(frame, tracker=self.config.tracker)
                else:
                    detections = self.detector.detect(frame)

                annotated = frame.copy()
                line_y = int(frame.shape[0] * self.config.counting_line_y_frac)

                for det in detections:
                    track_id = det.track_id if det.track_id is not None else -(frame_index * 1000)
                    abs_id = abs(track_id)
                    colour_box = get_box_colour(abs_id)

                    # Colour / OCR only until a value is known for this track
                    est_colour = session.colour_for(abs_id)
                    if self.config.estimate_colour and not est_colour:
                        est_colour = estimate_vehicle_colour(
                            frame, det.x1, det.y1, det.x2, det.y2
                        )

                    plate_text = session.plate_for(abs_id)
                    if self.config.run_plate_ocr and not plate_text:
                        from plate_recognition import attempt_plate_ocr
                        plate_text = attempt_plate_ocr(
                            frame, det.x1, det.y1, det.x2, det.y2
                        )

                    session.record_detection(
                        track_id=abs_id,
                        vehicle_class=det.class_name,
                        confidence=det.confidence,
                        video_timestamp=video_ts,
                        est_colour=est_colour,
                        plate_text=plate_text,
                    )

                    # Counting line logic
                    if self.config.use_counting_line and det.track_id is not None:
                        cy = det.center[1]
                        prev_cy = prev_centres.get(det.track_id)
                        if prev_cy is not None:
                            crossed = False
                            if self.config.counting_direction in ("down", "both"):
                                crossed = crossed or (prev_cy < line_y <= cy)
                            if self.config.counting_direction in ("up", "both"):
                                crossed = crossed or (prev_cy > line_y >= cy)
                            if crossed:
                                session.record_line_crossing(
                                    abs(det.track_id), video_ts
                                )
                        prev_centres[det.track_id] = cy

                    # Build label
                    label_parts = [f"{det.class_name}"]
                    if det.track_id is not None:
                        label_parts.append(f"#{det.track_id}")
                    label_parts.append(f"{det.confidence:.2f}")
                    if est_colour:
                        label_parts.append(est_colour)
                    label = " ".join(label_parts)

                    draw_detection(
                        annotated,
                        det.x1, det.y1, det.x2, det.y2,
                        label,
                        colour=colour_box,
                    )

                if self.config.use_counting_line:
                    draw_counting_line(annotated, line_y)

                processed_count += 1
                elapsed = time.perf_counter() - proc_start
                processing_fps = processed_count / elapsed if elapsed > 0 else 0.0

                yield FrameResult(
                    frame_rgb=bgr_to_rgb(annotated),
                    detections=detections,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    fps=fps,
                    unique_vehicles=session.unique_vehicles,
                    class_counts=dict(session.class_counts()),
                    line_crossings=session.line_crossing_count,
                    video_timestamp=video_ts,
                    processing_fps=processing_fps,
                )

        finally:
            cap.release()

        logger.info(
            f"Processing complete — {frame_index} frames, "
            f"{session.unique_vehicles} unique vehicles."
        )

    @property
    def session(self) -> Optional[SessionAnalytics]:
        return self._session
