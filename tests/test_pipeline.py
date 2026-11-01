"""Video pipeline, skip-frame, and persistence tests."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import pytest

from analytics import SessionAnalytics
from database import load_all_records
from detector import Detection, preset_config
from video_processor import ProcessingConfig, VideoProcessor


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def reset_tracker(self):
        pass

    def track(self, frame, tracker=None):
        self.calls += 1
        h, w = frame.shape[:2]
        return [Detection(10, 10, min(80, w), min(80, h), 2, "car", 0.9, track_id=1)]

    def detect(self, frame):
        return self.track(frame)


def _write_dummy_video(path: Path, n_frames: int = 10, fps: int = 10) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120)
    )
    for i in range(n_frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:] = (i * 20) % 255
        writer.write(frame)
    writer.release()


def test_preset_config_keys():
    cfg = preset_config("Balanced")
    assert cfg["imgsz"] == 640
    assert "skip_frames" in cfg
    unknown = preset_config("nope")
    assert unknown["imgsz"] == 640


def test_frame_skip_reduces_inference(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_dummy_video(video, n_frames=10)
    detector = FakeDetector()
    config = ProcessingConfig(skip_frames=1, estimate_colour=False, use_tracking=True)
    processor = VideoProcessor(detector, config)
    results = list(processor.process(str(video)))
    # 10 frames, skip=1 → process frames 1,3,5,7,9 → 5 inferences
    assert detector.calls == 5
    assert len(results) == 5
    assert processor.session.unique_vehicles == 1


def test_video_session_source_is_up_vid(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_dummy_video(video, n_frames=4)
    db_path = tmp_path / "pipe.db"

    detector = FakeDetector()
    config = ProcessingConfig(skip_frames=0, estimate_colour=False, db_path=db_path)
    processor = VideoProcessor(detector, config)
    list(processor.process(str(video)))
    df = load_all_records(db_path)
    assert not df.empty
    assert set(df["source"].unique()) == {"Up Vid"}
    assert len(df) == 1  # one unique track


def test_session_persist_survives_new_connection(tmp_path):
    db_path = tmp_path / "persist.db"
    s = SessionAnalytics("live1", "camera", persist=True, db_path=db_path)
    s.record_detection(7, "truck", 0.7, "00:02.000", est_colour="grey")
    s.record_detection(7, "truck", 0.7, "00:02.100", est_colour="grey")
    df = load_all_records(db_path)
    assert len(df) == 1
    assert df.iloc[0]["source"] == "Live Cam"
    assert df.iloc[0]["vehicle_class"] == "truck"


def test_interrupt_keeps_committed_rows(tmp_path):
    db_path = tmp_path / "partial.db"
    s = SessionAnalytics("sess", "upload", persist=True, db_path=db_path)
    s.record_detection(1, "car", 0.8, "00:01.000")
    s.record_detection(2, "bus", 0.7, "00:02.000")
    df = load_all_records(db_path)
    assert len(df) == 2
