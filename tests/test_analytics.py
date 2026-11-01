"""
tests/test_analytics.py — Unit tests for analytics, counting, and CSV export.

Run with:
    ./venv/bin/python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from analytics import SessionAnalytics, DetectionHistory
from utils import frame_to_timestamp, seconds_to_timestamp


# ── Timestamp helpers ─────────────────────────────────────────────────────────

class TestTimestamps:
    def test_frame_to_timestamp_zero(self):
        assert frame_to_timestamp(0, 25.0) == "00:00.000"

    def test_frame_to_timestamp_one_second(self):
        result = frame_to_timestamp(25, 25.0)
        assert result == "00:01.000"

    def test_frame_to_timestamp_one_minute(self):
        result = frame_to_timestamp(1500, 25.0)
        assert result == "01:00.000"

    def test_frame_to_timestamp_invalid_fps(self):
        result = frame_to_timestamp(100, 0)
        assert result == "00:00.000"

    def test_seconds_to_timestamp(self):
        assert seconds_to_timestamp(0.0) == "00:00.000"
        assert seconds_to_timestamp(61.5) == "01:01.500"


# ── Session analytics ─────────────────────────────────────────────────────────

class TestSessionAnalytics:
    def _make_session(self) -> SessionAnalytics:
        return SessionAnalytics(session_id="test_001", source="upload", persist=False)

    def test_initial_state(self):
        s = self._make_session()
        assert s.unique_vehicles == 0
        assert s.line_crossing_count == 0

    def test_record_single_vehicle(self):
        s = self._make_session()
        s.record_detection(
            track_id=1,
            vehicle_class="car",
            confidence=0.85,
            video_timestamp="00:01.000",
        )
        assert s.unique_vehicles == 1
        counts = s.class_counts()
        assert counts["car"] == 1

    def test_same_track_id_counted_once(self):
        s = self._make_session()
        for _ in range(10):
            s.record_detection(1, "car", 0.80, "00:01.000")
        assert s.unique_vehicles == 1

    def test_multiple_classes(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.80, "00:01.000")
        s.record_detection(2, "truck", 0.75, "00:02.000")
        s.record_detection(3, "motorcycle", 0.60, "00:03.000")
        assert s.unique_vehicles == 3
        counts = s.class_counts()
        assert counts["car"] == 1
        assert counts["truck"] == 1
        assert counts["motorcycle"] == 1

    def test_colour_update(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.80, "00:01.000", est_colour=None)
        s.record_detection(1, "car", 0.80, "00:01.500", est_colour="blue")
        df = s.records_dataframe()
        assert df.iloc[0]["est_colour"] == "blue"

    def test_confidence_updates_to_max(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.50, "00:01.000")
        s.record_detection(1, "car", 0.90, "00:01.500")
        df = s.records_dataframe()
        assert df.iloc[0]["confidence"] == pytest.approx(0.90)

    def test_line_crossing_counted_once(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.80, "00:01.000")
        assert s.record_line_crossing(1, "00:01.500") is True
        assert s.record_line_crossing(1, "00:02.000") is False
        assert s.line_crossing_count == 1

    def test_multiple_line_crossings(self):
        s = self._make_session()
        for tid in [1, 2, 3]:
            s.record_detection(tid, "car", 0.80, "00:01.000")
            s.record_line_crossing(tid, "00:01.500")
        assert s.line_crossing_count == 3

    def test_records_dataframe_columns(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.80, "00:01.000")
        df = s.records_dataframe()
        assert "track_id" in df.columns
        assert "vehicle_class" in df.columns
        assert "confidence" in df.columns

    def test_to_csv_nonempty(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.80, "00:01.000")
        csv = s.to_csv()
        assert "track_id" in csv
        assert "car" in csv

    def test_to_csv_empty(self):
        s = self._make_session()
        csv = s.to_csv()
        assert "No records" in csv

    def test_reset(self):
        s = self._make_session()
        s.record_detection(1, "car", 0.80, "00:01.000")
        s.reset()
        assert s.unique_vehicles == 0


# ── Detection history ─────────────────────────────────────────────────────────
# NOTE: DetectionHistory now reads from SQLite. These tests use a fresh
# isolated in-memory-only session to verify the in-memory layer. Full DB
# persistence tests are in tests/test_database.py.

class TestDetectionHistory:
    def test_empty_history(self):
        """A new DetectionHistory with no sessions should have 0 in-memory sessions."""
        h = DetectionHistory()
        # In-memory session list should be empty
        assert h._sessions == [] or len(h._sessions) == 0

    def test_add_session_increments_in_memory(self):
        """add_session registers the session object in memory."""
        h = DetectionHistory()
        initial_count = len(h._sessions)
        s = SessionAnalytics(f"test_hist_{id(h)}", "upload", persist=False)
        s.record_detection(1, "car", 0.8, "00:01.000")
        h.add_session(s)
        assert len(h._sessions) == initial_count + 1

    def test_multiple_sessions_in_memory(self):
        """Multiple add_session calls accumulate in _sessions."""
        h = DetectionHistory()
        initial_count = len(h._sessions)
        for i in range(3):
            s = SessionAnalytics(f"test_multi_{id(h)}_{i}", "upload", persist=False)
            s.record_detection(i + 1, "truck", 0.75, "00:01.000")
            h.add_session(s)
        assert len(h._sessions) == initial_count + 3

    def test_reset_all(self, tmp_path):
        """reset_all clears only the provided database, not the project DB."""
        db_path = tmp_path / "hist.db"
        h = DetectionHistory(db_path=db_path)
        s = SessionAnalytics(f"test_reset_{id(h)}", "upload", persist=True, db_path=db_path)
        s.record_detection(1, "car", 0.8, "00:01.000")
        h.add_session(s)
        h.reset_all()
        assert len(h._sessions) == 0
        assert h.total_unique_vehicles() == 0

    def test_csv_export(self):
        """CSV export from a fresh session contains the recorded data."""
        # Use SessionAnalytics directly (in-memory) to avoid DB side-effects
        s = SessionAnalytics(f"test_csv_{id(object())}", "upload", persist=False)
        s.record_detection(1, "car", 0.8, "00:01.000", est_colour="blue")
        csv = s.to_csv()
        assert "car" in csv
        assert "blue" in csv
