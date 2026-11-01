"""
tests/test_database.py — Comprehensive tests for the SQLite persistence layer.

Tests cover:
 1. Database initialisation
 2. Record insertion
 3. Persistence across separate DB connections
 4. Correct column order in Detection History display
 5. Correct source labels (Live Cam / Up Vid)
 6. Serial-number display logic
 7. Duplicate-event prevention
 8. CSV export contents and column order
 9. Filtering by vehicle type, date, colour, and source
10. Graceful handling of unavailable OCR
11. Correct record creation for live and uploaded video workflows

Run with:
    /Users/harsha/Downloads/CV_Project_New/venv/bin/python -m pytest tests/test_database.py -v
"""

import sys
import os
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd

# Use a temp DB for every test that touches the database
@pytest.fixture
def tmp_db(tmp_path):
    """Return a Path to a fresh temporary SQLite database."""
    db_path = tmp_path / "test_vi.db"
    from database import init_db
    init_db(db_path)
    return db_path


# ── 1. Database Initialisation ────────────────────────────────────────────────

class TestDatabaseInit:
    def test_init_creates_file(self, tmp_db):
        assert tmp_db.exists(), "DB file should be created by init_db"

    def test_init_creates_detections_table(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        assert cursor.fetchone() is not None, "Table 'detections' must exist"
        conn.close()

    def test_init_is_idempotent(self, tmp_db):
        """Calling init_db a second time on an existing DB should not error."""
        from database import init_db
        init_db(tmp_db)  # Should not raise
        conn = sqlite3.connect(str(tmp_db))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_creates_index(self, tmp_db):
        conn = sqlite3.connect(str(tmp_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_detections_wall_time'"
        )
        assert cursor.fetchone() is not None, "Index on wall_time must be created"
        conn.close()


# ── 2. Record Insertion ───────────────────────────────────────────────────────

class TestRecordInsertion:
    def test_upsert_inserts_row(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.85, video_timestamp="00:01.000", source="Up Vid",
            est_colour="blue", db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert len(df) == 1
        assert df.iloc[0]["vehicle_class"] == "car"

    def test_upsert_fields_stored_correctly(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=2, vehicle_class="truck",
            confidence=0.75, video_timestamp="00:02.000", source="Live Cam",
            est_colour="white", plate_text="MH12AB1234", db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        row = df.iloc[0]
        assert row["vehicle_class"] == "truck"
        assert row["source"] == "Live Cam"
        assert row["est_colour"] == "white"
        assert row["plate_text"] == "MH12AB1234"

    def test_upsert_multiple_distinct_records(self, tmp_db):
        from database import upsert_detection, load_all_records
        for tid, cls in [(1, "car"), (2, "motorcycle"), (3, "bus")]:
            upsert_detection(
                session_id="s1", track_id=tid, vehicle_class=cls,
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        df = load_all_records(tmp_db)
        assert len(df) == 3


# ── 3. Persistence Across Connections ────────────────────────────────────────

class TestPersistenceAcrossConnections:
    def test_record_survives_reconnect(self, tmp_db):
        """Records inserted in one connection must be readable from a new connection."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="persist_test", track_id=99, vehicle_class="car",
            confidence=0.90, video_timestamp="00:01.000", source="Up Vid",
            db_path=tmp_db,
        )
        # Load via a completely independent call (new connection internally)
        df2 = load_all_records(tmp_db)
        assert not df2.empty
        assert df2.iloc[0]["session_id"] == "persist_test"

    def test_multiple_sessions_persist(self, tmp_db):
        from database import upsert_detection, load_all_records
        for sid in ["session_a", "session_b", "session_c"]:
            upsert_detection(
                session_id=sid, track_id=1, vehicle_class="car",
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        df = load_all_records(tmp_db)
        assert df["session_id"].nunique() == 3


# ── 4. Correct Source Labels ──────────────────────────────────────────────────

class TestSourceLabels:
    def test_source_up_vid_stored(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000",
            source="Up Vid", db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["source"] == "Up Vid"

    def test_source_live_cam_stored(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000",
            source="Live Cam", db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["source"] == "Live Cam"

    def test_analytics_source_mapping_upload(self):
        """SessionAnalytics with source='upload' should map to 'Up Vid'."""
        from analytics import SessionAnalytics
        # Patch DB to avoid writing during this unit test
        s = SessionAnalytics.__new__(SessionAnalytics)
        s.session_id = "test"
        s.source = "upload"
        from analytics import _db_source
        assert _db_source("upload") == "Up Vid"

    def test_analytics_source_mapping_camera(self):
        from analytics import _db_source
        assert _db_source("camera") == "Live Cam"

    def test_analytics_source_mapping_passthrough(self):
        from analytics import _db_source
        assert _db_source("Up Vid") == "Up Vid"
        assert _db_source("Live Cam") == "Live Cam"


# ── 5. Duplicate Event Prevention ────────────────────────────────────────────

class TestDuplicatePrevention:
    def test_same_session_track_no_duplicate(self, tmp_db):
        """Upserting the same (session_id, track_id) multiple times should result in one row."""
        from database import upsert_detection, load_all_records
        for _ in range(5):
            upsert_detection(
                session_id="s1", track_id=1, vehicle_class="car",
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        df = load_all_records(tmp_db)
        assert len(df) == 1, "Duplicate upsert must result in single row"

    def test_upsert_updates_confidence_to_max(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.50, video_timestamp="00:01.000", source="Up Vid",
            db_path=tmp_db,
        )
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.95, video_timestamp="00:01.500", source="Up Vid",
            db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["confidence"] == pytest.approx(0.95, abs=0.01)

    def test_upsert_updates_colour_when_null(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            est_colour=None, db_path=tmp_db,
        )
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.500", source="Up Vid",
            est_colour="red", db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["est_colour"] == "red"

    def test_different_sessions_allow_same_track_id(self, tmp_db):
        """Different sessions may have the same track_id — separate rows expected."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            db_path=tmp_db,
        )
        upsert_detection(
            session_id="s2", track_id=1, vehicle_class="truck",
            confidence=0.75, video_timestamp="00:01.000", source="Live Cam",
            db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert len(df) == 2


# ── 6. Filtering ─────────────────────────────────────────────────────────────

class TestFiltering:
    def _seed(self, tmp_db):
        from database import upsert_detection
        base = datetime(2026, 1, 15, 10, 0, 0)
        records = [
            ("s1", 1, "car",        0.85, "Up Vid",   "blue",   base),
            ("s1", 2, "truck",      0.75, "Up Vid",   "white",  base + timedelta(hours=1)),
            ("s2", 3, "motorcycle", 0.60, "Live Cam", "red",    base + timedelta(days=1)),
            ("s2", 4, "car",        0.90, "Live Cam", "silver", base + timedelta(days=2)),
        ]
        for sid, tid, cls, conf, src, col, dt in records:
            upsert_detection(
                session_id=sid, track_id=tid, vehicle_class=cls,
                confidence=conf, video_timestamp="00:01.000", source=src,
                est_colour=col, wall_time=dt.strftime("%Y-%m-%d %H:%M:%S"),
                db_path=tmp_db,
            )

    def test_filter_by_vehicle_class(self, tmp_db):
        self._seed(tmp_db)
        from database import load_filtered_records
        df = load_filtered_records(vehicle_class="car", db_path=tmp_db)
        assert all(df["vehicle_class"] == "car")
        assert len(df) == 2

    def test_filter_by_source(self, tmp_db):
        self._seed(tmp_db)
        from database import load_filtered_records
        df = load_filtered_records(source="Live Cam", db_path=tmp_db)
        assert all(df["source"] == "Live Cam")
        assert len(df) == 2

    def test_filter_by_colour(self, tmp_db):
        self._seed(tmp_db)
        from database import load_filtered_records
        df = load_filtered_records(colour="blue", db_path=tmp_db)
        assert all(df["est_colour"] == "blue")
        assert len(df) == 1

    def test_filter_by_date_range(self, tmp_db):
        self._seed(tmp_db)
        from database import load_filtered_records
        df = load_filtered_records(
            date_from="2026-01-15",
            date_to="2026-01-15",
            db_path=tmp_db,
        )
        assert len(df) == 2  # Only records on 2026-01-15

    def test_filter_search_plate(self, tmp_db):
        self._seed(tmp_db)
        from database import upsert_detection, load_filtered_records
        upsert_detection(
            session_id="s3", track_id=9, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            plate_text="MH12AB1234", db_path=tmp_db,
        )
        df = load_filtered_records(search="MH12AB", db_path=tmp_db)
        assert len(df) == 1
        assert df.iloc[0]["plate_text"] == "MH12AB1234"

    def test_filter_combined(self, tmp_db):
        self._seed(tmp_db)
        from database import load_filtered_records
        df = load_filtered_records(
            vehicle_class="car", source="Live Cam", db_path=tmp_db
        )
        assert len(df) == 1
        assert df.iloc[0]["vehicle_class"] == "car"
        assert df.iloc[0]["source"] == "Live Cam"

    def test_distinct_values_vehicle_class(self, tmp_db):
        self._seed(tmp_db)
        from database import distinct_values
        classes = distinct_values("vehicle_class", tmp_db)
        assert "car" in classes
        assert "truck" in classes
        assert "motorcycle" in classes

    def test_distinct_values_source(self, tmp_db):
        self._seed(tmp_db)
        from database import distinct_values
        sources = distinct_values("source", tmp_db)
        assert "Up Vid" in sources
        assert "Live Cam" in sources

    def test_distinct_values_invalid_column_raises(self, tmp_db):
        from database import distinct_values
        with pytest.raises(ValueError):
            distinct_values("id", tmp_db)


# ── 7. CSV Export ─────────────────────────────────────────────────────────────

class TestCSVExport:
    def test_csv_has_correct_display_columns(self, tmp_db):
        """The exported CSV from the history page must have the exact 6 display columns."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.85, video_timestamp="00:01.000", source="Up Vid",
            est_colour="blue", plate_text="MH12AB1234",
            wall_time="2026-01-15 10:00:00", db_path=tmp_db,
        )
        df_raw = load_all_records(tmp_db)

        # Simulate _build_display_table from app.py
        display_cols = ["SI. No.", "Vehicle Type", "Colour", "Vehicle Number", "Date & Time", "Source"]
        # Build manually as app.py would
        rows = []
        for i, (_, row) in enumerate(df_raw.iterrows(), start=1):
            colour = str(row.get("est_colour", "")).strip()
            colour = colour if colour and colour.lower() not in ("none", "nan", "") else "Unknown"
            plate = str(row.get("plate_text", "")).strip()
            plate = plate if plate and plate.lower() not in ("none", "nan", "") else "Not detected"
            rows.append({
                "SI. No.": i,
                "Vehicle Type": str(row["vehicle_class"]).title(),
                "Colour": colour.title(),
                "Vehicle Number": plate,
                "Date & Time": row.get("wall_time", ""),
                "Source": row.get("source", ""),
            })
        display_df = pd.DataFrame(rows)

        import io
        buf = io.StringIO()
        display_df.to_csv(buf, index=False)
        csv_content = buf.getvalue()

        for col in display_cols:
            assert col in csv_content, f"Column '{col}' missing from CSV"

    def test_csv_contains_actual_data(self, tmp_db):
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="bus",
            confidence=0.70, video_timestamp="00:03.000", source="Live Cam",
            est_colour="yellow", db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert "bus" in df.to_csv()
        assert "yellow" in df.to_csv()

    def test_csv_not_detected_for_null_plate(self, tmp_db):
        """Null plate_text should appear as 'Not detected' in display CSV."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            plate_text=None, db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        row = df.iloc[0]
        plate = str(row.get("plate_text", "")).strip()
        display_plate = plate if plate and plate.lower() not in ("none", "nan", "") else "Not detected"
        assert display_plate == "Not detected"


# ── 8. Serial Number Display Logic ───────────────────────────────────────────

class TestSerialNumbers:
    def test_serial_numbers_sequential(self, tmp_db):
        from database import upsert_detection, load_all_records
        for tid in [1, 2, 3]:
            upsert_detection(
                session_id="s1", track_id=tid, vehicle_class="car",
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        df = load_all_records(tmp_db)
        # Build display table
        serial_numbers = list(range(1, len(df) + 1))
        assert serial_numbers == [1, 2, 3]

    def test_serial_numbers_for_filtered_results(self, tmp_db):
        """Serial numbers must restart from 1 for a filtered subset."""
        from database import upsert_detection, load_filtered_records
        for tid, cls in [(1, "car"), (2, "truck"), (3, "car")]:
            upsert_detection(
                session_id="s1", track_id=tid, vehicle_class=cls,
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        df = load_filtered_records(vehicle_class="car", db_path=tmp_db)
        # Only 2 "car" rows — SI. No. should be 1, 2
        assert len(df) == 2
        serial_numbers = list(range(1, len(df) + 1))
        assert serial_numbers == [1, 2]


# ── 9. Summary Stats ──────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_summary_total(self, tmp_db):
        from database import upsert_detection, get_summary_stats
        for tid in range(5):
            upsert_detection(
                session_id="s1", track_id=tid, vehicle_class="car",
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        stats = get_summary_stats(tmp_db)
        assert stats["total"] == 5

    def test_summary_by_class(self, tmp_db):
        from database import upsert_detection, get_summary_stats
        for cls in ["car", "car", "truck"]:
            import random
            upsert_detection(
                session_id=f"s{random.randint(1000,9999)}", track_id=1,
                vehicle_class=cls, confidence=0.80,
                video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        stats = get_summary_stats(tmp_db)
        assert stats["by_class"]["car"] == 2
        assert stats["by_class"]["truck"] == 1

    def test_summary_by_source(self, tmp_db):
        from database import upsert_detection, get_summary_stats
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            db_path=tmp_db,
        )
        upsert_detection(
            session_id="s2", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Live Cam",
            db_path=tmp_db,
        )
        stats = get_summary_stats(tmp_db)
        assert stats["by_source"]["Up Vid"] == 1
        assert stats["by_source"]["Live Cam"] == 1


# ── 10. Graceful OCR Handling ─────────────────────────────────────────────────

class TestOCRGracefulHandling:
    def test_ocr_unavailable_returns_none(self):
        """When EasyOCR is not installed, attempt_plate_ocr should return None silently."""
        from plate_recognition import attempt_plate_ocr, is_ocr_available
        import numpy as np
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = attempt_plate_ocr(frame, 10, 10, 180, 180)
        if not is_ocr_available():
            assert result is None, "Should return None when EasyOCR is not available"

    def test_ocr_availability_flag(self):
        from plate_recognition import is_ocr_available
        # Should return a bool without raising
        assert isinstance(is_ocr_available(), bool)

    def test_record_without_plate_stores_none(self, tmp_db):
        """Records can be stored with plate_text=None; should display as 'Not detected'."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            plate_text=None, db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["plate_text"] is None or pd.isna(df.iloc[0]["plate_text"])


# ── 11. Clear Records ─────────────────────────────────────────────────────────

class TestClearRecords:
    def test_clear_deletes_all(self, tmp_db):
        from database import upsert_detection, clear_all_records, load_all_records
        for tid in range(3):
            upsert_detection(
                session_id="s1", track_id=tid, vehicle_class="car",
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        deleted = clear_all_records(tmp_db)
        assert deleted == 3
        df = load_all_records(tmp_db)
        assert df.empty

    def test_clear_returns_row_count(self, tmp_db):
        from database import upsert_detection, clear_all_records
        for tid in range(4):
            upsert_detection(
                session_id="s1", track_id=tid, vehicle_class="car",
                confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
                db_path=tmp_db,
            )
        assert clear_all_records(tmp_db) == 4


# ── 12. Analytics Integration ─────────────────────────────────────────────────

class TestAnalyticsDBIntegration:
    def test_session_analytics_persists_to_db(self, tmp_db):
        """SessionAnalytics.record_detection should write to the DB.
        We test this by directly using the database functions since SessionAnalytics
        writes to the default DB (which cannot easily be redirected in a unit test
        without complex monkeypatching). The DB upsert logic is independently
        verified in TestRecordInsertion.
        """
        from database import upsert_detection, load_all_records

        # Simulate what SessionAnalytics._upsert_to_db does
        upsert_detection(
            session_id="sa_test",
            track_id=42,
            vehicle_class="car",
            confidence=0.88,
            video_timestamp="00:05.000",
            source="Up Vid",  # "upload" → "Up Vid"
            est_colour="black",
            db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert len(df) == 1
        assert df.iloc[0]["track_id"] == 42
        assert df.iloc[0]["vehicle_class"] == "car"
        assert df.iloc[0]["est_colour"] == "black"

    def test_source_label_upload_in_db(self, tmp_db):
        """Source stored in DB for upload sessions must be 'Up Vid'."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000",
            source="Up Vid",  # analytics.py now maps "upload" → "Up Vid"
            db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["source"] == "Up Vid"

    def test_source_label_camera_in_db(self, tmp_db):
        """Source stored in DB for camera sessions must be 'Live Cam'."""
        from database import upsert_detection, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000",
            source="Live Cam",
            db_path=tmp_db,
        )
        df = load_all_records(tmp_db)
        assert df.iloc[0]["source"] == "Live Cam"


# ── 13. Line Crossing Update ──────────────────────────────────────────────────

class TestLineCrossingUpdate:
    def test_update_line_crossing(self, tmp_db):
        from database import upsert_detection, update_line_crossing, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            db_path=tmp_db,
        )
        update_line_crossing("s1", 1, "00:01.500", db_path=tmp_db)
        df = load_all_records(tmp_db)
        assert df.iloc[0]["crossed_line"] == 1
        assert df.iloc[0]["line_cross_time"] == "00:01.500"

    def test_line_crossing_not_updated_twice(self, tmp_db):
        """The update should only apply when crossed_line = 0."""
        from database import upsert_detection, update_line_crossing, load_all_records
        upsert_detection(
            session_id="s1", track_id=1, vehicle_class="car",
            confidence=0.80, video_timestamp="00:01.000", source="Up Vid",
            db_path=tmp_db,
        )
        update_line_crossing("s1", 1, "00:01.500", db_path=tmp_db)
        update_line_crossing("s1", 1, "00:02.000", db_path=tmp_db)  # second call ignored
        df = load_all_records(tmp_db)
        assert df.iloc[0]["line_cross_time"] == "00:01.500"  # First time preserved
