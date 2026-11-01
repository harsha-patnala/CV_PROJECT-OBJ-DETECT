"""Tests for the six-column history display and CSV export."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from analytics import HISTORY_COLUMNS, build_history_display, DetectionHistory, SessionAnalytics


def test_history_column_order():
    df = pd.DataFrame([{
        "vehicle_class": "car",
        "est_colour": "blue",
        "plate_text": None,
        "wall_time": "2026-01-15 10:00:00",
        "source": "Up Vid",
    }])
    display = build_history_display(df)
    assert list(display.columns) == HISTORY_COLUMNS
    assert list(display.columns) == [
        "SI. No.", "Vehicle Type", "Colour", "Vehicle Number", "Date & Time", "Source",
    ]


def test_history_placeholders_and_labels():
    df = pd.DataFrame([
        {
            "vehicle_class": "motorcycle",
            "est_colour": None,
            "plate_text": None,
            "wall_time": "2026-01-15 10:00:00",
            "source": "camera",
        },
        {
            "vehicle_class": "bus",
            "est_colour": "white",
            "plate_text": "KA01CD9999",
            "wall_time": "2026-01-15 11:00:00",
            "source": "upload",
        },
    ])
    display = build_history_display(df)
    assert display.iloc[0]["Vehicle Type"] == "Motorcycle"
    assert display.iloc[0]["Colour"] == "Unknown"
    assert display.iloc[0]["Vehicle Number"] == "Not detected"
    assert display.iloc[0]["Source"] == "Live Cam"
    assert display.iloc[1]["Source"] == "Up Vid"
    assert display.iloc[1]["Vehicle Number"] == "KA01CD9999"
    assert display.iloc[0]["SI. No."] == 1
    assert display.iloc[1]["SI. No."] == 2


def test_empty_history_has_columns():
    display = build_history_display(pd.DataFrame())
    assert list(display.columns) == HISTORY_COLUMNS
    assert display.empty


def test_history_csv_six_columns(tmp_path):
    db_path = tmp_path / "hist.db"
    s = SessionAnalytics("sess1", "camera", persist=True, db_path=db_path)
    s.record_detection(1, "car", 0.9, "00:01.000", est_colour="red")
    h = DetectionHistory(db_path=db_path)
    csv = h.to_csv()
    header = csv.splitlines()[0]
    assert header == ",".join(HISTORY_COLUMNS)
    assert "Live Cam" in csv
    assert "Car" in csv
    assert "Red" in csv
    assert "Not detected" in csv
