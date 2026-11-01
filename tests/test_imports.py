"""Smoke tests for module imports and Streamlit source parse."""

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = Path(__file__).resolve().parents[1]


def test_core_modules_import():
    import analytics
    import database
    import detector
    import plate_recognition
    import utils
    import vehicle_attributes
    import video_processor
    assert detector.DEFAULT_MODEL == "yolov8n.pt"
    assert plate_recognition.is_ocr_available() in (True, False)


def test_app_source_parses():
    tree = ast.parse((ROOT / "app.py").read_text())
    assert tree is not None


def test_database_init_default_path_exists():
    from database import DB_PATH, init_db
    init_db()
    assert DB_PATH.parent.name == "data"
    assert DB_PATH.name == "vehicle_intelligence.db"
