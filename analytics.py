"""
analytics.py — Session record management, analytics, and CSV export.

Persistence: Every detection is written to SQLite via database.py immediately
upon recording. The DetectionHistory class reads from the database.
"""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("vehicle_intelligence.analytics")

# Source label mapping: internal name → display/DB name
_SOURCE_LABELS = {
    "upload": "Up Vid",
    "camera": "Live Cam",
    "Up Vid": "Up Vid",
    "Live Cam": "Live Cam",
}


def _db_source(source: str) -> str:
    """Translate internal source string to the canonical DB label."""
    return _SOURCE_LABELS.get(source, source)


HISTORY_COLUMNS = [
    "SI. No.",
    "Vehicle Type",
    "Colour",
    "Vehicle Number",
    "Date & Time",
    "Source",
]


def build_history_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the six-column history table from a raw detections DataFrame.

    SI. No. is display-only and is not the database primary key.
    Missing colour → Unknown. Missing plate → Not detected.
    Source labels are normalised to 'Live Cam' / 'Up Vid'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        vehicle_type = str(row.get("vehicle_class", "Unknown")).title()

        colour = str(row.get("est_colour", "") or "").strip()
        colour = colour if colour and colour.lower() not in ("none", "nan") else "Unknown"
        colour = colour.title()

        plate = str(row.get("plate_text", "") or "").strip()
        plate = plate if plate and plate.lower() not in ("none", "nan") else "Not detected"

        wall_time = str(row.get("wall_time", "") or "").strip()
        try:
            dt = datetime.strptime(wall_time, "%Y-%m-%d %H:%M:%S")
            date_time_str = dt.strftime("%d %b %Y  %H:%M:%S")
        except Exception:
            date_time_str = wall_time

        source = str(row.get("source", "") or "").strip()
        source = _db_source(source)
        if source.lower() == "upload":
            source = "Up Vid"
        elif source.lower() == "camera":
            source = "Live Cam"

        rows.append({
            "SI. No.": i,
            "Vehicle Type": vehicle_type,
            "Colour": colour,
            "Vehicle Number": plate,
            "Date & Time": date_time_str,
            "Source": source,
        })
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


@dataclass
class VehicleRecord:
    """Single observation record for a tracked vehicle."""
    session_id: str
    track_id: int
    vehicle_class: str
    confidence: float
    est_colour: Optional[str]
    plate_text: Optional[str]
    wall_time: str             # Real-world timestamp
    video_timestamp: str       # Video-relative timestamp (MM:SS.mmm)
    source: str                # "Up Vid" | "Live Cam"
    crossed_line: bool = False
    line_cross_time: Optional[str] = None


class SessionAnalytics:
    """
    Maintains per-session detection records and computes aggregate statistics.

    One instance is created per processing session (video upload or camera run).
    Records are keyed by track_id; only the first observation per track ID is
    stored to represent unique vehicles. Subsequent frames update the record
    if new information (colour, plate) becomes available.

    Every record_detection call also upserts to the SQLite database via
    database.py for immediate persistence.
    """

    def __init__(
        self,
        session_id: str,
        source: str = "upload",
        persist: bool = True,
        db_path: Optional[Path] = None,
    ):
        self.session_id = session_id
        self.source = source
        self._db_source = _db_source(source)
        self._records: Dict[int, VehicleRecord] = {}   # track_id → record
        self._line_crossings: Dict[int, bool] = {}     # track_id → crossed
        self.persist = persist
        self._db_path = db_path
        self._db_available = self._try_init_db() if persist else False

    def _try_init_db(self) -> bool:
        """Attempt to initialise the database. Returns True on success."""
        try:
            from database import init_db, DB_PATH
            init_db(self._db_path or DB_PATH)
            return True
        except Exception as exc:
            logger.warning(f"Database init failed — history will be memory-only: {exc}")
            return False

    def has_track(self, track_id: int) -> bool:
        return track_id in self._records

    def colour_for(self, track_id: int) -> Optional[str]:
        rec = self._records.get(track_id)
        return rec.est_colour if rec else None

    def plate_for(self, track_id: int) -> Optional[str]:
        rec = self._records.get(track_id)
        return rec.plate_text if rec else None

    # ── Recording ────────────────────────────────────────────────────────────

    def record_detection(
        self,
        track_id: int,
        vehicle_class: str,
        confidence: float,
        video_timestamp: str,
        est_colour: Optional[str] = None,
        plate_text: Optional[str] = None,
    ) -> None:
        wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = VehicleRecord(
            session_id=self.session_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            confidence=round(confidence, 3),
            est_colour=est_colour,
            plate_text=plate_text,
            wall_time=wall_time,
            video_timestamp=video_timestamp,
            source=self._db_source,
        )
        # First observation per track_id = unique vehicle record
        if track_id not in self._records:
            self._records[track_id] = record
            # Write to DB immediately
            if self._db_available:
                self._upsert_to_db(record)
        else:
            existing = self._records[track_id]
            # Update with richer information when newly available
            changed = False
            if est_colour and not existing.est_colour:
                existing.est_colour = est_colour
                changed = True
            if plate_text and not existing.plate_text:
                existing.plate_text = plate_text
                changed = True
            # Update confidence to maximum observed
            if confidence > existing.confidence:
                existing.confidence = round(confidence, 3)
                changed = True
            # Push updates to DB
            if changed and self._db_available:
                self._upsert_to_db(existing)

    def _upsert_to_db(self, record: VehicleRecord) -> None:
        """Write a record to SQLite using the upsert function."""
        try:
            from database import upsert_detection
            kwargs = dict(
                session_id=record.session_id,
                track_id=record.track_id,
                vehicle_class=record.vehicle_class,
                confidence=record.confidence,
                video_timestamp=record.video_timestamp,
                source=record.source,
                est_colour=record.est_colour,
                plate_text=record.plate_text,
                wall_time=record.wall_time,
                crossed_line=record.crossed_line,
                line_cross_time=record.line_cross_time,
            )
            if self._db_path is not None:
                kwargs["db_path"] = self._db_path
            upsert_detection(**kwargs)
        except Exception as exc:
            logger.warning(f"DB upsert failed for track {record.track_id}: {exc}")

    def record_line_crossing(self, track_id: int, video_timestamp: str) -> bool:
        """
        Mark a vehicle as having crossed the counting line.
        Returns True if this is the first crossing (i.e., a new count).
        """
        if track_id in self._line_crossings:
            return False  # Already counted
        self._line_crossings[track_id] = True
        wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if track_id in self._records:
            self._records[track_id].crossed_line = True
            self._records[track_id].line_cross_time = video_timestamp
        if self._db_available:
            try:
                from database import update_line_crossing
                if self._db_path is not None:
                    update_line_crossing(
                        self.session_id, track_id, video_timestamp, db_path=self._db_path
                    )
                else:
                    update_line_crossing(self.session_id, track_id, video_timestamp)
            except Exception as exc:
                logger.warning(f"DB line crossing update failed: {exc}")
        logger.debug(f"Line crossing recorded: track_id={track_id}")
        return True

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def unique_vehicles(self) -> int:
        return len(self._records)

    @property
    def line_crossing_count(self) -> int:
        return len(self._line_crossings)

    def class_counts(self) -> Counter:
        return Counter(r.vehicle_class for r in self._records.values())

    def records_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        rows = [asdict(r) for r in self._records.values()]
        df = pd.DataFrame(rows)
        return df[[
            "track_id", "vehicle_class", "confidence",
            "est_colour", "plate_text",
            "video_timestamp", "wall_time", "crossed_line", "line_cross_time",
            "source",
        ]]

    def timeline_dataframe(self) -> pd.DataFrame:
        """Unique vehicle records (one row per track in this session)."""
        return self.records_dataframe()

    # ── Export ────────────────────────────────────────────────────────────────

    def to_csv(self) -> str:
        """Export unique vehicle records as CSV string."""
        df = self.records_dataframe()
        if df.empty:
            return "No records to export."
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def reset(self) -> None:
        self._records.clear()
        self._line_crossings.clear()
        logger.info(f"Session '{self.session_id}' reset.")


# ── Global history across sessions ───────────────────────────────────────────

class DetectionHistory:
    """
    Provides access to the persistent detection history stored in SQLite.
    Also accumulates in-memory sessions for the current app run.
    Data persists across app restarts via the database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._sessions: List[SessionAnalytics] = []
        self._db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        try:
            from database import init_db, DB_PATH
            init_db(self._db_path or DB_PATH)
        except Exception as exc:
            logger.warning(f"DetectionHistory: DB init failed: {exc}")

    def _kwargs(self) -> dict:
        return {"db_path": self._db_path} if self._db_path is not None else {}

    def add_session(self, session: SessionAnalytics) -> None:
        """Register a completed session (records already in DB)."""
        self._sessions.append(session)

    def all_records_dataframe(self) -> pd.DataFrame:
        """Load all records from the persistent database."""
        try:
            from database import load_all_records
            return load_all_records(**self._kwargs())
        except Exception as exc:
            logger.warning(f"DB load failed, falling back to memory: {exc}")
            # Fallback: concatenate in-memory session data
            dfs = [s.records_dataframe() for s in self._sessions if s.unique_vehicles > 0]
            if not dfs:
                return pd.DataFrame()
            return pd.concat(dfs, ignore_index=True)

    def filtered_records_dataframe(
        self,
        vehicle_class: Optional[str] = None,
        source: Optional[str] = None,
        colour: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load filtered records from the database."""
        try:
            from database import load_filtered_records
            return load_filtered_records(
                vehicle_class=vehicle_class,
                source=source,
                colour=colour,
                date_from=date_from,
                date_to=date_to,
                search=search,
                **self._kwargs(),
            )
        except Exception as exc:
            logger.warning(f"Filtered DB load failed: {exc}")
            return pd.DataFrame()

    def total_unique_vehicles(self) -> int:
        try:
            from database import get_summary_stats
            return get_summary_stats(**self._kwargs())["total"]
        except Exception:
            return sum(s.unique_vehicles for s in self._sessions)

    def combined_class_counts(self) -> Counter:
        try:
            from database import get_summary_stats
            return Counter(get_summary_stats(**self._kwargs())["by_class"])
        except Exception:
            total: Counter = Counter()
            for s in self._sessions:
                total.update(s.class_counts())
            return total

    def get_summary_stats(self) -> dict:
        """Return full summary stats from the database."""
        try:
            from database import get_summary_stats
            return get_summary_stats(**self._kwargs())
        except Exception as exc:
            logger.warning(f"Summary stats failed: {exc}")
            return {"total": 0, "by_class": {}, "by_source": {}, "by_colour": {}, "daily": []}

    def session_count(self) -> int:
        """Return number of sessions recorded in DB."""
        try:
            from database import load_all_records
            df = load_all_records(**self._kwargs())
            if df.empty:
                return 0
            return df["session_id"].nunique()
        except Exception:
            return len(self._sessions)

    def distinct_values(self, column: str) -> List[str]:
        """Return distinct non-null values for filter dropdowns."""
        try:
            from database import distinct_values
            if self._db_path is not None:
                return distinct_values(column, db_path=self._db_path)
            return distinct_values(column)
        except Exception:
            return []

    def to_csv(self) -> str:
        """Export the six-column history table as CSV."""
        display = build_history_display(self.all_records_dataframe())
        if display.empty:
            return "No records to export."
        buf = io.StringIO()
        display.to_csv(buf, index=False)
        return buf.getvalue()

    def reset_all(self) -> None:
        try:
            from database import clear_all_records
            cleared = clear_all_records(**self._kwargs())
            logger.info(f"Cleared {cleared} records from database.")
        except Exception as exc:
            logger.warning(f"DB clear failed: {exc}")
        self._sessions.clear()
        logger.info("Detection history cleared.")
