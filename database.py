"""
database.py — SQLite persistence layer for Vehicle Intelligence.

Schema
------
Table: detections
  id              INTEGER  PRIMARY KEY AUTOINCREMENT
  session_id      TEXT     NOT NULL
  track_id        INTEGER  NOT NULL
  vehicle_class   TEXT     NOT NULL
  confidence      REAL     NOT NULL
  est_colour      TEXT
  plate_text      TEXT
  wall_time       TEXT     NOT NULL
  video_timestamp TEXT     NOT NULL
  source          TEXT     NOT NULL   -- "Live Cam" | "Up Vid"
  crossed_line    INTEGER  NOT NULL DEFAULT 0
  line_cross_time TEXT

Duplicate prevention: UNIQUE(session_id, track_id) — one row per unique
tracked vehicle per session.  ON CONFLICT REPLACE updates with richer data.

Database location: <project_dir>/data/vehicle_intelligence.db
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import pandas as pd

logger = logging.getLogger("vehicle_intelligence.database")

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
DB_DIR = _PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "vehicle_intelligence.db"

# ── Schema ────────────────────────────────────────────────────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    track_id        INTEGER NOT NULL,
    vehicle_class   TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    est_colour      TEXT,
    plate_text      TEXT,
    wall_time       TEXT    NOT NULL,
    video_timestamp TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    crossed_line    INTEGER NOT NULL DEFAULT 0,
    line_cross_time TEXT,
    UNIQUE(session_id, track_id)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_detections_wall_time
    ON detections(wall_time DESC);
"""


# ── Connection context ────────────────────────────────────────────────────────

@contextmanager
def _get_conn(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with WAL mode and row factory."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> None:
    """Create data directory and database tables if they do not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
    logger.info(f"Database initialised at {db_path}")


# ── Write ─────────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO detections
    (session_id, track_id, vehicle_class, confidence,
     est_colour, plate_text, wall_time, video_timestamp,
     source, crossed_line, line_cross_time)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(session_id, track_id) DO UPDATE SET
    confidence      = MAX(confidence, excluded.confidence),
    est_colour      = COALESCE(
                        CASE WHEN est_colour IS NULL
                             THEN excluded.est_colour
                             ELSE est_colour END,
                        excluded.est_colour),
    plate_text      = COALESCE(
                        CASE WHEN plate_text IS NULL
                             THEN excluded.plate_text
                             ELSE plate_text END,
                        excluded.plate_text),
    crossed_line    = MAX(crossed_line, excluded.crossed_line),
    line_cross_time = COALESCE(line_cross_time, excluded.line_cross_time)
"""


def _upsert_params(
    session_id: str,
    track_id: int,
    vehicle_class: str,
    confidence: float,
    video_timestamp: str,
    source: str,
    est_colour: Optional[str],
    plate_text: Optional[str],
    wall_time: Optional[str],
    crossed_line: bool,
    line_cross_time: Optional[str],
) -> tuple:
    if wall_time is None:
        wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        session_id, track_id, vehicle_class, round(confidence, 4),
        est_colour, plate_text, wall_time, video_timestamp,
        source, int(crossed_line), line_cross_time,
    )


def upsert_detection(
    session_id: str,
    track_id: int,
    vehicle_class: str,
    confidence: float,
    video_timestamp: str,
    source: str,
    est_colour: Optional[str] = None,
    plate_text: Optional[str] = None,
    wall_time: Optional[str] = None,
    crossed_line: bool = False,
    line_cross_time: Optional[str] = None,
    db_path: Path = DB_PATH,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """
    Insert or update one detection record.

    ON CONFLICT (session_id, track_id): update est_colour and plate_text
    only when richer data becomes available; update confidence to max.

    Commits immediately so a browser refresh is not required to save the event.
    Pass ``conn`` to reuse an open session connection.
    """
    params = _upsert_params(
        session_id, track_id, vehicle_class, confidence, video_timestamp,
        source, est_colour, plate_text, wall_time, crossed_line, line_cross_time,
    )
    if conn is not None:
        conn.execute(_UPSERT_SQL, params)
        conn.commit()
        return
    with _get_conn(db_path) as owned:
        owned.execute(_UPSERT_SQL, params)


def update_line_crossing(
    session_id: str,
    track_id: int,
    line_cross_time: str,
    db_path: Path = DB_PATH,
) -> None:
    """Mark an existing record as having crossed the counting line."""
    with _get_conn(db_path) as conn:
        conn.execute("""
            UPDATE detections
               SET crossed_line = 1, line_cross_time = ?
             WHERE session_id = ? AND track_id = ?
               AND crossed_line = 0
        """, (line_cross_time, session_id, track_id))


# ── Read ──────────────────────────────────────────────────────────────────────

def load_all_records(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Return all detection records as a DataFrame, newest first.
    Returns the exact six display columns plus id for internal use.
    """
    with _get_conn(db_path) as conn:
        rows = conn.execute("""
            SELECT id, session_id, track_id, vehicle_class, confidence,
                   est_colour, plate_text, wall_time, video_timestamp,
                   source, crossed_line, line_cross_time
              FROM detections
             ORDER BY wall_time DESC, id DESC
        """).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def _filter_clause(
    vehicle_class: Optional[str] = None,
    source: Optional[str] = None,
    colour: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []

    if vehicle_class:
        clauses.append("vehicle_class = ?")
        params.append(vehicle_class)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if colour:
        clauses.append("est_colour = ?")
        params.append(colour)
    if date_from:
        clauses.append("wall_time >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("wall_time <= ?")
        params.append(date_to + " 23:59:59")
    if search:
        term = f"%{search.strip()}%"
        clauses.append(
            "(IFNULL(plate_text,'') LIKE ? OR IFNULL(vehicle_class,'') LIKE ? "
            "OR IFNULL(est_colour,'') LIKE ? OR IFNULL(source,'') LIKE ? "
            "OR CAST(track_id AS TEXT) LIKE ?)"
        )
        params.extend([term, term, term, term, term])

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def load_filtered_records(
    vehicle_class: Optional[str] = None,
    source: Optional[str] = None,
    colour: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Load records with optional filters applied in SQL."""
    where, params = _filter_clause(
        vehicle_class, source, colour, date_from, date_to, search
    )

    with _get_conn(db_path) as conn:
        rows = conn.execute(f"""
            SELECT id, session_id, track_id, vehicle_class, confidence,
                   est_colour, plate_text, wall_time, video_timestamp,
                   source, crossed_line, line_cross_time
              FROM detections
             {where}
             ORDER BY wall_time DESC, id DESC
        """, params).fetchall()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_summary_stats(db_path: Path = DB_PATH) -> dict:
    """Return aggregate counts used by the Analytics page."""
    with _get_conn(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        by_class = dict(conn.execute(
            "SELECT vehicle_class, COUNT(*) FROM detections GROUP BY vehicle_class"
        ).fetchall())
        by_source = dict(conn.execute(
            "SELECT source, COUNT(*) FROM detections GROUP BY source"
        ).fetchall())
        by_colour = dict(conn.execute(
            "SELECT COALESCE(est_colour,'Unknown'), COUNT(*) "
            "FROM detections GROUP BY est_colour"
        ).fetchall())
        # Daily event counts (last 30 days)
        daily = conn.execute("""
            SELECT substr(wall_time, 1, 10) as day, COUNT(*) as cnt
              FROM detections
             WHERE wall_time >= date('now', '-30 days')
             GROUP BY day ORDER BY day
        """).fetchall()
    return {
        "total": total,
        "by_class": by_class,
        "by_source": by_source,
        "by_colour": by_colour,
        "daily": [dict(r) for r in daily],
    }


def clear_all_records(db_path: Path = DB_PATH) -> int:
    """Delete all detection records. Returns number of rows deleted."""
    with _get_conn(db_path) as conn:
        cur = conn.execute("DELETE FROM detections")
        deleted = cur.rowcount
    logger.info(f"Cleared {deleted} detection records from database.")
    return deleted


def distinct_values(column: str, db_path: Path = DB_PATH) -> List[str]:
    """Return sorted distinct non-null values for a column (for filter dropdowns)."""
    allowed = {"vehicle_class", "source", "est_colour"}
    if column not in allowed:
        raise ValueError(f"Column '{column}' not allowed for distinct query.")
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM detections "
            f"WHERE {column} IS NOT NULL ORDER BY {column}"
        ).fetchall()
    return [r[0] for r in rows]
