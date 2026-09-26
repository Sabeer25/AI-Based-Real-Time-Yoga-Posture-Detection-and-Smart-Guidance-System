"""
session_store.py
-----------------
Module 3 - SQLite session history for the yoga posture detection system.

Records every practice session and the per-frame pose results (asana,
correct/incorrect, geometry score, and a few joint angles) into a local
SQLite database. No cloud, no configuration: the file is created next to the
scripts on first use and can be upgraded to a networked DB later without
changing the rest of the app.

Schema:
    sessions     - one row per practice run
    pose_events  - one row per detected-pose frame inside a session

Design choices (matching the "SQLite + Streamlit over full cloud stack"
justification):
  * zero-configuration single-user storage,
  * the model-training pipeline (practice_model.py) only ever reads from this
    module, so swapping SQLite for Postgres later touches one file.
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    duration_s  REAL DEFAULT 0,
    frame_count INTEGER DEFAULT 0,
    pose_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pose_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    ts             REAL NOT NULL,             -- seconds since session start
    asana          TEXT NOT NULL,
    is_correct     INTEGER NOT NULL,          -- 0 / 1
    score          REAL NOT NULL,             -- 0-100 geometry match
    left_knee      REAL,
    right_knee     REAL,
    torso_lean     REAL,
    leg_spread     REAL,
    hip_level_diff REAL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON pose_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_asana   ON pose_events(asana);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db(db_path: str) -> None:
    """Create the database file and tables if they don't exist yet."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


class SessionRecorder:
    """
    Context-manager wrapper around one practice run.

    Usage:
        with SessionRecorder("sessions.db") as rec:
            rec.log_event(ts, "Tadasana", True, 92.0, features)

    Can also be driven manually with start()/stop() so a single app run can
    record several sessions (start/stop buttons in the webcam UI):

        rec = SessionRecorder("sessions.db")
        rec.start()
        rec.log_event(ts, ...)
        rec.stop()
    """

    def __init__(self, db_path: str, session_id: Optional[int] = None):
        self.db_path = db_path
        self.session_id = session_id
        self._conn: Optional[sqlite3.Connection] = None
        self._start_wall = time.time()

    def start(self) -> "SessionRecorder":
        """Open the DB and insert this session's row. Safe to call once."""
        if self._conn is not None:
            return self
        ensure_db(self.db_path)
        self._conn = _connect(self.db_path)
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (_now_iso(),)
        )
        self.session_id = cur.lastrowid
        self._conn.commit()
        self._start_wall = time.time()
        return self

    def __enter__(self) -> "SessionRecorder":
        return self.start()

    def stop(self) -> None:
        """Finalize this session's row and close the connection."""
        self.finish()
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def log_event(
        self,
        ts: float,
        asana: str,
        is_correct: bool,
        score: float,
        features: Any = None,
    ) -> None:
        """Persist one detected-pose frame. `features` is a PoseFeatures
        object (or anything with the same attribute names) - the angles we
        keep are enough for the prediction model and later analysis."""
        if self._conn is None:
            raise RuntimeError("SessionRecorder not open (use it as a context manager)")
        f = features
        self._conn.execute(
            """
            INSERT INTO pose_events
                (session_id, ts, asana, is_correct, score,
                 left_knee, right_knee, torso_lean, leg_spread, hip_level_diff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id, ts, asana, int(bool(is_correct)), float(score),
                getattr(f, "left_knee", None),
                getattr(f, "right_knee", None),
                getattr(f, "torso_lean", None),
                getattr(f, "leg_spread_ratio", None),
                getattr(f, "hip_level_diff", None),
            ),
        )

    def finish(self) -> None:
        """Finalize the session row (duration / frame / pose counts)."""
        if self._conn is None or self.session_id is None:
            return
        duration = time.time() - self._start_wall
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM pose_events WHERE session_id = ?",
            (self.session_id,),
        ).fetchone()
        pose_count = row["n"] if row else 0
        self._conn.execute(
            """
            UPDATE sessions
               SET ended_at = ?, duration_s = ?, pose_count = ?
             WHERE id = ?
            """,
            (_now_iso(), duration, pose_count, self.session_id),
        )
        self._conn.commit()

    def reset(self) -> None:
        """Drop all data (used by the test suite / when re-demoing)."""
        if self._conn is None:
            ensure_db(self.db_path)
            self._conn = _connect(self.db_path)
        self._conn.executescript("DROP TABLE IF EXISTS pose_events; DROP TABLE IF EXISTS sessions;")
        self._conn.commit()


# --------------------------------------------------------------------------
# Read-side helpers (used by the dashboard and the prediction model)
# --------------------------------------------------------------------------

def list_sessions(db_path: str) -> List[Dict[str, Any]]:
    ensure_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def session_events(db_path: str, session_id: int) -> List[Dict[str, Any]]:
    ensure_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM pose_events WHERE session_id = ? ORDER BY ts", (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def practice_history(db_path: str) -> "List[Dict[str, Any]]":
    """
    All pose events joined with their session timestamps, ordered by when the
    session happened. This is the raw table the prediction model builds its
    feature matrix from.
    """
    ensure_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT e.*, s.started_at
              FROM pose_events e
              JOIN sessions s ON s.id = e.session_id
             ORDER BY s.started_at, e.ts
            """
        ).fetchall()
    return [dict(r) for r in rows]


def stats(db_path: str) -> Dict[str, Any]:
    """Cheap overview numbers for the dashboard header."""
    ensure_db(db_path)
    with _connect(db_path) as conn:
        sessions = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        poses = conn.execute(
            "SELECT COUNT(DISTINCT asana) AS n FROM pose_events"
        ).fetchone()["n"]
        total_time = conn.execute(
            "SELECT COALESCE(SUM(duration_s), 0) AS t FROM sessions"
        ).fetchone()["t"]
        correct = conn.execute(
            "SELECT COALESCE(SUM(is_correct), 0) AS c, COUNT(*) AS n FROM pose_events"
        ).fetchone()
    return {
        "sessions": sessions,
        "poses": poses,
        "total_time_s": total_time,
        "events": correct["n"] or 0,
        "correct_events": correct["c"] or 0,
        "db_path": os.path.abspath(db_path),
    }


def last_session_stats(db_path: str) -> Optional[Dict[str, Any]]:
    """Stats for the most recent session only (or None when no data yet)."""
    ensure_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        sid = row["id"]
        poses = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(is_correct), 0) AS c "
            "FROM pose_events WHERE session_id = ?",
            (sid,),
        ).fetchone()
        asanas = conn.execute(
            "SELECT COUNT(DISTINCT asana) AS n FROM pose_events WHERE session_id = ?",
            (sid,),
        ).fetchone()
    return {
        "id": sid,
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "duration_s": row["duration_s"] or 0.0,
        "events": poses["n"] or 0,
        "correct_events": poses["c"] or 0,
        "asana_count": asanas["n"] or 0,
    }
