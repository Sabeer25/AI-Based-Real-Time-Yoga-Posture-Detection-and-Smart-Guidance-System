"""
test_session_model.py
----------------------
Camera-free sanity check for Module 3 (SQLite session history) and Module 4 /
Model 2 (scikit-learn practice prediction). Seeds a temporary database with a
few synthetic sessions, then verifies storage round-trips and that the model
trains and produces per-asana predictions.

Run:
    python test_session_model.py

(You can also run it with pytest:  python -m pytest test_session_model.py)
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import session_store
from session_store import SessionRecorder, list_sessions, session_events, practice_history, stats
from pose_classifier import SUPPORTED_ASANAS
from practice_model import (
    ASANAS, load_blocks, prepare_training_set, train_and_evaluate, predict_next,
    explain_model, recommend_next,
)


def _seed(db_path: str, n_sessions: int = 4) -> None:
    """Create n sessions spread over time, each practicing all supported asanas."""
    conn = sqlite3.connect(db_path)
    conn.executescript(session_store.SCHEMA)
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(n_sessions):
        start = base + timedelta(days=i * 4)
        conn.execute(
            "INSERT INTO sessions (started_at, ended_at, duration_s) VALUES (?,?,?)",
            (start.isoformat(), (start + timedelta(minutes=3)).isoformat(), 180),
        )
        sid = i + 1
        for j, asana in enumerate(ASANAS):
            for k in range(20):
                ts = j * 60.0 + k * 2.0
                score = 50.0 + i * 10.0 + j * 3.0 + k
                conn.execute(
                    """INSERT INTO pose_events
                           (session_id, ts, asana, is_correct, score,
                            left_knee, right_knee, torso_lean, leg_spread, hip_level_diff)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (sid, ts, asana, int(score >= 80), score, 165.0, 165.0, 5.0, 1.5, 0.02),
                )
    conn.commit()
    conn.close()


def test_storage_roundtrip(db_path: str) -> None:
    _seed(db_path, n_sessions=3)
    assert len(list_sessions(db_path)) == 3
    assert len(session_events(db_path, 1)) == 20 * len(ASANAS)
    assert len(practice_history(db_path)) == 60 * len(ASANAS)
    info = stats(db_path)
    assert info["sessions"] == 3 and info["events"] == 60 * len(ASANAS)
    print("[ok] storage round-trip (sessions, events, history, stats)")


def test_recorder_context(db_path: str) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    with SessionRecorder(db_path) as rec:
        assert rec.session_id == 1
        rec.log_event(0.0, "Tadasana", True, 90.0)
        rec.log_event(0.1, "Vrikshasana", False, 55.0, None)
    assert len(session_events(db_path, 1)) == 2
    assert stats(db_path)["correct_events"] == 1
    print("[ok] SessionRecorder context manager logs events")


def test_model_pipeline(db_path: str) -> None:
    _seed(db_path, n_sessions=8)
    prepared = prepare_training_set(db_path)
    assert prepared is not None
    X, blocks, y_score, y_correct = prepared
    assert len(X) == len(blocks) == 8 * len(ASANAS)
    model = train_and_evaluate(X, y_score, y_correct)
    assert model["metrics"]["test_blocks"] > 0
    assert model["forest"] is not None

    preds = predict_next(model, db_path)
    assert len(preds) == len(ASANAS)
    for p in preds:
        assert 0.0 <= p["score"] <= 100.0

    rules = explain_model(model)
    assert isinstance(rules, list)

    recs = recommend_next(model, db_path, top_n=3)
    assert 0 < len(recs) <= 3
    for r in recs:
        assert r["asana"] in ASANAS
        assert 0.0 <= r["priority"] <= 100.0
    print("[ok] model pipeline (features, train, evaluate, predict, recommend, explain)")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test_sessions.db")
        test_storage_roundtrip(db)
        test_recorder_context(db)
        test_model_pipeline(db)
    print("\nAll session + model tests passed.")


if __name__ == "__main__":
    main()
