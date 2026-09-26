"""
practice_model.py
------------------
Model 2 - the second "model" of the project: a lightweight, interpretable
scikit-learn model that predicts how well the user will perform in their
next practice, per asana, based on their SQLite session history.

Why scikit-learn (from the design justification): the per-user dataset is
small, so a proportional model is fast and explainable - unlike a deep model
which would overfit on a few dozen sessions. A random forest handles the
mostly-categorical per-asana features well and its feature importances read
as plain-English rules in the dashboard.

What it predicts, for each asana:
  * expected geometry score (0-100)      -> RandomForestRegressor
  * probability of a CORRECT posture     -> RandomForestClassifier

The model is retrained on demand from the history table (training takes
milliseconds on this data), so there is nothing to keep "in sync". Both
models are kept deliberately simple so their coefficients read as plain
English rules in the dashboard ("more attempts -> +x score").

Run:
    python practice_model.py --db sessions.db

Module 4 prediction is fully camera-free: it only reads the database written
by session_store.py.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pose_classifier import SUPPORTED_ASANAS
from session_store import practice_history

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.db")

# The asanas the prediction model knows about = the ones the detector supports.
ASANAS = list(SUPPORTED_ASANAS)

# Numeric feature columns, in a fixed order so trained coefficients stay
# aligned with their names when we explain the model.
NUMERIC_FEATURES = [
    "attempts",        # total prior events for this asana
    "sessions",        # distinct prior sessions it appeared in
    "avg_score",       # mean geometry score in prior practice
    "max_score",       # best score ever
    "score_std",       # consistency (low = steady)
    "correct_rate",    # fraction of prior events marked CORRECT
    "hold_s",          # total time it was held before
    "days_since",      # recency (days since last practiced)
    "last_score",      # score of the most recent prior event
]

FEATURE_COLUMNS = NUMERIC_FEATURES + [f"asana__{a}" for a in ASANAS]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def _block_stats(events: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate the events of one (session, asana) block into a target row."""
    ts = events["ts"].to_numpy()
    hold = 0.0
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        if 0.0 < dt <= 5.0:  # ignore pauses so hold time reflects actual holding
            hold += dt
    return {
        "n": int(len(events)),
        "score": float(events["score"].mean()),
        "is_correct": bool(events["is_correct"].mean() >= 0.5),
        "hold_s": float(hold),
    }


def _history_features(prior: pd.DataFrame, asana: str, cut: pd.Timestamp) -> Dict[str, float]:
    """Stats for one asana computed from events strictly before `cut`."""
    ev = prior[prior["asana"] == asana]
    if ev.empty:
        return {
            "attempts": 0.0, "sessions": 0.0, "avg_score": 0.0, "max_score": 0.0,
            "score_std": 0.0, "correct_rate": 0.0, "hold_s": 0.0,
            "days_since": 999.0, "last_score": 0.0,
        }
    n = len(ev)
    return {
        "attempts": float(n),
        "sessions": float(ev["session_id"].nunique()),
        "avg_score": float(ev["score"].mean()),
        "max_score": float(ev["score"].max()),
        "score_std": float(ev["score"].std()) if n > 1 else 0.0,
        "correct_rate": float(ev["is_correct"].mean()),
        "hold_s": float(ev["hold_s"].sum()),
        "days_since": max(0.0, (cut - ev["started_at"].max()).days),
        "last_score": float(ev.sort_values(["started_at", "ts"]).iloc[-1]["score"]),
    }


def load_blocks(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (blocks, history):
      blocks  - one row per (session, asana) practice block, with target
                columns `score` and `is_correct`, plus `started_at`,
                `session_id`, `asana`.
      history - the per-asana-per-session hold times, keyed by
                (session_id, asana), used by the feature builder.
    """
    rows = practice_history(db_path)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, format="mixed")

    # hold time per (session, asana) block
    hold_rows: List[Dict[str, Any]] = []
    for (sid, asana), grp in df.groupby(["session_id", "asana"]):
        grp = grp.sort_values("ts")
        stats = _block_stats(grp)
        hold_rows.append({"session_id": sid, "asana": asana, **stats})
    blocks = pd.DataFrame(hold_rows)
    blocks["started_at"] = pd.to_datetime(
        blocks["session_id"].map(df.groupby("session_id")["started_at"].first()),
        utc=True,
    )

    # merge hold_s back into the per-event frame for the feature builder
    hist = df.merge(blocks[["session_id", "asana", "hold_s"]], on=["session_id", "asana"], how="left")
    return blocks, hist


def prepare_training_set(
    db_path: str, since: Optional[pd.Timestamp] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Builds (X, blocks) ready for training. X is the feature matrix aligned
    with blocks (same index); targets live in blocks: `score`, `is_correct`.
    `since` optionally limits the data to practice from that date onwards
    (e.g. "last week / month / year"). Returns None when there is no data.
    """
    blocks, hist = load_blocks(db_path)
    if blocks.empty:
        return None
    if since is not None:
        blocks = blocks[blocks["started_at"] >= since]
        hist = hist[hist["started_at"] >= since]
        if blocks.empty:
            return None
    blocks = blocks.sort_values(["started_at", "asana"]).reset_index(drop=True)
    hist = hist.sort_values(["started_at", "ts"])

    X = _features_without_lookahead(blocks, hist)
    return X, blocks, blocks["score"], blocks["is_correct"]


def _features_without_lookahead(blocks: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Per-block features computed from events strictly before that block's
    session start (mask applied row-by-row; data is tiny, clarity > speed)."""
    feats: List[Dict[str, Any]] = []
    for _, row in blocks.iterrows():
        cut = row["started_at"]
        prior = hist[hist["started_at"] < cut]
        asana = row["asana"]
        f = _history_features(prior, asana, cut)
        f.update({name: float(f.get(name, 0.0)) for name in NUMERIC_FEATURES})
        f["asana__" + asana] = 1.0
        for other in ASANAS:
            f.setdefault("asana__" + other, 0.0)
        feats.append(f)
    return pd.DataFrame(feats, index=blocks.index)[FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Model training + evaluation
# --------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y_score: pd.Series,
    y_correct: pd.Series,
) -> Dict[str, Any]:
    """Fit a random forest (score regression + correctness classification)
    using a time-aware split: the earliest ~70% of blocks train, the most
    recent ~30% test, which mirrors how the model is actually used
    (predicting the future)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score

    n = len(X)
    n_test = 0
    X_tr, X_te = X, pd.DataFrame()
    y_tr, y_te = y_score, pd.Series(dtype=float)
    c_tr, c_te = y_correct, pd.Series(dtype=float)

    if n >= 6:
        n_test = max(1, n // 3)
        split = n - n_test
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y_score.iloc[:split], y_score.iloc[split:]
        c_tr, c_te = y_correct.iloc[:split], y_correct.iloc[split:]

    # --- score regression -------------------------------------------------
    forest = RandomForestRegressor(
        n_estimators=200, random_state=42, min_samples_leaf=1, n_jobs=-1,
    )
    forest.fit(X_tr.values, y_tr.to_numpy(dtype=float))

    # --- correctness classification ----------------------------------------
    clf = None
    if c_tr.nunique() >= 2 and c_tr.sum() > 0 and (c_tr.sum() < len(c_tr)):
        clf = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1,
        )
        clf.fit(X_tr.values, c_tr.to_numpy())

    metrics: Dict[str, Any] = {}
    if n_test > 0:
        pred_score = np.clip(forest.predict(X_te.values), 0, 100)
        metrics["test_r2"] = float(r2_score(y_te, pred_score))
        if clf is not None:
            pred_cls = clf.predict(X_te.values)
            metrics["test_accuracy"] = float(accuracy_score(c_te, pred_cls))
        else:
            metrics["test_accuracy"] = None
        metrics["test_blocks"] = n_test
    else:
        metrics["test_blocks"] = 0
        metrics["test_r2"] = None
        metrics["test_accuracy"] = None

    metrics["train_blocks"] = len(X_tr)
    return {
        "forest": forest,
        "clf": clf,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "asana_names": ASANAS,
        "metrics": metrics,
    }


def predict_next(
    model: Dict[str, Any], db_path: str, since: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    """
    Predict next-session outcome for every supported asana using the history
    in `db_path` as features (optionally only within `since`). Returns rows
    for the dashboard / CLI:
        {asana, score, prob_correct}
    """
    blocks, hist = load_blocks(db_path)
    if since is not None:
        hist = hist[hist["started_at"] >= since]
        blocks = blocks[blocks["started_at"] >= since]
    # Predicting the *next* session needs at least one completed prior
    # session to learn from; otherwise every asana would collapse to the
    # same degenerate value, so report n/a honestly instead.
    if hist.empty or hist["session_id"].nunique() < 2:
        return [{"asana": a, "score": None, "prob_correct": None} for a in ASANAS]

    # Cut just AFTER the last recorded session, so the most recent practice
    # counts as history for the next-session prediction (it is not lookahead:
    # the predicted session is always the one that has not happened yet).
    cut = hist["started_at"].max() + pd.Timedelta(microseconds=1)
    rows = []
    for asana in ASANAS:
        f = _history_features(hist, asana, cut)
        f.update({name: float(f.get(name, 0.0)) for name in NUMERIC_FEATURES})
        f["asana__" + asana] = 1.0
        for other in ASANAS:
            f.setdefault("asana__" + other, 0.0)
        row = pd.DataFrame([f])[FEATURE_COLUMNS]
        score = float(np.clip(model["forest"].predict(row.values)[0], 0, 100))
        prob = None
        if model["clf"] is not None:
            prob = float(model["clf"].predict_proba(row.values)[0][1])
        rows.append({"asana": asana, "score": round(score, 1), "prob_correct": prob})
    return rows


def explain_model(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn the random-forest's feature importances into human-readable rules
    (the interpretable part of the "lightweight + explainable" justification).
    Sorted by how much each feature influences the expected score.
    """
    importances = model["forest"].feature_importances_
    explanation = []
    for name, imp in zip(model["feature_columns"], importances):
        if imp < 1e-4:
            continue
        label = name
        if name.startswith("asana__"):
            label = f"practicing {name.split('__')[1]}"
        explanation.append(
            {"feature": label, "impact": float(imp), "direction": "drives"}
        )
    return sorted(explanation, key=lambda r: r["impact"], reverse=True)[:8]


def recommend_next(
    model: Dict[str, Any],
    db_path: str,
    since: Optional[pd.Timestamp] = None,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Recommend which asana to practise next from the *previous* data in the
    chosen window (`since` = e.g. last week / last month / last year, None =
    all history). Uses the trained random forest's expected score and
    correctness probability, blended with how long it has been since each
    asana was last done, so the top pick is a useful next pose to work on and
    not just "the pose you are already best at".

    Returns rows sorted by priority (descending):
        {asana, priority, score, prob_correct, days_since, reason}
    """
    preds = predict_next(model, db_path, since=since)
    if not preds or all(p["score"] is None for p in preds):
        return []
    asana_map = {p["asana"]: p for p in preds}

    _blocks, hist = load_blocks(db_path)
    if since is not None:
        hist = hist[hist["started_at"] >= since]
    if hist.empty:
        return []
    cut = hist["started_at"].max()

    rows: List[Dict[str, Any]] = []
    for asana in ASANAS:
        p = asana_map[asana]
        if p["score"] is None:
            continue
        ev = hist[hist["asana"] == asana]
        days_since = 999.0
        if not ev.empty:
            days_since = max(0.0, (cut - ev["started_at"].max()).days)

        weakness = max(0.0, 100.0 - p["score"])     # 0-100, higher = needs work
        risk = (1.0 - p["prob_correct"]) if p["prob_correct"] is not None else 0.5
        recency = min(days_since / 30.0, 1.0) if days_since < 999 else 1.0
        priority = 0.4 * weakness + 0.3 * recency * 100.0 + 0.3 * risk * 100.0

        rows.append({
            "asana": asana,
            "priority": round(priority, 1),
            "score": p["score"],
            "prob_correct": p["prob_correct"],
            "days_since": days_since,
            "reason": _recommend_reason(p, days_since),
        })
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows[:top_n]


def _recommend_reason(p: Dict[str, Any], days_since: float) -> str:
    """Explain the top reason an asana scored high on the next-practice rank."""
    weakness = max(0.0, 100.0 - p["score"])
    risk = (1.0 - p["prob_correct"]) * 100.0 if p["prob_correct"] is not None else 50.0
    recency = min(days_since / 30.0, 1.0) * 100.0 if days_since < 999 else 100.0
    if recency >= weakness and recency >= risk:
        if days_since >= 999:
            return "never practised in this window"
        return f"not practised for {int(days_since)} day(s)"
    if risk >= weakness:
        prob = f"{p['prob_correct']:.0%}" if p["prob_correct"] is not None else "n/a"
        return f"lowest P(correct) ({prob}) - needs form work"
    return f"weakest expected score ({p['score']:.0f}/100)"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train / evaluate the practice-prediction model (Model 2)")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Path to session DB (default: {DEFAULT_DB})")
    args = parser.parse_args()

    prepared = prepare_training_set(args.db)
    if prepared is None:
        print(f"[model] No practice history in {args.db} yet.")
        print("        Run `python yoga_pose_detection.py` and do a few sessions first.")
        return

    X, blocks, y_score, y_correct = prepared
    model = train_and_evaluate(X, y_score, y_correct)
    m = model["metrics"]

    print(f"[model] Training set: {m['train_blocks']} practice blocks")
    if m["test_blocks"]:
        print(f"[model] Held-out (most recent {m['test_blocks']}) blocks:")
        print(f"        score R^2        = {m['test_r2']:.2f}")
        acc = m["test_accuracy"]
        print(f"        correctness acc  = {acc:.2f}" if acc is not None else "        correctness acc  = n/a (need both correct+incorrect history)")
    else:
        print("[model] Too few blocks for a hold-out split - trained on all data.")

    print("\nPredicted next-session outcome per asana:")
    print(f"{'Asana':14} {'Expected score':>15} {'P(correct)':>11}")
    preds = predict_next(model, args.db)
    for row in preds:
        prob = f"{row['prob_correct']:.0%}" if row["prob_correct"] is not None else "n/a"
        score = f"{row['score']:.1f}" if row["score"] is not None else "n/a"
        print(f"{row['asana']:14} {score:>15} {prob:>11}")
    if preds and preds[0]["score"] is None:
        print("\n[model] Predictions are n/a: you need at least 2 completed")
        print("        sessions before the model can predict your next practice.")

    print("\nWhat the model learned (top score drivers):")
    for r in explain_model(model):
        print(f"  {r['feature']:22} influence on expected score: {r['impact']:.3f}")


if __name__ == "__main__":
    main()
