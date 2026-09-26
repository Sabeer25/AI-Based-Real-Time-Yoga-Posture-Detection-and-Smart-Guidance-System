"""
app.py
------
Streamlit dashboard (Module 3/4 front end) - "SQLite + Streamlit over full
cloud stack" from the design justification.

Shows:
  * practice history stored in the SQLite DB (sessions, scores per asana),
  * the scikit-learn model's next-session predictions for each asana,
  * what the model learned (interpretable coefficients).

Run:
    streamlit run app.py

The dashboard only reads the database written by yoga_pose_detection.py -
it never touches the webcam or the detection pipeline.
"""

import os

import pandas as pd
import streamlit as st

from session_store import last_session_stats, list_sessions, session_events, stats
from practice_model import (
    DEFAULT_DB, load_blocks, prepare_training_set,
    train_and_evaluate, predict_next, recommend_next, explain_model,
)

st.set_page_config(page_title="Yoga Practice Dashboard", page_icon="", layout="wide")

st.title("AI-Based Yoga Posture Detection - Practice Dashboard")
st.caption("SQLite session history + scikit-learn practice-prediction model (Modules 3-4).")

with st.sidebar:
    st.header("Data source")
    db_path = st.text_input("SQLite database", value=DEFAULT_DB)
    if not os.path.exists(db_path):
        st.warning(f"Database not found: {db_path}\n\nRun `python yoga_pose_detection.py` "
                   "and do a practice session first.")
        st.stop()
    info = stats(db_path)
    last = last_session_stats(db_path)
    st.metric("Sessions", info["sessions"])
    st.metric("Pose events logged", info["events"])
    st.metric("Correct", f"{info['correct_events']} ({info['correct_events'] / max(info['events'],1):.0%})")
    st.metric("Total practice time", f"{info['total_time_s']/60:.1f} min")
    if last is not None:
        st.metric("Last session practice time",
                  f"{last['duration_s']/60:.1f} min",
                  help=f"Session #{last['id']} [{last['started_at']}]")
        st.metric("Last session poses",
                  f"{last['events']} events, {last['asana_count']} asanas")
    else:
        st.metric("Last session practice time", "n/a")
    st.caption(f"File: `{info['db_path']}`")

tab_hist, tab_model = st.tabs(["Practice History", "Practice Prediction (Model 2)"])

# --------------------------------------------------------------------------
# Tab 1 - history
# --------------------------------------------------------------------------
with tab_hist:
    blocks, _ = load_blocks(db_path)
    if blocks.empty:
        st.info("No pose events logged yet - do a practice session first.")
    else:
        sessions = list_sessions(db_path)
        st.subheader("Sessions")
        cols = ["id", "started_at", "ended_at", "duration_s", "pose_count"]
        df_s = pd.DataFrame(sessions)[cols].rename(columns={
            "id": "Session", "started_at": "Started", "ended_at": "Ended",
            "duration_s": "Duration (s)", "pose_count": "Pose events",
        })
        st.dataframe(df_s, width="stretch", hide_index=True)

        st.subheader("Score per asana (across practice blocks)")
        trend = blocks[["started_at", "asana", "score"]].copy()
        trend["started_at"] = pd.to_datetime(trend["started_at"]).dt.tz_localize(None)
        pivot = trend.pivot_table(index="started_at", columns="asana", values="score", aggfunc="mean")
        if not pivot.empty:
            st.line_chart(pivot)

        st.subheader("Time held per asana")
        hold = blocks.groupby("asana")["hold_s"].sum().sort_values(ascending=False)
        if not hold.empty:
            st.bar_chart(hold)

        sel = st.selectbox("Inspect a session", [f"#{s['id']} - {s['started_at']}" for s in sessions][::-1])
        if sel:
            sid = int(sel.split(" - ")[0][1:])
            ev = session_events(db_path, sid)
            if ev:
                df_e = pd.DataFrame(ev)[["ts", "asana", "is_correct", "score"]].rename(columns={
                    "ts": "Time (s)", "is_correct": "Correct",
                })
                df_e["Correct"] = df_e["Correct"].map({1: "yes", 0: "no"})
                st.dataframe(df_e, width="stretch", hide_index=True)
                st.line_chart(df_e.set_index("Time (s)")["score"])

# --------------------------------------------------------------------------
# Tab 2 - the scikit-learn model
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Which asana should you practise next?")
    st.caption(
        "A random forest (Model 2 - `RandomForestRegressor` for the expected "
        "score, `RandomForestClassifier` for the chance of a CORRECT posture) "
        "is retrained from your session history every time the page refreshes. "
        "Pick a window of *previous* data below - the model then learns from "
        "that history and recommends the most useful asana for the next "
        "practice."
    )

    period = st.selectbox(
        "Base the prediction on previous data from",
        ["All time (whole history)", "Last year", "Last month", "Last week"],
        index=0,
    )
    since = None
    if period != "All time (whole history)":
        days = {"Last year": 365, "Last month": 30, "Last week": 7}[period]
        since = pd.Timestamp.now("UTC") - pd.Timedelta(days=days)

    prepared = prepare_training_set(db_path, since=since)
    if prepared is None:
        st.info(
            "Not enough practice history in the selected window to train the "
            "model yet - do at least two sessions inside that window."
        )
    else:
        X, _blocks, y_score, y_correct = prepared
        model = train_and_evaluate(X, y_score, y_correct)

        picks = recommend_next(model, db_path, since=since, top_n=5)
        if picks:
            top = picks[0]
            st.success(
                f"**Recommended for your next practice: {top['asana']}** - "
                f"{top['reason']}."
            )
            df_picks = pd.DataFrame(picks).rename(columns={
                "asana": "Asana", "priority": "Priority (0-100)",
                "score": "Expected score (0-100)", "prob_correct": "P(correct)",
                "days_since": "Days since last done", "reason": "Why",
            })
            df_picks["Expected score (0-100)"] = df_picks["Expected score (0-100)"].map(
                lambda s: f"{s:.1f}"
            )
            df_picks["P(correct)"] = df_picks["P(correct)"].map(
                lambda p: "n/a" if p is None else f"{p:.0%}"
            )
            df_picks["Days since last done"] = df_picks["Days since last done"].map(
                lambda d: "never" if d >= 900 else f"{int(d)}"
            )
            st.dataframe(df_picks, width="stretch", hide_index=True)
            st.caption(
                "Priority = 0.4 x how far the expected score is from 100 + "
                "0.3 x how many weeks since you last did the asana + "
                "0.3 x how likely the pose comes out wrong."
            )
        else:
            st.info(
                "The model needs at least two completed sessions in the chosen "
                "window before it can rank your next asana."
            )

        st.divider()
        st.subheader("Next-session prediction per asana")
        preds = predict_next(model, db_path, since=since)
        df_p = pd.DataFrame(preds).rename(columns={
            "asana": "Asana", "score": "Expected score (0-100)",
            "prob_correct": "P(correct)",
        })
        df_p["Expected score (0-100)"] = df_p["Expected score (0-100)"].apply(
            lambda s: "n/a" if s is None else f"{s:.1f}"
        )
        df_p["P(correct)"] = df_p["P(correct)"].apply(
            lambda p: "n/a" if p is None else f"{p:.0%}"
        )
        if picks:
            top_asana = picks[0]["asana"]
            df_p["Next pick"] = df_p["Asana"].map(
                lambda a: "" if a != top_asana else "RECOMMENDED"
            )
        st.dataframe(df_p, width="stretch", hide_index=True)

        m = model["metrics"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Training blocks", m["train_blocks"])
        col_b.metric("Held-out blocks", m["test_blocks"] if m["test_blocks"] else "n/a")
        r2 = m["test_r2"]
        col_c.metric("Score R-squared (held-out)", f"{r2:.2f}" if r2 is not None else "n/a")

        if m["test_accuracy"] is not None:
            st.caption(f"Correctness classification accuracy (held-out): {m['test_accuracy']:.0%}")
        else:
            st.caption("Correctness accuracy not shown: history so far is mostly one class.")

        st.subheader("What the model learned")
        st.caption("Random-forest feature importances - which factors carry "
                   "the most weight when predicting your next practice.")
        rules = explain_model(model)
        if rules:
            df_r = pd.DataFrame(rules)[["feature", "impact"]].rename(columns={
                "feature": "Factor", "impact": "Model influence",
            })
            st.dataframe(df_r, width="stretch", hide_index=True)
        else:
            st.caption("No rules to show yet - practice a bit more.")
