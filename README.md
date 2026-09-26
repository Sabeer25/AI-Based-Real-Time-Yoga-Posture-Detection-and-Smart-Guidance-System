# AI-Based Real-Time Yoga Posture Detection

Real-time webcam system that detects **which asana** you're doing and whether
you're doing it **correctly**, using MediaPipe pose landmarks + coordinate
geometry (no training data for validation), with live voice/text feedback,
SQLite session history, a scikit-learn practice-prediction model, and a
Streamlit dashboard.

**26 supported asanas** (classic, easy/healthy set - see `pose_classifier.py`):

`Tadasana, Vrikshasana, Trikonasana, Virabhadrasana I/II/III, Utkatasana,
Uttanasana, Prasarita Padottanasana, Ardha Chandrasana, Natarajasana,
Anjaneyasana, Utthita Parsvakonasana, Garudasana, Sukhasana, Baddha Konasana,
Paschimottanasana, Vajrasana, Balasana, Ustrasana, Marjaryasana, Bitilasana,
Adho Mukha Svanasana, Bhujangasana, Setu Bandhasana, Savasana`

## Modules implemented

| Module | What it does | Files |
|---|---|---|
| **1. Pose detection** | MediaPipe `PoseLandmarker` extracts 33 landmarks; coordinate geometry computes joint angles/distances; rule-based classifier names the asana and marks it Correct/Incorrect | `yoga_pose_detection.py`, `pose_geometry.py`, `pose_classifier.py` |
| **2. Voice + text alerts** | pyttsx3 speaks joint-specific corrections; on-screen warnings with a cooldown | `yoga_pose_detection.py` (`VoiceFeedback`) |
| **3. Session history** | Every detected pose frame is logged to SQLite (asana, correct/incorrect, geometry score, key joint angles) | `session_store.py` |
| **4. Practice prediction** | Scikit-learn RandomForest (score regression + correctness classification) predicts next-session outcome per asana, with explainable feature importances | `practice_model.py`, `app.py` |

The two models in the design justification: **Model 1** = MediaPipe + coordinate
geometry (training-free validation), **Model 2** = the scikit-learn practice
prediction model (lightweight + interpretable on small per-user data).

## How it works

1. **MediaPipe Pose** extracts 33 body landmarks per frame from the webcam.
2. **Coordinate geometry** (`pose_geometry.py`) turns landmarks into joint
   angles (elbow, shoulder, hip, knee), torso lean, stance width, etc. — pure
   dot-product angle + Euclidean distance math.
3. **Rule-based classification** (`pose_classifier.py`) identifies the asana
   and produces specific feedback like *"Straighten your left knee"*. It is
   two-stage: first the *posture family* is found from gross geometry (seated,
   kneeling, prone, lying, standing, balance, lunge, forward fold, inverted),
   then the specific asana is matched within that family - which is what keeps
   26 asanas distinguishable from a handful of angles.
4. **Voice + text** (`Module 2`) speaks and shows that feedback live.
5. **SQLite** (`session_store.py`) records each detected-pose frame.
6. **Scikit-learn** (`practice_model.py`) aggregates that history into
   per-asana features (attempts, avg/best score, consistency, hold time,
   recency) and predicts what your next practice will look like.
7. **Streamlit** (`app.py`) visualizes history and predictions.

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt
```

## Run

### Live detection + history logging

```bash
python yoga_pose_detection.py
```

- Opens a camera preview but **does not detect anything until you press
  `SPACE` to start a practice session**. Press `SPACE` again to end it - only
  the time inside a started session is logged to `sessions.db`.
- While a session is running, correctly placed joints and skeleton lines are
  drawn **green** and any joint/line that is off is drawn **red**.
- On first run it auto-downloads the small MediaPipe pose model
  (`pose_landmarker_lite.task`, ~5–6 MB) into the project folder — this needs
  an internet connection once; after that it runs fully offline.

**Controls** (with the video window focused):
| Key | Action |
|---|---|
| `Space` | Start / end the practice session (toggles detection + logging) |
| `q` | Quit (saves the session + prints predictions) |
| `v` | Toggle voice feedback on/off |
| `s` | Save a screenshot to `screenshots/` |
| `p` | Print next-session predictions to the console |

**Useful flags:**
```bash
python yoga_pose_detection.py --camera 1        # use a different webcam
python yoga_pose_detection.py --no-voice        # start with voice off
python yoga_pose_detection.py --no-log          # skip history logging
python yoga_pose_detection.py --db my_data.db   # use a different DB file
```

### Model 2 - practice prediction (no webcam needed)

```bash
python practice_model.py            # trains on sessions.db, prints metrics + predictions
```

### Dashboard

```bash
streamlit run app.py
```

Shows session history, per-asana score trends, the model's next-session
predictions, and the interpretable rules the model learned. A small demo
`sessions.db` is included so the dashboard has data on first run — delete it
(or `--db`) to start recording your own history.

## Testing the logic without a webcam

```bash
python test_pose_logic.py      # geometry + classifier sanity checks
python test_session_model.py   # SQLite storage + model pipeline sanity checks
```

Both are deterministic and camera-free — handy for the review demo or report.

## Tuning

- **Geometry thresholds** (`pose_classifier.py`): what counts as a "straight
  knee", "wide stance", etc. Adjust if detection feels too strict/loose.
- **Model** (`practice_model.py`): features, asana list, and train/test split
  are near the top. Predictions are recomputed on demand, so no retraining step.

## Design decisions (from the review deck)

- **MediaPipe + coordinate geometry over deep learning** — training-free,
  explainable, real-time; fits a single-semester project with no dataset.
- **arctan2-based angle computation** — scale/distance-invariant, accurate at
  any camera distance.
- **pyttsx3 over cloud TTS** — fully offline, zero API cost, no network latency.
- **SQLite + Streamlit over a cloud stack** — zero-config, sufficient for
  single-user history + demo dashboard, upgradable without redesign.
- **Scikit-learn over deep learning for prediction** — proportionate to the
  small per-user dataset, fast, interpretable (coefficients shown in dashboard).
