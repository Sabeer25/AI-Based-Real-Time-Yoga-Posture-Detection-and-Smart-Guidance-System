"""
yoga_pose_detection.py
----------------------
AI-Based Real-Time Yoga Posture Detection - Module 1 (First Review)

Real-time webcam application that:
  1. Extracts 33 body landmarks per frame using MediaPipe Pose (Tasks API).
  2. Computes joint angles and distances using coordinate geometry
     (pose_geometry.py) - no ML model, no training data needed for the
     posture-validation step itself.
  3. Identifies which asana is being attempted and whether it is being
     performed correctly, with specific corrective feedback
     (pose_classifier.py).
  4. Displays the result live on screen (pose name, Correct/Incorrect,
     feedback) and optionally speaks corrective feedback aloud.

Detection only runs while a practice session is active: press SPACE to start
the session (camera shows a live preview before that, nothing is logged or
classified) and SPACE again to end it. Joints and skeleton lines are drawn
green where the pose is right and red where it has drifted off.

Supported asanas (Module 1 scope): 26 classic asanas - see SUPPORTED_ASANAS
in pose_classifier.py (Tadasana through Savasana).

Run:
    python yoga_pose_detection.py
Controls (while the window is focused):
    Space - start / end the practice session (detection + logging)
    q     - quit
    v     - toggle voice feedback on/off
    s     - save a screenshot to screenshots/
    p     - print next-session predictions to the console
"""

import argparse
import collections
import os
import sys
import threading
import time
import urllib.request

import cv2

from pose_geometry import (
    Point, POSE_CONNECTIONS, VISIBILITY_THRESHOLD, CORE_LANDMARKS, extract_features,
)
from pose_classifier import classify, UNKNOWN
from session_store import SessionRecorder
from practice_model import predict_next, prepare_training_set, train_and_evaluate

# Friendly names for the landmarks extract_features needs - shown on screen
# when MediaPipe sees a body but part of it is cropped out of the frame.
LANDMARK_NAMES = {
    11: "left shoulder", 12: "right shoulder",
    13: "left elbow", 14: "right elbow",
    15: "left hand", 16: "right hand",
    23: "left hip", 24: "right hip",
    25: "left knee", 26: "right knee",
    27: "left foot", 28: "right foot",
}

# --------------------------------------------------------------------------
# MediaPipe Tasks API setup
# --------------------------------------------------------------------------
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Colors (BGR, since we draw with OpenCV)
COLOR_OK = (80, 200, 80)
COLOR_BAD = (60, 60, 230)
COLOR_TEXT = (255, 255, 255)


def ensure_model(model_path: str) -> str:
    """Download the PoseLandmarker model on first run if it isn't present."""
    if os.path.exists(model_path):
        return model_path
    print(f"[setup] Pose model not found locally - downloading (~5-6 MB) ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[setup] Saved model to {model_path}")
    except Exception as exc:
        print(
            "[error] Could not download the MediaPipe pose model.\n"
            f"        Reason: {exc}\n"
            f"        Manually download it from:\n        {MODEL_URL}\n"
            f"        and place it next to this script as '{MODEL_FILENAME}'."
        )
        sys.exit(1)
    return model_path


def build_landmarker(model_path: str) -> mp_vision.PoseLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


class VoiceFeedback:
    """
    Thin, non-blocking wrapper around pyttsx3. Speaks at most one message
    every `cooldown` seconds so it doesn't stack up alerts and fall behind
    real time. Silently disables itself if pyttsx3 / a TTS engine isn't
    available on this machine (e.g. missing OS speech driver).
    """

    def __init__(self, cooldown: float = 4.0):
        self.enabled = True
        self.cooldown = cooldown
        self._last_spoken = ""
        self._last_time = 0.0
        self._lock = threading.Lock()
        try:
            import pyttsx3  # imported lazily so the app still runs without it
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 165)
            self._available = True
        except Exception:
            self._engine = None
            self._available = False
            print("[info] Voice feedback unavailable (pyttsx3 / TTS engine not found) - continuing without it.")

    def maybe_speak(self, message: str):
        if not (self.enabled and self._available and message):
            return
        now = time.time()
        if message == self._last_spoken and (now - self._last_time) < self.cooldown:
            return
        if (now - self._last_time) < 1.5:
            return
        self._last_spoken = message
        self._last_time = now
        threading.Thread(target=self._speak, args=(message,), daemon=True).start()

    def _speak(self, message: str):
        with self._lock:
            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception:
                pass


def draw_skeleton(frame, landmarks_px, bad_landmarks=frozenset()):
    """Draw the skeleton colour-coded by correctness: joints/lines touched by
    a failing rule are drawn red (COLOR_BAD), everything else stays green."""
    bad = set(bad_landmarks or ())
    for a_idx, b_idx in POSE_CONNECTIONS:
        a, b = landmarks_px.get(a_idx), landmarks_px.get(b_idx)
        if a and b and a.visibility >= VISIBILITY_THRESHOLD and b.visibility >= VISIBILITY_THRESHOLD:
            color = COLOR_BAD if a_idx in bad or b_idx in bad else COLOR_OK
            cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), color, 2)
    for p_idx, p in landmarks_px.items():
        if p.visibility >= VISIBILITY_THRESHOLD:
            color = COLOR_BAD if p_idx in bad else COLOR_OK
            cv2.circle(frame, (int(p.x), int(p.y)), 4, color, -1)


def draw_hud(frame, asana, is_correct, feedback, score, fps, voice_on,
             session_active, elapsed_s=0.0, notice=""):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if session_active:
        status_color = COLOR_OK if is_correct else COLOR_BAD
        status_text = "CORRECT" if asana != UNKNOWN and is_correct else ("INCORRECT" if asana != UNKNOWN else "")

        cv2.putText(frame, f"Asana: {asana}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2)
        if status_text:
            cv2.putText(frame, status_text, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.putText(frame, f"Match: {score:.0f}%", (230, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)

        cv2.putText(frame, f"Session: {elapsed_s:.0f}s", (w - 300, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OK, 1)
        cv2.putText(frame, "SPACE: end session", (w - 300, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1)

        if asana == UNKNOWN:
            # Say WHY nothing is detected instead of leaving a silent screen:
            # usually part of the body is cropped out of the camera view.
            msg = notice or "Step fully into the camera view"
            cv2.putText(frame, msg, (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2)
            cv2.putText(frame, "Keep your head, hands, hips and feet inside the frame",
                        (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
        else:
            # Feedback lines (max 2 shown on-screen to avoid clutter)
            y = 95
            for msg in feedback[:2]:
                cv2.putText(frame, f"- {msg}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1)
                y += 22
    else:
        cv2.putText(frame, "IDLE - detection paused", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 210, 255), 2)
        cv2.putText(frame, "Press SPACE to start the session",
                    (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    voice_label = "Voice: ON" if voice_on else "Voice: OFF"
    cv2.putText(frame, voice_label, (w - 140, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    cv2.putText(frame, "SPACE: start/end session  q: quit  v: voice  s: shot  p: predict",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="Real-time Yoga Posture Detection (MediaPipe + Geometry)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice feedback at startup")
    parser.add_argument(
        "--smoothing", type=int, default=7,
        help="Number of recent frames to majority-vote over, to reduce flicker"
    )
    parser.add_argument(
        "--db", default=None,
        help="SQLite file to log this session's history into (default: sessions.db)"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip session history logging entirely"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = args.db or os.path.join(script_dir, "sessions.db")
    model_path = ensure_model(os.path.join(script_dir, MODEL_FILENAME))
    landmarker = build_landmarker(model_path)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[error] Could not open webcam index {args.camera}. Try a different --camera value.")
        sys.exit(1)

    voice = VoiceFeedback()
    voice.enabled = not args.no_voice

    asana_history = collections.deque(maxlen=args.smoothing)

    os.makedirs(os.path.join(script_dir, "screenshots"), exist_ok=True)

    print("[info] Starting camera feed. Press 'q' in the window to quit.")
    print("[info] Detection is OFF until you press SPACE to start a session.")

    recorder = SessionRecorder(db_path) if not args.no_log else None

    try:
        _run_loop(
            landmarker, cap, voice, args, asana_history,
            script_dir, recorder,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if recorder is not None:
            try:
                recorder.finish()  # no-op if the session was ended with SPACE
            except Exception:
                pass
        _print_session_summary(recorder, db_path)


def _run_loop(landmarker, cap, voice, args, asana_history, script_dir, recorder):
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    start_ts = time.time()
    session_active = False
    session_start_ts = None
    bad_feature_run = 0   # consecutive frames where a body was seen but unusable

    def toggle_session():
        """Start or end the practice session. Detection + logging only run
        while a session is active, so a fresh app launch detects nothing."""
        nonlocal session_active, session_start_ts
        if session_active:
            session_active = False
            session_start_ts = None
            if recorder is not None:
                recorder.stop()
                print(f"[info] Session #{recorder.session_id} ended and saved to {recorder.db_path}.")
            else:
                print("[info] Session ended.")
        else:
            session_active = True
            session_start_ts = time.time()
            asana_history.clear()
            if recorder is not None:
                recorder.start()
            print("[info] Session started - detection + logging active. Press SPACE to end.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] Frame grab failed, retrying...")
            continue

        frame = cv2.flip(frame, 1)  # mirror for a natural selfie view

        asana, is_correct, feedback, score, bad = UNKNOWN, False, [], 0.0, set()
        notice = ""

        if session_active:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_ts) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                norm_pts = {i: Point(p.x, p.y, p.visibility if p.visibility is not None else 1.0)
                            for i, p in enumerate(lm)}
                px_pts = {i: Point(p.x * w, p.y * h, p.visibility if p.visibility is not None else 1.0)
                          for i, p in enumerate(lm)}
                features = extract_features(norm_pts)

                if features is None:
                    # MediaPipe sees a body, but a required joint is cropped or
                    # occluded (usually ankles or wrists near the frame edge).
                    # Show the user WHICH joints are missing instead of a
                    # silent "No Pose" that looks like the pose isn't supported.
                    missing = [LANDMARK_NAMES.get(i, f"joint {i}")
                               for i in CORE_LANDMARKS
                               if norm_pts.get(i) is None
                               or norm_pts[i].visibility < VISIBILITY_THRESHOLD]
                    if missing:
                        notice = "Can't see your " + ", ".join(missing[:3])
                    else:
                        notice = "Body partly out of frame - step back from the camera"
                    if frame_count % 30 == 0:
                        print(f"[debug] pose visible but frame unusable: {notice}")
                    bad_feature_run += 1
                    if bad_feature_run >= 5:
                        asana_history.clear()  # stale votes shouldn't outlive the pose
                else:
                    bad_feature_run = 0
                    res = classify(features)
                    asana, is_correct, feedback, score = res.asana, res.is_correct, res.feedback, res.score
                    bad = res.bad_landmarks
                    draw_skeleton(frame, px_pts, bad)

                    asana_history.append(asana)
                    # Majority vote over recent frames to stop the label flickering
                    if asana_history:
                        asana = collections.Counter(asana_history).most_common(1)[0][0]

                    if recorder is not None and asana != UNKNOWN:
                        recorder.log_event(
                            ts=time.time() - session_start_ts,
                            asana=asana,
                            is_correct=is_correct,
                            score=score,
                            features=features,
                        )

                    if asana != UNKNOWN:
                        if is_correct:
                            voice.maybe_speak(f"{asana} correct")
                        elif feedback:
                            voice.maybe_speak(feedback[0])
            else:
                notice = "No body detected - step fully into the camera view"
                if frame_count % 30 == 0:
                    print(f"[debug] {notice}")

        elapsed_s = (time.time() - session_start_ts) if session_active else 0.0
        draw_hud(frame, asana, is_correct, feedback, score, fps, voice.enabled,
                 session_active, elapsed_s, notice)
        cv2.imshow("Yoga Posture Detection - Modules 1-3", frame)

        frame_count += 1
        now = time.time()
        if now - prev_time >= 0.5:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("v"):
            voice.enabled = not voice.enabled
        elif key == ord("s"):
            fname = os.path.join(script_dir, "screenshots", f"pose_{int(time.time())}.png")
            cv2.imwrite(fname, frame)
            print(f"[info] Saved screenshot: {fname}")
        elif key == ord("p"):
            _print_next_predictions(recorder)
        elif key == ord(" "):  # SPACE toggles start / end of the session
            toggle_session()


def _print_next_predictions(recorder):
    db = recorder.db_path
    prepared = prepare_training_set(db)
    if prepared is None:
        print("[model] No history logged yet - predictions not available.")
        return
    X, blocks, y_score, y_correct = prepared
    model = train_and_evaluate(X, y_score, y_correct)
    print("\n[model] Predicted next-session outcome per asana:")
    for row in predict_next(model, db):
        prob = f"{row['prob_correct']:.0%}" if row["prob_correct"] is not None else "n/a"
        score = f"{row['score']:.1f}" if row["score"] is not None else "n/a"
        print(f"  {row['asana']:14} expected score: {score:>5}   P(correct): {prob}")


def _print_session_summary(recorder, db_path):
    if recorder is None:
        print("[info] Session logging disabled (--no-log).")
        return
    if recorder.session_id is None:
        print("[info] No session was started (press SPACE in the window to begin one) - nothing saved.")
        return
    sid = recorder.session_id
    print(f"[info] Session #{sid} saved to {db_path}.")
    try:
        prepared = prepare_training_set(db_path)
        if prepared is None:
            print("[info] No pose events were detected this session.")
            return
        X, blocks, y_score, y_correct = prepared
        model = train_and_evaluate(X, y_score, y_correct)
        print("[model] Updated practice-prediction model from your history:")
        for row in predict_next(model, db_path):
            prob = f"{row['prob_correct']:.0%}" if row["prob_correct"] is not None else "n/a"
            score = f"{row['score']:.1f}" if row["score"] is not None else "n/a"
            print(f"  {row['asana']:14} expected score: {score:>5}   P(correct): {prob}")
        print("[info] View the full dashboard with:  streamlit run app.py")
    except Exception as exc:
        print(f"[info] Prediction summary skipped ({exc}).")


if __name__ == "__main__":
    main()
