"""
test_pose_logic.py
--------------------
Quick, camera-free sanity check for the geometry + classification engine.
Builds synthetic landmark coordinates for each supported asana (one clean
attempt, a few flawed attempts) and prints what the classifier decides.

Run:
    python test_pose_logic.py

This is useful to show in your review / demo that the posture-validation
logic itself is deterministic coordinate geometry and can be verified
without needing MediaPipe or a live webcam at all.

All 26 supported asanas are covered with a "clean" synthetic attempt that
must be identified correctly AND scored CORRECT. A handful of intentionally
flawed attempts must be identified as the right asana but scored INCORRECT.
"""

from pose_geometry import (
    Point, extract_features,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
)
from pose_classifier import classify, SUPPORTED_ASANAS


def mk(coords):
    return {idx: Point(x, y, 1.0) for idx, (x, y) in coords.items()}


# --------------------------------------------------------------------------
# Clean attempts - one per supported asana. Coordinates are hand-tuned so the
# geometry matches the ideal shape of each pose (see pose_geometry.py for the
# feature definitions, pose_classifier.py for the thresholds).
# --------------------------------------------------------------------------
CASES = {
    "Tadasana (clean)": mk({
        LEFT_SHOULDER: (0.45, 0.30), RIGHT_SHOULDER: (0.55, 0.30),
        LEFT_ELBOW: (0.44, 0.42), RIGHT_ELBOW: (0.56, 0.42),
        LEFT_WRIST: (0.44, 0.54), RIGHT_WRIST: (0.56, 0.54),
        LEFT_HIP: (0.47, 0.55), RIGHT_HIP: (0.53, 0.55),
        LEFT_KNEE: (0.47, 0.75), RIGHT_KNEE: (0.53, 0.75),
        LEFT_ANKLE: (0.47, 0.95), RIGHT_ANKLE: (0.53, 0.95),
        NOSE: (0.50, 0.24),
    }),
    "Vrikshasana (clean)": mk({
        LEFT_SHOULDER: (0.45, 0.28), RIGHT_SHOULDER: (0.55, 0.28),
        LEFT_ELBOW: (0.40, 0.15), RIGHT_ELBOW: (0.60, 0.15),
        LEFT_WRIST: (0.50, 0.05), RIGHT_WRIST: (0.50, 0.05),
        LEFT_HIP: (0.47, 0.55), RIGHT_HIP: (0.53, 0.55),
        LEFT_KNEE: (0.47, 0.75), RIGHT_KNEE: (0.70, 0.55),
        LEFT_ANKLE: (0.47, 0.95), RIGHT_ANKLE: (0.46, 0.62),
        NOSE: (0.50, 0.22),
    }),
    "Trikonasana (clean)": mk({
        LEFT_SHOULDER: (0.25, 0.42), RIGHT_SHOULDER: (0.48, 0.28),
        LEFT_ELBOW: (0.15, 0.60), RIGHT_ELBOW: (0.55, 0.15),
        LEFT_WRIST: (0.10, 0.80), RIGHT_WRIST: (0.60, 0.02),
        LEFT_HIP: (0.45, 0.55), RIGHT_HIP: (0.58, 0.55),
        LEFT_KNEE: (0.35, 0.75), RIGHT_KNEE: (0.68, 0.75),
        LEFT_ANKLE: (0.25, 0.95), RIGHT_ANKLE: (0.75, 0.95),
        NOSE: (0.35, 0.32),
    }),
    "Virabhadrasana I (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.30), RIGHT_SHOULDER: (0.53, 0.30),
        LEFT_ELBOW: (0.47, 0.18), RIGHT_ELBOW: (0.53, 0.18),
        LEFT_WRIST: (0.47, 0.06), RIGHT_WRIST: (0.53, 0.06),
        LEFT_HIP: (0.45, 0.60), RIGHT_HIP: (0.55, 0.60),
        LEFT_KNEE: (0.40, 0.70), RIGHT_KNEE: (0.68, 0.60),
        LEFT_ANKLE: (0.34, 0.95), RIGHT_ANKLE: (0.68, 0.90),
        NOSE: (0.50, 0.24),
    }),
    "Virabhadrasana II (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.30), RIGHT_SHOULDER: (0.53, 0.30),
        LEFT_ELBOW: (0.30, 0.30), RIGHT_ELBOW: (0.70, 0.30),
        LEFT_WRIST: (0.15, 0.30), RIGHT_WRIST: (0.85, 0.30),
        LEFT_HIP: (0.45, 0.60), RIGHT_HIP: (0.55, 0.60),
        LEFT_KNEE: (0.40, 0.70), RIGHT_KNEE: (0.68, 0.60),
        LEFT_ANKLE: (0.34, 0.95), RIGHT_ANKLE: (0.68, 0.90),
        NOSE: (0.50, 0.24),
    }),
    "Virabhadrasana III (clean)": mk({
        LEFT_SHOULDER: (0.35, 0.48), RIGHT_SHOULDER: (0.42, 0.48),
        LEFT_ELBOW: (0.28, 0.48), RIGHT_ELBOW: (0.35, 0.48),
        LEFT_WRIST: (0.20, 0.48), RIGHT_WRIST: (0.27, 0.48),
        LEFT_HIP: (0.45, 0.50), RIGHT_HIP: (0.55, 0.50),
        LEFT_KNEE: (0.45, 0.68), RIGHT_KNEE: (0.70, 0.50),
        LEFT_ANKLE: (0.45, 0.86), RIGHT_ANKLE: (0.83, 0.50),
        NOSE: (0.40, 0.48),
    }),
    "Utkatasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.45), RIGHT_SHOULDER: (0.53, 0.45),
        LEFT_ELBOW: (0.47, 0.33), RIGHT_ELBOW: (0.53, 0.33),
        LEFT_WRIST: (0.47, 0.20), RIGHT_WRIST: (0.53, 0.20),
        LEFT_HIP: (0.45, 0.75), RIGHT_HIP: (0.52, 0.75),
        LEFT_KNEE: (0.50, 0.75), RIGHT_KNEE: (0.57, 0.75),
        LEFT_ANKLE: (0.50, 0.93), RIGHT_ANKLE: (0.57, 0.93),
        NOSE: (0.50, 0.40),
    }),
    "Uttanasana (clean)": mk({
        LEFT_SHOULDER: (0.48, 0.60), RIGHT_SHOULDER: (0.52, 0.60),
        LEFT_ELBOW: (0.47, 0.73), RIGHT_ELBOW: (0.53, 0.73),
        LEFT_WRIST: (0.47, 0.88), RIGHT_WRIST: (0.53, 0.88),
        LEFT_HIP: (0.47, 0.52), RIGHT_HIP: (0.53, 0.52),
        LEFT_KNEE: (0.47, 0.70), RIGHT_KNEE: (0.53, 0.70),
        LEFT_ANKLE: (0.47, 0.90), RIGHT_ANKLE: (0.53, 0.90),
        NOSE: (0.50, 0.65),
    }),
    "Prasarita Padottanasana (clean)": mk({
        LEFT_SHOULDER: (0.42, 0.60), RIGHT_SHOULDER: (0.52, 0.60),
        LEFT_ELBOW: (0.39, 0.73), RIGHT_ELBOW: (0.56, 0.73),
        LEFT_WRIST: (0.35, 0.86), RIGHT_WRIST: (0.65, 0.86),
        LEFT_HIP: (0.45, 0.52), RIGHT_HIP: (0.55, 0.52),
        LEFT_KNEE: (0.42, 0.70), RIGHT_KNEE: (0.58, 0.70),
        LEFT_ANKLE: (0.30, 0.90), RIGHT_ANKLE: (0.70, 0.90),
        NOSE: (0.47, 0.66),
    }),
    "Ardha Chandrasana (clean)": mk({
        LEFT_SHOULDER: (0.58, 0.35), RIGHT_SHOULDER: (0.62, 0.45),
        LEFT_ELBOW: (0.15, 0.18), RIGHT_ELBOW: (0.62, 0.60),
        LEFT_WRIST: (0.10, 0.10), RIGHT_WRIST: (0.62, 0.80),
        LEFT_HIP: (0.42, 0.50), RIGHT_HIP: (0.52, 0.50),
        LEFT_KNEE: (0.30, 0.44), RIGHT_KNEE: (0.58, 0.70),
        LEFT_ANKLE: (0.20, 0.42), RIGHT_ANKLE: (0.61, 0.90),
        NOSE: (0.55, 0.32),
    }),
    "Natarajasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.30), RIGHT_SHOULDER: (0.53, 0.30),
        LEFT_ELBOW: (0.47, 0.18), RIGHT_ELBOW: (0.62, 0.48),
        LEFT_WRIST: (0.47, 0.06), RIGHT_WRIST: (0.55, 0.42),
        LEFT_HIP: (0.47, 0.50), RIGHT_HIP: (0.53, 0.50),
        LEFT_KNEE: (0.47, 0.70), RIGHT_KNEE: (0.60, 0.62),
        LEFT_ANKLE: (0.47, 0.90), RIGHT_ANKLE: (0.52, 0.45),
        NOSE: (0.50, 0.24),
    }),
    "Anjaneyasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.35), RIGHT_SHOULDER: (0.53, 0.35),
        LEFT_ELBOW: (0.47, 0.23), RIGHT_ELBOW: (0.53, 0.23),
        LEFT_WRIST: (0.47, 0.12), RIGHT_WRIST: (0.53, 0.12),
        LEFT_HIP: (0.45, 0.55), RIGHT_HIP: (0.55, 0.55),
        LEFT_KNEE: (0.40, 0.72), RIGHT_KNEE: (0.68, 0.55),
        LEFT_ANKLE: (0.40, 0.88), RIGHT_ANKLE: (0.68, 0.88),
        NOSE: (0.50, 0.30),
    }),
    "Utthita Parsvakonasana (clean)": mk({
        LEFT_SHOULDER: (0.62, 0.30), RIGHT_SHOULDER: (0.68, 0.42),
        LEFT_ELBOW: (0.58, 0.22), RIGHT_ELBOW: (0.70, 0.50),
        LEFT_WRIST: (0.52, 0.12), RIGHT_WRIST: (0.72, 0.66),
        LEFT_HIP: (0.45, 0.60), RIGHT_HIP: (0.55, 0.60),
        LEFT_KNEE: (0.40, 0.70), RIGHT_KNEE: (0.68, 0.60),
        LEFT_ANKLE: (0.34, 0.95), RIGHT_ANKLE: (0.68, 0.90),
        NOSE: (0.60, 0.30),
    }),
    "Garudasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.30), RIGHT_SHOULDER: (0.53, 0.30),
        LEFT_ELBOW: (0.56, 0.38), RIGHT_ELBOW: (0.44, 0.38),
        LEFT_WRIST: (0.48, 0.46), RIGHT_WRIST: (0.52, 0.46),
        LEFT_HIP: (0.45, 0.62), RIGHT_HIP: (0.52, 0.62),
        LEFT_KNEE: (0.55, 0.72), RIGHT_KNEE: (0.58, 0.66),
        LEFT_ANKLE: (0.47, 0.92), RIGHT_ANKLE: (0.50, 0.72),
        NOSE: (0.50, 0.25),
    }),
    "Sukhasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.40), RIGHT_SHOULDER: (0.53, 0.40),
        LEFT_ELBOW: (0.46, 0.52), RIGHT_ELBOW: (0.54, 0.52),
        LEFT_WRIST: (0.46, 0.64), RIGHT_WRIST: (0.54, 0.64),
        LEFT_HIP: (0.47, 0.65), RIGHT_HIP: (0.53, 0.65),
        LEFT_KNEE: (0.36, 0.58), RIGHT_KNEE: (0.64, 0.58),
        LEFT_ANKLE: (0.44, 0.72), RIGHT_ANKLE: (0.56, 0.72),
        NOSE: (0.50, 0.34),
    }),
    "Baddha Konasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.40), RIGHT_SHOULDER: (0.53, 0.40),
        LEFT_ELBOW: (0.46, 0.52), RIGHT_ELBOW: (0.54, 0.52),
        LEFT_WRIST: (0.46, 0.64), RIGHT_WRIST: (0.54, 0.64),
        LEFT_HIP: (0.47, 0.65), RIGHT_HIP: (0.53, 0.65),
        LEFT_KNEE: (0.35, 0.58), RIGHT_KNEE: (0.65, 0.58),
        LEFT_ANKLE: (0.49, 0.70), RIGHT_ANKLE: (0.51, 0.70),
        NOSE: (0.50, 0.34),
    }),
    "Paschimottanasana (clean)": mk({
        LEFT_SHOULDER: (0.52, 0.60), RIGHT_SHOULDER: (0.56, 0.58),
        LEFT_ELBOW: (0.45, 0.60), RIGHT_ELBOW: (0.49, 0.58),
        LEFT_WRIST: (0.38, 0.60), RIGHT_WRIST: (0.42, 0.58),
        LEFT_HIP: (0.60, 0.62), RIGHT_HIP: (0.68, 0.62),
        LEFT_KNEE: (0.50, 0.62), RIGHT_KNEE: (0.57, 0.62),
        LEFT_ANKLE: (0.35, 0.62), RIGHT_ANKLE: (0.40, 0.62),
        NOSE: (0.50, 0.60),
    }),
    "Vajrasana (clean)": mk({
        LEFT_SHOULDER: (0.47, 0.45), RIGHT_SHOULDER: (0.53, 0.45),
        LEFT_ELBOW: (0.46, 0.52), RIGHT_ELBOW: (0.54, 0.52),
        LEFT_WRIST: (0.46, 0.60), RIGHT_WRIST: (0.54, 0.60),
        LEFT_HIP: (0.47, 0.72), RIGHT_HIP: (0.53, 0.72),
        LEFT_KNEE: (0.47, 0.76), RIGHT_KNEE: (0.53, 0.76),
        LEFT_ANKLE: (0.47, 0.80), RIGHT_ANKLE: (0.53, 0.80),
        NOSE: (0.50, 0.40),
    }),
    "Balasana (clean)": mk({
        LEFT_SHOULDER: (0.55, 0.70), RIGHT_SHOULDER: (0.60, 0.70),
        LEFT_ELBOW: (0.63, 0.72), RIGHT_ELBOW: (0.68, 0.72),
        LEFT_WRIST: (0.72, 0.74), RIGHT_WRIST: (0.77, 0.74),
        LEFT_HIP: (0.45, 0.72), RIGHT_HIP: (0.52, 0.72),
        LEFT_KNEE: (0.45, 0.74), RIGHT_KNEE: (0.52, 0.74),
        LEFT_ANKLE: (0.45, 0.75), RIGHT_ANKLE: (0.52, 0.75),
        NOSE: (0.55, 0.74),
    }),
    "Ustrasana (clean)": mk({
        LEFT_SHOULDER: (0.32, 0.48), RIGHT_SHOULDER: (0.44, 0.48),
        LEFT_ELBOW: (0.39, 0.66), RIGHT_ELBOW: (0.49, 0.65),
        LEFT_WRIST: (0.47, 0.80), RIGHT_WRIST: (0.53, 0.80),
        LEFT_HIP: (0.47, 0.72), RIGHT_HIP: (0.53, 0.72),
        LEFT_KNEE: (0.47, 0.76), RIGHT_KNEE: (0.53, 0.76),
        LEFT_ANKLE: (0.47, 0.80), RIGHT_ANKLE: (0.53, 0.80),
        NOSE: (0.38, 0.36),
    }),
    "Marjaryasana (clean)": mk({
        LEFT_SHOULDER: (0.42, 0.55), RIGHT_SHOULDER: (0.48, 0.55),
        LEFT_ELBOW: (0.38, 0.60), RIGHT_ELBOW: (0.44, 0.60),
        LEFT_WRIST: (0.34, 0.66), RIGHT_WRIST: (0.40, 0.66),
        LEFT_HIP: (0.55, 0.56), RIGHT_HIP: (0.62, 0.56),
        LEFT_KNEE: (0.55, 0.70), RIGHT_KNEE: (0.62, 0.70),
        LEFT_ANKLE: (0.63, 0.70), RIGHT_ANKLE: (0.70, 0.70),
        NOSE: (0.40, 0.64),
    }),
    "Bitilasana (clean)": mk({
        LEFT_SHOULDER: (0.42, 0.55), RIGHT_SHOULDER: (0.48, 0.55),
        LEFT_ELBOW: (0.38, 0.60), RIGHT_ELBOW: (0.44, 0.60),
        LEFT_WRIST: (0.34, 0.66), RIGHT_WRIST: (0.40, 0.66),
        LEFT_HIP: (0.55, 0.56), RIGHT_HIP: (0.62, 0.56),
        LEFT_KNEE: (0.55, 0.70), RIGHT_KNEE: (0.62, 0.70),
        LEFT_ANKLE: (0.63, 0.70), RIGHT_ANKLE: (0.70, 0.70),
        NOSE: (0.44, 0.48),
    }),
    "Adho Mukha Svanasana (clean)": mk({
        LEFT_SHOULDER: (0.40, 0.55), RIGHT_SHOULDER: (0.48, 0.55),
        LEFT_ELBOW: (0.39, 0.68), RIGHT_ELBOW: (0.47, 0.68),
        LEFT_WRIST: (0.36, 0.80), RIGHT_WRIST: (0.44, 0.80),
        LEFT_HIP: (0.45, 0.40), RIGHT_HIP: (0.55, 0.40),
        LEFT_KNEE: (0.55, 0.62), RIGHT_KNEE: (0.65, 0.62),
        LEFT_ANKLE: (0.65, 0.84), RIGHT_ANKLE: (0.73, 0.84),
        NOSE: (0.40, 0.62),
    }),
    "Bhujangasana (clean)": mk({
        LEFT_SHOULDER: (0.40, 0.52), RIGHT_SHOULDER: (0.50, 0.52),
        LEFT_ELBOW: (0.40, 0.60), RIGHT_ELBOW: (0.50, 0.60),
        LEFT_WRIST: (0.50, 0.60), RIGHT_WRIST: (0.60, 0.60),
        LEFT_HIP: (0.55, 0.65), RIGHT_HIP: (0.65, 0.65),
        LEFT_KNEE: (0.67, 0.65), RIGHT_KNEE: (0.74, 0.65),
        LEFT_ANKLE: (0.78, 0.65), RIGHT_ANKLE: (0.86, 0.65),
        NOSE: (0.36, 0.48),
    }),
    "Setu Bandhasana (clean)": mk({
        LEFT_SHOULDER: (0.45, 0.62), RIGHT_SHOULDER: (0.53, 0.62),
        LEFT_ELBOW: (0.45, 0.66), RIGHT_ELBOW: (0.53, 0.66),
        LEFT_WRIST: (0.45, 0.70), RIGHT_WRIST: (0.53, 0.70),
        LEFT_HIP: (0.45, 0.48), RIGHT_HIP: (0.55, 0.48),
        LEFT_KNEE: (0.52, 0.58), RIGHT_KNEE: (0.60, 0.58),
        LEFT_ANKLE: (0.52, 0.66), RIGHT_ANKLE: (0.60, 0.66),
        NOSE: (0.49, 0.66),
    }),
    "Savasana (clean)": mk({
        LEFT_SHOULDER: (0.25, 0.60), RIGHT_SHOULDER: (0.32, 0.60),
        LEFT_ELBOW: (0.28, 0.57), RIGHT_ELBOW: (0.35, 0.57),
        LEFT_WRIST: (0.30, 0.55), RIGHT_WRIST: (0.37, 0.55),
        LEFT_HIP: (0.48, 0.60), RIGHT_HIP: (0.55, 0.60),
        LEFT_KNEE: (0.65, 0.60), RIGHT_KNEE: (0.70, 0.60),
        LEFT_ANKLE: (0.80, 0.60), RIGHT_ANKLE: (0.85, 0.60),
        NOSE: (0.20, 0.60),
    }),
    # ----------------------------------------------------------------------
    # Flawed attempts - must still be IDENTIFIED as the right asana but must
    # be scored INCORRECT (at least one correctness check fails).
    # ----------------------------------------------------------------------
    "Tadasana (bent knees - flawed)": mk({
        LEFT_SHOULDER: (0.45, 0.30), RIGHT_SHOULDER: (0.55, 0.30),
        LEFT_ELBOW: (0.44, 0.42), RIGHT_ELBOW: (0.56, 0.42),
        LEFT_WRIST: (0.44, 0.54), RIGHT_WRIST: (0.56, 0.54),
        LEFT_HIP: (0.47, 0.55), RIGHT_HIP: (0.53, 0.55),
        LEFT_KNEE: (0.44, 0.68), RIGHT_KNEE: (0.56, 0.68),
        LEFT_ANKLE: (0.47, 0.95), RIGHT_ANKLE: (0.53, 0.95),
        NOSE: (0.50, 0.24),
    }),
    "Vrikshasana (arms not overhead, hips uneven - flawed)": mk({
        LEFT_SHOULDER: (0.45, 0.28), RIGHT_SHOULDER: (0.55, 0.28),
        LEFT_ELBOW: (0.40, 0.35), RIGHT_ELBOW: (0.60, 0.35),
        LEFT_WRIST: (0.40, 0.45), RIGHT_WRIST: (0.60, 0.45),
        LEFT_HIP: (0.47, 0.55), RIGHT_HIP: (0.53, 0.50),
        LEFT_KNEE: (0.47, 0.75), RIGHT_KNEE: (0.70, 0.55),
        LEFT_ANKLE: (0.47, 0.95), RIGHT_ANKLE: (0.46, 0.62),
        NOSE: (0.50, 0.22),
    }),
    "Trikonasana (front knee bent, hips uneven - flawed)": mk({
        LEFT_SHOULDER: (0.25, 0.42), RIGHT_SHOULDER: (0.48, 0.28),
        LEFT_ELBOW: (0.15, 0.60), RIGHT_ELBOW: (0.55, 0.15),
        LEFT_WRIST: (0.10, 0.80), RIGHT_WRIST: (0.60, 0.02),
        LEFT_HIP: (0.45, 0.52), RIGHT_HIP: (0.58, 0.58),
        LEFT_KNEE: (0.28, 0.72), RIGHT_KNEE: (0.68, 0.75),
        LEFT_ANKLE: (0.25, 0.95), RIGHT_ANKLE: (0.75, 0.95),
        NOSE: (0.35, 0.32),
    }),
    "Uttanasana (hips uneven - flawed)": mk({
        LEFT_SHOULDER: (0.48, 0.60), RIGHT_SHOULDER: (0.52, 0.60),
        LEFT_ELBOW: (0.47, 0.73), RIGHT_ELBOW: (0.53, 0.73),
        LEFT_WRIST: (0.47, 0.86), RIGHT_WRIST: (0.53, 0.86),
        LEFT_HIP: (0.47, 0.56), RIGHT_HIP: (0.53, 0.50),
        LEFT_KNEE: (0.47, 0.70), RIGHT_KNEE: (0.53, 0.70),
        LEFT_ANKLE: (0.47, 0.90), RIGHT_ANKLE: (0.53, 0.90),
        NOSE: (0.50, 0.65),
    }),
    "Random pose (not a supported asana)": mk({
        LEFT_SHOULDER: (0.45, 0.30), RIGHT_SHOULDER: (0.55, 0.30),
        LEFT_ELBOW: (0.30, 0.40), RIGHT_ELBOW: (0.70, 0.40),
        LEFT_WRIST: (0.20, 0.30), RIGHT_WRIST: (0.80, 0.30),
        LEFT_HIP: (0.47, 0.55), RIGHT_HIP: (0.53, 0.55),
        LEFT_KNEE: (0.40, 0.60), RIGHT_KNEE: (0.60, 0.60),
        LEFT_ANKLE: (0.40, 0.65), RIGHT_ANKLE: (0.60, 0.65),
        NOSE: (0.50, 0.24),
    }),
}


def main():
    print(f"{'Case':38} {'Detected Asana':16} {'Status':10} {'Match%':7}  Feedback")
    print("-" * 100)

    clean_ok = flawed_ok = unknown_ok = 0
    clean_total = flawed_total = 0

    for name, coords in CASES.items():
        features = extract_features(coords)
        result = classify(features)
        status = "CORRECT" if result.is_correct else "INCORRECT"
        fb = "; ".join(result.feedback) if result.feedback else "-"
        print(f"{name:38} {result.asana:16} {status:10} {result.score:6.1f}%  {fb}")

        is_flawed = "flawed" in name
        is_unknown = "Random" in name
        expected = name.split(" (")[0]
        if is_unknown:
            unknown_ok += result.asana.startswith("No Pose")
        elif is_flawed:
            flawed_total += 1
            flawed_ok += result.asana == expected and not result.is_correct
        else:
            clean_total += 1
            clean_ok += result.asana == expected and result.is_correct

    print()
    print(f"Supported asanas: {len(SUPPORTED_ASANAS)} "
          f"({', '.join(sorted(SUPPORTED_ASANAS))})")
    print(f"Clean attempts correctly identified + scored CORRECT: "
          f"{clean_ok}/{clean_total}")
    print(f"Flawed attempts identified but scored INCORRECT: "
          f"{flawed_ok}/{flawed_total}")
    print(f"Random pose rejected: {'PASS' if unknown_ok else 'FAIL'}")
    if clean_ok != clean_total or flawed_ok != flawed_total or not unknown_ok:
        print("RESULT: FAIL")
    else:
        print("RESULT: PASS")


if __name__ == "__main__":
    main()
