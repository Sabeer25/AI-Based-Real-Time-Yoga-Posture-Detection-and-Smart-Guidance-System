"""
pose_geometry.py
-----------------
Coordinate-geometry helpers for the yoga posture detection system.

Everything here is pure math (no MediaPipe / OpenCV calls) so it can be
unit-tested on its own with plain (x, y) points.

Landmark indices follow the standard 33-point BlazePose / MediaPipe Pose
topology (same numbering in both the legacy `mp.solutions.pose` API and
the newer `mediapipe.tasks` PoseLandmarker API used in this project).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Landmark indices we actually use (out of the 33 MediaPipe provides)
# --------------------------------------------------------------------------
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Skeleton edges for drawing (MediaPipe's `solutions.drawing_utils` isn't
# shipped with every mediapipe build any more, so we draw the skeleton
# ourselves using this fixed connection list).
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12),
]

# Minimum visibility score for a landmark to be drawn / used. Deliberately
# permissive: desk webcams and tight rooms crop ankles/knees constantly, and
# a strict gate here means EVERY frame returns "no pose" - nothing is ever
# detected, which is far worse than an occasionally shaky joint estimate.
VISIBILITY_THRESHOLD = 0.3

# The 12 landmarks extract_features requires before it will analyse a frame.
CORE_LANDMARKS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
]

# Family-detection thresholds (used by extract_features).
KNEE_ON_FLOOR = 0.18      # |knee.y - ankle.y| / torso below this = knee on ground
HIPS_ON_HEELS_RATIO = 0.5 # vertical leg reach / torso below this = hips near ankles


@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


def angle_at(a: Point, b: Point, c: Point) -> float:
    """
    Angle ABC (in degrees, 0-180) at vertex B, formed by rays B->A and B->C.
    Pure 2D coordinate geometry using the dot-product formula:

        cos(theta) = (u . v) / (|u| * |v|)
    """
    ux, uy = a.x - b.x, a.y - b.y
    vx, vy = c.x - b.x, c.y - b.y
    dot = ux * vx + uy * vy
    mag_u = math.hypot(ux, uy)
    mag_v = math.hypot(vx, vy)
    if mag_u * mag_v == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag_u * mag_v)))
    return math.degrees(math.acos(cos_theta))


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def lean_angle_from_vertical(top: Point, bottom: Point) -> float:
    """
    Angle (degrees) between the vector bottom->top and the vertical axis.
    0  = perfectly upright, 90 = perfectly horizontal.
    Used to measure how far the torso is leaning sideways (Trikonasana).
    """
    vx, vy = top.x - bottom.x, top.y - bottom.y
    # Vertical reference vector pointing "up" in image coords is (0, -1)
    dot = vx * 0 + vy * (-1)
    mag_v = math.hypot(vx, vy)
    if mag_v == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / mag_v))
    return math.degrees(math.acos(cos_theta))


def midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2, min(a.visibility, b.visibility))


# --------------------------------------------------------------------------
# Feature extraction: turn 33 raw landmarks into the handful of geometric
# quantities the classifier and correctness checker actually reason about.
# --------------------------------------------------------------------------
@dataclass
class PoseFeatures:
    left_elbow: float = 0.0
    right_elbow: float = 0.0
    left_shoulder: float = 0.0
    right_shoulder: float = 0.0
    left_hip: float = 0.0
    right_hip: float = 0.0
    left_knee: float = 0.0
    right_knee: float = 0.0
    torso_lean: float = 0.0          # degrees from vertical
    leg_spread_ratio: float = 0.0    # ankle distance / hip distance
    knee_diff: float = 0.0           # |left_knee - right_knee|
    l_ankle_to_r_knee: float = 0.0   # normalized by torso length
    r_ankle_to_l_knee: float = 0.0
    left_wrist_y: float = 0.0        # normalized, smaller = higher on screen
    right_wrist_y: float = 0.0
    left_wrist_x: float = 0.0
    right_wrist_x: float = 0.0
    shoulder_y: float = 0.0
    shoulder_x: float = 0.0
    hip_y: float = 0.0
    hip_x: float = 0.0
    knee_x: float = 0.0
    ankle_x: float = 0.0
    hip_level_diff: float = 0.0      # |left_hip.y - right_hip.y|, balance cue
    torso_len: float = 0.0           # |shoulder_mid - hip_mid| (raw length)
    visible: bool = True

    # --- posture-family features (added to support ~26 asanas) -------------
    knee_y: float = 0.0              # midpoint knee y
    ankle_y: float = 0.0             # midpoint ankle y
    left_knee_y: float = 0.0
    right_knee_y: float = 0.0
    left_ankle_y: float = 0.0
    right_ankle_y: float = 0.0
    nose_y: float = 0.0              # head height cue
    torso_vert: float = 0.0          # |shoulder_mid.y - hip_mid.y|
    leg_vert: float = 0.0            # avg |hip.y - ankle.y| per leg
    lower_reach: float = 0.0         # leg_vert / torso_vert (>1 = standing, <1 = folded/seated)
    torso_leg_ratio: float = 0.0     # torso length / avg leg length (<0.5 = torso folded)
    uprightness: float = 0.0         # torso_vert / torso length (~1 upright, ~0 horizontal)
    arm_span: float = 0.0            # |lw.x - rw.x| / torso length
    wrist_ankle_gap: float = 0.0     # avg |wrist.y - ankle.y| / torso length (small = hands near feet)
    hip_angle: float = 0.0           # deg, angle at hip midpoint (shoulder-mid - hip-mid - ankle-mid)
    knee_span: float = 0.0           # |lk.x - rk.x| / torso length (wide knees = seated/kneeling)
    ankle_span: float = 0.0          # |la.x - ra.x| / torso length (feet together = butterfly/eagle)
    one_ankle_lift: float = 0.0      # |la.y - ra.y| / torso length (raised foot in balance poses)
    wrists_above: bool = False       # both wrists above shoulders (arms overhead)
    wrists_wide: bool = False        # both wrists near shoulder height and far apart (arms horizontal)
    hands_low: bool = False          # both wrists below knee level (hands near/on floor)
    hips_above_shoulders: bool = False  # inverted-V / bridge / arch signatures
    shoulders_above_hips: bool = False  # chest raised (cobra / cow)
    knees_on_floor: bool = False     # both knees near ankle height (kneeling / all-fours)
    hips_on_heels: bool = False      # hips near ankles (seated / kneeling / lying)


def extract_features(pts: Dict[int, Point]) -> Optional[PoseFeatures]:
    """
    pts: dict mapping landmark index -> Point (x, y in normalized [0,1]
    image coordinates, or pixel coordinates - ratios stay valid either way).
    Returns None if too many key landmarks are missing/low-visibility.
    """
    for idx in CORE_LANDMARKS:
        p = pts.get(idx)
        if p is None or p.visibility < VISIBILITY_THRESHOLD:
            return None

    ls, rs = pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER]
    le, re = pts[LEFT_ELBOW], pts[RIGHT_ELBOW]
    lw, rw = pts[LEFT_WRIST], pts[RIGHT_WRIST]
    lh, rh = pts[LEFT_HIP], pts[RIGHT_HIP]
    lk, rk = pts[LEFT_KNEE], pts[RIGHT_KNEE]
    la, ra = pts[LEFT_ANKLE], pts[RIGHT_ANKLE]

    shoulder_mid = midpoint(ls, rs)
    hip_mid = midpoint(lh, rh)
    torso_len = max(distance(shoulder_mid, hip_mid), 1e-6)

    f = PoseFeatures()
    f.left_elbow = angle_at(ls, le, lw)
    f.right_elbow = angle_at(rs, re, rw)
    f.left_shoulder = angle_at(le, ls, lh)
    f.right_shoulder = angle_at(re, rs, rh)
    f.left_hip = angle_at(ls, lh, lk)
    f.right_hip = angle_at(rs, rh, rk)
    f.left_knee = angle_at(lh, lk, la)
    f.right_knee = angle_at(rh, rk, ra)
    f.torso_lean = lean_angle_from_vertical(shoulder_mid, hip_mid)

    hip_width = max(distance(lh, rh), 1e-6)
    f.leg_spread_ratio = distance(la, ra) / hip_width
    f.knee_diff = abs(f.left_knee - f.right_knee)
    f.l_ankle_to_r_knee = distance(la, rk) / torso_len
    f.r_ankle_to_l_knee = distance(ra, lk) / torso_len
    f.left_wrist_y = lw.y
    f.right_wrist_y = rw.y
    f.left_wrist_x = lw.x
    f.right_wrist_x = rw.x
    f.shoulder_y = shoulder_mid.y
    f.shoulder_x = shoulder_mid.x
    f.hip_y = hip_mid.y
    f.hip_x = hip_mid.x
    f.hip_level_diff = abs(lh.y - rh.y) / torso_len
    f.torso_len = torso_len

    # --- posture-family features -------------------------------------------
    knee_mid = midpoint(lk, rk)
    ankle_mid = midpoint(la, ra)
    nose = pts.get(NOSE, Point(shoulder_mid.x, shoulder_mid.y, 0.0))
    leg_len = max((distance(lh, la) + distance(rh, ra)) / 2, 1e-6)

    f.knee_x = knee_mid.x
    f.ankle_x = ankle_mid.x

    f.knee_y = knee_mid.y
    f.ankle_y = ankle_mid.y
    f.left_knee_y = lk.y
    f.right_knee_y = rk.y
    f.left_ankle_y = la.y
    f.right_ankle_y = ra.y
    f.nose_y = nose.y
    f.torso_vert = abs(shoulder_mid.y - hip_mid.y)
    f.leg_vert = (abs(lh.y - la.y) + abs(rh.y - ra.y)) / 2
    f.lower_reach = f.leg_vert / max(f.torso_vert, 1e-6)
    f.torso_leg_ratio = torso_len / leg_len
    f.uprightness = f.torso_vert / max(torso_len, 1e-6)
    f.arm_span = abs(lw.x - rw.x) / torso_len
    f.wrist_ankle_gap = (abs(lw.y - la.y) + abs(rw.y - ra.y)) / 2 / torso_len
    f.hip_angle = angle_at(shoulder_mid, hip_mid, ankle_mid)
    f.knee_span = abs(lk.x - rk.x) / torso_len
    f.ankle_span = abs(la.x - ra.x) / torso_len
    f.one_ankle_lift = abs(la.y - ra.y) / torso_len
    f.wrists_above = lw.y < shoulder_mid.y and rw.y < shoulder_mid.y
    f.wrists_wide = (
        abs(lw.y - shoulder_mid.y) < 0.3 * torso_len
        and abs(rw.y - shoulder_mid.y) < 0.3 * torso_len
        and f.arm_span > 0.8
    )
    f.hands_low = lw.y > knee_mid.y and rw.y > knee_mid.y
    f.hips_above_shoulders = hip_mid.y < shoulder_mid.y
    f.shoulders_above_hips = shoulder_mid.y < hip_mid.y
    f.knees_on_floor = (
        abs(lk.y - la.y) / torso_len < KNEE_ON_FLOOR
        and abs(rk.y - ra.y) / torso_len < KNEE_ON_FLOOR
    )
    f.hips_on_heels = f.leg_vert / max(torso_len, 1e-6) < HIPS_ON_HEELS_RATIO
    f.visible = True
    return f
