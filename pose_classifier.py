"""
pose_classifier.py
------------------
Training-free asana classification and correctness validation, built purely
on the joint-angle / distance features computed in pose_geometry.py.

Detection is two-stage:
  1. Identify the *posture family* (standing, lunge, balance, forward-fold,
     seated, kneeling, all-fours, lying, inverted) from gross geometric
     signatures - the biggest cue is whether the hips are near the ground.
  2. Within the family, match the finer geometric signature of each specific
     asana (priority-ordered, most distinctive first).

26 asanas are supported - a mix of the most popular, easy and healthy classic
asanas. Adding another one means adding one rule in `identify_asana` plus one
correctness function - no retraining, no dataset.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from pose_geometry import (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    PoseFeatures,
)

# ---------------------------------------------------------------------
# Landmark groups used to colour the on-screen skeleton: a joint (and the
# lines meeting at it) turns red when any correctness rule that involves it
# fails, otherwise it stays green.
# ---------------------------------------------------------------------
JOINT_HEAD = {NOSE}
JOINT_SPINE = {LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP}
JOINT_ARM_L = {LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST}
JOINT_ARM_R = {RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST}
JOINT_ARMS = JOINT_ARM_L | JOINT_ARM_R
JOINT_LEG_L = {LEFT_HIP, LEFT_KNEE, LEFT_ANKLE}
JOINT_LEG_R = {RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE}
JOINT_LEGS = JOINT_LEG_L | JOINT_LEG_R
JOINT_KNEES = {LEFT_KNEE, RIGHT_KNEE}

UNKNOWN = "No Pose / Unknown"

SUPPORTED_ASANAS = [
    "Adho Mukha Svanasana",      # Downward Dog
    "Anjaneyasana",              # Low Lunge
    "Ardha Chandrasana",         # Half Moon
    "Baddha Konasana",           # Butterfly
    "Balasana",                  # Child's Pose
    "Bhujangasana",              # Cobra
    "Bitilasana",                # Cow
    "Garudasana",                # Eagle
    "Marjaryasana",              # Cat
    "Natarajasana",              # Dancer
    "Paschimottanasana",         # Seated Forward Bend
    "Prasarita Padottanasana",   # Wide-Legged Forward Fold
    "Savasana",                  # Corpse
    "Setu Bandhasana",           # Bridge
    "Sukhasana",                 # Easy Seat
    "Tadasana",                  # Mountain
    "Trikonasana",               # Triangle
    "Ustrasana",                 # Camel
    "Utkatasana",                # Chair
    "Uttanasana",                # Standing Forward Bend
    "Utthita Parsvakonasana",    # Extended Side Angle
    "Vajrasana",                 # Thunderbolt
    "Virabhadrasana I",          # Warrior I
    "Virabhadrasana II",         # Warrior II
    "Virabhadrasana III",        # Warrior III
    "Vrikshasana",               # Tree
]


@dataclass
class ClassificationResult:
    asana: str
    is_correct: bool
    feedback: List[str]
    score: float  # 0-100, how closely the pose matches its ideal geometry
    bad_landmarks: Set[int] = field(default_factory=set)


# ---------------------------------------------------------------------
# Identification thresholds (tunable). These describe the *shape* of each
# asana in terms of the geometric features - not appearance/pixels - which
# is what makes this training-free.
# ---------------------------------------------------------------------
STRAIGHT_KNEE = 160        # a knee angle above this counts as "straight leg"
BENT_KNEE_MAX = 130        # a knee angle below this counts as "bent leg"
DEEP_BENT_KNEE = 125       # below this = a deep (lunge/chair) bend
NARROW_STANCE_RATIO = 1.35 # ankle-distance/hip-distance below this = feet together
WIDE_STANCE_RATIO = 1.7    # above this = wide stance (Triangle / Wide Fold)
FOOT_NEAR_KNEE_RATIO = 0.6 # raised ankle near the standing knee (Tree)
UPRIGHT_LEAN_MAX = 15      # torso_lean below this = standing upright
SIDE_BEND_MIN = 20         # torso_lean above this = bent sideways
RAISED_ANKLE = 0.25        # one ankle this much higher than the other = balance pose
HAND_TO_FLOOR_GAP = 0.5    # wrist-ankle gap below this = hands reach the feet
TORSO_FOLDED = 0.55        # uprightness below this = torso horizontal/folded
TORSO_UPRIGHT = 0.6        # uprightness above this = torso tall
ELBOW_STRAIGHT = 150
ELBOW_BENT_MAX = 120       # below this = arms strongly folded (Eagle)


def _hips_low(f: PoseFeatures) -> bool:
    """Hips near the ground: hips in the lower half of the frame AND the rest
    of the body reads as grounded - ankles at roughly hip height (seated /
    kneeling upright), the torso horizontal (lying / all-fours / folded),
    both knees on the floor (camel / cobra / tabletop) or the legs folded
    under the hips (child's pose). Standing / lunge / forward-fold poses
    satisfy none of these cues, so they fall through to the standing family.
    The ankle-height gate uses 0.12 (not 0.15): real seated/kneeling poses sit
    0.07-0.08 from the ankles, while balance poses like Ardha Chandrasana (hips
    at mid-frame, one leg horizontal) jitter across a 0.15 gate and get
    misrouted into this family, where they match nothing."""
    return f.hip_y > 0.5 and (
        abs(f.hip_y - f.ankle_y) < 0.12
        or (f.uprightness < 0.5 and f.hip_y > 0.55)
        or f.hips_on_heels
        or f.knees_on_floor
    )


def _raised_side(f: PoseFeatures) -> str:
    """Which leg has the higher ankle (returns 'left' or 'right')."""
    return "left" if f.left_ankle_y < f.right_ankle_y else "right"


# ---------------------------------------------------------------------
# Asana identification
# ---------------------------------------------------------------------
def identify_asana(f: PoseFeatures) -> str:
    k = f
    avg_knee = (k.left_knee + k.right_knee) / 2
    min_knee = min(k.left_knee, k.right_knee)
    max_knee = max(k.left_knee, k.right_knee)
    min_elbow = min(k.left_elbow, k.right_elbow)
    max_elbow = max(k.left_elbow, k.right_elbow)
    torso = max(k.torso_len, 1e-6)

    # --- Setu Bandhasana (bridge): hips lifted above the shoulders with both
    # knees bent, feet flat on the floor. Checked first because the raised
    # hips put its hip_y right on the seated/standing boundary - with no
    # family gate the bridge is picked up no matter how the frame is framed. --
    # Knees < 165 (not 155): bridge knees are only moderately bent and webcam
    # jitter swings them past 155, dropping real bridges to Unknown.
    # torso_lean > 120: the bridge torso lies nearly horizontal (shoulders and
    # hips at similar heights), which no upright standing/seated pose and no
    # cat/cow tabletop matches.
    if (
        k.hips_above_shoulders
        and k.hip_y < k.knee_y
        and not k.knees_on_floor
        and k.leg_spread_ratio < NARROW_STANCE_RATIO   # a wide fold is never a bridge
        and k.torso_lean > 120
        and min_knee < 165
        and avg_knee < 165
    ):
        return "Setu Bandhasana"

    # ============ HIPS-LOW FAMILY (seated / kneeling / prone / flat) =========
    # NOTE: this block never falls through into the standing family below - a
    # grounded pose that matches nothing here is UNKNOWN, never hijacked by a
    # standing rule (which is what let a child's pose turn into a side angle).
    if _hips_low(k):

        # --- All-fours family (cat / cow): hands + knees on the floor, torso
        # horizontal, hips held UP off the heels (a child's pose rests the
        # hips on the heels instead, so hips_not on heels keeps it out). The
        # leg-reach gate adds margin: a child's pose folded fully onto the
        # heels can jitter past the boolean, but never past 0.6 of the torso.
        # Elbows only need to be near-straight (140): short arm segments make
        # the elbow angle noisy, and 150 dropped real cat/cow to Unknown. -----
        if (
            k.uprightness < 0.55
            and not k.hips_on_heels
            and k.leg_vert / max(k.torso_len, 1e-6) > 0.6
            and max_knee < 155
            and min_elbow > 140
        ):
            if k.nose_y > k.shoulder_y:
                return "Marjaryasana"     # cat: head dropped, spine rounded
            return "Bitilasana"           # cow: chest open, head lifted

        # --- Bhujangasana (cobra): prone, knees on floor, chest+head lifted
        # well above the shoulders, arms bent (hands under the shoulders).
        # The wrist-ankle gap keeps T-pose-like frames out: a cobra's hands rest
        # on the floor at ankle height, so the gap stays small. --------------------
        if (
            k.shoulders_above_hips
            and k.nose_y < k.shoulder_y
            and max_elbow < 150
            and k.wrist_ankle_gap < 0.5
            and k.hip_angle < 160   # lying flat (Savasana) has a straight hip line
            and (k.hips_on_heels or k.knees_on_floor)
        ):
            return "Bhujangasana"

        # --- Balasana (child's pose): hips folded back onto the heels, torso
        # resting over the thighs, head dropped. The shoulders sit roughly level
        # with (or just above) the hips once the torso is horizontal, so there is
        # no chest-height gate here - the dropped head (nose below shoulders) is
        # what separates it from the cobra, which is checked before it. The
        # ankles-under-hips gate separates it from the seated forward bend
        # (Paschimottanasana), whose extended legs push the ankles well in
        # front of the hips. -------------------------------------------------------
        if (
            k.hips_on_heels
            and k.uprightness < 0.7
            and k.nose_y > k.shoulder_y
            and abs(k.ankle_x - k.hip_x) < 0.5 * k.torso_len
        ):
            return "Balasana"

        # --- Kneeling upright family (Vajrasana / Ustrasana): knees on the
        # floor under the hips, hips sitting back toward the ankles, spine tall.
        # The knee-span gate excludes cross-legged seating (knees wide open) and
        # generic low frames with the legs apart; knees below hips excludes
        # cross-legged poses whose knees are raised to hip height. -----------------
        if (
            k.hips_on_heels
            and k.shoulders_above_hips
            and k.uprightness >= 0.7
            and k.knee_span < 0.7
            and (k.knees_on_floor or k.knee_y > k.hip_y - 0.03)
        ):
            if k.nose_y < k.shoulder_y - 0.15 or k.wrist_ankle_gap < 0.4:
                return "Ustrasana"              # camel: chest lifted, hands reaching heels
            return "Vajrasana"                  # thunderbolt: kneeling tall

        # --- Paschimottanasana (seated forward bend): seated, straight legs
        # extended forward away from the hips, torso folded over the legs,
        # hands reaching the feet. ------------------------------------------------
        if (
            avg_knee > 155
            and k.hip_angle < 135
            and k.wrist_ankle_gap < HAND_TO_FLOOR_GAP
            and abs(k.ankle_x - k.hip_x) > 0.7 * torso
        ):
            return "Paschimottanasana"

        # --- Savasana (corpse): body flat, legs straight, body in one line. --
        if (
            k.uprightness < TORSO_FOLDED
            and avg_knee > 155
            and k.hip_angle > 150
            and not k.hands_low
        ):
            return "Savasana"

        # --- Seated family (knees raised, spine tall). Baddha Konasana keeps
        # the feet planted on the floor ("soles together") so the ankles are
        # at the same height; a cross-legged Sukhasana stacks them instead. -----
        if (
            k.knee_y < k.hip_y
            and k.uprightness > 0.5
            and k.torso_lean < 25
        ):
            if (
                k.ankle_span < 0.4
                and k.one_ankle_lift < 0.15
                and k.knee_span > 0.4
            ):
                return "Baddha Konasana"        # butterfly: feet together on the floor
            if k.knee_span > 0.4:
                return "Sukhasana"              # easy seat: cross-legged, knees open
            return UNKNOWN

        return UNKNOWN

    else:
        # ============ STANDING FAMILY (weight on the feet) =====================
        # --- Adho Mukha Svanasana (downward dog): hips above shoulders, hands
        # on the floor, legs + arms straight, torso clearly diagonal (shoulders
        # horizontally offset from the hips - unlike a hanging forward fold
        # where the shoulders sit directly below the hips), feet not wide. -------
        if (
            k.hips_above_shoulders
            and k.hands_low
            and min_knee > 155
            and min_elbow > 140
            and abs(k.shoulder_x - k.hip_x) > 0.3 * torso
            and k.leg_spread_ratio < WIDE_STANCE_RATIO
        ):
            return "Adho Mukha Svanasana"

        # Balance poses: one foot lifted well above the other. The not-hands-low
        # gate keeps folded frames out: in a forward fold the hands hang below
        # the knees, and there the tiny folded torso makes one_ankle_lift so
        # jitter-sensitive that real folds wandered into this block and died
        # as Unknown. A genuine one-foot balance never has both hands down.
        # --------------------
        if k.one_ankle_lift > RAISED_ANKLE and not k.hands_low:
            raised = _raised_side(k)
            raised_knee = k.left_knee if raised == "left" else k.right_knee
            standing_knee = k.right_knee if raised == "left" else k.left_knee
            foot_tuck = k.l_ankle_to_r_knee if raised == "left" else k.r_ankle_to_l_knee
            wrist_high_gap = abs(k.left_wrist_y - k.right_wrist_y)

            if max_elbow < 120 and wrist_high_gap < 0.3 and avg_knee < 160 and standing_knee < 160:
                return "Garudasana"             # eagle: arms + legs wrapped, standing leg bent
            if foot_tuck < FOOT_NEAR_KNEE_RATIO and standing_knee > STRAIGHT_KNEE:
                return "Vrikshasana"            # tree: foot on inner thigh
            if raised_knee < 130:
                return "Natarajasana"           # dancer: bent leg lifted behind
            if raised_knee > 150 and wrist_high_gap > 0.3:
                return "Ardha Chandrasana"      # half moon: straight raised leg, one arm down + one up
            if raised_knee > 150 and wrist_high_gap < 0.2 and k.shoulder_y >= k.hip_y - 0.08 and not k.hands_low:
                return "Virabhadrasana III"     # warrior III: straight raised leg, torso forward
            if raised_knee > 130 and wrist_high_gap > 0.2:
                return "Ardha Chandrasana"      # half moon: jitter-tolerant catch (meets Natarajasana at 130)
            return UNKNOWN

        # Standing forward folds: both feet down, legs straight, hands to the
        # floor. avg_knee > 150 (not 155): jitter bends straight knees a few
        # degrees and 155 dropped real folds to Unknown. min_knee > 145 keeps
        # genuinely bent-knee frames (bridge) out of the fold family. The
        # fold-depth gate is deliberately loose - a camera in front
        # of a folded body foreshortens the torso, so a strict ratio would let
        # a real Uttanasana fall through to UNKNOWN. ------------------------------
        if avg_knee > 150 and min_knee > 145 and k.hands_low and k.torso_leg_ratio < 0.75:
            if k.leg_spread_ratio > WIDE_STANCE_RATIO:
                return "Prasarita Padottanasana"     # wide-legged forward fold
            return "Uttanasana"                      # standing forward bend

        # Lunges: one knee deeply bent, the other leg extended, both feet down. --
        if min_knee < DEEP_BENT_KNEE and max_knee > 140:
            back_knee_on_floor = (
                abs(k.left_knee_y - k.left_ankle_y) if k.left_knee > k.right_knee
                else abs(k.right_knee_y - k.right_ankle_y)
            )
            if back_knee_on_floor < 0.18 and k.wrists_above:
                return "Anjaneyasana"                # low lunge: back knee down, arms up
            if k.wrists_wide:
                return "Virabhadrasana II"           # warrior II: arms horizontal
            if k.wrists_above:
                return "Virabhadrasana I"            # warrior I: arms overhead
            if k.torso_lean > SIDE_BEND_MIN:
                return "Utthita Parsvakonasana"      # side angle: side bend over the leg
            return UNKNOWN

        # Both knees bent (feet on the floor, weight low): chair / eagle.
        # knee_diff allows 60: knee angles are noisy, and a strict symmetry
        # gate dropped real chair poses to Unknown. ----------------------------
        if avg_knee < 155 and k.knee_diff < 60:
            if (
                max_elbow < ELBOW_BENT_MAX
                and abs(k.left_wrist_y - k.right_wrist_y) < 0.3
                and not k.wrists_above
                and k.knee_span < 0.6
                and k.ankle_span < 0.6
            ):
                return "Garudasana"                  # eagle: arms + legs wrapped
            if k.wrists_above and 60 <= avg_knee <= 155:
                return "Utkatasana"                  # chair: arms overhead, knees bent
            return UNKNOWN

        # Trikonasana (triangle): wide stance, straight legs, side bend. ---------
        if k.leg_spread_ratio > WIDE_STANCE_RATIO and avg_knee > 150 and k.torso_lean > SIDE_BEND_MIN:
            return "Trikonasana"

        # Tadasana (mountain): straight legs, narrow stance, tall torso. ---------
        if avg_knee > STRAIGHT_KNEE and k.knee_diff < 15 and k.torso_lean < UPRIGHT_LEAN_MAX:
            return "Tadasana"

        return UNKNOWN


# ---------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------

_CHECK_T = Tuple[bool, str, Set[int]]


def _build(checks: List[_CHECK_T]) -> Tuple[bool, List[str], float, Set[int]]:
    """Evaluate the (passed, message, affected landmarks) rules for an asana.

    Returns (is_correct, feedback, score, bad_landmarks) where
    bad_landmarks is the union of every landmark touched by a rule that
    failed - used to draw those joints/lines red on the live skeleton.
    """
    passed = [ok for ok, _msg, _j in checks]
    bad: Set[int] = set()
    for ok, _msg, joints in checks:
        if not ok:
            bad |= set(joints)
    feedback = [msg for ok, msg, _j in checks if not ok]
    score = 100.0 * sum(passed) / len(checks)
    return score >= 80.0, feedback, score, bad


def _check_tadasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.left_knee > 165, "Straighten your left knee", {LEFT_KNEE}),
        (f.right_knee > 165, "Straighten your right knee", {RIGHT_KNEE}),
        (f.leg_spread_ratio < NARROW_STANCE_RATIO, "Bring your feet closer together",
         {LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.torso_lean < 10, "Stand tall - avoid leaning your torso", JOINT_SPINE),
        (abs(f.left_shoulder - f.right_shoulder) < 20, "Keep both shoulders level and relaxed",
         {LEFT_SHOULDER, RIGHT_SHOULDER}),
    ])


def _check_vrikshasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    raised = _raised_side(f)
    standing_is_left = raised == "right"
    standing_knee = LEFT_KNEE if standing_is_left else RIGHT_KNEE
    bent_knee = RIGHT_KNEE if standing_is_left else LEFT_KNEE
    foot_tuck = {RIGHT_ANKLE, LEFT_KNEE} if standing_is_left else {LEFT_ANKLE, RIGHT_KNEE}
    return _build([
        (f.left_knee > 165 if standing_is_left else f.right_knee > 165,
         "Fully straighten your standing leg", {standing_knee}),
        (f.right_knee < 130 if standing_is_left else f.left_knee < 130,
         "Lift the bent knee out to the side more", {bent_knee}),
        (f.r_ankle_to_l_knee < FOOT_NEAR_KNEE_RATIO if standing_is_left else
         f.l_ankle_to_r_knee < FOOT_NEAR_KNEE_RATIO,
         "Press your raised foot higher into the standing thigh", foot_tuck),
        (f.hip_level_diff < 0.12, "Keep your hips level - avoid tilting to one side",
         {LEFT_HIP, RIGHT_HIP}),
        (f.torso_lean < 20, "Keep your torso upright, avoid leaning", JOINT_SPINE),
        (f.wrists_above, "Raise your arms overhead (or bring palms to the chest)", JOINT_ARMS),
    ])


def _check_trikonasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.left_knee > 155, "Keep your left leg straight", {LEFT_KNEE}),
        (f.right_knee > 155, "Keep your right leg straight", {RIGHT_KNEE}),
        (f.leg_spread_ratio > WIDE_STANCE_RATIO, "Widen your stance further",
         {LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE}),
        (30 <= f.torso_lean <= 65, "Adjust your side bend - not too little, not too much", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Keep your hips level", {LEFT_HIP, RIGHT_HIP}),
        (abs(f.left_wrist_y - f.right_wrist_y) > 0.25, "Reach one arm toward the floor and the other straight up",
         {LEFT_WRIST, RIGHT_WRIST}),
    ])


def _check_virabhadrasana_i(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    straight_knee = max(f.left_knee, f.right_knee)
    return _build([
        (80 <= min(f.left_knee, f.right_knee) <= 130, "Bend your front knee to roughly 90 degrees", JOINT_KNEES),
        (straight_knee > 150, "Straighten your back leg, press the heel down", JOINT_KNEES),
        (f.wrists_above and min(f.left_elbow, f.right_elbow) > 150,
         "Raise both arms overhead with straight elbows", JOINT_ARMS),
        (f.torso_lean < 15, "Keep your torso upright over your hips", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Square your hips and keep them level", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_virabhadrasana_ii(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (80 <= min(f.left_knee, f.right_knee) <= 130, "Bend your front knee to roughly 90 degrees", JOINT_KNEES),
        (130 <= max(f.left_knee, f.right_knee) <= 170, "Keep your back leg active but slightly bent", JOINT_KNEES),
        (f.wrists_wide and min(f.left_elbow, f.right_elbow) > 150,
         "Extend both arms horizontal, fingertips reaching out", JOINT_ARMS),
        (f.torso_lean < 12, "Keep your torso upright, don't lean toward the front leg", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Keep your hips level and open to the side", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_utkatasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (80 <= (f.left_knee + f.right_knee) / 2 <= 140, "Bend your knees deeper, sit your hips back", JOINT_KNEES),
        (f.knee_diff < 40, "Keep both knees bent evenly", JOINT_KNEES),
        (f.leg_spread_ratio < NARROW_STANCE_RATIO, "Keep your feet together or hip-width apart",
         {LEFT_ANKLE, RIGHT_ANKLE}),
        (f.wrists_above and min(f.left_elbow, f.right_elbow) > 150,
         "Raise your arms overhead with straight elbows", JOINT_ARMS),
        (f.torso_lean < 25 and f.uprightness > 0.6, "Lengthen your spine - avoid slumping forward", JOINT_SPINE),
    ])


def _check_uttanasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.left_knee > 155 and f.right_knee > 155, "Straighten your knees", JOINT_KNEES),
        (f.wrist_ankle_gap < HAND_TO_FLOOR_GAP, "Reach your hands toward the floor",
         {LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.torso_leg_ratio < 0.55, "Fold forward from the hips, keep the spine long", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Keep your hips level", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_prasarita_padottanasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.leg_spread_ratio > WIDE_STANCE_RATIO, "Widen your stance further",
         {LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.left_knee > 155 and f.right_knee > 155, "Straighten your knees", JOINT_KNEES),
        (f.wrist_ankle_gap < HAND_TO_FLOOR_GAP, "Reach your hands to the floor",
         {LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.torso_leg_ratio < 0.55, "Fold forward from the hips", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Keep your hips level", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_ardha_chandrasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    raised = _raised_side(f)
    raised_knee = LEFT_KNEE if raised == "left" else RIGHT_KNEE
    standing_knee = RIGHT_KNEE if raised == "left" else LEFT_KNEE
    raised_leg = {LEFT_HIP, LEFT_KNEE, LEFT_ANKLE} if raised == "left" else {RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE}
    standing_leg = JOINT_LEG_R if raised == "left" else JOINT_LEG_L
    return _build([
        (f.left_knee > 160 if raised == "right" else f.right_knee > 160,
         "Straighten your standing leg", standing_leg),
        (f.left_knee > 150 if raised == "left" else f.right_knee > 150,
         "Extend your lifted leg straight out", {raised_knee}),
        (f.one_ankle_lift > 0.3, "Lift your top leg higher - aim for hip height", raised_leg),
        (abs(f.left_wrist_y - f.right_wrist_y) > 0.3,
         "Press your bottom hand to the floor, reach the top arm up", JOINT_ARMS),
        (f.torso_lean > SIDE_BEND_MIN, "Open your chest and stack your shoulder over your wrist", JOINT_SPINE),
    ])


def _check_virabhadrasana_iii(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    raised = _raised_side(f)
    raised_knee = LEFT_KNEE if raised == "left" else RIGHT_KNEE
    standing_knee = RIGHT_KNEE if raised == "left" else LEFT_KNEE
    developed_leg = JOINT_LEG_R if raised == "left" else JOINT_LEG_L
    return _build([
        (f.left_knee > 160 if raised == "right" else f.right_knee > 160,
         "Fully straighten your standing leg", developed_leg),
        (f.left_knee > 150 if raised == "left" else f.right_knee > 150,
         "Extend your back leg straight behind you", {raised_knee}),
        (f.one_ankle_lift > 0.3, "Lift your back leg to hip height",
         {LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE}),
        (abs(f.left_wrist_y - f.right_wrist_y) < 0.2,
         "Reach your arms straight forward, level with the body", JOINT_ARMS),
        (f.torso_leg_ratio < 0.55, "Bring your torso parallel to the floor", JOINT_SPINE),
    ])


def _check_natarajasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    raised = _raised_side(f)
    raised_knee = LEFT_KNEE if raised == "left" else RIGHT_KNEE
    standing_knee = RIGHT_KNEE if raised == "left" else LEFT_KNEE
    raised_ankle_y = f.left_ankle_y if raised == "left" else f.right_ankle_y
    near_hand = f.left_wrist_y if raised == "left" else f.right_wrist_y
    raised_ankle_idx = LEFT_ANKLE if raised == "left" else RIGHT_ANKLE
    near_hand_idx = LEFT_WRIST if raised == "left" else RIGHT_WRIST
    raised_leg = {LEFT_HIP, LEFT_KNEE, LEFT_ANKLE} if raised == "left" else {RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE}
    standing_leg = JOINT_LEG_R if raised == "left" else JOINT_LEG_L
    return _build([
        (f.left_knee > 160 if raised == "right" else f.right_knee > 160,
         "Fully straighten your standing leg", standing_leg),
        ((f.left_knee < 130 if raised == "left" else f.right_knee < 130) and f.one_ankle_lift > 0.3,
         "Lift your back foot higher toward your head", raised_leg),
        (abs(near_hand - raised_ankle_y) < 0.4, "Reach your hand back to grasp the foot",
         {near_hand_idx, raised_ankle_idx}),
        (f.torso_lean < 45, "Lift your chest - don't collapse forward", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Keep your hips level and squared", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_anjaneyasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    straight_side = "left" if f.left_knee > f.right_knee else "right"
    back_knee_floor = (
        abs(f.left_knee_y - f.left_ankle_y)
        if straight_side == "left" else abs(f.right_knee_y - f.right_ankle_y)
    )
    back_leg = {LEFT_KNEE, LEFT_ANKLE} if straight_side == "left" else {RIGHT_KNEE, RIGHT_ANKLE}
    return _build([
        (80 <= min(f.left_knee, f.right_knee) <= 120, "Bend your front knee to 90 degrees", JOINT_KNEES),
        (back_knee_floor < 0.18, "Lower your back knee to the floor", back_leg),
        (f.wrists_above, "Raise your arms overhead", JOINT_ARMS),
        (f.torso_lean < 20, "Lift your torso tall", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Square your hips toward the front", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_parsvakonasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (80 <= min(f.left_knee, f.right_knee) <= 120, "Bend your front knee to 90 degrees", JOINT_KNEES),
        (max(f.left_knee, f.right_knee) > 150, "Straighten your back leg", JOINT_KNEES),
        (f.torso_lean > SIDE_BEND_MIN + 5, "Reach your torso over the bent leg", JOINT_SPINE),
        (abs(f.left_wrist_y - f.right_wrist_y) > 0.3, "Press your bottom hand to the floor, top arm overhead",
         {LEFT_WRIST, RIGHT_WRIST}),
        (f.hip_level_diff < 0.12, "Keep your hips level", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_garudasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (80 <= (f.left_knee + f.right_knee) / 2 <= 145, "Bend your knees deeper to wrap the legs", JOINT_KNEES),
        (f.knee_span < 0.6 and f.ankle_span < 0.6, "Cross your legs, hook the top foot behind the calf",
         JOINT_LEGS),
        (max(f.left_elbow, f.right_elbow) < 120, "Cross your arms and bend the elbows",
         {LEFT_ELBOW, RIGHT_ELBOW}),
        (abs(f.left_wrist_y - f.right_wrist_y) < 0.3, "Bring your palms together",
         {LEFT_WRIST, RIGHT_WRIST}),
        (f.torso_lean < 15, "Keep your spine tall", JOINT_SPINE),
    ])


def _check_sukhasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.uprightness > 0.6 and f.torso_lean < 15, "Lengthen your spine and sit tall", JOINT_SPINE),
        (f.knee_span > 0.4, "Relax your knees out to the sides", JOINT_KNEES),
        (f.hip_level_diff < 0.15, "Sit evenly on both sit bones", {LEFT_HIP, RIGHT_HIP}),
        (abs(f.left_shoulder - f.right_shoulder) < 20, "Keep your shoulders relaxed and level",
         {LEFT_SHOULDER, RIGHT_SHOULDER}),
    ])


def _check_baddha_konasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.ankle_span < 0.4, "Bring the soles of your feet together", {LEFT_ANKLE, RIGHT_ANKLE}),
        (f.knee_span > 0.5, "Open your knees toward the floor", JOINT_KNEES),
        (f.uprightness > 0.6 and f.torso_lean < 15, "Lengthen your spine", JOINT_SPINE),
        (abs(f.left_shoulder - f.right_shoulder) < 20, "Relax your shoulders down your back",
         {LEFT_SHOULDER, RIGHT_SHOULDER}),
    ])


def _check_paschimottanasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.left_knee > 155 and f.right_knee > 155, "Straighten your legs", JOINT_KNEES),
        (f.wrist_ankle_gap < HAND_TO_FLOOR_GAP, "Reach your hands toward your feet",
         {LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.uprightness < TORSO_FOLDED, "Fold forward from the hips, keep the spine long", JOINT_SPINE),
        (abs(f.left_shoulder - f.right_shoulder) < 20, "Relax your shoulders away from your ears",
         {LEFT_SHOULDER, RIGHT_SHOULDER}),
    ])


def _check_vajrasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.knees_on_floor, "Kneel with both knees on the floor", JOINT_KNEES),
        (abs(f.hip_y - f.ankle_y) < 0.25, "Sit your hips down toward your heels",
         {LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.knee_span < 0.45, "Keep your knees together", JOINT_KNEES),
        (f.leg_vert / max(f.torso_len, 1e-6) < 0.45, "Fold your legs fully under you", JOINT_LEGS),
        (f.uprightness > 0.6 and f.torso_lean < 15, "Lengthen your spine and sit tall", JOINT_SPINE),
    ])


def _check_balasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.leg_vert / f.torso_len < 0.4, "Sit your hips back onto your heels", JOINT_LEGS),
        (f.uprightness < 0.5, "Rest your torso over your thighs", JOINT_SPINE),
        (min(f.left_elbow, f.right_elbow) > 150, "Reach your arms forward along the floor", JOINT_ARMS),
        (f.hip_level_diff < 0.12, "Keep your hips even", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_ustrasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.knees_on_floor, "Kneel with your knees under your hips", JOINT_KNEES),
        (f.wrist_ankle_gap < HAND_TO_FLOOR_GAP, "Reach your hands back toward your heels",
         {LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.shoulders_above_hips, "Lift your chest up and open", JOINT_SPINE),
        (f.torso_lean > 25, "Lengthen your spine into a backbend", JOINT_SPINE),
        (f.knee_span < 0.5, "Keep your knees hip-width apart", JOINT_KNEES),
    ])


def _check_marjaryasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.knees_on_floor, "Place your knees on the floor under your hips", JOINT_KNEES),
        (min(f.left_elbow, f.right_elbow) > 150, "Stack your wrists under your shoulders, arms straight", JOINT_ARMS),
        (f.nose_y > f.shoulder_y, "Round your spine up, tuck the chin toward the chest", JOINT_HEAD),
        (f.torso_lean > 60, "Keep your spine level like a table top", JOINT_SPINE),
        (f.hip_level_diff < 0.12, "Keep your hips level", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_bitilasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.knees_on_floor, "Place your knees on the floor under your hips", JOINT_KNEES),
        (min(f.left_elbow, f.right_elbow) > 150, "Stack your wrists under your shoulders, arms straight", JOINT_ARMS),
        (f.nose_y < f.shoulder_y, "Arch your spine, lift the chest and gaze up", JOINT_HEAD),
        (f.uprightness < 0.5, "Keep your torso moving in a gentle curve", JOINT_SPINE),
    ])


def _check_adho_mukha_svanasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (min(f.left_elbow, f.right_elbow) > 150, "Straighten your arms, push the floor away", JOINT_ARMS),
        (f.left_knee > 150 and f.right_knee > 150, "Straighten your legs", JOINT_KNEES),
        (f.hips_above_shoulders, "Lift your hips up and back into an inverted V",
         {LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER}),
        (f.hip_y < f.ankle_y, "Press your heels toward the floor", JOINT_LEGS),
        (f.hands_low, "Press your palms firmly into the floor", {LEFT_WRIST, RIGHT_WRIST}),
    ])


def _check_bhujangasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.shoulders_above_hips, "Lift your chest off the floor", JOINT_SPINE),
        (60 <= max(f.left_elbow, f.right_elbow) <= 150, "Keep your elbows slightly bent",
         {LEFT_ELBOW, RIGHT_ELBOW}),
        (abs(f.hip_y - f.ankle_y) < 0.2, "Keep your hips pressed toward the floor",
         {LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE}),
        (f.nose_y < f.shoulder_y, "Look forward and keep the neck long", JOINT_HEAD),
        (f.hip_level_diff < 0.12, "Keep your hips even", {LEFT_HIP, RIGHT_HIP}),
    ])


def _check_setu_bandhasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.hips_above_shoulders, "Lift your hips toward the ceiling",
         {LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER}),
        (f.hip_y < f.knee_y, "Press through your feet to lift the hips higher",
         {LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE}),
        (80 <= (f.left_knee + f.right_knee) / 2 <= 165, "Bend your knees, feet flat on the floor", JOINT_KNEES),
        (f.hip_level_diff < 0.12, "Keep your hips level", {LEFT_HIP, RIGHT_HIP}),
        (abs(f.left_shoulder - f.right_shoulder) < 20, "Keep your shoulders relaxed on the floor",
         {LEFT_SHOULDER, RIGHT_SHOULDER}),
    ])


def _check_savasana(f: PoseFeatures) -> Tuple[bool, List[str], float, Set[int]]:
    return _build([
        (f.uprightness < TORSO_FOLDED, "Lie flat, let the body settle", JOINT_SPINE),
        (f.left_knee > 155 and f.right_knee > 155, "Let your legs lengthen, feet fall open", JOINT_LEGS),
        (abs(f.left_wrist_y - f.right_wrist_y) < 0.3, "Rest your arms at your sides, palms up",
         {LEFT_WRIST, RIGHT_WRIST}),
        (f.hip_level_diff < 0.12, "Let the body rest symmetrically", {LEFT_HIP, RIGHT_HIP}),
        (abs(f.left_shoulder - f.right_shoulder) < 20, "Release the shoulders toward the floor",
         {LEFT_SHOULDER, RIGHT_SHOULDER}),
    ])


CORRECTNESS_CHECKS: Dict[str, Callable[[PoseFeatures], Tuple[bool, List[str], float, Set[int]]]] = {
    "Tadasana": _check_tadasana,
    "Vrikshasana": _check_vrikshasana,
    "Trikonasana": _check_trikonasana,
    "Virabhadrasana I": _check_virabhadrasana_i,
    "Virabhadrasana II": _check_virabhadrasana_ii,
    "Virabhadrasana III": _check_virabhadrasana_iii,
    "Utkatasana": _check_utkatasana,
    "Uttanasana": _check_uttanasana,
    "Prasarita Padottanasana": _check_prasarita_padottanasana,
    "Ardha Chandrasana": _check_ardha_chandrasana,
    "Natarajasana": _check_natarajasana,
    "Anjaneyasana": _check_anjaneyasana,
    "Utthita Parsvakonasana": _check_parsvakonasana,
    "Garudasana": _check_garudasana,
    "Sukhasana": _check_sukhasana,
    "Baddha Konasana": _check_baddha_konasana,
    "Paschimottanasana": _check_paschimottanasana,
    "Vajrasana": _check_vajrasana,
    "Balasana": _check_balasana,
    "Ustrasana": _check_ustrasana,
    "Marjaryasana": _check_marjaryasana,
    "Bitilasana": _check_bitilasana,
    "Adho Mukha Svanasana": _check_adho_mukha_svanasana,
    "Bhujangasana": _check_bhujangasana,
    "Setu Bandhasana": _check_setu_bandhasana,
    "Savasana": _check_savasana,
}


def classify(f: Optional[PoseFeatures]) -> ClassificationResult:
    if f is None:
        return ClassificationResult(UNKNOWN, False, ["Move fully into the camera frame"], 0.0)

    asana = identify_asana(f)
    if asana == UNKNOWN:
        return ClassificationResult(
            UNKNOWN, False,
            ["Hold a supported asana - 26 poses are in the classifier"],
            0.0,
        )

    is_correct, feedback, score, bad_landmarks = CORRECTNESS_CHECKS[asana](f)
    return ClassificationResult(asana, is_correct, feedback, score, bad_landmarks)
