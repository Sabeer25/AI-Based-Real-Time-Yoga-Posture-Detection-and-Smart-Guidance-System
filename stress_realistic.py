"""
stress_realistic.py
-------------------
Models how a LIVE webcam actually frames a pose - the missing piece behind
"only 10 asanas ever register in practice":

  * the person fills a variable fraction of the frame (scale 0.65x - 1.5x),
  * the body drifts up/down/left/right in the frame (framing),
  * every joint jitters with realistic MediaPipe noise (up to ~3%),
  * limbs foreshorten (a leg pointing away from the camera reads shorter).

Each clean template from test_pose_logic is re-scaled, re-shifted and
jittered; `identify_asana` must still name the right pose. Low rates point at
rules that only work for one canned size/centring and break live.
"""
import random

from pose_geometry import Point, extract_features
from pose_classifier import identify_asana, SUPPORTED_ASANAS, UNKNOWN
from test_pose_logic import CASES


def realistic(coords, seed):
    rng = random.Random(seed)
    scale = rng.uniform(0.65, 1.5)
    # anchor the anchor, then scale the torso length
    cx = rng.uniform(-0.12, 0.12)
    cy = rng.uniform(-0.10, 0.10)
    cx = rng.uniform(-0.15, 0.15)
    cy = rng.uniform(-0.12, 0.12)
    jitter = rng.uniform(0.008, 0.03)
    out = {}
    for idx, p in coords.items():
        # scale around the frame centre so the whole body grows/shrinks
        x = 0.5 + (p.x - 0.5) * scale + cx
        y = 0.5 + (p.y - 0.5) * scale + cy
        x += rng.uniform(-jitter, jitter)
        y += rng.uniform(-jitter, jitter)
        out[idx] = Point(min(0.999, max(0.001, x)),
                         min(0.999, max(0.002, y)), 1.0)
    return out


def main():
    print(f"{'asana':32}{'rate':>7}  (confusions when it misses)")
    for name in sorted(SUPPORTED_ASANAS):
        key = f"{name} (clean)"
        if key not in CASES:
            continue
        conf = {}
        for seed in range(400):
            feats = extract_features(realistic(CASES[key], seed))
            if feats is None:
                got = UNKNOWN
            else:
                got = identify_asana(feats)
            conf[got] = conf.get(got, 0) + 1
        n = sum(conf.values())
        right = conf.get(name, 0)
        misses = " ".join(f"{o}={c}" for o, c in sorted(conf.items(), key=lambda kv: -kv[1])
                          if o != name)[:70]
        flag = "  <-- FRAGILE at realistic framing" if right / n < 0.85 else ""
        print(f"{name:32}{right/n:6.1%}{flag}")
        if right / n < 0.85:
            print(f"   misses: {misses}")


if __name__ == "__main__":
    main()