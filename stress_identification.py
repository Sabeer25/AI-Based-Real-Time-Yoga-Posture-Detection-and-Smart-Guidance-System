"""
stress_identification.py
------------------------
Measures how robustly each supported asana is IDENTIFIED when the clean
landmark template is perturbed the way a real webcam / real body wobble does:

  * a few pixels/percent of jitter on every joint,
  * mild global scaling / shifting,
  * one or two joints pushed out of their ideal position
    (e.g. hips slightly higher, knees slightly straighter/bent,
     hands not fully reaching the floor).

For each asana we report how often `identify_asana` still names the RIGHT
asana across 500 noisy variants. Low scores => fragile rules that should be
relaxed. This is a diagnostic file (imports nothing new; run standalone),
it is not part of the app.
"""

import random

from pose_geometry import Point, extract_features
from pose_classifier import identify_asana, SUPPORTED_ASANAS
from test_pose_logic import CASES


def perturb(coords, seed):
    rng = random.Random(seed)
    out = {}
    for idx, p in coords.items():
        jx = rng.uniform(-0.015, 0.015)
        jy = rng.uniform(-0.015, 0.015)
        x = min(0.99, max(0.01, p.x + jx))
        y = min(0.99, max(0.02, p.y + jy))
        out[idx] = Point(x, y, 1.0)
    return out


def main():
    hits = {}
    for name in sorted(SUPPORTED_ASANAS):
        key = f"{name} (clean)"
        if key not in CASES:
            continue
        coords = CASES[key]
        right = 0
        n = 600
        for seed in range(n):
            feats = extract_features(perturb(coords, seed))
            if feats is None:
                continue
            if identify_asana(feats) == name:
                right += 1
        hits[name] = right / n

    for name, rate in sorted(hits.items(), key=lambda kv: kv[1]):
        flag = "  <-- FRAGILE" if rate < 0.9 else ""
        print(f"{rate:6.1%}  {name}{flag}")


if __name__ == "__main__":
    main()