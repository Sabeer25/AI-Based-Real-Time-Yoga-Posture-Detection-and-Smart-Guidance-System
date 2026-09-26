"""Report the top misclassifications for each fragile asana under jitter."""
import collections
import random

from pose_geometry import Point, extract_features
from pose_classifier import identify_asana, SUPPORTED_ASANAS, UNKNOWN
from test_pose_logic import CASES


def perturb(coords, seed):
    rng = random.Random(seed)
    out = {}
    for idx, p in coords.items():
        jx = rng.uniform(-0.015, 0.015)
        jy = rng.uniform(-0.015, 0.015)
        out[idx] = Point(min(0.99, max(0.01, p.x + jx)),
                         min(0.99, max(0.02, p.y + jy)), 1.0)
    return out


def main():
    for name in sorted(SUPPORTED_ASANAS, key=lambda n: n):
        key = f"{name} (clean)"
        if key not in CASES:
            continue
        conf = collections.Counter()
        for seed in range(600):
            feats = extract_features(perturb(CASES[key], seed))
            if feats is None:
                conf[UNKNOWN] += 1
                continue
            conf[identify_asana(feats)] += 1
        right = conf[name]
        row = [f"{right / 600:.0%} {name}"]
        for other, n in conf.most_common(5):
            if other == name:
                continue
            row.append(f"{other}={n}")
        print("  ".join(row))


if __name__ == "__main__":
    main()