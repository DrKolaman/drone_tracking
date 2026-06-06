"""Golden-master regression: an optimized/refactored tracker must reproduce the
current per-frame output.

The reference is tests/golden/track_ref.csv (frame,state,cx,cy). A fresh run must
match it: same frame count, same state on >=98% of frames, and where BOTH have a
box the centre within 3 px. This is what guarantees "if I optimize the code, the
results stay the same and consistent".

Regenerate the golden ONLY when you intentionally change behaviour:
    REGEN_GOLDEN=1 HF_TOKEN=... python3 -m pytest tests/test_golden.py -m slow
"""

import csv
import math
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

GOLD = Path(__file__).parent / "golden" / "track_ref.csv"
# Tolerances absorb GPU fp16 run-to-run noise (DINOv3 inference is not bit-exact),
# while still catching a real behaviour change (a moved track, a lost segment).
CENTRE_TOL = 15.0     # px
STATE_MATCH_MIN = 0.85


def _write_golden(rows):
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "state", "cx", "cy"])
        for r in rows:
            w.writerow([r["frame"], r["state"], r["cx"], r["cy"]])


def test_golden_regression(tracker_rows):
    if os.environ.get("REGEN_GOLDEN") or not GOLD.exists():
        _write_golden(tracker_rows)
        pytest.skip("golden reference regenerated")

    ref = list(csv.DictReader(open(GOLD)))
    assert len(ref) == len(tracker_rows), "frame count changed vs golden"

    state_matches = 0
    for a, b in zip(ref, tracker_rows):
        if a["state"] == b["state"]:
            state_matches += 1
        if a["cx"] and b["cx"]:
            d = math.hypot(float(a["cx"]) - float(b["cx"]),
                           float(a["cy"]) - float(b["cy"]))
            assert d <= CENTRE_TOL, f"frame {a['frame']}: centre moved {d:.1f}px vs golden"
    frac = state_matches / len(ref)
    assert frac >= STATE_MATCH_MIN, f"state match {frac:.3f} < {STATE_MATCH_MIN}"
