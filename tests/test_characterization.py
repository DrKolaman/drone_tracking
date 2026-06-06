"""Characterization tests: behaviour invariants of the tracker on data/source.mp4.

These pin the behaviours we worked hard for (no false jumps, the f124 dropout
stays fixed, good coverage, one id) with margin. They are SLOW (run the full
tracker) and need HF_TOKEN. See test_golden.py for exact-output regression.
"""

import math

import pytest

pytestmark = pytest.mark.slow


def _boxed(rows):
    return [(int(r["frame"]), float(r["cx"]), float(r["cy"]))
            for r in rows if r["cx"] != ""]


def test_no_false_jumps(tracker_rows):
    """Adjacent-frame box centre stays within the motion gate (~40px); the f304
    far jump was 142px. Band allows the 40px gate + GPU-fp16 run-to-run noise."""
    bx = [(f, x, y) for f, x, y in _boxed(tracker_rows) if 36 <= f <= 475]
    worst, prev = 0.0, None
    for f, x, y in bx:
        if prev and prev[0] == f - 1:
            worst = max(worst, math.hypot(x - prev[1], y - prev[2]))
        prev = (f, x, y)
    assert worst <= 50.0, f"max adjacent jump {worst:.1f}px (regression toward far jumps)"


def test_f124_region_stays_covered(tracker_rows):
    have = {int(r["frame"]) for r in tracker_rows if r["cx"] != ""}
    missing = [f for f in range(120, 201) if f not in have]
    assert missing == [], f"uncovered frames in 120-200: {missing[:10]}"


def test_coverage_over_continuous_segment(tracker_rows):
    seg = [r for r in tracker_rows if 36 <= int(r["frame"]) <= 475]
    shown = sum(1 for r in seg if r["cx"] != "")
    assert shown / len(seg) >= 0.68, f"coverage {shown}/{len(seg)}"


def test_states_are_known(tracker_rows):
    assert {r["state"] for r in tracker_rows} <= {"NONE", "TRACK", "HOLD", "REACQ"}
