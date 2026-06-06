"""Golden-master / regression tests on the real clip.

These pin the CURRENT pipeline outputs so a future optimisation or refactor must
reproduce them. RANSAC is seeded (see conftest._determinism), so the values below
are reproducible. If you intentionally change an algorithm, update the golden value
here and note why in the commit.

CPU only (no DINOv3). DINOv3 golden values live in test_dinov3_gpu.py.
"""
import cv2
import numpy as np
import pytest

from colorfix import to_bw
from build_map import blur_scores, chain, fit_canvas
from registration import GlobalMotionEstimator
from conftest import needs_video


@needs_video
def test_golden_blur_skip_set(read_frames):
    """Blur-skip on frames 0-646 @ 0.19x median == the motion-blur burst only."""
    sc = blur_scores([to_bw(f) for f in read_frames(0, 646)])
    skip = [i for i, s in enumerate(sc) if s < 0.19 * np.median(sc)]
    assert skip == [610, 611, 612]                      # GOLDEN


@needs_video
def test_golden_jump_vs_continuous(read_frames):
    """744->745 is a jump (inlier collapse); 742->743 is continuous."""
    f = [to_bw(x) for x in read_frames(742, 745)]       # -> idx 0..3 == frames 742..745
    est = GlobalMotionEstimator(min_inliers=25)

    def inl(a, b):
        return est.estimate(cv2.cvtColor(f[a], cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(f[b], cv2.COLOR_BGR2GRAY)).n_inliers

    cont, jump = inl(0, 1), inl(2, 3)
    assert cont > 400 and jump < 25                     # semantic (robust)
    assert cont == pytest.approx(588, rel=0.15)         # GOLDEN 588
    assert jump == pytest.approx(10, abs=8)             # GOLDEN 10


@needs_video
@pytest.mark.slow
def test_golden_chain_canvas(read_frames):
    """Chaining frames 300-450 -> deterministic kept count + canvas size."""
    sub = [to_bw(f) for f in read_frames(300, 450)]
    Hs, kept = chain(sub, 25)
    h, w = sub[0].shape[:2]
    _, cw, ch = fit_canvas(Hs, w, h)
    assert len(kept) == 151                             # GOLDEN (all frames kept)
    assert abs(cw - 550) <= 3 and abs(ch - 722) <= 3    # GOLDEN 550x722
