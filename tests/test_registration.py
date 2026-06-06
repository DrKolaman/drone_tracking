"""Unit tests for the optical-flow global-motion estimator."""

import numpy as np

from registration import GlobalMotionEstimator


def test_identity_on_same_frame(textured):
    g = textured(1)
    r = GlobalMotionEstimator().estimate(g, g)
    assert r.ok
    assert abs(r.H[0, 2]) < 1.0 and abs(r.H[1, 2]) < 1.0      # no translation
    assert abs(r.H[0, 0] - 1) < 0.05 and abs(r.H[1, 1] - 1) < 0.05  # no scale


def test_recovers_known_translation(textured):
    big = textured(2, h=260, w=380)
    g = big[10:250, 10:330]
    shifted = big[10:250, 18:338]          # same content, +8 px in x (no wrap)
    r = GlobalMotionEstimator().estimate(g, shifted)
    assert r.ok
    mag = (r.H[0, 2] ** 2 + r.H[1, 2] ** 2) ** 0.5
    assert 6.0 <= mag <= 10.0              # ~8 px translation recovered


def test_unrelated_frames_flagged_not_ok(textured):
    a = textured(3)
    b = np.random.default_rng(99).integers(0, 255, a.shape).astype(np.uint8)
    r = GlobalMotionEstimator().estimate(a, b)
    assert not r.ok                        # discontinuity -> callers re-anchor
    assert np.allclose(r.H, np.eye(3))     # identity fallback
