"""GlobalMotionEstimator must recover a known camera motion (no GPU).

cur = warp(prev, H_true) => the estimator's H maps cur->prev ~= H_true^-1,
so H_est @ H_true ~= identity.
"""
import cv2
import numpy as np
import pytest

from registration import GlobalMotionEstimator
from conftest import sim_matrix


def _corner_err(make_texture, H_true, model):
    """Max corner reprojection error (px) after composing the estimate with H_true."""
    prev = make_texture(seed=3)
    h, w = prev.shape[:2]
    cur = cv2.warpPerspective(prev, H_true, (w, h))
    est = GlobalMotionEstimator(min_inliers=20, model=model)
    res = est.estimate(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    assert res.ok and res.n_inliers >= 20
    M = res.H @ H_true                       # ~= identity if recovered correctly
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    return float(np.linalg.norm(mapped - corners.reshape(-1, 2), axis=1).max())


@pytest.mark.parametrize("model", ["homography", "similarity"])
def test_recovers_translation(make_texture, model):
    assert _corner_err(make_texture, sim_matrix(tx=14, ty=-9), model) < 2.0


@pytest.mark.parametrize("model", ["homography", "similarity"])
def test_recovers_similarity(make_texture, model):
    # similarity model is sub-pixel; the 8-DOF homography has a touch more slack
    tol = 1.5 if model == "similarity" else 4.0
    assert _corner_err(make_texture, sim_matrix(scale=1.15, deg=7, tx=10, ty=6), model) < tol


def test_similarity_has_no_shear(make_texture):
    # a similarity fit must produce a (scaled-rotation) 2x2 block: a=d, b=-c
    prev = make_texture(seed=4)
    h, w = prev.shape[:2]
    cur = cv2.warpPerspective(prev, sim_matrix(scale=1.1, deg=10, tx=8, ty=4), (w, h))
    res = GlobalMotionEstimator(min_inliers=20, model="similarity").estimate(
        cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    a, b, c, d = res.H[0, 0], res.H[0, 1], res.H[1, 0], res.H[1, 1]
    assert a == pytest.approx(d, abs=1e-6) and b == pytest.approx(-c, abs=1e-6)
    assert res.H[2, 0] == 0 and res.H[2, 1] == 0          # no projective term


def test_fails_on_unrelated_frames(make_texture):
    a = make_texture(seed=5)
    b = make_texture(seed=999)            # different scene
    res = GlobalMotionEstimator(min_inliers=25).estimate(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    assert not res.ok                      # discontinuity -> caller re-anchors
