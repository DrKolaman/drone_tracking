"""Pure-geometry unit tests (no GPU). Pin the math the map/zoom code relies on."""
import numpy as np
import pytest

import zoom_geometry as zg
from build_map import fit_canvas, _feather_weight
from test_scale_warp import implied_scale


def _scale_about(c, s):
    """Homography: scale by s about point c=(cx,cy)."""
    cx, cy = c
    return np.array([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy], [0, 0, 1]], np.float64)


def test_scale_at_pure_scale():
    H = np.diag([2.0, 2.0, 1.0])
    assert zg._scale_at(H, 0, 0) == pytest.approx(2.0, rel=1e-6)
    assert zg._scale_at(H, 123, 45) == pytest.approx(2.0, rel=1e-6)


def test_foe_is_scale_center():
    c = (137.0, 88.0)
    H = _scale_about(c, 3.0)
    foe = zg._foe(H)
    assert foe is not None
    assert foe[0] == pytest.approx(c[0], abs=1e-3)
    assert foe[1] == pytest.approx(c[1], abs=1e-3)
    # scale measured at the FoE equals the true scale
    assert zg._scale_at(H, *foe) == pytest.approx(3.0, rel=1e-4)


def test_implied_scale_matches_zoom():
    H = _scale_about((180, 320), 2.5)
    assert implied_scale(H, 360, 640) == pytest.approx(2.5, rel=1e-3)


def test_sym_transfer_err_zero_for_exact_homography():
    H = _scale_about((50, 50), 1.7)
    pa = np.array([[10, 10], [80, 20], [30, 90], [60, 60]], np.float32)
    pb = (zg.cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H)).reshape(-1, 2)
    err = zg._sym_transfer_err(H, pa, pb)
    assert np.all(err < 1e-3)


def test_sampson_zero_on_consistent_epipolar():
    # points related by a pure horizontal shift have F = [[0,0,0],[0,0,-1],[0,1,0]]
    F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
    pa = np.array([[10, 20], [50, 20], [90, 35]], np.float32)
    pb = pa + np.array([15, 0], np.float32)        # same y -> epipolar-consistent
    assert np.all(zg._sampson(F, pa, pb) < 1e-6)


def test_fit_canvas_translation():
    w, h = 100, 60
    Hs = [np.eye(3), np.array([[1, 0, 40], [0, 1, 25], [0, 0, 1]], np.float64)]
    T, cw, ch = fit_canvas(Hs, w, h)
    assert (cw, ch) == (w + 40, h + 25)
    assert T[0, 2] == pytest.approx(0.0) and T[1, 2] == pytest.approx(0.0)


def test_feather_weight_center_gt_edge():
    wt = _feather_weight(40, 60)
    assert wt.shape == (40, 60)
    assert wt[20, 30] > wt[0, 30]          # centre brighter than top edge
    assert wt[20, 30] > wt[20, 0]          # centre brighter than left edge
    assert wt.max() == pytest.approx(1.0, abs=1e-6)


def _corr_from_H(H, n=120, seed=1):
    rng = np.random.default_rng(seed)
    pa = rng.uniform(20, 300, size=(n, 2)).astype(np.float32)
    pb = zg.cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2).astype(np.float32)
    return pa, pb


def test_characterize_planar_is_step_zoom():
    H = _scale_about((180, 160), 2.5)
    pa, pb = _corr_from_H(H)
    rep = zg.characterize_from_correspondences(pa, pb, (320, 360))
    assert rep.verdict == "step-zoom"
    assert rep.zoom == pytest.approx(2.5, rel=0.1)
    assert len(rep.parallax_pts_a) == 0


def test_characterize_flags_parallax_as_dual_camera():
    H = _scale_about((180, 160), 2.0)
    pa, pb = _corr_from_H(H, n=120)
    # inject a coherent off-plane cluster: shift 25 pts horizontally only (parallax),
    # which a single homography cannot explain but a fundamental matrix can.
    pb = pb.copy()
    pb[:25, 0] += 18.0
    rep = zg.characterize_from_correspondences(pa, pb, (320, 360))
    assert rep.n_inliers_F >= rep.n_inliers_H
    assert len(rep.parallax_pts_a) >= 8
