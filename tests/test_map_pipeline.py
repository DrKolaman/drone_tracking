"""Map-pipeline tests (no GPU): chain recovers motion, skips blur; mosaic composites."""
import cv2
import numpy as np
import pytest

from build_map import chain, build_mosaic, fit_canvas
from conftest import sim_matrix


def _pan_sequence(make_texture, n=6, step=6):
    """frames[i] = base translated right by step*i (pure pan)."""
    base = make_texture(seed=11)
    h, w = base.shape[:2]
    return [cv2.warpPerspective(base, sim_matrix(tx=step * i), (w, h)) for i in range(n)]


def test_chain_recovers_pan(make_texture):
    n, step = 6, 6
    frames = _pan_sequence(make_texture, n, step)
    Hs, kept = chain(frames, min_inliers=20)
    assert kept == list(range(n))                    # all sharp -> all kept
    # frame_i content is shifted +step*i; mapping frame_i->frame_0 shifts by -step*i
    assert Hs[-1][0, 2] == pytest.approx(-step * (n - 1), abs=2.0)
    assert Hs[-1][1, 2] == pytest.approx(0.0, abs=2.0)


def test_chain_skips_blurred_frame(make_texture):
    frames = _pan_sequence(make_texture, n=6, step=5)
    frames[3] = cv2.GaussianBlur(frames[3], (0, 0), 6)   # blur the middle frame
    Hs, kept = chain(frames, min_inliers=20)
    assert 3 not in kept                                  # blurred frame dropped
    assert kept[0] == 0 and kept[-1] == 5                 # chain bridges across it


def test_build_mosaic_canvas_and_smoke(make_texture):
    frames = _pan_sequence(make_texture, n=5, step=8)
    Hs, kept = chain(frames, min_inliers=20)
    T, cw, ch = fit_canvas(Hs, *frames[0].shape[1::-1])
    mosaic = build_mosaic(frames, Hs, kept)
    assert mosaic.shape == (ch, cw, 3)
    assert mosaic.dtype == np.uint8
    assert cw >= frames[0].shape[1]                       # canvas grew with the pan
    assert mosaic.max() > 0                               # something was drawn
