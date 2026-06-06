"""Colour-mode filter tests: red-thermal frames must normalise to B/W with contrast kept."""
import cv2
import numpy as np
import pytest

from colorfix import color_spread, to_bw


def test_color_spread_zero_for_grayscale():
    g = np.random.default_rng(0).integers(0, 255, (60, 80), np.uint8)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    assert color_spread(bgr) == pytest.approx(0.0, abs=1e-6)


def test_color_spread_high_for_colour():
    img = np.zeros((60, 80, 3), np.uint8)
    img[..., 2] = 200            # strong red only
    assert color_spread(img) > 50


def test_to_bw_outputs_3ch_grayscale():
    img = np.random.default_rng(1).integers(0, 255, (40, 50, 3), np.uint8)
    out = to_bw(img)
    assert out.shape == img.shape
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_to_bw_preserves_red_thermal_contrast():
    # red-colormapped thermal: signal in R, flat G/B -> luminance would crush it.
    rng = np.random.default_rng(2)
    r = rng.integers(0, 255, (80, 100), np.uint8)
    img = cv2.merge([np.full_like(r, 10), np.full_like(r, 10), r])   # B,G flat; R=signal
    out_var = to_bw(img)[..., 0].var()
    lum_var = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).var()
    assert out_var > 5 * lum_var                 # filter keeps the contrast luminance loses
    assert out_var == pytest.approx(r.var(), rel=1e-6)
