"""Blur detection (synthetic) + real-clip blur/colour characterization (CPU)."""
import cv2
import numpy as np
import pytest

from build_map import blur_scores
from colorfix import color_spread, to_bw
from conftest import needs_video


def test_blur_scores_lower_for_blurred(make_texture):
    sharp = make_texture(seed=7)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    s_sharp, s_blur = blur_scores([sharp, blurred])
    assert s_blur < 0.3 * s_sharp


@needs_video
def test_motion_blur_burst_is_minimum(grab):
    # frames 605..619: the sharpness minimum must fall in the 610-612 burst
    frames = [to_bw(grab(i)) for i in range(605, 620)]
    scores = blur_scores(frames)
    argmin_frame = 605 + int(np.argmin(scores))
    assert argmin_frame in (610, 611, 612)


@needs_video
def test_burst_below_quarter_of_baseline(grab):
    baseline = np.median(blur_scores([to_bw(grab(i)) for i in range(615, 647)]))
    burst = np.median(blur_scores([to_bw(grab(i)) for i in (610, 611, 612)]))
    assert burst < 0.25 * baseline          # justifies blur_frac=0.19..0.25


@needs_video
def test_colour_switch_detected(grab):
    assert color_spread(grab(300)) < 12      # B/W segment
    assert color_spread(grab(990)) > 30      # red thermal segment
