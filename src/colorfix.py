"""Colour-normalisation filter: keep the video in B/W across colour-mode switches.

The clip switches B/W <-> red (thermal colormap) at 957/1030. The red frames are the
SAME thermal scene, just colour-mapped, so they SHOULD match the B/W frames -- but
cv2.BGR2GRAY weights red at only 0.30 and crushes a red-colormapped frame's contrast
(measured: a red thermal frame's luminance variance collapses from ~4700 in the red
channel to ~100 in BGR2GRAY). That wrecks registration / loop closure across the switch.

Filter: if a frame is colour (high per-pixel channel spread), convert it to B/W using
the highest-variance channel (the one carrying the thermal signal) instead of luminance;
B/W frames pass through as plain grayscale. Output is always 3-channel grayscale, so the
whole pipeline sees one consistent modality.

    from colorfix import to_bw   # to_bw(frame_bgr) -> 3-channel grayscale
"""
from __future__ import annotations

import cv2
import numpy as np


def color_spread(bgr: np.ndarray) -> float:
    """Mean per-pixel channel spread; ~0 for grayscale, large for colour."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    return float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)


def to_bw(bgr: np.ndarray, spread_thresh: float = 12.0) -> np.ndarray:
    """Normalise a frame to 3-channel B/W, recovering thermal contrast on colour frames."""
    if color_spread(bgr) <= spread_thresh:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        # colour-mapped thermal: the channel with the most variance carries the signal
        g = max(cv2.split(bgr), key=lambda c: c.var())
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
