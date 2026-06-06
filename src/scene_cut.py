"""Scene-discontinuity detection.

The source clip is only *continuous* for roughly its first half. After that it
exhibits a sharp resolution/focus change, a hard cut to a different scene, and a
switch from a (near-)grayscale thermal look to full colour.

A motion/appearance tracker assumes temporal continuity: object identity is
propagated from frame t-1 to t via a motion model + appearance. Carrying IDs
across a hard cut is meaningless and produces ID-switch garbage. This module
gives the pipeline a cheap, dependency-free signal so it can *reset* the tracker
at a discontinuity instead of hallucinating continuity across it.

Two complementary signals, both computed on a downscaled frame so the cost is
negligible relative to the detector:

1. Hard-cut: HSV-histogram correlation between consecutive frames. A value far
   below the running norm means the global appearance changed abruptly.
2. Colour-mode flip: mean per-pixel channel spread (|R-G|,|G-B|,|R-B|). A
   near-grayscale frame (thermal) has ~0 spread; colour footage is well above
   it. A flip across the threshold marks the thermal->colour transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CutEvent:
    """A detected discontinuity at `frame_idx`."""

    frame_idx: int
    kind: str          # "hard_cut" | "color_mode_flip"
    hist_corr: float   # histogram correlation vs previous frame
    is_color: bool     # colour mode of the *current* frame


class SceneCutDetector:
    """Flags abrupt scene cuts and grayscale<->colour transitions.

    Parameters
    ----------
    hist_corr_thresh:
        Correlation below this between consecutive HSV histograms is a hard cut.
        1.0 = identical, ~0 = unrelated. 0.6 is a conservative default that fires
        on real cuts without tripping on ordinary motion.
    color_spread_thresh:
        Mean channel spread above this => the frame is treated as colour.
    downscale:
        Long edge (px) the frame is resized to before measuring. Keeps the check
        cheap and insensitive to the resolution change itself.
    """

    def __init__(
        self,
        hist_corr_thresh: float = 0.6,
        color_spread_thresh: float = 12.0,
        downscale: int = 256,
    ) -> None:
        self.hist_corr_thresh = hist_corr_thresh
        self.color_spread_thresh = color_spread_thresh
        self.downscale = downscale
        self._prev_hist: np.ndarray | None = None
        self._prev_is_color: bool | None = None

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.downscale / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _hsv_hist(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _color_spread(self, frame: np.ndarray) -> float:
        b, g, r = cv2.split(frame.astype(np.int16))
        spread = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0
        return float(spread.mean())

    def update(self, frame: np.ndarray, frame_idx: int) -> CutEvent | None:
        """Feed the next frame. Returns a CutEvent if a discontinuity is seen."""
        small = self._resize(frame)
        hist = self._hsv_hist(small)
        is_color = self._color_spread(small) > self.color_spread_thresh

        event: CutEvent | None = None
        if self._prev_hist is not None:
            corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
            if self._prev_is_color is not None and is_color != self._prev_is_color:
                event = CutEvent(frame_idx, "color_mode_flip", corr, is_color)
            elif corr < self.hist_corr_thresh:
                event = CutEvent(frame_idx, "hard_cut", corr, is_color)

        self._prev_hist = hist
        self._prev_is_color = is_color
        return event
