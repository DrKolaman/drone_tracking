"""Camera-motion-compensated MOG2 detector.

`cv2.MOG2` keeps its background model on a fixed pixel grid, so it only works if
the background is stationary in the frame it sees. Under a moving camera we make
that true by warping every frame into one common *reference canvas* via the
cumulative homography from registration.py, and running MOG2 there. Because the
background is (nearly) static in the reference, the only residual foreground is
the independently-moving person.

Reference handling:
  * The reference canvas is larger than the frame (padded by `margin_frac`) so the
    warped frame has room to pan before falling off the canvas.
  * `H_cum` maps current-frame coords -> reference-canvas coords. Each frame it is
    updated as H_cum = H_cum_prev @ H_rel (H_rel maps current -> previous).
  * We re-anchor (reset H_cum, recreate MOG2 with the current frame as the new
    reference) when registration fails, or the warped frame would overflow the
    canvas. The pipeline also calls reanchor() on scene cuts.

Detections are found in reference coords (MOG2 + morphology + connected
components) and mapped back to current-frame coords via inv(H_cum) for tracking
and display.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from registration import GlobalMotionEstimator


@dataclass
class DetectorDebug:
    compensated: np.ndarray  # current frame warped into the reference canvas (BGR)
    fgmask: np.ndarray       # post-processed foreground mask in canvas coords (uint8)
    reanchored: bool         # True if this frame forced a reference reset
    n_inliers: int           # registration inliers this frame
    n_matches: int = 0       # tracked correspondences fed to RANSAC
    reason: str = ""         # why a reanchor happened: "init"|"reg_fail"|"overflow"


class CompensatedMOG2Detector:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        margin_frac: float = 0.5,
        history: int = 120,
        var_threshold: float = 24.0,
        min_area_px: float = 5.0,
        max_area_frac: float = 0.05,
        aspect_min: float = 0.25,
        aspect_max: float = 4.0,
        border_guard: int = 12,
        validity_erode_px: int = 25,
        estimator: GlobalMotionEstimator | None = None,
    ) -> None:
        self.fw, self.fh = frame_w, frame_h
        self.ox, self.oy = int(margin_frac * frame_w), int(margin_frac * frame_h)
        self.cw = frame_w + 2 * self.ox
        self.ch = frame_h + 2 * self.oy
        # H_cum at anchor: place the anchor frame at the canvas offset.
        self._offset = np.array([[1, 0, self.ox], [0, 1, self.oy], [0, 0, 1]],
                                dtype=np.float64)

        self.history = history
        self.var_threshold = var_threshold
        # The target is a small top-down blob, so the floor is an absolute pixel
        # area (not a frame fraction) and the aspect window is near-square.
        self.min_area = min_area_px
        self.max_area = max_area_frac * frame_w * frame_h
        self.aspect_min, self.aspect_max = aspect_min, aspect_max
        self.border_guard = border_guard

        self.est = estimator or GlobalMotionEstimator()
        # Small kernel for fg morphology: must NOT erase few-pixel targets.
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Large kernel to trim the warped-frame border, where the rotating seam
        # otherwise leaks in as foreground. ~validity_erode_px inward.
        e = max(1, validity_erode_px)
        self._k_validity = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
        self._prev_gray: np.ndarray | None = None
        self.H_cum = self._offset.copy()
        self._new_mog2()

    # --- reference management -------------------------------------------------
    def _new_mog2(self) -> None:
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=False
        )

    def reanchor(self, gray: np.ndarray) -> None:
        """Reset the reference to the current frame (new MOG2, identity offset)."""
        self.H_cum = self._offset.copy()
        self._new_mog2()
        self._prev_gray = gray

    def _would_overflow(self, H_cum: np.ndarray) -> bool:
        corners = np.float32([[0, 0], [self.fw, 0], [self.fw, self.fh], [0, self.fh]])
        warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_cum).reshape(-1, 2)
        g = self.border_guard
        return bool(
            (warped[:, 0] < g).any() or (warped[:, 0] > self.cw - g).any()
            or (warped[:, 1] < g).any() or (warped[:, 1] > self.ch - g).any()
        )

    # --- main -----------------------------------------------------------------
    def process(self, frame: np.ndarray, gray: np.ndarray):
        reanchored = False
        n_inliers = n_matches = 0
        reason = ""

        if self._prev_gray is None:
            self.reanchor(gray)
            reanchored = True
            reason = "init"
        else:
            res = self.est.estimate(self._prev_gray, gray)
            n_inliers, n_matches = res.n_inliers, res.n_matches
            if not res.ok:
                self.reanchor(gray)               # registration failed / discontinuity
                reanchored = True
                reason = "reg_fail"
            else:
                H_cum = self.H_cum @ res.H
                if self._would_overflow(H_cum):
                    self.reanchor(gray)           # panned off the canvas
                    reanchored = True
                    reason = "overflow"
                else:
                    self.H_cum = H_cum

        # Warp current frame into the reference canvas and run MOG2 there.
        aligned = cv2.warpPerspective(frame, self.H_cum, (self.cw, self.ch))
        validity = cv2.warpPerspective(
            np.full((self.fh, self.fw), 255, np.uint8), self.H_cum, (self.cw, self.ch)
        )
        # Trim the warped-frame border hard: the rotating seam otherwise leaks in
        # as a large false foreground strip.
        validity = cv2.erode(validity, self._k_validity)

        lr = 1.0 if reanchored else -1.0       # fast-learn the new reference frame
        fg = self.mog2.apply(aligned, learningRate=lr)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]   # drop shadows/uncertain
        fg = cv2.bitwise_and(fg, validity)

        # Gentle morphology: open(3x3) kills single-pixel speckle but keeps a few-
        # pixel target; close+dilate then consolidate it into one blob.
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k3, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k3, iterations=2)
        fg = cv2.dilate(fg, self._k3, iterations=1)

        boxes = [] if reanchored else self._boxes_from_mask(fg)

        self._prev_gray = gray
        return boxes, DetectorDebug(aligned, fg, reanchored, n_inliers, n_matches, reason)

    # --- blob -> boxes (canvas coords -> current-frame coords) ----------------
    def _boxes_from_mask(self, fg: np.ndarray) -> list[list[int]]:
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        H_inv = np.linalg.inv(self.H_cum)
        out: list[list[int]] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area or area > self.max_area:
                continue
            aspect = h / max(w, 1)
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue
            box = self._canvas_box_to_frame([x, y, w, h], H_inv)
            if box is not None:
                out.append(box)
        return out

    def _canvas_box_to_frame(self, xywh, H_inv) -> list[int] | None:
        x, y, w, h = xywh
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        fr = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        x1, y1 = fr[:, 0].min(), fr[:, 1].min()
        x2, y2 = fr[:, 0].max(), fr[:, 1].max()
        x1 = int(np.clip(x1, 0, self.fw - 1)); x2 = int(np.clip(x2, 0, self.fw - 1))
        y1 = int(np.clip(y1, 0, self.fh - 1)); y2 = int(np.clip(y2, 0, self.fh - 1))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        return [x1, y1, x2, y2]
