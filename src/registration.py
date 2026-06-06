"""Global camera-motion estimation via sparse optical flow.

The camera moves, so before MOG2 can isolate the *independently* moving person we
must cancel the global (camera-induced) motion. This module estimates the
frame-to-frame global transform: track Shi-Tomasi corners with Lucas-Kanade and
fit a homography with RANSAC. The homography maps the *current* frame's points to
the *previous* frame's coordinates; chaining these maps any frame back to a
reference (see motion_detector.CompensatedMOG2Detector).

Homography (8-DOF) is used because the camera pans/rotates; it models a planar
scene or pure rotation exactly. Residual error under camera translation in a 3-D
scene (parallax) is a documented limitation of the global-motion assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionResult:
    H: np.ndarray        # 3x3, maps current-frame points -> previous-frame coords
    n_inliers: int       # RANSAC inliers fed to the fit
    n_matches: int       # tracked point correspondences fed to RANSAC
    ok: bool = True      # False => fell back to identity (failure / discontinuity
                         #          such as a hard zoom or cut). Callers should
                         #          RE-ANCHOR, not apply the identity transform.


_IDENTITY = np.eye(3, dtype=np.float64)


class GlobalMotionEstimator:
    """Estimate the inter-frame global motion homography (current -> previous)."""

    def __init__(
        self,
        max_corners: int = 600,
        quality_level: float = 0.01,
        min_distance: int = 7,
        block_size: int = 7,
        lk_win: int = 21,
        lk_levels: int = 3,
        ransac_thresh: float = 3.0,
        min_inliers: int = 25,
    ) -> None:
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers

    def estimate(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> MotionResult:
        """Return the homography mapping cur_gray points to prev_gray coords.

        On failure (too few features/matches/inliers) returns identity with
        n_inliers=0 so the caller can re-anchor the reference.
        """
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        if prev_pts is None or len(prev_pts) < 4:
            return MotionResult(_IDENTITY.copy(), 0, 0, ok=False)

        cur_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pts, None, **self.lk_params
        )
        if cur_pts is None or status is None:
            return MotionResult(_IDENTITY.copy(), 0, 0, ok=False)

        good = status.ravel() == 1
        p_prev = prev_pts[good].reshape(-1, 2)
        p_cur = cur_pts[good].reshape(-1, 2)
        n_matches = len(p_cur)
        if n_matches < 4:
            return MotionResult(_IDENTITY.copy(), 0, n_matches, ok=False)

        # Map current -> previous so cumulative product warps current -> reference.
        H, mask = cv2.findHomography(p_cur, p_prev, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return MotionResult(_IDENTITY.copy(), 0, n_matches, ok=False)

        n_inliers = int(mask.sum()) if mask is not None else 0
        # Too few inliers => a discontinuity the small-motion model can't bridge
        # (e.g. the ~7x zoom at frame 647): the fit collapses onto the static
        # zoom-centre points and would wrongly read as identity. Flag, don't trust.
        if n_inliers < self.min_inliers:
            return MotionResult(_IDENTITY.copy(), n_inliers, n_matches, ok=False)

        return MotionResult(H.astype(np.float64), n_inliers, n_matches, ok=True)
