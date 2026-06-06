"""Test: can a homography register two loop-closure frames across a scale change?

The loop closure at frames (632, 1094) shows the *same* tree at *different
scales* (the camera is closer in one). Those frames live in different continuous
segments, so the within-segment stitch never registered them to each other. This
script asks the direct question: estimate a homography straight between the two
frames and see whether ORB matching + RANSAC can (a) find correspondences across
the scale gap at all, and (b) produce a warp that actually lands the tree on the
tree.

A homography *can* represent scale (it is projective). The practical question is
whether the feature matcher survives the scale difference well enough to fit one.

Usage:
    python3 src/test_scale_warp.py --i 632 --j 1094
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def grab(video: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx}")
    return frame


def direct_homography(a: np.ndarray, b: np.ndarray, n_features: int, ratio: float):
    """Estimate H mapping image `a` -> image `b` via ORB + Lowe-ratio + RANSAC."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # 8 pyramid levels @ 1.2x = ORB sees ~3.6x of scale; that is the cross-scale budget.
    orb = cv2.ORB_create(nfeatures=n_features, nlevels=8, scaleFactor=1.2)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None:
        return None, 0, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]

    if len(good) < 4:
        return None, len(good), 0, 0.0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, len(good), inliers, 0.0


def implied_scale(H: np.ndarray, w: int, h: int) -> float:
    """Local *linear* scale of H at the image centre = sqrt(|det of the Jacobian|).

    det(J) is the area magnification; its square root is the linear scale factor.
    """
    cx, cy = w / 2.0, h / 2.0
    # Jacobian of the projective map at the centre.
    p = H @ np.array([cx, cy, 1.0])
    wz = p[2]
    J = (H[:2, :2] * wz - np.outer(p[:2], H[2, :2])) / (wz * wz)
    return float(np.sqrt(abs(np.linalg.det(J))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--i", type=int, default=632)
    ap.add_argument("--j", type=int, default=1094)
    ap.add_argument("--features", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    a = grab(args.video, args.i)
    b = grab(args.video, args.j)
    h, w = b.shape[:2]

    H, n_good, inliers, _ = direct_homography(a, b, args.features, args.ratio)
    print(f"Direct homography  frame {args.i} -> frame {args.j}")
    print(f"  good matches (ratio test): {n_good}")
    if H is None:
        print("  RESULT: no homography -- matcher could not bridge the scale gap.")
        return
    ratio_in = inliers / n_good if n_good else 0.0
    sc = implied_scale(H, w, h)
    print(f"  RANSAC inliers: {inliers}  ({ratio_in:.1%} of matches)")
    print(f"  implied linear scale a->b: {sc:.2f}x")
    verdict = "HANDLES IT" if inliers >= 15 else "FAILS (too few inliers to trust)"
    print(f"  VERDICT: homography {verdict}")

    # Warp a onto b and build a blended overlay to judge registration by eye.
    warped = cv2.warpPerspective(a, H, (w, h))
    valid = cv2.warpPerspective(np.ones((a.shape[0], a.shape[1]), np.uint8), H, (w, h))
    overlay = b.copy()
    m = valid > 0
    overlay[m] = (0.5 * b[m] + 0.5 * warped[m]).astype(np.uint8)

    panel = np.hstack([a, b, warped, overlay])
    path = f"{args.out}/scale_warp_test.png"
    cv2.imwrite(path, panel)
    print(f"  panel [A | B | A-warped-to-B | 50/50 overlay] -> {path}")


if __name__ == "__main__":
    main()
