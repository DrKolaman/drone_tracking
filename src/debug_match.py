"""Debug why matching fails across the 646->647 FOV switch.

Decomposes the failure into its possible causes by controlled experiments:
  1. control   : 605<->606 (same scene, same scale, same sensor) -> matcher sanity.
  2. scale-only : 647 vs its own 4x downscale (same appearance, pure scale gap).
  3. target     : 646<->647 raw (scale + thermal-appearance + resolution).
  4. target+CLAHE: 646<->647 after per-frame contrast normalisation
                   (tests the thermal AGC / sensor-appearance hypothesis).

For each: DINOv3 mutual-NN correspondences, RANSAC homography inliers, median
match cosine, and a viz with green=inlier / red=outlier lines so we can SEE
whether the matches are real or noise.
"""
from __future__ import annotations

import cv2
import numpy as np

from dinov3_match import DenseMatcher
from test_scale_warp import grab


def match_stats(a, b, dm):
    """DINOv3 correspondences + geometric-inlier mask + median cosine."""
    pa, pb = dm.match(a, b, sim_floor=0.5)
    n = len(pa)
    mask = np.zeros(n, bool)
    if n >= 4:
        H, m = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        if m is not None:
            mask = m.ravel().astype(bool)
    return pa, pb, mask


def viz(a, b, pa, pb, mask, title, path):
    ha = max(a.shape[0], b.shape[0])
    A = cv2.copyMakeBorder(a, 0, ha - a.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    B = cv2.copyMakeBorder(b, 0, ha - b.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    panel = np.hstack([A, B])
    ox = A.shape[1]
    idx = np.linspace(0, len(pa) - 1, min(80, len(pa))).astype(int) if len(pa) else []
    for i in idx:
        col = (0, 200, 0) if mask[i] else (0, 0, 255)
        p = tuple(np.round(pa[i]).astype(int))
        q = (int(round(pb[i][0])) + ox, int(round(pb[i][1])))
        cv2.circle(panel, p, 2, col, -1)
        cv2.circle(panel, q, 2, col, -1)
        cv2.line(panel, p, q, col, 1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(panel, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, panel)


def clahe(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def run(name, a, b, dm, out):
    pa, pb, mask = match_stats(a, b, dm)
    n, inl = len(pa), int(mask.sum())
    ratio = inl / n if n else 0.0
    print(f"{name:24s} matches={n:5d}  inliers={inl:5d}  inlier_ratio={ratio:.2f}")
    viz(a, b, pa, pb, mask, f"{name}  n={n} inl={inl}", f"{out}/dbg_{name}.png")
    return n, inl


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    dm = DenseMatcher(longside=1024)

    f646, f647 = grab(args.video, 646), grab(args.video, 647)
    f605, f606 = grab(args.video, 605), grab(args.video, 606)
    H, W = f647.shape[:2]
    f647_q = cv2.resize(f647, (W // 4, H // 4), interpolation=cv2.INTER_AREA)  # 4x downscale

    print(f"{'experiment':24s} {'':5s}  result")
    run("control_605_606", f605, f606, dm, args.out)            # sanity: easy pair
    run("scale_only_647_self4x", f647, f647_q, dm, args.out)    # pure 4x scale, same appearance
    run("target_646_647_raw", f646, f647, dm, args.out)         # scale + appearance + resolution
    run("target_646_647_clahe", clahe(f646), clahe(f647), dm, args.out)  # appearance-normalised


if __name__ == "__main__":
    main()
