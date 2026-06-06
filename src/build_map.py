"""Build a stitched map (mosaic) of the continuous segment from the start to the zoom.

Per docs/REQUIREMENTS.md the clip is continuous over ~frames 0-609, then the 610-614
motion blur, the 615-646 static hold, and the 647 FOV switch ("zoom"). This builds
one map of that opening continuous run.

Pipeline (matches the proven stitch in the stich_yolo_bytetrack worktree, with a
better compositor):
  * Global motion: Shi-Tomasi + Lucas-Kanade + RANSAC homography per consecutive
    pair (`registration.GlobalMotionEstimator`, current->previous). Chain to the
    first frame. Stop at the first frame registration can't bridge (a discontinuity)
    so the map is the clean run from the start.
  * Canvas fit to the whole trajectory (`fit_canvas`).
  * FEATHER blending instead of last-wins or naive average: each frame is weighted
    by its distance-to-border, so frame centres stay sharp while seams and per-frame
    thermal-AGC brightness jumps blend out. This removes the visible frame boundaries
    that last-wins leaves, without the all-average blur.

    python3 src/build_map.py --lo 0 --hi 609 --stride 1
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from registration import GlobalMotionEstimator


def fit_canvas(Hs, w, h):
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    xs, ys = [], []
    for H in Hs:
        c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs += [c[:, 0].min(), c[:, 0].max()]
        ys += [c[:, 1].min(), c[:, 1].max()]
    T = np.array([[1, 0, -min(xs)], [0, 1, -min(ys)], [0, 0, 1]], np.float64)
    return T, int(np.ceil(max(xs) - min(xs))), int(np.ceil(max(ys) - min(ys)))


def blur_scores(frames):
    """Variance-of-Laplacian sharpness per frame (low = blurred)."""
    return [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            for f in frames]


def chain(frames, min_inliers, blur_frac=0.19):
    """Homographies frame_i -> anchor via chained LK homographies, SKIPPING blurred
    frames so the motion-blur burst (e.g. 610-614) doesn't inject a skew into the
    chain. A blurred frame is never made the reference and never multiplied in; the
    chain bridges from the last sharp frame straight to the next sharp one.

    Returns (Hs, kept) where kept[i] is the frame index for Hs[i]. Stops at the
    first un-registerable sharp frame (a true discontinuity).
    """
    scores = blur_scores(frames)
    thresh = blur_frac * float(np.median(scores))
    est = GlobalMotionEstimator(min_inliers=min_inliers, model="similarity")
    start = 0
    while start < len(frames) and scores[start] < thresh:
        start += 1
    prev = cv2.cvtColor(frames[start], cv2.COLOR_BGR2GRAY)
    rel = np.eye(3)
    Hs, kept, skipped = [rel.copy()], [start], []
    for i in range(start + 1, len(frames)):
        if scores[i] < thresh:
            skipped.append(i)
            continue
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        res = est.estimate(prev, g)
        if not res.ok:
            print(f"  stop at frame offset {i}: registration failed "
                  f"(inliers={res.n_inliers}) -> discontinuity")
            break
        rel = rel @ res.H                 # frame_i -> anchor
        Hs.append(rel.copy())
        kept.append(i)
        prev = g
    if skipped:
        print(f"  skipped {len(skipped)} blurred frames (thresh={thresh:.1f}): "
              f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    return Hs, kept


def _feather_weight(h, w):
    """Distance-to-border weight: ~0 at edges, max at centre (for seam blending)."""
    m = np.full((h, w), 255, np.uint8)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return (d / (d.max() + 1e-6)).astype(np.float32)


def build_mosaic(frames, Hs, kept, max_canvas=4500):
    h, w = frames[0].shape[:2]
    T, cw, ch = fit_canvas(Hs, w, h)
    if max(cw, ch) > max_canvas:
        print(f"  WARNING: canvas {cw}x{ch} exceeds {max_canvas}; drift is large.")
    weight = _feather_weight(h, w)
    acc = np.zeros((ch, cw, 3), np.float32)
    wsum = np.zeros((ch, cw), np.float32)
    for k, H in zip(kept, Hs):
        f = frames[k]
        Hc = T @ H
        warp = cv2.warpPerspective(f, Hc, (cw, ch)).astype(np.float32)
        wt = cv2.warpPerspective(weight, Hc, (cw, ch))
        acc += warp * wt[..., None]
        wsum += wt
    mosaic = acc / np.maximum(wsum, 1e-6)[..., None]
    return mosaic.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=609)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > args.hi:
            break
        if idx >= args.lo and (idx - args.lo) % args.stride == 0:
            frames.append(f)
        idx += 1
    cap.release()
    print(f"read {len(frames)} frames [{args.lo}:{args.hi}:{args.stride}]")

    Hs, kept = chain(frames, args.min_inliers)
    mosaic = build_mosaic(frames, Hs, kept)
    out = f"{args.out}/map_{args.lo}_{args.hi}.png"
    cv2.imwrite(out, mosaic)
    print(f"stitched {len(kept)} frames (through offset {kept[-1]}) -> "
          f"map {mosaic.shape[1]}x{mosaic.shape[0]} -> {out}")


if __name__ == "__main__":
    main()
