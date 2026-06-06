#!/usr/bin/env python3
"""Offline ego-motion-compensated motion tracker (reference / re-detector).

The target is a faint warm blob that MOVES INDEPENDENTLY of the ground; the
big appearance distractor (the bright 'ladder' structure) is static. So we
track independent motion:

  per frame:
    1. estimate global camera motion prev->cur (ORB+RANSAC homography),
    2. warp prev into cur and take the residual = independent motion,
    3. predict the target's image position (apply H to last pos + velocity),
    4. pick the independent-motion peak within a small window of the prediction,
    5. blend warmth (top-hat) lightly to disambiguate; update velocity.

Outputs a trajectory txt and (optionally) an overlay video for verification.
This is deliberately offline/slow — accuracy over speed.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

_ORB = cv2.ORB_create(2000)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def homography(prev_gray, cur_gray):
    k0, d0 = _ORB.detectAndCompute(prev_gray, None)
    k1, d1 = _ORB.detectAndCompute(cur_gray, None)
    if d0 is None or d1 is None:
        return None
    m = _BF.match(d0, d1)
    if len(m) < 15:
        return None
    m = sorted(m, key=lambda x: x.distance)[:300]
    src = np.float32([k0[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([k1[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("x", type=float); ap.add_argument("y", type=float)
    ap.add_argument("w", type=float); ap.add_argument("h", type=float)
    ap.add_argument("--stop", type=int, default=0)
    ap.add_argument("--out_txt", default="output/motion_gt.txt")
    ap.add_argument("--out_vid", default="output/motion_gt.mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = args.stop if args.stop > 0 else total
    bw, bh = args.w, args.h

    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    ftxt = open(args.out_txt, "w")

    ok, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pos = np.array([args.x + bw / 2, args.y + bh / 2], np.float32)
    vel = np.array([0, 0], np.float32)

    def emit(bgr, p):
        x, y = int(p[0] - bw / 2), int(p[1] - bh / 2)
        cv2.rectangle(bgr, (x, y), (x + int(bw), y + int(bh)), (0, 255, 255), 2)
        writer.write(bgr)
        ftxt.write(f"{p[0]:.1f}\t{p[1]:.1f}\n")

    emit(frame.copy(), pos)
    idx = 1
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Hm = homography(prev_gray, cur_gray)
        # predicted image position after camera motion + own velocity
        if Hm is not None:
            p_h = cv2.perspectiveTransform(pos.reshape(1, 1, 2), Hm).reshape(2)
            warp = cv2.warpPerspective(prev_gray, Hm, (W, H))
            resid = cv2.absdiff(warp, cur_gray); resid[warp == 0] = 0
        else:
            p_h = pos
            resid = cv2.absdiff(prev_gray, cur_gray)
        # cap velocity and keep the prediction on-frame (guards runaway divergence)
        spd = float(np.hypot(*vel))
        if spd > 25.0:
            vel = vel * (25.0 / spd)
        pred = p_h + vel
        pred[0] = min(max(pred[0], 0), W - 1)
        pred[1] = min(max(pred[1], 0), H - 1)
        # independent motion, denoised (median kills sharp-edge registration speckle)
        resid = cv2.medianBlur(resid, 5).astype(np.float32)
        resid = cv2.GaussianBlur(resid, (0, 0), 2)
        bright = cv2.morphologyEx(cur_gray, cv2.MORPH_TOPHAT,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(np.float32)

        radius = 38
        x0, y0 = int(max(pred[0] - radius, 0)), int(max(pred[1] - radius, 0))
        x1, y1 = int(min(pred[0] + radius, W)), int(min(pred[1] + radius, H))
        x1 = max(x1, x0 + 4); y1 = max(y1, y0 + 4)  # never-empty ROI
        rm = resid[y0:y1, x0:x1]; rb = bright[y0:y1, x0:x1]
        if rm.size:
            def nrm(a):
                mx = a.max(); return a / mx if mx > 1e-6 else a
            score = 0.75 * nrm(rm) + 0.25 * nrm(rb)
            score = cv2.GaussianBlur(score, (0, 0), 1.5)
            _, _, _, loc = cv2.minMaxLoc(score)
            newp = np.array([loc[0] + x0, loc[1] + y0], np.float32)
        else:
            newp = pred
        vel = 0.6 * vel + 0.4 * (newp - p_h)  # velocity in compensated frame
        pos = newp
        emit(frame.copy(), pos)
        prev_gray = cur_gray
        idx += 1

    cap.release(); writer.release(); ftxt.close()
    print(f"done: {idx} frames -> {args.out_vid}, {args.out_txt}")


if __name__ == "__main__":
    main()
