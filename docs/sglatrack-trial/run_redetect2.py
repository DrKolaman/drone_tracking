#!/usr/bin/env python3
"""SGLATrack + motion-aware re-detection (v2).

Improvements over run_redetect.py:
  * Re-acquisition uses ego-motion-compensated frame differencing (the moving
    target) combined with white top-hat (the warm blob) — so it re-seeds onto
    the actual mover, not static bright terrain.
  * Re-detect is triggered not only by low confidence but also by box blow-up
    or a large center jump (catches slow drift before it wanders off).
  * Search window grows the longer we stay lost, and recenters on the
    motion-predicted position.

SGLATrack stays the short-term appearance core; motion is the re-detector.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.test.evaluation import Tracker  # noqa: E402

_ORB = cv2.ORB_create(1200)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _norm(a):
    a = a.astype(np.float32)
    m = a.max()
    return a / m if m > 1e-6 else a


def motion_compensated_diff(prev_gray, cur_gray):
    """Warp prev->cur (global motion) and return residual = independent motion."""
    k0, d0 = _ORB.detectAndCompute(prev_gray, None)
    k1, d1 = _ORB.detectAndCompute(cur_gray, None)
    if d0 is None or d1 is None:
        return cv2.absdiff(prev_gray, cur_gray)
    m = _BF.match(d0, d1)
    if len(m) < 12:
        return cv2.absdiff(prev_gray, cur_gray)
    m = sorted(m, key=lambda x: x.distance)[:200]
    src = np.float32([k0[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([k1[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return cv2.absdiff(prev_gray, cur_gray)
    warp = cv2.warpPerspective(prev_gray, H, (cur_gray.shape[1], cur_gray.shape[0]))
    d = cv2.absdiff(warp, cur_gray)
    d[warp == 0] = 0
    return d


def reacquire(cur_gray, motion, cx, cy, radius, bw, bh):
    """Strongest moving+warm blob in a window around (cx,cy)."""
    H, W = cur_gray.shape
    x0, y0 = max(int(cx - radius), 0), max(int(cy - radius), 0)
    x1, y1 = min(int(cx + radius), W), min(int(cy + radius), H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = cur_gray[y0:y1, x0:x1]
    mroi = motion[y0:y1, x0:x1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k)
    score = 0.6 * _norm(mroi) + 0.4 * _norm(bright)
    score = cv2.GaussianBlur(score, (0, 0), 2)
    _, mx, _, loc = cv2.minMaxLoc(score)
    if mx < 0.25:
        return None
    px, py = loc[0] + x0, loc[1] + y0
    return [px - bw / 2.0, py - bh / 2.0, float(bw), float(bh)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("x", type=float); ap.add_argument("y", type=float)
    ap.add_argument("w", type=float); ap.add_argument("h", type=float)
    ap.add_argument("--stop", type=int, default=0)
    ap.add_argument("--conf_thr", type=float, default=0.5)
    ap.add_argument("--out", default="output/source_redetect2.mp4")
    ap.add_argument("--boxes_txt", default="output/source_redetect2.txt")
    args = ap.parse_args()

    t = Tracker("sglatrack", "deit_distilled", "video")
    params = t.get_parameters(); params.debug = 0
    params.tracker_name = "sglatrack"; params.param_name = "deit_distilled"
    trk = t.create_tracker(params)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = args.stop if args.stop > 0 else total

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fbox = open(args.boxes_txt, "w")

    bw, bh = args.w, args.h
    init_area = bw * bh
    ok, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    init_box = [args.x, args.y, bw, bh]
    trk.initialize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {"init_bbox": init_box})
    last_cx, last_cy = args.x + bw / 2, args.y + bh / 2
    lost = 0
    n_redetect = 0

    def emit(bgr, box, status):
        x, y, w, h = [int(round(v)) for v in box]
        colour = {"track": (0, 255, 0), "redetect": (0, 165, 255), "lost": (0, 0, 255)}[status]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), colour, 2)
        writer.write(bgr)
        fbox.write("\t".join(f"{v:.1f}" for v in box) + f"\t{status}\n")

    emit(frame, init_box, "track")
    idx = 1
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = trk.track(rgb)
        conf = out.get("conf", 1.0)
        box = out["target_bbox"]
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        area = box[2] * box[3]
        jump = np.hypot(cx - last_cx, cy - last_cy)

        bad = (conf < args.conf_thr) or (area > 3 * init_area) or (jump > 55)
        if not bad:
            status = "track"
            last_cx, last_cy = cx, cy
            lost = 0
        else:
            lost += 1
            radius = min(55 + 20 * lost, max(W, H))
            motion = motion_compensated_diff(prev_gray, cur_gray)
            cand = reacquire(cur_gray, motion, last_cx, last_cy, radius, bw, bh)
            if cand is not None:
                box = cand
                trk.initialize(rgb, {"init_bbox": box})
                last_cx, last_cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                lost = 0
                n_redetect += 1
                status = "redetect"
            else:
                box = [last_cx - bw / 2, last_cy - bh / 2, bw, bh]
                status = "lost"
        emit(frame, box, status)
        prev_gray = cur_gray
        idx += 1

    cap.release(); writer.release(); fbox.close()
    print(f"done: {idx} frames, re-detections={n_redetect} -> {args.out}")


if __name__ == "__main__":
    main()
