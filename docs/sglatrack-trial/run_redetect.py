#!/usr/bin/env python3
"""SGLATrack + re-detection loop for the faint thermal target.

SGLATrack alone loses this ~30px low-contrast blob after ~5s (its peak response
collapses from ~0.9 to ~0.3). Here we watch that confidence: while it's healthy
we trust SGLATrack; when it collapses we re-acquire the warm target by white
top-hat (small bright blob) in an expanding window around the last good
position, and RE-INITIALISE the tracker there. This is the detector+tracker
pattern that keeps the track alive on a target the SOT core can't hold solo.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.test.evaluation import Tracker  # noqa: E402


def reacquire(gray, cx, cy, radius, box_w, box_h, min_resp=4.0):
    """Find the strongest small bright blob in a window around (cx,cy)."""
    H, W = gray.shape
    x0, y0 = max(int(cx - radius), 0), max(int(cy - radius), 0)
    x1, y1 = min(int(cx + radius), W), min(int(cy + radius), H)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k)
    th = cv2.GaussianBlur(th.astype(np.float32), (0, 0), 2)
    _, mx, _, loc = cv2.minMaxLoc(th)
    if mx < min_resp:
        return None
    px, py = loc[0] + x0, loc[1] + y0
    return [px - box_w / 2.0, py - box_h / 2.0, float(box_w), float(box_h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("x", type=float); ap.add_argument("y", type=float)
    ap.add_argument("w", type=float); ap.add_argument("h", type=float)
    ap.add_argument("--stop", type=int, default=0)
    ap.add_argument("--conf_thr", type=float, default=0.45)
    ap.add_argument("--out", default="output/source_redetect.mp4")
    ap.add_argument("--boxes_txt", default="output/source_redetect.txt")
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

    box_w, box_h = args.w, args.h
    ok, frame = cap.read()
    init_box = [args.x, args.y, box_w, box_h]
    trk.initialize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {"init_bbox": init_box})
    last_cx, last_cy = args.x + box_w / 2, args.y + box_h / 2
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
        out = trk.track(rgb)
        conf = out.get("conf", 1.0)
        box = out["target_bbox"]

        if conf >= args.conf_thr:
            status = "track"
            last_cx, last_cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            lost = 0
        else:
            lost += 1
            radius = min(60 + 18 * lost, max(W, H))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cand = reacquire(gray, last_cx, last_cy, radius, box_w, box_h)
            if cand is not None:
                box = cand
                trk.initialize(rgb, {"init_bbox": box})  # reseed template at re-acquired blob
                last_cx, last_cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                lost = 0
                n_redetect += 1
                status = "redetect"
            else:
                box = [last_cx - box_w / 2, last_cy - box_h / 2, box_w, box_h]
                status = "lost"
        emit(frame, box, status)
        idx += 1

    cap.release(); writer.release(); fbox.close()
    print(f"done: {idx} frames, re-detections={n_redetect} -> {args.out}")


if __name__ == "__main__":
    main()
