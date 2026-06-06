#!/usr/bin/env python3
"""Headless SGLATrack runner: track one target through a video → annotated MP4.

No GUI (works under WSL): uses the programmatic Tracker API instead of
tracking/video_demo.py (which needs cv2.imshow / cv2.selectROI).

Feeds RGB frames to the tracker to match the model's training-time path
(lib/test/evaluation/tracker.py::_read_image converts BGR->RGB for the
benchmark path; only the interactive video demo feeds raw BGR).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Ensure repo root is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.test.evaluation import Tracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Headless SGLATrack on a video.")
    ap.add_argument("video")
    ap.add_argument("x", type=float)
    ap.add_argument("y", type=float)
    ap.add_argument("w", type=float)
    ap.add_argument("h", type=float)
    ap.add_argument("--tracker_param", default="deit_distilled")
    ap.add_argument("--stop", type=int, default=0, help="last frame index (exclusive); 0 = whole video")
    ap.add_argument("--out", default="output/source_annotated.mp4")
    ap.add_argument("--boxes_txt", default="output/source_boxes.txt")
    args = ap.parse_args()

    # Build tracker via the programmatic API.
    t = Tracker("sglatrack", args.tracker_param, "video")
    params = t.get_parameters()
    params.debug = 0
    params.tracker_name = "sglatrack"
    params.param_name = args.tracker_param
    tracker = t.create_tracker(params)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = args.stop if args.stop > 0 else total

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    fbox = open(args.boxes_txt, "w")

    def emit(bgr_frame, box, locked=True):
        x, y, w, h = [int(round(v)) for v in box]
        colour = (0, 255, 0) if locked else (0, 165, 255)
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), colour, 2)
        writer.write(bgr_frame)
        fbox.write("\t".join(f"{v:.2f}" for v in box) + "\n")

    ok, frame = cap.read()
    assert ok, "empty video"
    init_box = [args.x, args.y, args.w, args.h]
    tracker.initialize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {"init_bbox": init_box})
    emit(frame, init_box)

    idx = 1
    while idx < stop:
        ok, frame = cap.read()
        if not ok:
            break
        out = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emit(frame, out["target_bbox"])
        idx += 1

    cap.release()
    writer.release()
    fbox.close()
    print(f"done: {idx} frames (stop={stop}, total={total}) -> {args.out}")
    print(f"boxes -> {args.boxes_txt}")


if __name__ == "__main__":
    main()
