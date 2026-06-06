"""Motion-compensated MOG2 + ByteTrack pipeline.

Detect and track the independently-moving person in a clip shot with a moving
camera. Per frame:

  scene-cut?  -> reset reference + MOG2 + ByteTrack (don't track across cuts)
  register    -> estimate global camera motion (homography), warp into reference
  MOG2        -> foreground in reference coords == the person; map boxes back
  ByteTrack   -> persistent IDs on the boxes (current-frame coords)
  draw/write  -> annotated output (+ optional side-by-side debug video)

Run (system python3; container has all deps):
  python3 src/pipeline.py --source /project/data/source.mp4 --debug
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from bytetrack_shim import detections_from_boxes, make_tracker
from motion_detector import CompensatedMOG2Detector
from registration import GlobalMotionEstimator
from scene_cut import SceneCutDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compensated MOG2 + ByteTrack.")
    p.add_argument("--source", default="/project/data/source.mp4")
    p.add_argument("--output", default="output/tracked.mp4")
    p.add_argument("--debug", action="store_true",
                   help="Also write output/debug.mp4: [orig+boxes | fg mask | compensated].")
    p.add_argument("--debug-output", default="output/debug.mp4")
    p.add_argument("--max-frames", type=int, default=None)
    # tuning knobs
    p.add_argument("--history", type=int, default=120)
    p.add_argument("--var-threshold", type=float, default=24.0)
    p.add_argument("--min-area-px", type=float, default=5.0,
                   help="Min blob area in pixels (target is a small aerial blob).")
    p.add_argument("--validity-erode-px", type=int, default=25,
                   help="Inward trim of the warped-frame border to kill seam leakage.")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--no-scene-cut", action="store_true",
                   help="Disable scene-cut resets (process the clip as one segment).")
    return p.parse_args()


_PALETTE = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 255), (0, 128, 255), (255, 255, 0)]


def color_for(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


def draw_tracks(frame, tracks):
    """tracks: array rows [x1,y1,x2,y2,id,score,cls,idx]."""
    for row in tracks:
        x1, y1, x2, y2 = (int(v) for v in row[:4])
        tid = int(row[4])
        c = color_for(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, f"id{tid}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return frame


def make_debug_panel(annotated, dbg, size):
    """Stack [annotated | fg mask | compensated] resized to frame height."""
    h, w = size[1], size[0]

    def fit(img):
        img = cv2.resize(img, (w, h))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    return np.hstack([annotated, fit(dbg.fgmask), fit(dbg.compensated)])


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(w * h)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    dbg_writer = (cv2.VideoWriter(args.debug_output, fourcc, fps, (w * 3, h))
                  if args.debug else None)

    detector = CompensatedMOG2Detector(
        w, h, history=args.history, var_threshold=args.var_threshold,
        min_area_px=args.min_area_px, validity_erode_px=args.validity_erode_px,
        estimator=GlobalMotionEstimator(min_inliers=args.min_inliers),
    )
    scene_cut = None if args.no_scene_cut else SceneCutDetector()
    tracker = make_tracker(fps)

    frame_idx, latencies, n_cuts, n_reanchor = 0, [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if scene_cut is not None:
            ev = scene_cut.update(frame, frame_idx)
            if ev is not None:
                n_cuts += 1
                print(f"[f{frame_idx}] scene cut: {ev.kind} (corr={ev.hist_corr:.2f}) -> reset")
                detector.reanchor(gray)
                tracker = make_tracker(fps)   # drop identities across the cut

        t0 = time.perf_counter()
        boxes, dbg = detector.process(frame, gray)
        dets = detections_from_boxes(boxes, frame_area)
        tracks = tracker.update(dets)
        latencies.append((time.perf_counter() - t0) * 1000)
        if dbg.reanchored:
            n_reanchor += 1

        annotated = draw_tracks(frame.copy(), tracks)
        cv2.putText(annotated, f"f{frame_idx} det:{len(boxes)} trk:{len(tracks)}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        writer.write(annotated)
        if dbg_writer is not None:
            dbg_writer.write(make_debug_panel(annotated, dbg, (w, h)))
        frame_idx += 1

    cap.release()
    writer.release()
    if dbg_writer is not None:
        dbg_writer.release()

    if latencies:
        avg = float(np.mean(latencies))
        print(f"\nFrames: {frame_idx} | mean {avg:.1f} ms/frame ({1000/avg:.1f} FPS)")
    print(f"Scene cuts: {n_cuts} | re-anchors: {n_reanchor}")
    print(f"Output: {args.output}" + (f" | Debug: {args.debug_output}" if dbg_writer else ""))


if __name__ == "__main__":
    main()
