"""Detection + precision tracking pipeline.

Detect a walking person in outdoor footage and keep a stable identity on them
through partial/full occlusion and appearance changes, in real time.

Architecture (see README for the block diagram):

    frames --> [Detector: YOLO] --> boxes --> [Tracker: ByteTrack/BoT-SORT]
                                                   |
                          [SceneCutDetector] ------+--> reset on discontinuity
                                                   |
                                          [Target lock] --> annotated video + log

Design choices, scoped for a finite-time take-home:
  * Detector is an off-the-shelf YOLO (COCO `person`, class 0). It is the single
    deep-learning block. Swappable via --model.
  * Tracking is delegated to Ultralytics' built-in trackers. BoT-SORT (default)
    adds appearance ReID on top of a Kalman motion model, which is what carries
    identity through occlusion and clothing/scale changes; ByteTrack is the
    lighter, motion-only fallback (--tracker bytetrack).
  * Discontinuities (hard cut, thermal->colour flip) are *detected* and used to
    reset the tracker rather than tracked through. Robust cross-cut re-ID is
    explicitly out of scope (see README "What I did not solve").
  * "Target lock": the scenario has one subject of interest. We lock onto the
    most persistent track and highlight it, so the deliverable shows a single
    tracked target rather than every passer-by.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import cv2

from scene_cut import SceneCutDetector

# COCO class id for "person".
PERSON_CLASS = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect + track a person in video.")
    p.add_argument("--source", required=True,
                   help="Path to input video (download the clip first; see README).")
    p.add_argument("--model", default="yolo11n.pt",
                   help="Ultralytics detection weights (auto-downloaded).")
    p.add_argument("--tracker", default="botsort.yaml",
                   choices=["botsort.yaml", "bytetrack.yaml"],
                   help="botsort.yaml = motion+ReID (default); bytetrack.yaml = motion-only.")
    p.add_argument("--conf", type=float, default=0.3,
                   help="Detection confidence threshold.")
    p.add_argument("--output", default="output/tracked.mp4",
                   help="Annotated output video path.")
    p.add_argument("--stop-at-cut", action="store_true",
                   help="Stop processing at the first hard discontinuity "
                        "(the continuous segment is ~first half of the clip).")
    p.add_argument("--device", default=None,
                   help="'cpu', '0' for GPU, etc. Default: Ultralytics auto.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames processed (debugging).")
    return p.parse_args()


def open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)


def draw(frame, boxes, ids, confs, target_id):
    """Overlay boxes; the locked target is drawn thicker and in a hot colour."""
    for box, tid, conf in zip(boxes, ids, confs):
        x1, y1, x2, y2 = (int(v) for v in box)
        is_target = tid == target_id
        color = (0, 0, 255) if is_target else (0, 200, 0)
        thick = 3 if is_target else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        tag = f"TARGET id{tid} {conf:.2f}" if is_target else f"id{tid} {conf:.2f}"
        cv2.putText(frame, tag, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thick)
    return frame


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # imported late so --help is instant

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(args.output, fps, (w, h))

    cuts = SceneCutDetector()
    # how many frames each track id has been seen -> used to lock the target
    seen = collections.Counter()
    frame_idx = 0
    inference_ms: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break

        event = cuts.update(frame, frame_idx)
        if event is not None:
            print(f"[frame {frame_idx}] discontinuity: {event.kind} "
                  f"(hist_corr={event.hist_corr:.2f}, color={event.is_color})")
            if args.stop_at_cut:
                print("Stopping at discontinuity (continuous segment ended).")
                break
            # Reset tracker state so identities don't bleed across the cut.
            model.predictor.trackers[0].reset() if getattr(
                model, "predictor", None) and getattr(
                model.predictor, "trackers", None) else None
            seen.clear()

        t0 = time.perf_counter()
        results = model.track(
            frame, persist=True, tracker=args.tracker, conf=args.conf,
            classes=[PERSON_CLASS], device=args.device, verbose=False,
        )[0]
        inference_ms.append((time.perf_counter() - t0) * 1000)

        boxes, ids, confs = [], [], []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            for tid in ids:
                seen[tid] += 1

        # Lock onto the most persistent identity = the subject we follow.
        target_id = seen.most_common(1)[0][0] if seen else None

        writer.write(draw(frame, boxes, ids, confs, target_id))
        frame_idx += 1

    cap.release()
    writer.release()

    if inference_ms:
        avg = sum(inference_ms) / len(inference_ms)
        print(f"\nProcessed {frame_idx} frames.")
        print(f"Mean detector+tracker latency: {avg:.1f} ms "
              f"({1000 / avg:.1f} FPS) on {args.device or 'auto'} device.")
    print(f"Annotated video written to: {args.output}")


if __name__ == "__main__":
    main()
