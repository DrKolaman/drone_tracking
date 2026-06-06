"""Slick, jump-free stabilization/stitching preview (homography only).

Unlike stitch_preview's re-anchoring view, this renders ONE smooth mosaic with no
snaps, intended for the continuous segment up to the zoom:

  * Two-pass: pass 1 chains the homographies and sizes the canvas to fit every
    frame exactly, so the frame never pans off and never triggers an overflow
    reset (the main source of "jumps").
  * On a registration failure (e.g. the 610-614 motion-blur burst) it HOLDS the
    last good cumulative transform instead of re-anchoring, so the view doesn't
    snap — the blurred frames just sit briefly until tracking recovers.

Output panels: [ raw frame | stabilized mosaic, current frame outlined ].

  python3 src/stabilize_preview.py --source /project/data/source.mp4 --max-frames 647
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from registration import GlobalMotionEstimator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smooth stabilization mosaic (no jumps).")
    p.add_argument("--source", default="/project/data/source.mp4")
    p.add_argument("--output", default="output/stabilized.mp4")
    p.add_argument("--max-frames", type=int, default=647)
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--panel-h", type=int, default=720)
    return p.parse_args()


def chain(source, max_frames, min_inliers):
    """Pass 1: cumulative homography per frame, HOLDING last good H on failure."""
    cap = cv2.VideoCapture(source)
    est = GlobalMotionEstimator(min_inliers=min_inliers)
    Hs, holds = [], []
    H_cum = np.eye(3)
    prev = None
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        held = False
        if prev is not None:
            res = est.estimate(prev, g)
            if res.ok:
                H_cum = H_cum @ res.H
            else:
                held = True            # hold last transform: no snap through blur
        Hs.append(H_cum.copy())
        holds.append(held)
        prev = g
        idx += 1
    cap.release()
    return Hs, holds


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    Hs, holds = chain(args.source, args.max_frames, args.min_inliers)

    # canvas sized to fit every warped frame exactly (no clip, no overflow)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    xs, ys = [], []
    for H in Hs:
        c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs += [c[:, 0].min(), c[:, 0].max()]
        ys += [c[:, 1].min(), c[:, 1].max()]
    T = np.array([[1, 0, -min(xs)], [0, 1, -min(ys)], [0, 0, 1]], np.float64)
    cw, ch = int(np.ceil(max(xs) - min(xs))), int(np.ceil(max(ys) - min(ys)))
    print(f"mosaic canvas {cw}x{ch} from {len(Hs)} frames; holds(blur)={sum(holds)}")

    ph = args.panel_h
    raw_w = int(w * ph / h)
    mos_w = int(cw * ph / ch)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (raw_w + mos_w, ph))

    mosaic = np.zeros((ch, cw, 3), np.uint8)
    cap = cv2.VideoCapture(args.source)
    for i, (H, held) in enumerate(zip(Hs, holds)):
        ok, f = cap.read()
        if not ok:
            break
        Hc = T @ H
        warped = cv2.warpPerspective(f, Hc, (cw, ch))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hc, (cw, ch)) > 0
        mosaic[mask] = warped[mask]

        view = mosaic.copy()
        quad = cv2.perspectiveTransform(corners, Hc).astype(np.int32)
        cv2.polylines(view, [quad], True, (0, 255, 0), 2)
        raw = cv2.resize(f, (raw_w, ph))
        tag = f"f{i}" + ("  (blur: holding)" if held else "")
        cv2.putText(raw, tag, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(np.hstack([raw, cv2.resize(view, (mos_w, ph))]))
    cap.release()
    writer.release()
    print(f"Output: {args.output}  ({raw_w + mos_w}x{ph})")


if __name__ == "__main__":
    main()
