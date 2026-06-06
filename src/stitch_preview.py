"""Visualise ONLY the homography registration / stitching — no MOG2, no tracking.

Builds the panorama mosaic: every frame is warped into a common coordinate via
the cumulative frame-to-frame homography and composited, so you watch the scene
stitch together as the camera moves.

The chain re-anchors (starts a fresh mosaic segment) whenever registration
reports a discontinuity it cannot bridge — a hard zoom (e.g. the ~7x jump at
frame 647), a scene cut, or panning off the canvas. Without this, one bad link
(mis-read as identity) corrupts the whole mosaic from that point on.

Output panels: [ raw frame | current mosaic segment, current frame outlined ].

  python3 src/stitch_preview.py --source /project/data/source.mp4
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
    p = argparse.ArgumentParser(description="Stitching-only preview (mosaic) with re-anchor.")
    p.add_argument("--source", default="/project/data/source.mp4")
    p.add_argument("--output", default="output/stitch_only.mp4")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--margin-frac", type=float, default=0.6,
                   help="Canvas padding each side (fraction of frame size).")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--panel-h", type=int, default=640)
    p.add_argument("--no-reanchor", action="store_true",
                   help="Disable all re-anchoring: pure cumulative chaining into one "
                        "fixed canvas (content that pans out is clipped).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ox, oy = int(args.margin_frac * w), int(args.margin_frac * h)
    cw, ch = w + 2 * ox, h + 2 * oy
    offset = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    est = GlobalMotionEstimator(min_inliers=args.min_inliers)

    ph = args.panel_h
    raw_w = int(w * ph / h)
    mos_w = int(cw * ph / ch)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (raw_w + mos_w, ph))

    quad0 = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def overflow(H):
        c = cv2.perspectiveTransform(quad0, H).reshape(-1, 2)
        return (c[:, 0].min() < 0 or c[:, 0].max() > cw
                or c[:, 1].min() < 0 or c[:, 1].max() > ch)

    mosaic = np.zeros((ch, cw, 3), np.uint8)
    H_cum = offset.copy()
    prev = None
    seg = 0
    flash = 0  # frames to keep the RE-ANCHOR banner up
    idx = 0
    n_reanchor = 0

    while True:
        ok, f = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        reason = ""
        if prev is None:
            reason = "init"
        elif args.no_reanchor:
            # Pure chaining: always accept the estimate (identity on failure),
            # never reset. Lets the raw drift / corruption show.
            H_cum = H_cum @ est.estimate(prev, g).H
        else:
            res = est.estimate(prev, g)
            cand = H_cum @ res.H
            if not res.ok:
                reason = f"discontinuity (inliers={res.n_inliers})"
            elif overflow(cand):
                reason = "canvas overflow"
            else:
                H_cum = cand

        if reason and prev is not None and not args.no_reanchor:
            seg += 1
            n_reanchor += 1
            flash = 12
            mosaic[:] = 0
            H_cum = offset.copy()
            print(f"[f{idx}] re-anchor: {reason} -> new mosaic segment #{seg}")

        warped = cv2.warpPerspective(f, H_cum, (cw, ch))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H_cum, (cw, ch)) > 0
        mosaic[mask] = warped[mask]

        view = mosaic.copy()
        quad = cv2.perspectiveTransform(quad0, H_cum).astype(np.int32)
        cv2.polylines(view, [quad], True, (0, 255, 0), 3)
        if flash > 0:
            cv2.putText(view, "RE-ANCHOR", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            flash -= 1

        raw = cv2.resize(f, (raw_w, ph))
        cv2.putText(raw, f"f{idx} seg{seg}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(np.hstack([raw, cv2.resize(view, (mos_w, ph))]))

        prev = g
        idx += 1

    cap.release()
    writer.release()
    print(f"\nFrames: {idx} | mosaic segments: {seg + 1} | re-anchors: {n_reanchor}")
    print(f"Output: {args.output}  ({raw_w + mos_w}x{ph})")


if __name__ == "__main__":
    main()
