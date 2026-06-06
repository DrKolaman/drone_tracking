"""Jump-free person tracking on the stabilized (stitched) video.

Combines the smooth stabilization of stabilize_preview with MOG2 + ByteTrack:

  pass 1  chain homographies (hold last H through blur), size a fit canvas
  pass 2  for each frame:
            warp into the fit canvas (stabilized mosaic coords)
            MOG2 on the stabilized frame  -> moving-person blobs (canvas coords)
            ByteTrack in canvas coords     -> persistent IDs
            draw on [ raw | stabilized mosaic ]

Tracking happens in the *stabilized* frame on purpose: there the person moves
smoothly across a static background while registration-residual noise stays
pinned to terrain, which is exactly what ByteTrack's motion model wants. A single
fit canvas means no re-anchoring and therefore no jumps.

  python3 src/track_stabilized.py --source /project/data/source.mp4 --max-frames 647
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from registration import GlobalMotionEstimator
from bytetrack_shim import detections_from_boxes, make_tracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track person on stabilized video (no jumps).")
    p.add_argument("--source", default="/project/data/source.mp4")
    p.add_argument("--output", default="output/track_stabilized.mp4")
    p.add_argument("--max-frames", type=int, default=647)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--var-threshold", type=float, default=50.0)
    p.add_argument("--min-area-px", type=float, default=5.0)
    p.add_argument("--max-area-frac", type=float, default=0.05)
    p.add_argument("--validity-erode-px", type=int, default=25)
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--panel-h", type=int, default=720)
    return p.parse_args()


_PALETTE = [(0, 200, 0), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (0, 128, 255), (255, 255, 0), (0, 255, 128)]


def chain(source, max_frames, min_inliers):
    cap = cv2.VideoCapture(source)
    est = GlobalMotionEstimator(min_inliers=min_inliers)
    Hs = []
    H_cum = np.eye(3)
    prev = None
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            res = est.estimate(prev, g)
            if res.ok:                       # else hold last H (no snap)
                H_cum = H_cum @ res.H
        Hs.append(H_cum.copy())
        prev = g
        idx += 1
    cap.release()
    return Hs


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    Hs = chain(args.source, args.max_frames, args.min_inliers)

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    xs, ys = [], []
    for H in Hs:
        c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs += [c[:, 0].min(), c[:, 0].max()]
        ys += [c[:, 1].min(), c[:, 1].max()]
    T = np.array([[1, 0, -min(xs)], [0, 1, -min(ys)], [0, 0, 1]], np.float64)
    cw, ch = int(np.ceil(max(xs) - min(xs))), int(np.ceil(max(ys) - min(ys)))

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=args.history, varThreshold=args.var_threshold, detectShadows=False)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    e = max(1, args.validity_erode_px)
    kval = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
    tracker = make_tracker(fps)
    min_area, max_area = args.min_area_px, args.max_area_frac * w * h
    frame_area = float(w * h)
    idlife: Counter = Counter()

    ph = args.panel_h
    raw_w = int(w * ph / h)
    mos_w = int(cw * ph / ch)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (raw_w + mos_w, ph))

    mosaic = np.zeros((ch, cw, 3), np.uint8)
    cap = cv2.VideoCapture(args.source)
    for i, H in enumerate(Hs):
        ok, f = cap.read()
        if not ok:
            break
        Hc = T @ H
        aligned = cv2.warpPerspective(f, Hc, (cw, ch))
        validity = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hc, (cw, ch))
        validity = cv2.erode(validity, kval)
        mask = validity > 0
        mosaic[mask] = aligned[mask]                     # update mosaic for display

        fg = mog2.apply(aligned)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.bitwise_and(fg, validity)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3, iterations=2)
        fg = cv2.dilate(fg, k3, iterations=1)

        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        boxes = []
        for c in range(1, n):
            x, y, bw, bh, ar = stats[c]
            if ar < min_area or ar > max_area:
                continue
            if not (0.25 <= bh / max(bw, 1) <= 4.0):
                continue
            boxes.append([x, y, x + bw, y + bh])

        tracks = tracker.update(detections_from_boxes(boxes, frame_area))
        for r in tracks:
            idlife[int(r[4])] += 1
        target = idlife.most_common(1)[0][0] if idlife else None

        view = mosaic.copy()
        cv2.polylines(view, [cv2.perspectiveTransform(corners, Hc).astype(np.int32)],
                      True, (60, 60, 60), 1)
        raw = cv2.resize(f, (raw_w, ph))
        H_inv = np.linalg.inv(Hc)
        sx, sy = raw_w / w, ph / h
        for r in tracks:
            x1, y1, x2, y2 = r[:4]
            tid = int(r[4])
            is_t = tid == target
            col = (0, 0, 255) if is_t else _PALETTE[tid % len(_PALETTE)]
            th = 3 if is_t else 1
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), col, th)
            lbl = f"TARGET id{tid}" if is_t else f"id{tid}"
            cv2.putText(view, lbl, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, th)
            # map box back to raw frame
            q = cv2.perspectiveTransform(
                np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2),
                H_inv).reshape(-1, 2)
            rx1, ry1 = q[:, 0].min() * sx, q[:, 1].min() * sy
            rx2, ry2 = q[:, 0].max() * sx, q[:, 1].max() * sy
            cv2.rectangle(raw, (int(rx1), int(ry1)), (int(rx2), int(ry2)), col, th)

        cv2.putText(raw, f"f{i} det:{len(boxes)} trk:{len(tracks)}",
                    (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(np.hstack([raw, cv2.resize(view, (mos_w, ph))]))
    cap.release()
    writer.release()
    print(f"canvas {cw}x{ch} | frames {len(Hs)} | distinct IDs {len(idlife)}")
    if idlife:
        top = idlife.most_common(3)
        print("longest-lived IDs:", [(f"id{t}", n) for t, n in top])
    print(f"Output: {args.output}  ({raw_w + mos_w}x{ph})")


if __name__ == "__main__":
    main()
