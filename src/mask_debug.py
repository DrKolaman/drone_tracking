"""Diagnostic: visualise the MOG2 motion mask on the stabilized video.

Helps answer "why isn't the moving target detected every frame?". Per frame, four
panels in the stabilized (stitched) coordinate:

  [ raw | stabilized aligned | MOG2 raw fg (validity-masked) | post-morph + boxes ]

Comparing panel 3 (what MOG2 fires on) vs panel 4 (what survives morphology +
size/aspect filtering) shows whether the signal is missing, too weak, or filtered
out. HUD reports raw-fg pixel count, #blobs, and #blobs passing the filter.

  python3 src/mask_debug.py --max-frames 647
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from track_stabilized import chain
from track_dino_reid import fit_canvas


def parse_args():
    p = argparse.ArgumentParser(description="MOG2 motion-mask debug (stabilized).")
    p.add_argument("--source", default="/project/data/source.mp4")
    p.add_argument("--output", default="output/mask_debug.mp4")
    p.add_argument("--max-frames", type=int, default=647)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--var-threshold", type=float, default=50.0)
    p.add_argument("--min-area-px", type=float, default=5.0)
    p.add_argument("--max-area-frac", type=float, default=0.05)
    p.add_argument("--validity-erode-px", type=int, default=25)
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--panel-h", type=int, default=480)
    return p.parse_args()


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    Hs = chain(a.source, a.max_frames, a.min_inliers)
    T, cw, ch, corners = fit_canvas(Hs, w, h)

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=a.history, varThreshold=a.var_threshold, detectShadows=False)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    e = max(1, a.validity_erode_px)
    kval = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
    min_area, max_area = a.min_area_px, a.max_area_frac * w * h

    ph = a.panel_h
    rw = int(w * ph / h)
    mw = int(cw * ph / ch)

    def fit_gray(m):
        return cv2.cvtColor(cv2.resize(m, (mw, ph)), cv2.COLOR_GRAY2BGR)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (rw + 3 * mw, ph))

    cap = cv2.VideoCapture(a.source)
    for i, H in enumerate(Hs):
        ok, f = cap.read()
        if not ok:
            break
        Hc = T @ H
        aligned = cv2.warpPerspective(f, Hc, (cw, ch))
        validity = cv2.erode(cv2.warpPerspective(
            np.full((h, w), 255, np.uint8), Hc, (cw, ch)), kval)

        raw_fg = cv2.threshold(mog2.apply(aligned), 200, 255, cv2.THRESH_BINARY)[1]
        raw_fg = cv2.bitwise_and(raw_fg, validity)
        n_raw = int((raw_fg > 0).sum())

        m = cv2.morphologyEx(raw_fg, cv2.MORPH_OPEN, k3, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3, iterations=2)
        m = cv2.dilate(m, k3, iterations=1)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        post = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        n_blob = n - 1
        n_pass = 0
        for c in range(1, n):
            x, y, bw, bh, ar = stats[c]
            ok_blob = (min_area <= ar <= max_area) and (0.25 <= bh / max(bw, 1) <= 4.0)
            col = (0, 0, 255) if ok_blob else (0, 140, 255)
            cv2.rectangle(post, (x, y), (x + bw, y + bh), col, 1)
            n_pass += int(ok_blob)

        raw = cv2.resize(f, (rw, ph))
        cv2.putText(raw, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        p3 = fit_gray(raw_fg)
        cv2.putText(p3, f"MOG2 raw fg px={n_raw}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        p4 = cv2.resize(post, (mw, ph))
        cv2.putText(p4, f"blobs={n_blob} pass={n_pass}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        panel = np.hstack([raw, cv2.resize(aligned, (mw, ph)), p3, p4])
        writer.write(panel)
    cap.release()
    writer.release()
    print(f"Output: {a.output}  ({rw + 3 * mw}x{ph})  panels: raw | stabilized | MOG2 fg | post+boxes")


if __name__ == "__main__":
    main()
