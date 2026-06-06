"""Sliding-window match debug: where in the wide frame does the narrow frame live?

The narrow (zoomed-in) frame 647 corresponds to some sub-window of the wide
(zoomed-out) frame 646. We slide a window across 646 at a 10%-of-image stride;
for each position we crop the window, upscale it to 647's size (removing the zoom
so scales match), DINOv3-match it against 647, and score the match. The window
that scores highest is the footprint = where the camera zoomed in.

Each window becomes one video frame: [646 with the current window box | 647],
match lines drawn, score in the title. Watch the score spike at the right window.

    python3 src/debug_sliding.py --win 0.4 --stride 0.1
"""
from __future__ import annotations

import cv2
import numpy as np

from dinov3_match import DenseMatcher
from test_scale_warp import grab


def score_window(win_up, narrow, dm):
    """DINOv3 match window->narrow; return (pa_in_winup, pb, n_matches, n_inliers)."""
    pa, pb = dm.match(win_up, narrow, sim_floor=0.5)
    inl = 0
    if len(pa) >= 4:
        _, m = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        inl = int(m.sum()) if m is not None else 0
    return pa, pb, len(pa), inl


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--wide", type=int, default=646)
    ap.add_argument("--narrow", type=int, default=647)
    ap.add_argument("--win", type=float, default=0.4, help="window size as frac of image")
    ap.add_argument("--stride", type=float, default=0.1, help="stride as frac of image")
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--out", default="out/sliding_match.mp4")
    args = ap.parse_args()

    dm = DenseMatcher(longside=1024)
    wide = grab(args.video, args.wide)
    narrow = grab(args.video, args.narrow)
    H, W = wide.shape[:2]
    nH, nW = narrow.shape[:2]
    wf, hf = int(W * args.win), int(H * args.win)
    sx, sy = max(1, int(W * args.stride)), max(1, int(H * args.stride))

    xs = list(range(0, W - wf + 1, sx))
    ys = list(range(0, H - hf + 1, sy))
    print(f"window {wf}x{hf}, stride {sx}x{sy} -> {len(xs)*len(ys)} positions")

    # pre-pass: collect scores to set a colour scale and find the best window
    results = []
    for y0 in ys:
        for x0 in xs:
            win = wide[y0:y0 + hf, x0:x0 + wf]
            win_up = cv2.resize(win, (nW, nH), interpolation=cv2.INTER_CUBIC)
            pa, pb, n, inl = score_window(win_up, narrow, dm)
            results.append((x0, y0, win_up, pa, pb, n, inl))
            print(f"  win @({x0:3d},{y0:3d})  matches={n:3d} inliers={inl:3d}")
    max_inl = max((r[6] for r in results), default=1) or 1
    best = max(results, key=lambda r: r[6])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (W + nW, H))
    sxw, syw = nW / wf, nH / hf      # win_up px -> map back to window px
    for (x0, y0, win_up, pa, pb, n, inl) in results:
        canvas = np.hstack([wide.copy(), narrow.copy()])
        is_best = (x0, y0) == (best[0], best[1])
        heat = inl / max_inl
        box_col = (0, int(255 * heat), int(255 * (1 - heat)))   # blue->green by score
        cv2.rectangle(canvas, (x0, y0), (x0 + wf, y0 + hf), box_col, 3 if is_best else 2)
        # draw match lines: window_up pt -> back to window coords; narrow pt offset by W
        idx = np.linspace(0, len(pa) - 1, min(60, len(pa))).astype(int) if len(pa) else []
        for i in idx:
            px = int(x0 + pa[i][0] / sxw)
            py = int(y0 + pa[i][1] / syw)
            qx = int(pb[i][0]) + W
            qy = int(pb[i][1])
            cv2.line(canvas, (px, py), (qx, qy), (0, 200, 0), 1)
            cv2.circle(canvas, (px, py), 2, (0, 200, 0), -1)
            cv2.circle(canvas, (qx, qy), 2, (0, 200, 0), -1)
        cv2.rectangle(canvas, (0, 0), (W + nW, 24), (0, 0, 0), -1)
        tag = f"window @({x0},{y0})  matches={n} inliers={inl}" + ("   <-- BEST" if is_best else "")
        cv2.putText(canvas, tag, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for _ in range(2 if not is_best else 6):     # linger on the best frame
            vw.write(canvas)
    vw.release()
    print(f"\nBEST window @({best[0]},{best[1]}) inliers={best[6]}  -> footprint center "
          f"({best[0]+wf//2},{best[1]+hf//2})")
    print(f"video -> {args.out}")


if __name__ == "__main__":
    main()
