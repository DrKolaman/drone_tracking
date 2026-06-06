"""Render the map being built, frame by frame, as a video.

Shows the actual pipeline working: sharp frames accumulate onto the canvas (green
outline = current wide frame), motion-blurred frames flash "SKIP (blur)" and add
nothing, then at 647 the zoom segment is linked via DINOv3 sliding-window and its
frames drop into the red footprint (cyan outline). Uses the same transforms as
map_with_zoom.py (similarity model + blur-skip + feather blend).

    HF_TOKEN=... python3 src/build_map_video.py
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from build_map import chain, fit_canvas, _feather_weight, blur_scores
from map_with_zoom import link_via_sliding
from dinov3_match import DenseMatcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--wide-hi", type=int, default=646)
    ap.add_argument("--zoom-hi", type=int, default=744)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="out/map_build.mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > args.zoom_hi:
            break
        frames.append(f)
        idx += 1
    cap.release()
    wide = frames[:args.wide_hi + 1]
    zoom = frames[args.wide_hi + 1:args.zoom_hi + 1]

    Hw, keptw = chain(wide, 25)
    dm = DenseMatcher(longside=1024)
    L, inl, box = link_via_sliding(wide[keptw[-1]], zoom[0], dm)
    A647 = Hw[-1] @ L
    Hz, keptz = chain(zoom, 25)
    Az = [A647 @ Z for Z in Hz]

    wide_map = {keptw[j]: Hw[j] for j in range(len(keptw))}
    zoom_map = {args.wide_hi + 1 + keptz[j]: Az[j] for j in range(len(keptz))}
    all_H = list(Hw) + Az
    h, w = frames[0].shape[:2]
    T, cw, ch = fit_canvas(all_H, w, h)
    weight = _feather_weight(h, w)
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    foot647 = cv2.perspectiveTransform(quad, T @ A647).astype(np.int32)

    OH = 720
    OW = int(cw * OH / ch)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OW, OH))
    acc = np.zeros((ch, cw, 3), np.float32)
    wsum = np.zeros((ch, cw), np.float32)
    flash = ""
    flash_n = 0
    last_idx = args.zoom_hi

    for i in range(last_idx + 1):
        outline, col = None, None
        if i in wide_map:
            Hc = T @ wide_map[i]
            acc += cv2.warpPerspective(frames[i], Hc, (cw, ch)).astype(np.float32) * \
                cv2.warpPerspective(weight, Hc, (cw, ch))[..., None]
            wsum += cv2.warpPerspective(weight, Hc, (cw, ch))
            outline, col = cv2.perspectiveTransform(quad, Hc).astype(np.int32), (0, 230, 0)
        elif i in zoom_map:
            if i == args.wide_hi + 1:
                flash, flash_n = "ZOOM LINK via DINOv3 sliding-window", 30
            Hc = T @ zoom_map[i]
            acc += cv2.warpPerspective(frames[i], Hc, (cw, ch)).astype(np.float32) * \
                cv2.warpPerspective(weight, Hc, (cw, ch))[..., None]
            wsum += cv2.warpPerspective(weight, Hc, (cw, ch))
            outline, col = cv2.perspectiveTransform(quad, Hc).astype(np.int32), (255, 220, 0)
        else:
            flash, flash_n = "SKIP (blur)", 6

        render = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)
        if i >= args.wide_hi + 1:
            cv2.polylines(render, [foot647], True, (0, 0, 255), 2)
        if outline is not None:
            cv2.polylines(render, [outline], True, col, 2)
        view = cv2.resize(render, (OW, OH))
        seg = "WIDE" if i <= args.wide_hi else "ZOOM"
        cv2.rectangle(view, (0, 0), (OW, 26), (0, 0, 0), -1)
        cv2.putText(view, f"frame {i}  [{seg}]", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if flash_n > 0:
            c = (0, 0, 255) if flash.startswith("SKIP") else (0, 255, 255)
            cv2.putText(view, flash, (OW // 2 - 180, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
            flash_n -= 1
        vw.write(view)
    vw.release()
    print(f"video ({OW}x{OH}, {last_idx + 1} frames) -> {args.out}")


if __name__ == "__main__":
    main()
