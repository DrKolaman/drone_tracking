"""Map that links the 647 FOV-switch ("zoom") segment into the wide map.

`build_map.py` stops at 647 because frame-to-frame homography can't bridge the
~3x FOV switch. Here we bridge it with the DINOv3 sliding-window match (the method
that *did* find the correspondence): slide a window over the last wide frame (646),
upscale each window to the narrow frame's size, DINOv3-match against 647, take the
best window, and fit a similarity 647->646 from its correspondences. That similarity
anchors the zoomed segment into the wide map's coordinate frame; the zoomed frames
(647->744) then chain internally (LK) and are composited into the same canvas.

Result: the wide map of 0-646 with the higher-detail zoomed swath placed at its true
location/scale (middle-right). The link's scale ~= 1/zoom is an independent confirm
of the ~3x switch.

    HF_TOKEN=... python3 src/map_with_zoom.py
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from registration import GlobalMotionEstimator
from build_map import fit_canvas, _feather_weight, chain
from dinov3_match import DenseMatcher


def link_via_sliding(wide, narrow, dm, win=0.4, stride=0.1):
    """Similarity 647->646 from the best DINOv3 sliding-window match.

    Slides a `win`-fraction window over the wide frame at `stride`, upscales it to
    the narrow size, DINOv3-matches against the narrow frame, and keeps the window
    with the most homography inliers. Returns (L 3x3 narrow->wide, n_inliers, box).
    """
    H, W = wide.shape[:2]
    nH, nW = narrow.shape[:2]
    wf, hf = int(W * win), int(H * win)
    sx, sy = max(1, int(W * stride)), max(1, int(H * stride))
    best = (0, None, None, None)
    for y0 in range(0, H - hf + 1, sy):
        for x0 in range(0, W - wf + 1, sx):
            win_up = cv2.resize(wide[y0:y0 + hf, x0:x0 + wf], (nW, nH),
                                interpolation=cv2.INTER_CUBIC)
            pa, pb = dm.match(win_up, narrow, sim_floor=0.5)   # pa: win_up, pb: narrow
            if len(pa) < 4:
                continue
            _, m = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            inl = int(m.sum()) if m is not None else 0
            if inl > best[0]:
                pa_w = np.column_stack([x0 + pa[:, 0] * wf / nW, y0 + pa[:, 1] * hf / nH])
                best = (inl, pa_w.astype(np.float32), pb.astype(np.float32), (x0, y0, wf, hf))
    inl, pa_w, pb, box = best
    if inl < 4:
        return None, inl, box
    L2, _ = cv2.estimateAffinePartial2D(pb, pa_w, method=cv2.RANSAC,
                                        ransacReprojThreshold=4.0)   # narrow->wide
    if L2 is None:
        return None, inl, box
    return np.vstack([L2, [0, 0, 1]]), inl, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--wide-hi", type=int, default=646)   # last wide frame
    ap.add_argument("--zoom-hi", type=int, default=744)   # last zoomed frame (745 jumps)
    ap.add_argument("--out", default="out")
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
    wide = frames[:args.wide_hi + 1]            # 0..646
    zoom = frames[args.wide_hi + 1:args.zoom_hi + 1]   # 647..744
    print(f"wide {len(wide)} frames, zoom {len(zoom)} frames")

    # 1. wide segment -> homographies to frame 0 (blurred frames skipped)
    Hw, keptw = chain(wide, min_inliers=25)
    last_wide = wide[keptw[-1]]               # last SHARP wide frame (the link anchor)
    print(f"wide stitched {len(keptw)} frames, link-anchor frame {keptw[-1]}")

    # 2. link 647 -> last sharp wide frame via DINOv3 sliding-window, then -> anchor
    dm = DenseMatcher(longside=1024)
    L, inl, box = link_via_sliding(last_wide, zoom[0], dm)
    if L is None:
        raise SystemExit("link failed: no usable sliding-window match at 647")
    scale = float(np.sqrt(L[0, 0] ** 2 + L[0, 1] ** 2))
    print(f"link 647->{keptw[-1]}: {inl} inliers, scale={scale:.2f} (=> zoom ~{1/scale:.1f}x), box={box}")
    A647 = Hw[-1] @ L                            # 647 -> anchor(frame 0)

    # 3. zoom segment -> homographies to 647, then -> anchor (blurred frames skipped)
    Hz, keptz = chain(zoom, min_inliers=25)
    print(f"zoom stitched {len(keptz)} frames")
    Az = [A647 @ Z for Z in Hz]

    # 4. composite everything into one feather-blended canvas
    comp = [(wide[k], H) for k, H in zip(keptw, Hw)] + \
           [(zoom[k], H) for k, H in zip(keptz, Az)]
    h, w = comp[0][0].shape[:2]
    all_H = [H for _, H in comp]
    T, cw, ch = fit_canvas(all_H, w, h)
    weight = _feather_weight(h, w)
    acc = np.zeros((ch, cw, 3), np.float32)
    wsum = np.zeros((ch, cw), np.float32)
    for f, Hm in comp:
        Hc = T @ Hm
        warp = cv2.warpPerspective(f, Hc, (cw, ch)).astype(np.float32)
        wt = cv2.warpPerspective(weight, Hc, (cw, ch))
        acc += warp * wt[..., None]
        wsum += wt
    mosaic = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)

    # outline the zoomed swath's footprint on the map
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    foot = cv2.perspectiveTransform(quad, T @ A647).astype(np.int32)
    cv2.polylines(mosaic, [foot], True, (0, 0, 255), 2)

    out = f"{args.out}/map_with_zoom.png"
    cv2.imwrite(out, mosaic)
    print(f"map ({cw}x{ch}) -> {out}   (red = frame 647 footprint in the map)")


if __name__ == "__main__":
    main()
