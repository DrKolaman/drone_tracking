"""Final stitched map image for the entire clip (still, not video).

Same pipeline as full_map_video.py but composites every placed frame in one pass
and saves the final mosaic: Segment 1 (wide + 647 zoom link) on the left, Segment 2
(745 jump) to the right, the 957 revisit loop-closed into Segment 1. Colour-filtered
to B/W, blurred frames skipped.

    HF_TOKEN=... python3 src/full_map.py
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from build_map import chain, fit_canvas, _feather_weight
from map_with_zoom import link_via_sliding
from scene_analysis import Dinov3Embedder
from dinov3_match import DenseMatcher
from colorfix import to_bw

TRANS = lambda dx, dy: np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--hi", type=int, default=1193)
    ap.add_argument("--out", default="out/full_map.png")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > args.hi:
            break
        frames.append(to_bw(f))
        idx += 1
    h, w = frames[0].shape[:2]
    dm = DenseMatcher(longside=1024)
    emb = Dinov3Embedder()

    Hw, keptw = chain(frames[:647], 25)
    hw_map = {keptw[j]: Hw[j] for j in range(len(keptw))}
    L, inl, _ = link_via_sliding(frames[keptw[-1]], frames[647], dm)
    A647 = Hw[-1] @ (L if L is not None else np.eye(3))
    Hz, keptz = chain(frames[647:745], 25)
    seg1 = {k: hw_map[k] for k in keptw}
    for j, k in enumerate(keptz):
        seg1[647 + k] = A647 @ Hz[j]

    Hs2, kept2 = chain(frames[745:957], 25)
    seg2 = {745 + k: Hs2[j] for j, k in enumerate(kept2)}

    kf_idx = keptw[::20]
    sims = emb.embed([frames[k] for k in kf_idx]) @ emb.embed([frames[957]])[0]
    kf = kf_idx[int(sims.argmax())]
    pa, pb = dm.match(frames[957], frames[kf], sim_floor=0.5)
    L957 = np.eye(3)
    if len(pa) >= 3:
        M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is not None:
            L957 = np.vstack([M, [0, 0, 1]])
    A957 = hw_map[kf] @ L957
    Hr, keptr = chain(frames[957:], 25)
    revisit = {957 + k: A957 @ Hr[j] for j, k in enumerate(keptr)}

    T1, cw1, ch1 = fit_canvas(list(seg1.values()) + list(revisit.values()), w, h)
    T2, cw2, ch2 = fit_canvas(list(seg2.values()), w, h)
    gap = 60
    GW, GH = cw1 + gap + cw2, max(ch1, ch2)
    Toff = TRANS(cw1 + gap, 0) @ T2

    place = []
    for k, H in {**seg1, **revisit}.items():
        place.append((frames[k], T1 @ H))
    for k, H in seg2.items():
        place.append((frames[k], Toff @ H))

    weight = _feather_weight(h, w)
    acc = np.zeros((GH, GW, 3), np.float32)
    wsum = np.zeros((GH, GW), np.float32)
    for f, Hg in place:
        warp = cv2.warpPerspective(f, Hg, (GW, GH)).astype(np.float32)
        wt = cv2.warpPerspective(weight, Hg, (GW, GH))
        acc += warp * wt[..., None]
        wsum += wt
    mosaic = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)

    cv2.putText(mosaic, f"Segment 1 (0-744, +zoom +957 revisit)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mosaic, f"Segment 2 (745-956 jump)", (cw1 + gap + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(args.out, mosaic)
    print(f"final map ({GW}x{GH}) -> {args.out}  | seg1={len(seg1)} revisit={len(revisit)} "
          f"seg2={len(seg2)} frames, loop-closure cos={float(sims.max()):.3f}")


if __name__ == "__main__":
    main()
