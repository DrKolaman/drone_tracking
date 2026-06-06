"""Progressive map-build video for the ENTIRE clip (0-1193).

Integrates every discontinuity handler onto one global canvas:
  * colour filter (colorfix.to_bw) so 957/1030 B/W<->red is one modality.
  * Segment 1 (0-744): wide map + 647 zoom linked via DINOv3 sliding-window.
  * Segment 2 (745-956): a JUMP -> new segment laid out to the RIGHT (no overlap).
  * Revisit (957-1193): a jump-BACK -> LOOP CLOSURE, relinked into Segment 1.
  * motion-blurred frames are skipped.

Each frame is composited as it arrives; banners flash at zoom-link / jump / loop-closure.

    HF_TOKEN=... python3 src/full_map_video.py
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
    ap.add_argument("--out", default="out/full_map_build.mp4")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > args.hi:
            break
        frames.append(to_bw(f))          # colour-mode filter
        idx += 1
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"read {n} frames (colour-normalised)")

    dm = DenseMatcher(longside=1024)
    emb = Dinov3Embedder()

    # ---- Segment 1: wide 0-646 + zoom 647-744 linked ----
    Hw, keptw = chain(frames[:647], 25)
    hw_map = {keptw[j]: Hw[j] for j in range(len(keptw))}
    L, inl, _ = link_via_sliding(frames[keptw[-1]], frames[647], dm)
    A647 = Hw[-1] @ (L if L is not None else np.eye(3))
    Hz, keptz = chain(frames[647:745], 25)
    seg1 = {k: hw_map[k] for k in keptw}
    for j, k in enumerate(keptz):
        seg1[647 + k] = A647 @ Hz[j]
    print(f"segment 1: {len(seg1)} frames (zoom link inliers={inl})")

    # ---- Segment 2: jump 745-956 placed away ----
    Hs2, kept2 = chain(frames[745:957], 25)
    seg2 = {745 + k: Hs2[j] for j, k in enumerate(kept2)}
    print(f"segment 2: {len(seg2)} frames")

    # ---- Revisit 957-1193: loop-closure into Segment 1 ----
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
    print(f"revisit: {len(revisit)} frames, loop-closed to keyframe {kf} (cos={float(sims.max()):.3f})")

    # ---- global canvas: seg1+revisit on the left, seg2 to the right ----
    left_H = list(seg1.values()) + list(revisit.values())
    T1, cw1, ch1 = fit_canvas(left_H, w, h)
    T2, cw2, ch2 = fit_canvas(list(seg2.values()), w, h)
    gap = 60
    GW, GH = cw1 + gap + cw2, max(ch1, ch2)
    Toff = TRANS(cw1 + gap, 0) @ T2

    place = {}                                    # global_idx -> (H_global, group)
    for k, H in seg1.items():
        place[k] = (T1 @ H, "wide" if k <= 646 else "zoom")
    for k, H in revisit.items():
        place[k] = (T1 @ H, "revisit")
    for k, H in seg2.items():
        place[k] = (Toff @ H, "seg2")

    events = {647: "ZOOM LINK (DINOv3 sliding-window)",
              745: "JUMP -> new segment",
              957: "LOOP CLOSURE -> relink to Segment 1"}
    colors = {"wide": (0, 230, 0), "zoom": (255, 220, 0),
              "seg2": (0, 215, 255), "revisit": (255, 0, 255)}

    OH = 720
    OW = int(GW * OH / GH)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OW, OH))
    acc = np.zeros((GH, GW, 3), np.float32)
    wsum = np.zeros((GH, GW), np.float32)
    weight = _feather_weight(h, w)
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    flash, flash_n, fcol = "", 0, (255, 255, 255)

    for gi in range(n):
        outline = col = None
        if gi in events:
            flash, flash_n = events[gi], 45
            fcol = (0, 0, 255) if "JUMP" in events[gi] else (0, 255, 255)
        if gi in place:
            Hg, grp = place[gi]
            warp = cv2.warpPerspective(frames[gi], Hg, (GW, GH)).astype(np.float32)
            wt = cv2.warpPerspective(weight, Hg, (GW, GH))
            acc += warp * wt[..., None]
            wsum += wt
            outline = cv2.perspectiveTransform(quad, Hg).astype(np.int32)
            col = colors[grp]
        else:
            if flash_n <= 0:
                flash, flash_n, fcol = "SKIP (blur)", 5, (0, 0, 255)

        render = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)
        if outline is not None:
            cv2.polylines(render, [outline], True, col, 3)
        view = cv2.resize(render, (OW, OH))
        cv2.rectangle(view, (0, 0), (OW, 26), (0, 0, 0), -1)
        cv2.putText(view, f"frame {gi}/{n-1}", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if flash_n > 0:
            cv2.putText(view, flash, (OW // 2 - 250, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, fcol, 2, cv2.LINE_AA)
            flash_n -= 1
        vw.write(view)
    vw.release()
    print(f"full map-build video ({OW}x{OH}, {n} frames) -> {args.out}")


if __name__ == "__main__":
    main()
