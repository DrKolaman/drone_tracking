"""Loop closure: relink the 957 jump-back into Segment 1 instead of a new segment.

Per docs/REQUIREMENTS.md, at frame 957 the camera "jumps back to the earlier area"
and the colour switches B/W -> red (thermal). So unlike the 745 jump (genuinely new
area), 957 is a REVISIT of Segment 1's ground. We:

  1. Detect the revisit: embed Segment-1 keyframes and frame 957 with DINOv3 and find
     the best match. Because 957 is red thermal and Segment 1 is B/W, we embed and
     match on GRAYSCALE structure (cross-modal).
  2. Relink: register 957 to its best-matching keyframe (DINOv3 dense + similarity),
     compose with that keyframe's map transform to place 957 in Segment 1's canvas,
     then chain the 957-1029 frames from there and composite into the SAME map.

Contrast with map_segments.py: a true jump -> new segment placed away; a revisit ->
relinked into the existing map (loop closed).

    HF_TOKEN=... python3 src/loop_closure.py
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from build_map import chain, fit_canvas, _feather_weight, build_mosaic
from registration import GlobalMotionEstimator
from scene_analysis import Dinov3Embedder
from dinov3_match import DenseMatcher
from colorfix import to_bw


def read_range(video, lo, hi):
    """Read frames [lo, hi] and normalise each to B/W (colour-mode filter)."""
    cap = cv2.VideoCapture(video)
    out, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > hi:
            break
        if idx >= lo:
            out.append(to_bw(f))
        idx += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--seg1-hi", type=int, default=646)     # Segment 1 wide ground
    ap.add_argument("--revisit-lo", type=int, default=957)  # jump-back start (red)
    ap.add_argument("--revisit-hi", type=int, default=1029)
    ap.add_argument("--kf-stride", type=int, default=20)
    ap.add_argument("--out", default="out/loop_closure.png")
    args = ap.parse_args()

    seg1 = read_range(args.video, 0, args.seg1_hi)
    rev = read_range(args.video, args.revisit_lo, args.revisit_hi)
    dm = DenseMatcher(longside=1024)
    emb = Dinov3Embedder()

    # Segment 1 map transforms
    Hs1, kept1 = chain(seg1, 25)
    H1 = {kept1[j]: Hs1[j] for j in range(len(kept1))}

    # keyframes + grayscale embeddings
    kf_idx = kept1[::args.kf_stride]
    kf_emb = emb.embed([seg1[k] for k in kf_idx])
    e957 = emb.embed([rev[0]])[0]
    sims = kf_emb @ e957
    best = int(sims.argmax())
    kf = kf_idx[best]
    print(f"loop-closure detect: frame {args.revisit_lo} best matches Segment-1 "
          f"keyframe {kf}  cos={float(sims[best]):.3f}  (max over {len(kf_idx)} keyframes)")
    print(f"  cos range over keyframes: {float(sims.min()):.3f}..{float(sims.max()):.3f}")

    # register 957 -> keyframe (grayscale, cross-modal), then -> Segment-1 anchor
    pa, pb = dm.match(rev[0], seg1[kf], sim_floor=0.5)
    L = None
    if len(pa) >= 3:
        M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is not None:
            L = np.vstack([M, [0, 0, 1]])
            print(f"  register 957->kf{kf}: {int(inl.sum())}/{len(pa)} inliers")
    if L is None:
        L = np.eye(3)
        print(f"  register 957->kf{kf}: too few matches -> placing at keyframe location")
    A957 = H1[kf] @ L                       # 957 -> Segment-1 anchor

    # chain the revisit segment and bring it into Segment-1 coords
    Hr, keptr = chain(rev, 25)
    Ar = [A957 @ R for R in Hr]
    print(f"revisit stitched {len(keptr)} frames ({args.revisit_lo}-{args.revisit_lo+keptr[-1]})")

    # composite Segment 1 + relinked revisit into one canvas
    pairs = [(seg1[k], H1[k]) for k in kept1] + [(rev[k], A) for k, A in zip(keptr, Ar)]
    h, w = seg1[0].shape[:2]
    all_H = [H for _, H in pairs]
    T, cw, ch = fit_canvas(all_H, w, h)
    weight = _feather_weight(h, w)
    acc = np.zeros((ch, cw, 3), np.float32)
    wsum = np.zeros((ch, cw), np.float32)
    for f, Hm in pairs:
        Hc = T @ Hm
        warp = cv2.warpPerspective(f, Hc, (cw, ch)).astype(np.float32)
        wt = cv2.warpPerspective(weight, Hc, (cw, ch))
        acc += warp * wt[..., None]
        wsum += wt
    mosaic = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)

    # outline where the revisit landed (frame 957 footprint) in red
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    foot = cv2.perspectiveTransform(quad, T @ A957).astype(np.int32)
    cv2.polylines(mosaic, [foot], True, (0, 0, 255), 2)
    cv2.putText(mosaic, "957 revisit relinked", (8, ch - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, mosaic)
    print(f"loop-closed map ({cw}x{ch}) -> {args.out}  (red = relinked 957 footprint)")


if __name__ == "__main__":
    main()
