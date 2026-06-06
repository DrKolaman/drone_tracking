"""Multi-segment map: a jump spawns a NEW segment placed away from the last one.

A jump (e.g. 744->745) lands on a different area with NO spatial overlap, so it
cannot be stitched into the existing map. Detected by: registration collapse +
sharp frames + low DINOv3 global similarity (see build_map_video / the classifier).
We finalise the current segment and start a fresh one, laid out *beside* the
previous segment on a combined canvas (they share no coordinates).

Segment 1: frames 0-744 (wide map + DINOv3-linked 647 zoom).
Segment 2: frames 745-956 (the jumped-to area), stitched independently and placed
to the right. (Segment 2's own chain stops at the next discontinuity, 957.)

    HF_TOKEN=... python3 src/map_segments.py
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from build_map import chain, build_mosaic, blur_scores
from map_with_zoom import link_via_sliding
from registration import GlobalMotionEstimator
from scene_analysis import Dinov3Embedder
from dinov3_match import DenseMatcher


def composite_pairs(pairs):
    """Feather-composite a list of (frame, H_to_anchor) into one mosaic."""
    fr = [p[0] for p in pairs]
    Hs = [p[1] for p in pairs]
    return build_mosaic(fr, Hs, list(range(len(pairs))))


def linked_segment(frames, wide_hi, zoom_hi, dm):
    """Segment 0..zoom_hi: wide map with the zoom sub-segment linked in."""
    wide = frames[:wide_hi + 1]
    zoom = frames[wide_hi + 1:zoom_hi + 1]
    Hw, keptw = chain(wide, 25)
    L, inl, box = link_via_sliding(wide[keptw[-1]], zoom[0], dm)
    A647 = Hw[-1] @ L
    Hz, keptz = chain(zoom, 25)
    Az = [A647 @ Z for Z in Hz]
    pairs = [(wide[k], H) for k, H in zip(keptw, Hw)] + \
            [(zoom[k], H) for k, H in zip(keptz, Az)]
    return composite_pairs(pairs)


def is_jump(prev_bgr, cur_bgr, est, emb):
    """Classify a registration collapse as a JUMP (new area, no overlap)."""
    gp = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    inl = est.estimate(gp, gc).n_inliers
    e = emb.embed([prev_bgr, cur_bgr])
    cos = float(e[0] @ e[1])
    sharp = cv2.Laplacian(gc, cv2.CV_64F).var()
    # jump = registration collapse + sharp frame + low scene similarity
    return inl < 25 and sharp > 20 and cos < 0.75, inl, cos, sharp


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--wide-hi", type=int, default=646)
    ap.add_argument("--zoom-hi", type=int, default=744)
    ap.add_argument("--seg2-hi", type=int, default=956)
    ap.add_argument("--out", default="out/map_segments.png")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > args.seg2_hi:
            break
        frames.append(f)
        idx += 1
    cap.release()

    dm = DenseMatcher(longside=1024)
    emb = Dinov3Embedder()
    est = GlobalMotionEstimator(min_inliers=25)

    # confirm the jump at zoom_hi -> zoom_hi+1
    jump, inl, cos, sharp = is_jump(frames[args.zoom_hi], frames[args.zoom_hi + 1], est, emb)
    print(f"jump {args.zoom_hi}->{args.zoom_hi+1}: inliers={inl} cos={cos:.3f} sharp={sharp:.0f} "
          f"-> {'JUMP (new segment)' if jump else 'not a jump'}")

    seg1 = linked_segment(frames, args.wide_hi, args.zoom_hi, dm)
    print(f"segment 1 (0-{args.zoom_hi}): {seg1.shape[1]}x{seg1.shape[0]}")

    seg2_frames = frames[args.zoom_hi + 1:args.seg2_hi + 1]
    Hs2, kept2 = chain(seg2_frames, 25)
    seg2 = build_mosaic(seg2_frames, Hs2, kept2)
    end2 = args.zoom_hi + 1 + kept2[-1]
    print(f"segment 2 ({args.zoom_hi+1}-{end2}): {seg2.shape[1]}x{seg2.shape[0]} "
          f"({len(kept2)} frames, stopped at next discontinuity)")

    # lay segment 2 AWAY from segment 1 (to the right, with a gap)
    gap = 60
    seg1 = label(seg1.copy(), f"Segment 1: frames 0-{args.zoom_hi} (wide + linked zoom)")
    seg2 = label(seg2.copy(), f"Segment 2: frames {args.zoom_hi+1}-{end2} (jumped area)")
    H = max(seg1.shape[0], seg2.shape[0])
    W = seg1.shape[1] + gap + seg2.shape[1]
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:seg1.shape[0], :seg1.shape[1]] = seg1
    x2 = seg1.shape[1] + gap
    canvas[:seg2.shape[0], x2:x2 + seg2.shape[1]] = seg2
    # arrow marking the jump between the two segments
    y = H // 2
    cv2.arrowedLine(canvas, (seg1.shape[1] + 8, y), (x2 - 8, y), (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(canvas, "JUMP", (seg1.shape[1] + 6, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, canvas)
    print(f"two-segment map ({W}x{H}) -> {args.out}")


if __name__ == "__main__":
    main()
