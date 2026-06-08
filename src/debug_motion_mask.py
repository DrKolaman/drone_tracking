"""Debug video of the motion-detection masks at several MOG2 thresholds.

For each frame it lays out side-by-side panels:
  [ raw frame ] [ stabilized + fg@thr1 ] [ stabilized + fg@thr2 ] ...
The foreground is the MOG2 output AT THAT THRESHOLD, masked by the same `validity`
(coverage-age + erosion) the tracker uses, drawn in red over the dimmed stabilized
frame. Each panel prints the threshold and the foreground pixel count, so you can
see when the DRONE's motion floods the whole frame (registration / freshly-revealed
ground -> high count everywhere) versus the localized person motion under the tree.

Use it to choose --sens-var-threshold: the value that still lights up the person
under the tree but does NOT flood on camera moves.

  HF_TOKEN not needed (no DINO).  Runs detection only.
  python3 src/debug_motion_mask.py --start 460 --end 646 --thresholds 16,30,50 \
      --out output/motion_mask_debug.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colorfix
from track_full_clip import chain_segments, fit_canvas


def parse_args():
    p = argparse.ArgumentParser(description="Motion-mask debug video (multi-threshold).")
    p.add_argument("--source", default="data/source.mp4")
    p.add_argument("--out", default="output/motion_mask_debug.mp4")
    p.add_argument("--thresholds", default="16,30,50",
                   help="Comma list of MOG2 varThreshold values to visualise.")
    p.add_argument("--start", type=int, default=460)
    p.add_argument("--end", type=int, default=646)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--coverage-frames", type=int, default=12)
    p.add_argument("--min-area-px", type=float, default=3.0)
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--blur-frac", type=float, default=0.19)
    p.add_argument("--panel-h", type=int, default=460, help="Panel height in the output.")
    p.add_argument("--filter-compare", action="store_true",
                   help="Compare noise filters on ONE threshold (--base-threshold): "
                        "raw vs morphological OPEN (erosion) vs temporal persistence.")
    p.add_argument("--base-threshold", type=float, default=16.0)
    p.add_argument("--persist", type=int, default=3,
                   help="Temporal persistence: keep motion present in N consecutive frames.")
    p.add_argument("--persist-tol", type=int, default=5,
                   help="Dilation tolerance (px) for temporal matching of slow motion.")
    p.add_argument("--open-iter", type=int, default=1, help="OPEN iterations for the erosion variant.")
    return p.parse_args()


def detect_overlay(aligned_bgr, fg, k3, min_area, max_area):
    """Return (overlay_bgr, n_blobs, fg_px). fg already validity-masked & morphology'd."""
    n, _, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    fg_px = int((fg > 0).sum())
    base = (aligned_bgr * 0.45).astype(np.uint8)
    red = np.zeros_like(base)
    red[..., 2] = fg                                  # raw fg in red
    out = cv2.addWeighted(base, 1.0, red, 0.8, 0)
    kept = 0
    for c in range(1, n):
        x, y, bw, bh, ar = stats[c]
        if ar < min_area or ar > max_area or not (0.25 <= bh / max(bw, 1) <= 4.0):
            continue
        kept += 1
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 1)   # kept blobs in green
    return out, kept, fg_px


def main():
    a = parse_args()
    thrs = [float(t) for t in a.thresholds.split(",")]
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    Hs, segs, blur = chain_segments(a.source, a.max_frames, a.min_inliers)
    blur = np.array(blur)
    blur_thr = a.blur_frac * float(np.median(blur))
    seg_canvas = {}
    for s in sorted(set(segs)):
        Hseg = [Hs[i] for i in range(len(Hs)) if segs[i] == s]
        T, cw, ch, _ = fit_canvas(Hseg, w, h)
        seg_canvas[s] = (T, cw, ch)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kbase = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ones = np.full((h, w), 255, np.uint8)
    min_area, max_area = a.min_area_px, 0.05 * w * h

    models = {t: None for t in thrs}
    base_model = None
    thist = []                                          # temporal history of base masks
    k_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.persist_tol + 1,) * 2)
    coverage = cur_seg = T = None
    cw = ch = 0

    pw = int(a.panel_h * w / h)                        # raw panel width (keep frame aspect)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None

    cap = cv2.VideoCapture(a.source)
    for idx in range(len(Hs)):
        ok, f = cap.read()
        if not ok:
            break
        s = segs[idx]
        if s != cur_seg:
            cur_seg = s
            T, cw, ch = seg_canvas[s]
            coverage = np.zeros((ch, cw), np.int32)
            for t in thrs:
                models[t] = cv2.createBackgroundSubtractorMOG2(
                    history=a.history, varThreshold=t, detectShadows=False)
            base_model = cv2.createBackgroundSubtractorMOG2(
                history=a.history, varThreshold=a.base_threshold, detectShadows=False)
            thist = []

        bw = colorfix.to_bw(f)
        Hc = T @ Hs[idx]
        blurred = blur[idx] < blur_thr
        panels = []
        # panel 0: raw frame
        raw = f.copy()
        cv2.putText(raw, f"f{idx} {'BLUR-skip' if blurred else ''}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panels.append(cv2.resize(raw, (pw, a.panel_h)))

        if not blurred:
            aligned = cv2.warpPerspective(bw, Hc, (cw, ch))
            covered = cv2.warpPerspective(ones, Hc, (cw, ch)) > 0
            coverage[covered] += 1
            coverage[~covered] = 0
            validity = cv2.erode(((coverage >= a.coverage_frames).astype(np.uint8)) * 255, kbase)
        else:
            aligned = np.zeros((ch, cw, 3), np.uint8)
            validity = np.zeros((ch, cw), np.uint8)

        cpw = int(a.panel_h * cw / ch)                 # canvas panel width
        if a.filter_compare:
            # one threshold, three noise filters: raw | OPEN(erosion) | temporal-persist
            if blurred:
                closed = np.zeros((ch, cw), np.uint8)
            else:
                rfg = cv2.bitwise_and(
                    cv2.threshold(base_model.apply(aligned), 200, 255, cv2.THRESH_BINARY)[1], validity)
                closed = cv2.morphologyEx(rfg, cv2.MORPH_CLOSE, k3, iterations=2)
            base = cv2.dilate(closed, k3, iterations=1)
            # honest erosion: OPEN the pre-dilate mask, then dilate to compare on equal footing
            opened = cv2.dilate(cv2.morphologyEx(closed, cv2.MORPH_OPEN, k3, iterations=a.open_iter),
                                k3, iterations=1)
            persist = base.copy()                       # temporal: present in N consecutive frames
            for prev in thist[-(a.persist - 1):]:
                persist = cv2.bitwise_and(persist, cv2.dilate(prev, k_tol))
            if not blurred:
                thist.append(base)
                thist[:] = thist[-(a.persist - 1):] if a.persist > 1 else []
            for name, m in [(f"raw thr{int(a.base_threshold)}", base),
                            (f"OPEN x{a.open_iter} (erosion)", opened),
                            (f"temporal x{a.persist}", persist)]:
                ov, kept, fg_px = detect_overlay(aligned, m, k3, min_area, max_area)
                cv2.putText(ov, f"{name}  px={fg_px} blobs={kept}", (6, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panels.append(cv2.resize(ov, (cpw, a.panel_h)))
        else:
            for t in thrs:
                if blurred:
                    ov = np.zeros((ch, cw, 3), np.uint8)
                    kept = fg_px = 0
                else:
                    raw_fg = cv2.threshold(models[t].apply(aligned), 200, 255, cv2.THRESH_BINARY)[1]
                    fg = cv2.bitwise_and(raw_fg, validity)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3, iterations=2)  # no OPEN (show specks)
                    fg = cv2.dilate(fg, k3, iterations=1)
                    ov, kept, fg_px = detect_overlay(aligned, fg, k3, min_area, max_area)
                cv2.putText(ov, f"thr={int(t)} fg_px={fg_px} blobs={kept}",
                            (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(cv2.resize(ov, (cpw, a.panel_h)))

        frame = np.hstack(panels)
        if not (a.start <= idx <= a.end):
            continue
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
    print(f"wrote {a.out}  panels: raw + " + ", ".join(f"thr{int(t)}" for t in thrs))


if __name__ == "__main__":
    main()
