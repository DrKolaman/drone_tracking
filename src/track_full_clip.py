"""Full-clip person tracking: per-segment stabilization + cross-segment DINOv3 re-ID.

The clip has hard discontinuities (the ~647 zoom, the 745/957 map jumps, the
B/W<->red colour switch). This tracker:
  * colour-normalises every frame (colorfix.to_bw) so red<->B/W is one modality;
  * splits the clip into SEGMENTS at genuine discontinuities (registration
    collapse, ok=False) ONLY -- NOT on canvas overflow;
  * within each segment, fits the canvas to that segment's camera trajectory
    (two-pass, like track_dino_reid) so a long pan never overflows / re-anchors;
  * runs motion (MOG2) + DINOv3 association in stabilized canvas coords;
  * persists the DINOv3 target MEMORY across segments, so after a cut the same
    person is re-acquired by appearance.

Association (per frame, stabilized coords):
  RULE 1 motion  - nearest blob within a small gate of the predicted position.
  RULE 2 DINOv3  - else the best discriminative-margin blob (far jump needs the
                   strong margin); this is what re-acquires across segments.
A short freeze (HOLD, hard-limited to hold_seconds) bridges brief gaps.

  HF_TOKEN=... python3 src/track_full_clip.py --output output/track_full_clip.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colorfix
from bytetrack_shim import detections_from_boxes, make_tracker
from dino_embedder import DinoEmbedder
from registration import GlobalMotionEstimator
from target_memory import BackgroundMemory, TargetMemory
from track_dino_reid import detect, fit_canvas


def parse_args():
    p = argparse.ArgumentParser(description="Full-clip tracking (segmented + cross-segment re-ID).")
    p.add_argument("--source", default="data/source.mp4")
    p.add_argument("--output", default="output/track_full_clip.mp4")
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--var-threshold", type=float, default=50.0)
    p.add_argument("--min-area-px", type=float, default=5.0)
    p.add_argument("--max-area-frac", type=float, default=0.05)
    p.add_argument("--coverage-frames", type=int, default=12)
    p.add_argument("--blur-frac", type=float, default=0.19,
                   help="Skip motion detection on frames whose sharpness < this x median "
                        "(blurred frames give garbage detections; we already detect blur).")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--warmup-views", type=int, default=4)
    p.add_argument("--bank-size", type=int, default=24)
    p.add_argument("--add-thresh", type=float, default=0.92)
    p.add_argument("--bg-size", type=int, default=120)
    p.add_argument("--max-identities", type=int, default=6,
                   help="Cap on distinct target identities spawned over the clip.")
    p.add_argument("--reid-margin", type=float, default=0.10)
    p.add_argument("--strong-margin", type=float, default=0.15)
    p.add_argument("--gate-radius", type=float, default=40.0)
    p.add_argument("--max-jump", type=float, default=60.0)
    p.add_argument("--coast-frames", type=int, default=8)
    p.add_argument("--hold-seconds", type=float, default=2.0)
    p.add_argument("--size-alpha", type=float, default=0.15)
    p.add_argument("--track-relax-radius", type=float, default=45.0,
                   help="Around the predicted position of a target that was MISSED last "
                        "frame, relax the coverage/erosion validity gate so detection "
                        "survives at the leading edge of a follow-pan. 0 disables.")
    p.add_argument("--track-relax-frames", type=int, default=3,
                   help="Min coverage (MOG2 model age) required inside the relax disc.")
    return p.parse_args()


def chain_segments(source, max_frames, min_inliers):
    """Pass 1: per-frame cumulative homography (frame -> segment start) and segment id.
    A new segment starts ONLY when registration collapses (a real discontinuity)."""
    cap = cv2.VideoCapture(source)
    est = GlobalMotionEstimator(min_inliers=min_inliers)
    Hs, segs, blur = [], [], []
    H = np.eye(3)
    seg = 0
    prev = None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or i >= max_frames:
            break
        g = cv2.cvtColor(colorfix.to_bw(f), cv2.COLOR_BGR2GRAY)
        blur.append(cv2.Laplacian(g, cv2.CV_64F).var())   # variance-of-Laplacian sharpness
        if prev is not None:
            r = est.estimate(prev, g)
            if r.ok:
                H = H @ r.H
            else:
                seg += 1
                H = np.eye(3)          # re-anchor: real discontinuity only
        Hs.append(H.copy())
        segs.append(seg)
        prev = g
        i += 1
    cap.release()
    return Hs, segs, blur


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Pass 1: segments + chains + per-frame blur; per-segment canvas fitted to the pan.
    Hs, segs, blur = chain_segments(a.source, a.max_frames, a.min_inliers)
    blur = np.array(blur)
    blur_thr = a.blur_frac * float(np.median(blur))   # below => too blurred to detect on
    seg_canvas = {}      # seg id -> (T, cw, ch)
    for s in sorted(set(segs)):
        Hseg = [Hs[i] for i in range(len(Hs)) if segs[i] == s]
        T, cw, ch, _ = fit_canvas(Hseg, w, h)
        seg_canvas[s] = (T, cw, ch)

    emb = DinoEmbedder()
    identities = []                 # list of {"id": int, "mem": TargetMemory}; PERMANENT
    background = BackgroundMemory(capacity=a.bg_size)   # shared negatives
    next_id = 1
    active = None                   # index into identities of the active target
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kbase = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    min_area, max_area = a.min_area_px, a.max_area_frac * w * h
    hold_frames = int(a.hold_seconds * fps)
    ones = np.full((h, w), 255, np.uint8)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    palette = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255),
               (0, 215, 255), (128, 0, 255), (255, 255, 0)]

    def best_identity(e):
        """(index, margin) of the identity whose memory best matches e (vs background)."""
        if not identities:
            return -1, -1e9
        bs, bi = -1e9, -1
        bgs = background.score(e)
        for ii, idd in enumerate(identities):
            m = idd["mem"].score(e) - bgs
            if m > bs:
                bs, bi = m, ii
        return bi, bs

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cur_seg = -1
    mog2 = coverage = T = None
    cw = ch = 0
    btrack = make_tracker(fps)              # causal, every frame (bootstrap + spawn)
    bt_embs = defaultdict(list)
    bt_count = Counter()
    last_pos = last_box = None
    vel = np.zeros(2)
    since = 10 ** 9
    sm_size = None
    present = held = reacq = spawns = 0
    id_frames = Counter()

    cap = cv2.VideoCapture(a.source)
    fa = float(w * h)
    for idx in range(len(Hs)):
        ok, f = cap.read()
        if not ok:
            break
        s = segs[idx]
        if s != cur_seg:                       # entered a new segment
            cur_seg = s
            T, cw, ch = seg_canvas[s]
            mog2 = cv2.createBackgroundSubtractorMOG2(
                history=a.history, varThreshold=a.var_threshold, detectShadows=False)
            coverage = np.zeros((ch, cw), np.int32)
            last_pos, since, vel = None, 10 ** 9, np.zeros(2)   # position invalid in new canvas

        bw = colorfix.to_bw(f)
        Hc = T @ Hs[idx]
        # Where do we expect the active target? Only relax detection once it was
        # MISSED last frame (since > 0): while cleanly tracking (since == 0) the
        # validity mask is left untouched, so good tracking is provably unchanged.
        relax_pred = None
        if (a.track_relax_radius > 0 and active is not None and last_pos is not None
                and since > 0):
            relax_pred = last_pos + vel * min(since + 1, a.coast_frames)
        if blur[idx] < blur_thr:
            boxes, cents, embs = [], [], np.zeros((0, emb.dim), np.float32)   # blurred -> no detection
        else:
            aligned = cv2.warpPerspective(bw, Hc, (cw, ch))
            covered = cv2.warpPerspective(ones, Hc, (cw, ch)) > 0
            coverage[covered] += 1
            coverage[~covered] = 0
            # Strict gate everywhere; a relaxed disc around the predicted position lets
            # detection survive at the leading edge of a follow-pan (validity is a
            # SUPERSET of the strict mask -- it only adds area, never removes it).
            validity = cv2.erode(((coverage >= a.coverage_frames).astype(np.uint8)) * 255, kbase)
            if relax_pred is not None:
                relaxed = ((coverage >= a.track_relax_frames).astype(np.uint8)) * 255  # NOT eroded
                disc = np.zeros((ch, cw), np.uint8)
                cv2.circle(disc, (int(round(relax_pred[0])), int(round(relax_pred[1]))),
                           int(a.track_relax_radius), 255, -1)
                validity = cv2.bitwise_or(validity, cv2.bitwise_and(relaxed, disc))
            boxes, cents = detect(aligned, validity, mog2, k3, min_area, max_area)
            embs = emb.embed_boxes(aligned, boxes) if boxes else np.zeros((0, emb.dim), np.float32)

        # ByteTrack runs every frame (for bootstrap + spawning persistent novel objects)
        bt_assign = {}
        for r in btrack.update(detections_from_boxes(boxes, fa)):
            tid, di = int(r[4]), int(r[7])
            bt_count[tid] += 1
            if di < len(embs):
                bt_embs[tid].append(embs[di])
                bt_assign[di] = tid

        prev_active = active
        target_j, state, match_id = -1, None, None

        if not identities:
            # BOOTSTRAP: first persistent ByteTrack track -> identity 1
            for di, tid in bt_assign.items():
                if bt_count[tid] >= a.warmup_views:
                    mem = TargetMemory(capacity=a.bank_size, add_thresh=a.add_thresh)
                    mem.consolidate(bt_embs[tid], capacity=a.bank_size)
                    identities.append({"id": next_id, "mem": mem})
                    next_id += 1
                    target_j, state, match_id = di, "TRACK", 0
                    break
        else:
            # RULE 1 — motion-near on the ACTIVE id
            if active is not None and last_pos is not None:
                pred = last_pos + vel * min(since + 1, a.coast_frames)
                best_d = a.gate_radius
                for j in range(len(boxes)):
                    d = np.hypot(cents[j][0] - pred[0], cents[j][1] - pred[1])
                    if d <= best_d:
                        best_d, target_j = d, j
                if target_j >= 0:
                    state, match_id = "TRACK", active
            # RULE 2 — re-acquire the ACTIVE id by appearance (HOLD-protected)
            if target_j < 0 and boxes and active is not None:
                am = [identities[active]["mem"].score(e) - background.score(e) for e in embs]
                g = int(np.argmax(am))
                jump = (np.hypot(cents[g][0] - last_pos[0], cents[g][1] - last_pos[1])
                        if last_pos is not None else 1e9)
                holding = last_pos is not None and since <= hold_frames
                if holding:
                    accept = jump <= a.max_jump and am[g] >= a.reid_margin   # near-only (f515 fix)
                else:
                    accept = am[g] >= (a.strong_margin if jump > a.max_jump else a.reid_margin)
                if accept:
                    target_j, state, match_id = g, "REACQ", active
            # RULE 3 — re-acquire ANY existing id (global best) -> return-of-id1 path
            if target_j < 0 and boxes:
                bj, bii, bm = -1, -1, -1e9
                for j in range(len(boxes)):
                    ii, m = best_identity(embs[j])
                    if m > bm:
                        bm, bj, bii = m, j, ii
                if bii >= 0 and bm >= a.reid_margin:    # loosened so id1 re-catches across the zoom
                    target_j, state, match_id = bj, "REACQ", bii
            # RULE 4 — spawn a NEW id for a persistent, NOVEL object (e.g. 754)
            if target_j < 0 and len(identities) < a.max_identities:
                for j in range(len(boxes)):
                    tid = bt_assign.get(j)
                    if tid is None or bt_count[tid] < a.warmup_views:
                        continue
                    _, m = best_identity(embs[j])
                    if m < a.reid_margin:                       # matches no existing id
                        mem = TargetMemory(capacity=a.bank_size, add_thresh=a.add_thresh)
                        mem.consolidate(bt_embs[tid], capacity=a.bank_size)
                        identities.append({"id": next_id, "mem": mem})
                        next_id += 1
                        spawns += 1
                        target_j, state, match_id = j, "NEW", len(identities) - 1
                        break

        # ---- commit / hold ----
        cur_id = None
        cbox = None
        if target_j >= 0 and match_id is not None:
            c = np.array(cents[target_j])
            vel = (0.6 * vel + 0.4 * (c - last_pos)) if (match_id == active and state == "TRACK"
                                                         and last_pos is not None) else np.zeros(2)
            if state == "REACQ":
                reacq += 1
            if match_id != active:
                sm_size = None                                 # id switch -> reset box-size filter
            active = match_id
            last_pos, last_box, since = c, boxes[target_j], 0
            identities[active]["mem"].update(embs[target_j])
            for j in range(len(boxes)):
                if j != target_j:
                    background.add(embs[j])
            present += 1
            cur_id = identities[active]["id"]
            id_frames[cur_id] += 1
            cbox = boxes[target_j]
        elif active is not None and last_pos is not None and since <= hold_frames:
            since += 1
            bw_, bh_ = last_box[2] - last_box[0], last_box[3] - last_box[1]
            cbox = [last_pos[0] - bw_ / 2, last_pos[1] - bh_ / 2,
                    last_pos[0] + bw_ / 2, last_pos[1] + bh_ / 2]
            state = "HOLD"
            held += 1
            cur_id = identities[active]["id"]
        else:
            since += 1

        # ---- draw on the raw colour frame ----
        if cbox is not None:
            cxc, cyc = (cbox[0] + cbox[2]) / 2.0, (cbox[1] + cbox[3]) / 2.0
            meas = np.array([cbox[2] - cbox[0], cbox[3] - cbox[1]], float)
            sm_size = meas if sm_size is None else a.size_alpha * meas + (1 - a.size_alpha) * sm_size
            bwd, bhd = sm_size
            quad = cv2.perspectiveTransform(
                np.float32([[cxc - bwd / 2, cyc - bhd / 2], [cxc + bwd / 2, cyc - bhd / 2],
                            [cxc + bwd / 2, cyc + bhd / 2], [cxc - bwd / 2, cyc + bhd / 2]]).reshape(-1, 1, 2),
                np.linalg.inv(Hc)).reshape(-1, 2)
            x1, y1 = int(quad[:, 0].min()), int(quad[:, 1].min())
            x2, y2 = int(quad[:, 0].max()), int(quad[:, 1].max())
            col = palette[cur_id % len(palette)]
            cv2.rectangle(f, (x1, y1), (x2, y2), col, 2)
            cv2.putText(f, f"id{cur_id} {state}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        hud = (f"id{cur_id} {state}" if cur_id is not None else "searching")
        cv2.putText(f, f"f{idx} seg{s} {hud}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(f)

    cap.release()
    writer.release()
    n = len(Hs)
    shown = sum(id_frames.values()) + held
    print(f"frames {n} | segments {len(seg_canvas)} | identities {len(identities)} "
          f"(spawns {spawns}) | shown {shown} ({100*shown//max(n,1)}%) | re-acq {reacq}")
    for k in sorted(id_frames):
        print(f"  id{k}: tracked {id_frames[k]} frames")
    print(f"Output: {a.output}")


if __name__ == "__main__":
    main()
