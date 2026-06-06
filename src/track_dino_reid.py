"""One stable target ID via DINOv3 appearance-ReID on the stabilized video.

Extends the stabilized MOG2 tracker: it embeds every moving-target bbox with
DINOv3, locks onto the most-persistent track, and then re-assigns that single
identity each frame by MOTION-GATED APPEARANCE — bridging the MOG2 dropouts and
ByteTrack ID switches that fragmented the baseline into 117 IDs.

  pass 1  per frame: warp -> MOG2 blobs -> DINOv3 embed each blob -> ByteTrack
          (records boxes/centroids/embeddings + the bootstrap track ids)
  lock    longest ByteTrack id -> gallery = mean of its embeddings
  pass 2  per frame: pick the detection nearest the predicted position whose
          embedding matches the gallery (cosine >= sim_thresh); EMA-update the
          gallery; coast through gaps. One fixed id = TARGET. Render video.

  HF_TOKEN=... python3 src/track_dino_reid.py --max-frames 647
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bytetrack_shim import detections_from_boxes, make_tracker
from dino_embedder import DinoEmbedder
from target_memory import BackgroundMemory, TargetMemory
from track_stabilized import chain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3 ReID -> one stable target ID.")
    p.add_argument("--source", default="/project/data/source.mp4")
    p.add_argument("--output", default="output/track_dino_reid.mp4")
    p.add_argument("--max-frames", type=int, default=647)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--var-threshold", type=float, default=50.0)
    p.add_argument("--min-area-px", type=float, default=5.0)
    p.add_argument("--max-area-frac", type=float, default=0.05)
    p.add_argument("--validity-erode-px", type=int, default=10,
                   help="Small inward trim of the stable region (kills seam lines).")
    p.add_argument("--coverage-frames", type=int, default=20,
                   help="Only detect where the frame footprint has covered the map "
                        "for this many consecutive frames (so MOG2 has learnt it). "
                        "Masks out freshly-revealed frame-edge regions in map coords.")
    p.add_argument("--min-inliers", type=int, default=25)
    p.add_argument("--dino-repo", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--crop-min-px", type=int, default=64)
    p.add_argument("--longside", type=int, default=112)
    # Matching uses the DISCRIMINATIVE MARGIN  disc = target.score - background.score
    # (calibrated: real target disc >= ~0.10, f518 false blob disc = -0.02).
    p.add_argument("--near-margin", type=float, default=0.05,
                   help="Margin to accept a NEAR (gated) frame-to-frame match.")
    p.add_argument("--reid-margin", type=float, default=0.10,
                   help="#1 min margin to accept a FAR/global re-acquisition.")
    p.add_argument("--strong-margin", type=float, default=0.15,
                   help="A far match this strong (margin) commits immediately.")
    p.add_argument("--verify-margin", type=float, default=0.05,
                   help="Min margin at the held position to keep a HOLD.")
    p.add_argument("--hysteresis", type=float, default=0.05,
                   help="#2 a far match must reach (current margin - this) to move the box.")
    p.add_argument("--max-jump", type=float, default=60.0,
                   help="A re-acquisition farther than this (px from last pos) needs strong-margin.")
    p.add_argument("--confirm-frames", type=int, default=2,
                   help="#3 medium far matches must recur this many times (within window).")
    p.add_argument("--confirm-window", type=int, default=5,
                   help="#3 frames a pending re-acquisition candidate survives without support.")
    p.add_argument("--gate-radius", type=float, default=40.0)
    p.add_argument("--coast-frames", type=int, default=8,
                   help="Gate around the predicted position only this many gap frames; "
                        "after that, re-acquire globally by appearance (no gate).")
    p.add_argument("--bank-size", type=int, default=24)
    p.add_argument("--bg-size", type=int, default=120, help="Background bank capacity.")
    p.add_argument("--bootstrap-tracks", type=int, default=5,
                   help="Seed the target memory from the union of the top-K longest "
                        "ByteTrack tracks (the same person, fragmented).")
    p.add_argument("--bg-exclude", type=float, default=0.88,
                   help="Don't put a detection in the background bank if it looks like "
                        "the target (target.score >= this) — avoids self-competition.")
    p.add_argument("--warmup-views", type=int, default=8,
                   help="Track MOTION-ONLY until the memory holds this many unique views, "
                        "then arm DINOv3 (it needs enough frames to be reliable).")
    p.add_argument("--warmup-floor", type=int, default=15,
                   help="Never arm DINOv3 before this many tracked frames.")
    p.add_argument("--warmup-ceiling", type=int, default=45,
                   help="Always arm DINOv3 by this many tracked frames (even if <views).")
    p.add_argument("--dedup-thresh", type=float, default=0.95,
                   help="Consolidation: prune views more similar than this (keep unique).")
    p.add_argument("--stop-consolidate-seconds", type=float, default=2.0,
                   help="After the target is untracked this long, consolidate the track's features.")
    p.add_argument("--add-thresh", type=float, default=0.92,
                   help="Store a new target view only if its best bank match is below this.")
    p.add_argument("--hold-seconds", type=float, default=3.0,
                   help="Keep the box on a stopped target this long, verifying with DINOv3.")
    p.add_argument("--vicinity-radius", type=int, default=24,
                   help="During HOLD, DINOv3 searches +/- this many px around the last "
                        "position (keeps the box nearby, never jumps far).")
    p.add_argument("--vicinity-step", type=int, default=8,
                   help="Grid step (px) of the HOLD vicinity search.")
    p.add_argument("--size-alpha", type=float, default=0.15,
                   help="EMA weight on the box SIZE (lower = steadier size). The box "
                        "CENTRE is taken raw (no position filtering) so it sits on the target.")
    p.add_argument("--panel-h", type=int, default=720)
    p.add_argument("--log-csv", default="",
                   help="If set, write per-frame frame,state,cx,cy,sim (canvas coords).")
    return p.parse_args()


def fit_canvas(Hs, w, h):
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    xs, ys = [], []
    for H in Hs:
        c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs += [c[:, 0].min(), c[:, 0].max()]
        ys += [c[:, 1].min(), c[:, 1].max()]
    T = np.array([[1, 0, -min(xs)], [0, 1, -min(ys)], [0, 0, 1]], np.float64)
    return T, int(np.ceil(max(xs) - min(xs))), int(np.ceil(max(ys) - min(ys))), corners


def detect(aligned, validity, mog2, k3, min_area, max_area):
    fg = cv2.threshold(mog2.apply(aligned), 200, 255, cv2.THRESH_BINARY)[1]
    fg = cv2.bitwise_and(fg, validity)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3, iterations=2)
    fg = cv2.dilate(fg, k3, iterations=1)
    n, _, stats, cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
    boxes, cents = [], []
    for c in range(1, n):
        x, y, bw, bh, ar = stats[c]
        if ar < min_area or ar > max_area or not (0.25 <= bh / max(bw, 1) <= 4.0):
            continue
        boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
        cents.append([float(cent[c][0]), float(cent[c][1])])
    return boxes, cents


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    Hs = chain(args.source, args.max_frames, args.min_inliers)
    T, cw, ch, corners = fit_canvas(Hs, w, h)
    frame_area = float(w * h)

    embedder = DinoEmbedder(repo=args.dino_repo, longside=args.longside,
                            crop_min_px=args.crop_min_px)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    min_area, max_area = args.min_area_px, args.max_area_frac * w * h
    ve = max(1, args.validity_erode_px)
    kbase = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ve + 1, 2 * ve + 1))
    ones_full = np.full((h, w), 255, np.uint8)

    # Coverage age: how many CONSECUTIVE recent frames each map pixel has been
    # inside the frame footprint. A pixel only becomes a valid detection region
    # once it has been covered >= coverage_frames (so MOG2 has actually learnt
    # its background). This masks out the moving frame border AND the band it
    # freshly revealed (which lingers as false motion for ~history frames).
    coverage = np.zeros((ch, cw), np.int32)

    def stable_validity(Hc):
        covered = cv2.warpPerspective(ones_full, Hc, (cw, ch)) > 0
        coverage[covered] += 1
        coverage[~covered] = 0
        stable = ((coverage >= args.coverage_frames).astype(np.uint8)) * 255
        return cv2.erode(stable, kbase)

    # ---- pass 1: detect + embed + ByteTrack (for bootstrap) ----
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=args.history, varThreshold=args.var_threshold, detectShadows=False)
    tracker = make_tracker(fps)
    per_frame = []                         # (boxes, cents, embs)
    frame_assign = []                      # per frame: {det_index -> bytetrack id}
    id_frames = Counter()                  # bytetrack id -> #frames
    id_embs = defaultdict(list)            # bytetrack id -> [emb,...]
    cap = cv2.VideoCapture(args.source)
    for i, H in enumerate(Hs):
        ok, f = cap.read()
        if not ok:
            break
        Hc = T @ H
        aligned = cv2.warpPerspective(f, Hc, (cw, ch))
        validity = stable_validity(Hc)                 # coverage-age edge mask
        boxes, cents = detect(aligned, validity, mog2, k3, min_area, max_area)
        embs = embedder.embed_boxes(aligned, boxes)
        per_frame.append((boxes, cents, embs))
        tracks = tracker.update(detections_from_boxes(boxes, frame_area))
        amap = {}
        for r in tracks:
            tid, di = int(r[4]), int(r[7])
            id_frames[tid] += 1
            if di < len(embs):
                id_embs[tid].append(embs[di])
                amap[di] = tid
        frame_assign.append(amap)
    cap.release()

    if not id_frames:
        raise SystemExit("No tracks formed in pass 1; cannot bootstrap a target.")
    # Pass 1 only SELECTS the target (the top-K longest ByteTrack tracks are the
    # same fragmented person). The appearance memory is built ONLINE in pass 2:
    # motion-only warmup until it holds enough unique views, then DINOv3 arms.
    target_ids = {t for t, _ in id_frames.most_common(args.bootstrap_tracks)}
    memory = TargetMemory(capacity=args.bank_size, add_thresh=args.add_thresh)
    background = BackgroundMemory(capacity=args.bg_size)

    def disc(e):
        """Discriminative margin: how much more target-like than background."""
        return memory.score(e) - background.score(e)

    # Fixed display box size = median size of the target's own detections, so the
    # box doesn't "breathe" with the noisy MOG2 blob.
    tsz = [(b[2] - b[0], b[3] - b[1]) for i, (bx, _, _) in enumerate(per_frame)
           for j, b in enumerate(bx) if frame_assign[i].get(j) in target_ids]
    fixed_w = float(np.median([s[0] for s in tsz])) if tsz else 24.0
    fixed_h = float(np.median([s[1] for s in tsz])) if tsz else 24.0

    print(f"target = top-{args.bootstrap_tracks} ByteTrack tracks {sorted(target_ids)} "
          f"({sum(id_frames[t] for t in target_ids)} frames); memory built online "
          f"(motion-only warmup until >={args.warmup_views} views)")

    # ---- pass 2: motion-gated appearance assignment + render ----
    ph = args.panel_h
    raw_w = int(w * ph / h)
    mos_w = int(cw * ph / ch)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (raw_w + mos_w, ph))
    mosaic = np.zeros((ch, cw, 3), np.uint8)

    last_pos = None
    last_box = None
    hold_anchor = None
    vel = np.zeros(2)
    since = 0
    acquired = False
    conf = 0.0            # current confidence (margin) of the displayed box
    pending = None        # pending far re-acquisition candidate (temporal confirm)
    armed = False         # False during motion-only warmup; True once DINOv3 is reliable
    wframes = 0           # frames the target has been tracked (for warmup)
    track_embs = []       # features of the current track run (for consolidation)
    consolidated = True   # whether the last track run has been consolidated
    present = 0
    held = 0
    reacquisitions = 0
    sims_log = []
    bank_log = []
    center_log = [] if args.log_csv else None
    hold_frames = int(args.hold_seconds * fps)
    stop_frames = int(args.stop_consolidate_seconds * fps)

    sm_size = None        # EMA-smoothed box size [w, h]; centre is taken raw

    cap = cv2.VideoCapture(args.source)
    for i, H in enumerate(Hs):
        ok, f = cap.read()
        if not ok:
            break
        Hc = T @ H
        aligned = cv2.warpPerspective(f, Hc, (cw, ch))
        mask = cv2.warpPerspective(ones_full, Hc, (cw, ch)) > 0   # full footprint (display)
        mosaic[mask] = aligned[mask]

        # ===== Two-rule tracker ==========================================
        # RULE 1 (MOTION): while active, the target is the blob nearest the
        #   predicted position, within a gate. Pure motion association.
        # RULE 2 (DINOv3): only when motion can't find it (lost / big gap), accept
        #   the blob whose discriminative margin clearly beats background.
        # Warmup: before the memory has >= warmup_views, follow the target by motion
        #   only (via its ByteTrack id) so DINOv3 learns before it's trusted.
        # =================================================================
        boxes, cents, embs = per_frame[i]
        trained = len(memory) >= args.warmup_views
        margins = [disc(e) for e in embs]
        target_j, reason = -1, ""

        # RULE 1 — motion-near: nearest blob to the velocity-predicted position,
        # within a CONSTANT small gate (so it follows genuine continuous motion and
        # does NOT grab a far distractor during a gap). Beyond the gate -> RULE 2.
        if acquired and last_pos is not None:
            pred = last_pos + vel * min(since + 1, args.coast_frames)
            best_d = args.gate_radius
            for j in range(len(boxes)):
                d = np.hypot(cents[j][0] - pred[0], cents[j][1] - pred[1])
                if d <= best_d:
                    best_d, target_j = d, j
            if target_j >= 0:
                reason = "motion"

        # RULE 2 — DINOv3 re-acquisition (only once trained, and only if motion
        # failed). A FAR jump must clear the STRONG margin so a far weak blob can't
        # hijack the target (the f304 jump); a nearby re-acq uses the reid margin.
        if target_j < 0 and trained and boxes:
            g = int(np.argmax(margins))
            jump = (np.hypot(cents[g][0] - last_pos[0], cents[g][1] - last_pos[1])
                    if last_pos is not None else 1e9)
            need = args.strong_margin if jump > args.max_jump else args.reid_margin
            if margins[g] >= need:
                target_j, reason = g, "reid"

        # Warmup cold-start: follow the target by its motion track id
        if target_j < 0 and not trained:
            tj = next((j for j in range(len(boxes))
                       if frame_assign[i].get(j) in target_ids), -1)
            if tj >= 0:
                target_j, reason = tj, "warmup"

        state, disp_box, disp_sim = None, None, None
        if target_j >= 0:
            c = np.array(cents[target_j])
            if reason == "motion" and last_pos is not None:
                vel = 0.6 * vel + 0.4 * (c - last_pos)
            else:
                vel = np.zeros(2)                        # reacq / warmup -> snap
            last_pos, last_box, since, acquired = c, boxes[target_j], 0, True
            memory.update(embs[target_j])                # learn the target's appearance
            for j in range(len(boxes)):                  # everything else is background
                if j != target_j:
                    background.add(embs[j])
            present += 1
            if reason == "reid":
                reacquisitions += 1
            sims_log.append(margins[target_j])
            state, disp_box, disp_sim = ("REACQ" if reason == "reid" else "TRACK"), \
                boxes[target_j], margins[target_j]
        else:
            since += 1
            # FREEZE: bridge a short gap / brief stop by holding the box at the last
            # position (stabilized, so a stopped target stays put) for hold_frames.
            if acquired and last_pos is not None and since <= hold_frames:
                bw, bh = last_box[2] - last_box[0], last_box[3] - last_box[1]
                disp_box = [last_pos[0] - bw / 2, last_pos[1] - bh / 2,
                            last_pos[0] + bw / 2, last_pos[1] + bh / 2]
                state, disp_sim = "HOLD", None
                held += 1
            else:
                acquired = False                         # LOST
        bank_log.append(len(memory))

        # ---- render ----
        view = mosaic.copy()
        cv2.polylines(view, [cv2.perspectiveTransform(corners, Hc).astype(np.int32)],
                      True, (60, 60, 60), 1)
        raw = cv2.resize(f, (raw_w, ph))
        H_inv = np.linalg.inv(Hc)
        sx, sy = raw_w / w, ph / h

        # RAW centre (no position filtering, so the box stays on the target);
        # EMA-filter only the box SIZE so it doesn't breathe with the MOG2 blob.
        if disp_box is not None:
            cx = (disp_box[0] + disp_box[2]) / 2.0
            cy = (disp_box[1] + disp_box[3]) / 2.0
            meas = np.array([disp_box[2] - disp_box[0], disp_box[3] - disp_box[1]], float)
            if sm_size is None:
                sm_size = meas
            else:
                sm_size = args.size_alpha * meas + (1 - args.size_alpha) * sm_size
            bw, bh = sm_size
            disp_box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
        else:
            sm_size = None

        if disp_box is not None:
            col = (0, 0, 255) if state == "REACQ" else ((0, 200, 0) if state == "TRACK"
                                                        else (0, 165, 255))
            lbl = f"id1 {state}" + ("" if disp_sim is None else f" m{disp_sim:.2f}")
            x1, y1, x2, y2 = disp_box
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
            cv2.putText(view, lbl, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            q = cv2.perspectiveTransform(
                np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2),
                H_inv).reshape(-1, 2)
            cv2.rectangle(raw, (int(q[:, 0].min() * sx), int(q[:, 1].min() * sy)),
                          (int(q[:, 0].max() * sx), int(q[:, 1].max() * sy)), col, 2)
            cv2.putText(raw, lbl, (6, ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        hud = "TRACK" if state == "TRACK" else ("HOLD" if state == "HOLD"
              else (f"re-ID ({since})" if acquired else "searching"))
        cv2.putText(raw, f"f{i} det:{len(boxes)} {hud}", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(np.hstack([raw, cv2.resize(view, (mos_w, ph))]))

        if center_log is not None:
            if disp_box is not None:
                cx = (disp_box[0] + disp_box[2]) / 2
                cy = (disp_box[1] + disp_box[3]) / 2
                ms = "" if disp_sim is None else f"{disp_sim:.3f}"
                center_log.append(f"{i},{state},{cx:.1f},{cy:.1f},{ms}")
            else:
                center_log.append(f"{i},NONE,,,")
    cap.release()
    writer.release()

    n = len(Hs)
    if sims_log:
        shown = present + held
        print(f"frames {n} | tracked(motion) {present} ({100*present/n:.0f}%) | "
              f"held(stopped) {held} ({100*held/n:.0f}%) | box shown {shown} "
              f"({100*shown/n:.0f}%) | re-acquisitions {reacquisitions} | "
              f"sim mean {np.mean(sims_log):.2f} | bank {bank_log[-1]}/{args.bank_size}")
    else:
        print(f"frames {n} | target never matched")
    if center_log is not None:
        Path(args.log_csv).write_text("frame,state,cx,cy,sim\n" + "\n".join(center_log) + "\n")
        print(f"Per-frame centers -> {args.log_csv}")
    print(f"Output: {args.output}  ({raw_w + mos_w}x{ph})")


if __name__ == "__main__":
    main()
