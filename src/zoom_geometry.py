"""Characterize a discrete FOV/camera switch between two frames.

The 646->647 jump is a thermal-gimbal FOV switch, not a moving-camera dolly. So
the two views are related by a *fixed* transform. This module fits that transform
and answers three questions about a switch pair:

  * how big is the switch?  -> scale of the wide->narrow homography at its fixed
    point (the FOV ratio = the "zoom"). No intrinsics needed: the image-space
    scale of a same-optical-centre switch *is* the focal ratio.
  * where is its centre?    -> the focus-of-expansion = fixed point of H = the
    off-centre scaling centre caused by the boresight / principal-point offset.
  * step-zoom or dual-camera? -> fit a homography (one plane) AND a fundamental
    matrix. If a coherent set of points (the tall tree) is a homography OUTLIER
    but a fundamental-matrix INLIER, there is real parallax => separate sensors
    with a fixed baseline. If the tree fits the same homography as the ground =>
    same optical centre (step-zoom), homography is the complete model.

The hard part is correspondence sparsity across a ~4x scale gap: only a handful of
DINOv3 patches in the wide frame land inside the narrow frame's footprint. The
crop-refine pass fixes this -- bootstrap a rough scale, crop+upscale the wide
frame's footprint so both views are at ~equal scale, then re-match densely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from dinov3_match import DenseMatcher
from scene_analysis import Dinov3Embedder


@dataclass
class ZoomReport:
    frame_a: int
    frame_b: int
    n_matches: int
    n_inliers_H: int
    n_inliers_F: int
    zoom: float                       # scale of H at the FoE (FOV ratio)
    foe_xy: tuple[float, float] | None
    verdict: str                      # "step-zoom" | "dual-camera" | "insufficient"
    gric_H: float = float("nan")
    gric_F: float = float("nan")
    H: np.ndarray | None = None
    F: np.ndarray | None = None
    plane_pts_a: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    plane_pts_b: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    parallax_pts_a: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    parallax_pts_b: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))


# --------------------------------------------------------------------------- #
# geometry helpers                                                            #
# --------------------------------------------------------------------------- #
def _sym_transfer_err(H: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Per-point symmetric transfer error (px) of a->b homography."""
    Hi = np.linalg.inv(H)
    fa = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2)
    fb = cv2.perspectiveTransform(pb.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    d1 = np.linalg.norm(fa - pb, axis=1)
    d2 = np.linalg.norm(fb - pa, axis=1)
    return np.sqrt(0.5 * (d1 ** 2 + d2 ** 2))


def _sampson(F: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Per-point Sampson distance (px) for fundamental matrix a->b."""
    pah = np.c_[pa, np.ones(len(pa))]
    pbh = np.c_[pb, np.ones(len(pb))]
    Fx1 = (F @ pah.T).T          # epipolar lines in b
    Ftx2 = (F.T @ pbh.T).T       # epipolar lines in a
    num = np.sum(pbh * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def _foe(H: np.ndarray) -> tuple[float, float] | None:
    """Fixed point of H (eigenvector whose eigenvalue is closest to 1)."""
    w, v = np.linalg.eig(H)
    order = np.argsort(np.abs(w - 1.0))
    for k in order:
        vec = v[:, k]
        if abs(vec[2]) < 1e-9 or abs(vec.imag).max() > 1e-6:
            continue
        p = (vec.real / vec.real[2])
        return float(p[0]), float(p[1])
    return None


def _scale_at(H: np.ndarray, x: float, y: float) -> float:
    """Local linear scale of H at point (x, y) = sqrt(|det Jacobian|)."""
    p = H @ np.array([x, y, 1.0])
    wz = p[2]
    J = (H[:2, :2] * wz - np.outer(p[:2], H[2, :2])) / (wz * wz)
    return float(np.sqrt(abs(np.linalg.det(J))))


def _gric(resid_px: np.ndarray, dim: int, k: int, sigma: float) -> float:
    """Torr's GRIC score (lower = better). dim: H=2, F=3. k: params H=8, F=7."""
    n = len(resid_px)
    r = 4.0
    lam1, lam2, lam3 = np.log(r), np.log(r * n), 2.0
    e2 = (resid_px / sigma) ** 2
    rho = np.minimum(e2, lam3 * (r - dim))
    return float(rho.sum() + lam1 * dim * n + lam2 * k)


def _orb_corr(a: np.ndarray, b: np.ndarray, n=4000, ratio=0.75):
    """Ratio-test ORB correspondences (used after scale-normalisation)."""
    orb = cv2.ORB_create(nfeatures=n, nlevels=8, scaleFactor=1.2)
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for pr in knn if len(pr) == 2 for m, n2 in [pr] if m.distance < ratio * n2.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return pa, pb


# --------------------------------------------------------------------------- #
# correspondences with crop-refine                                           #
# --------------------------------------------------------------------------- #
def _search_boxes(wide, e_n, embedder, scales, stride_frac,
                  cx_lo, cx_hi, cy_lo, cy_hi):
    """Best (scale, cx, cy, bw, bh, sim) over a grid of boxes vs target embed e_n."""
    Hh, Ww = wide.shape[:2]
    crops, meta = [], []
    for s in scales:
        bw, bh = int(Ww / s), int(Hh / s)
        if bw < 16 or bh < 16:
            continue
        sx = max(1, int(bw * stride_frac))
        sy = max(1, int(bh * stride_frac))
        xc_lo = int(np.clip(cx_lo, bw / 2, Ww - bw / 2))
        xc_hi = int(np.clip(cx_hi, bw / 2, Ww - bw / 2))
        yc_lo = int(np.clip(cy_lo, bh / 2, Hh - bh / 2))
        yc_hi = int(np.clip(cy_hi, bh / 2, Hh - bh / 2))
        for yc in range(yc_lo, yc_hi + 1, sy):
            for xc in range(xc_lo, xc_hi + 1, sx):
                x0, y0 = int(xc - bw / 2), int(yc - bh / 2)
                crops.append(wide[y0:y0 + bh, x0:x0 + bw])
                meta.append((s, float(xc), float(yc), bw, bh))
    if not crops:
        return None
    sims = embedder.embed(crops) @ e_n
    k = int(sims.argmax())
    return (*meta[k], float(sims[k]))


def locate_narrow_in_wide(wide, narrow, embedder: Dinov3Embedder,
                          scales=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0)):
    """Locate (scale, centre) of the narrow frame inside the wide frame.

    Slides a box of size (W/s, H/s) for each candidate scale, upscales it to the
    narrow frame's size, and compares DINOv3 global descriptors (cosine).
    Semantic descriptors match the same place across the big resolution gap where
    raw feature matching fails. Two stages: coarse over the whole frame, then a
    fine scale+position refine around the coarse winner. Returns
    (scale, cx, cy, box_w, box_h, sim) or None.
    """
    Hh, Ww = wide.shape[:2]
    e_n = embedder.embed([narrow])[0]
    coarse = _search_boxes(wide, e_n, embedder, scales, 0.5, 0, Ww, 0, Hh)
    if coarse is None:
        return None
    s, cx, cy, bw, bh, _ = coarse
    fine_scales = [s * f for f in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)]
    fine = _search_boxes(wide, e_n, embedder, fine_scales, 0.25,
                         cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2)
    return fine if fine is not None else coarse


def correspondences(a_bgr, b_bgr, dm: DenseMatcher, embedder: Dinov3Embedder,
                    sim_floor=0.5, raw_inlier_skip=60):
    """Wide(a)->narrow(b) correspondences in original a/b pixel coords.

    Fast path: if a raw DINOv3 match already yields a strong homography (same-FOV
    frames), use it directly. Otherwise (big scale gap): locate the narrow frame
    in the wide frame by embedding search, crop+upscale that footprint so both are
    at ~equal scale, re-match (DINOv3 + ORB, which works once scales agree), and
    map correspondences back to original wide-frame coords.
    """
    pa, pb = dm.match(a_bgr, b_bgr, sim_floor=sim_floor)
    if len(pa) >= 4:
        H0, m0 = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 3.0)
        if H0 is not None and m0 is not None and int(m0.sum()) >= raw_inlier_skip:
            return pa, pb                              # same-FOV: raw match is fine

    loc = locate_narrow_in_wide(a_bgr, b_bgr, embedder)
    if loc is None:
        return pa, pb
    s0, cx, cy, bw, bh, _ = loc
    Ha, Wa = a_bgr.shape[:2]
    x0 = int(np.clip(cx - bw / 2, 0, Wa - bw))
    y0 = int(np.clip(cy - bh / 2, 0, Ha - bh))
    crop = a_bgr[y0:y0 + int(bh), x0:x0 + int(bw)]
    if crop.size == 0:
        return pa, pb
    Hb, Wb = b_bgr.shape[:2]
    crop_up = cv2.resize(crop, (Wb, Hb), interpolation=cv2.INTER_CUBIC)
    sx, sy = crop.shape[1] / Wb, crop.shape[0] / Hb    # crop_up px -> wide px

    pca, pcb = dm.match(crop_up, b_bgr, sim_floor=sim_floor)
    oa, ob = _orb_corr(crop_up, b_bgr)
    ca = np.vstack([pca, oa]) if len(oa) else pca
    cb = np.vstack([pcb, ob]) if len(ob) else pcb
    if len(ca) < 4:
        return pa, pb
    ca_orig = np.column_stack([x0 + ca[:, 0] * sx, y0 + ca[:, 1] * sy])
    return ca_orig.astype(np.float32), cb.astype(np.float32)


# --------------------------------------------------------------------------- #
# main entry                                                                  #
# --------------------------------------------------------------------------- #
def characterize_from_correspondences(pa, pb, wide_shape, frame_a=-1, frame_b=-1,
                                      thresh=2.0, parallax_min=10) -> ZoomReport:
    """Fit H + F to wide->narrow correspondences and classify the FOV switch.

    Matcher-agnostic: feed it DINOv3 or MASt3R correspondences. `wide_shape` is
    (H, W) of the wide frame, used for the FoE-centre fallback. Returns the zoom
    (scale of H at its fixed point), the focus-of-expansion, and the step-zoom vs
    dual-camera verdict (off-plane points that are H-outliers but F-inliers).
    """
    pa = np.asarray(pa, np.float32)
    pb = np.asarray(pb, np.float32)
    rep = ZoomReport(frame_a, frame_b, len(pa), 0, 0, float("nan"), None, "insufficient")
    if len(pa) < 8:
        return rep

    H, maskH = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, thresh)
    F, maskF = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, thresh, confidence=0.999)
    if H is None:
        return rep
    inH = maskH.ravel().astype(bool)
    rep.H, rep.F, rep.n_inliers_H = H, F, int(inH.sum())

    rH = _sym_transfer_err(H, pa, pb)
    rep.plane_pts_a, rep.plane_pts_b = pa[inH], pb[inH]

    foe = _foe(H)
    rep.foe_xy = foe
    Hh, Ww = wide_shape[:2]
    cx, cy = foe if foe else (Ww / 2, Hh / 2)
    rep.zoom = _scale_at(H, cx, cy)
    rep.gric_H = _gric(np.minimum(rH, 10 * thresh), dim=2, k=8, sigma=thresh)

    # --- parallax test: H-outliers that are F-inliers = off-plane (tree) ---
    if F is not None and F.shape == (3, 3):
        sF = _sampson(F, pa, pb)
        rep.n_inliers_F = int((sF < thresh).sum())
        rep.gric_F = _gric(np.minimum(sF, 10 * thresh), dim=3, k=7, sigma=thresh)
        off_plane = (rH > 3 * thresh) & (sF < thresh)
        rep.parallax_pts_a, rep.parallax_pts_b = pa[off_plane], pb[off_plane]
        is_3d = (int(off_plane.sum()) >= parallax_min and rep.gric_F < rep.gric_H)
        rep.verdict = "dual-camera" if is_3d else "step-zoom"
    else:
        rep.verdict = "step-zoom"
    return rep


def characterize_pair(a_bgr, b_bgr, dm: DenseMatcher, embedder: Dinov3Embedder,
                      frame_a=-1, frame_b=-1, thresh=2.0, parallax_min=10) -> ZoomReport:
    pa, pb = correspondences(a_bgr, b_bgr, dm, embedder)
    return characterize_from_correspondences(pa, pb, a_bgr.shape, frame_a, frame_b,
                                             thresh, parallax_min)


def draw_zoom_report(a_bgr, b_bgr, rep: ZoomReport) -> np.ndarray:
    """[a | b] with ground (green) / parallax (red) points, epipolar lines, FoE box."""
    a, b = a_bgr.copy(), b_bgr.copy()
    for p in rep.plane_pts_a.astype(int):
        cv2.circle(a, tuple(p), 2, (0, 200, 0), -1)
    for p in rep.plane_pts_b.astype(int):
        cv2.circle(b, tuple(p), 2, (0, 200, 0), -1)
    for p in rep.parallax_pts_a.astype(int):
        cv2.circle(a, tuple(p), 3, (0, 0, 255), -1)
    for p in rep.parallax_pts_b.astype(int):
        cv2.circle(b, tuple(p), 3, (0, 0, 255), -1)
    if rep.F is not None and len(rep.parallax_pts_a):
        lines = cv2.computeCorrespondEpilines(rep.parallax_pts_a.reshape(-1, 1, 2), 1, rep.F)
        for ln in lines.reshape(-1, 3)[:12]:
            x0v, x1v = 0, b.shape[1]
            y0v = int(-ln[2] / ln[1]) if abs(ln[1]) > 1e-6 else 0
            y1v = int(-(ln[2] + ln[0] * x1v) / ln[1]) if abs(ln[1]) > 1e-6 else b.shape[0]
            cv2.line(b, (x0v, y0v), (x1v, y1v), (0, 0, 255), 1)
    if rep.foe_xy:
        fx, fy = int(rep.foe_xy[0]), int(rep.foe_xy[1])
        cv2.drawMarker(a, (fx, fy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        if rep.zoom > 1.05:
            bw, bh = a.shape[1] / rep.zoom, a.shape[0] / rep.zoom
            cv2.rectangle(a, (int(fx - bw / 2), int(fy - bh / 2)),
                          (int(fx + bw / 2), int(fy + bh / 2)), (0, 255, 255), 2)
    panel = np.hstack([a, b])
    txt = (f"zoom={rep.zoom:.2f}x  H_in={rep.n_inliers_H} F_in={rep.n_inliers_F}  "
           f"parallax_pts={len(rep.parallax_pts_a)}  verdict={rep.verdict}")
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(panel, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel
