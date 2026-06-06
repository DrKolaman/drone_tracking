"""Measure the FOV switch (e.g. frame 646->647) with MASt3R.

Classical and DINOv3-patch matching collapse on this pair (~6 inliers) because the
wide frame under-resolves the region the narrow frame zoomed into. MASt3R (Naver,
DUSt3R's matching successor) regresses dense correspondences + per-view 3D
pointmaps + focal with unknown, differing intrinsics -- trained for exactly this
extreme scale/viewpoint/low-texture regime -- so it finds matches where detail
matchers cannot.

Zoom is read two independent ways:
  * primary  : similarity scale of the wide->narrow correspondences
               (cv2.estimateAffinePartial2D). For a same-optical-centre step-zoom
               the whole narrow frame maps to a centred sub-window of the wide
               frame, so a 2D similarity captures the zoom directly.
  * crosscheck: focal ratio f_narrow / f_wide (estimate_focal_knowing_depth),
               needing a swapped second inference (pred2 is in view1's frame).

The step-zoom vs dual-camera verdict reuses zoom_geometry's H-vs-F test, now
feedable because MASt3R supplies enough correspondences.

Setup: repo at /project/mast3r_repo, checkpoint at /project/mast3r/checkpoints/.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/project/mast3r_repo")
sys.path.insert(0, "/project/mast3r_repo/dust3r")

import cv2
import numpy as np
import torch
from PIL import Image

import mast3r.utils.path_to_dust3r  # noqa: F401  (registers dust3r on sys.path)
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import ImgNorm
from dust3r.post_process import estimate_focal_knowing_depth

from test_scale_warp import grab
from scene_analysis import register_pair
from zoom_geometry import characterize_from_correspondences, draw_zoom_report

CKPT = "/project/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


def load_model(device: str) -> AsymmetricMASt3R:
    return AsymmetricMASt3R.from_pretrained(CKPT).to(device).eval()


def _to_view(bgr: np.ndarray, idx: int, size: int):
    """dust3r-style view dict from a BGR frame, top-left cropped to /16.

    Top-left (not centre) crop keeps the coord mapping back to the original a
    trivial divide-by-scale. Returns (view, scale) with original = resized/scale.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]
    s = size / max(H0, W0)
    W, H = round(W0 * s), round(H0 * s)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    W -= W % 16
    H -= H % 16
    rgb = rgb[:H, :W]
    img = ImgNorm(Image.fromarray(rgb))[None]
    return dict(img=img, true_shape=np.int32([[H, W]]), idx=idx, instance=str(idx)), s


def _correspondences(view_a, view_b, model, device, subsample=8):
    """Dense reciprocal NN matches between two views, in resized pixel coords."""
    out = inference([(view_a, view_b)], model, device, batch_size=1, verbose=False)
    d1 = out["pred1"]["desc"].squeeze(0).detach()
    d2 = out["pred2"]["desc"].squeeze(0).detach()
    ma, mb = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=subsample,
                                 device=device, dist="dot", block_size=2 ** 13)
    H1, W1 = view_a["true_shape"][0]
    H2, W2 = view_b["true_shape"][0]
    keep = ((ma[:, 0] >= 3) & (ma[:, 0] < W1 - 3) & (ma[:, 1] >= 3) & (ma[:, 1] < H1 - 3) &
            (mb[:, 0] >= 3) & (mb[:, 0] < W2 - 3) & (mb[:, 1] >= 3) & (mb[:, 1] < H2 - 3))
    return ma[keep].astype(np.float32), mb[keep].astype(np.float32), out


def _spread_scale(pa, pb, trials=4000):
    """Robust apparent magnification = median ratio of pairwise distances (narrow/wide).

    Model-free: works under parallax where a single similarity/homography does not.
    A feature at distance d from the zoom centre in the wide frame lands at ~zoom*d
    in the narrow frame, so the ratio of inter-point distances estimates the zoom.
    """
    n = len(pa)
    if n < 3:
        return float("nan")
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, trials)
    j = rng.integers(0, n, trials)
    ok = i != j
    da = np.linalg.norm(pa[i[ok]] - pa[j[ok]], axis=1)
    db = np.linalg.norm(pb[i[ok]] - pb[j[ok]], axis=1)
    m = da > 2.0
    return float(np.median(db[m] / da[m])) if m.any() else float("nan")


def _focal(out, view) -> float:
    """Focal (px, resized frame) from the pointmap in view1's own frame."""
    p = out["pred1"]["pts3d"].detach().float()
    H, W = view["true_shape"][0]
    pp = torch.tensor([[W / 2.0, H / 2.0]], device=p.device, dtype=p.dtype)
    return float(estimate_focal_knowing_depth(p, pp, focal_mode="weiszfeld").squeeze())


def measure_zoom(wide_bgr, narrow_bgr, model, device, size, frame_a=-1, frame_b=-1):
    """Returns a dict with correspondences count, zoom (similarity + focal), verdict."""
    va, sa = _to_view(wide_bgr, 0, size)
    vb, sb = _to_view(narrow_bgr, 1, size)

    ma, mb, out_ab = _correspondences(va, vb, model, device)
    pa = ma / sa                     # -> original wide px
    pb = mb / sb                     # -> original narrow px

    res = {"n_corr": len(pa), "pa": pa, "pb": pb,
           "zoom_sim": float("nan"), "n_inliers": 0,
           "zoom_focal": float("nan"), "A": None,
           "zoom_spread": _spread_scale(pa, pb)}
    if len(pa) >= 3:
        A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if A is not None:
            res["zoom_sim"] = float(np.sqrt(A[0, 0] ** 2 + A[0, 1] ** 2))
            res["n_inliers"] = int(inl.sum())
            res["A"] = A

    # focal cross-check: f_wide from (wide,narrow); f_narrow needs swapped order
    try:
        f_wide = _focal(out_ab, va)
        out_ba = inference([(vb, va)], model, device, batch_size=1, verbose=False)
        f_narrow = _focal(out_ba, vb)
        res["zoom_focal"] = f_narrow / f_wide if f_wide else float("nan")
    except Exception as e:
        res["focal_err"] = str(e)[:120]

    res["report"] = characterize_from_correspondences(pa, pb, wide_bgr.shape,
                                                      frame_a, frame_b)
    return res


def _draw(wide, narrow, res, path):
    a, b = wide.copy(), narrow.copy()
    pa, pb = res["pa"], res["pb"]
    idx = np.linspace(0, len(pa) - 1, min(40, len(pa))).astype(int) if len(pa) else []
    panel = np.hstack([a, b])
    ox = a.shape[1]
    rng = np.random.default_rng(0)
    for i in idx:
        c = tuple(int(v) for v in rng.integers(60, 255, 3))
        pa_i = tuple(np.round(pa[i]).astype(int))
        pb_i = (int(round(pb[i][0])) + ox, int(round(pb[i][1])))
        cv2.circle(panel, pa_i, 2, c, -1)
        cv2.circle(panel, pb_i, 2, c, -1)
        cv2.line(panel, pa_i, pb_i, c, 1)
    txt = (f"corr={res['n_corr']}  zoom_sim={res['zoom_sim']:.2f}x  "
           f"zoom_focal={res['zoom_focal']:.2f}x  verdict={res['report'].verdict}")
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(panel, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, panel)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--pairs", default="646:647,605:606",
                    help="comma list of wide:narrow frame pairs")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    try:
        model = load_model(device)
    except RuntimeError as e:
        print(f"[load on {device} failed: {str(e)[:80]}] -> CPU")
        device = "cpu"
        model = load_model(device)

    for spec in args.pairs.split(","):
        fa, fb = (int(x) for x in spec.split(":"))
        wide, narrow = grab(args.video, fa), grab(args.video, fb)
        try:
            res = measure_zoom(wide, narrow, model, device, args.size, fa, fb)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM on cuda@{args.size}] retrying CPU@512")
            torch.cuda.empty_cache()
            model = load_model("cpu")
            device = "cpu"
            res = measure_zoom(wide, narrow, model, device, 512, fa, fb)

        # contrast with the original ORB failure
        _, _, orb_inl = register_pair(wide, narrow)
        rep = res["report"]
        print(f"\n=== {fa} -> {fb} ===")
        print(f"  MASt3R correspondences : {res['n_corr']}   (ORB register_pair inliers: {orb_inl})")
        print(f"  zoom (spread ratio)    : {res['zoom_spread']:.2f}x   <- robust, model-free")
        print(f"  zoom (similarity)      : {res['zoom_sim']:.2f}x   ({res['n_inliers']} inliers)")
        print(f"  zoom (focal ratio)     : {res['zoom_focal']:.2f}x")
        print(f"  zoom (H @ FoE)         : {rep.zoom:.2f}x   FoE={None if not rep.foe_xy else tuple(round(v) for v in rep.foe_xy)}")
        print(f"  verdict                : {rep.verdict}   (H_in={rep.n_inliers_H} F_in={rep.n_inliers_F} parallax={len(rep.parallax_pts_a)})")
        _draw(wide, narrow, res, f"{args.out}/mast3r_zoom_{fa}_{fb}.png")
        cv2.imwrite(f"{args.out}/mast3r_geom_{fa}_{fb}.png", draw_zoom_report(wide, narrow, rep))
        print(f"  viz -> {args.out}/mast3r_zoom_{fa}_{fb}.png , mast3r_geom_{fa}_{fb}.png")


if __name__ == "__main__":
    main()
