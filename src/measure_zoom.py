"""Measure the exact zoom factor across the frame 646 -> 647 jump.

A direct homography fails because the scale change is beyond ORB's pyramid, and a
feature scale-sweep also fails because the zoomed view maps to a tiny patch of the
wide frame (too few keypoints there). So we measure scale two independent ways
that do not rely on keypoint matching:

1. Multi-scale template matching (primary). After a Z x zoom, the entire post
   frame is a magnified sub-window of the pre frame. Shrink the post frame by a
   trial factor f and slide it over the pre frame (normalised cross-correlation);
   the f with the highest correlation peak is the zoom, and the peak location is
   *where* the camera zoomed in. Because post's full width W maps to a box of
   width W/f inside pre, the zoom factor is exactly f.

2. Fourier-Mellin (cross-check). Phase-correlate the log-polar-resampled FFT
   magnitudes; a global scale shows up as a shift along the log-radius axis. FFT
   magnitude is translation-invariant, so an off-centre zoom does not fool it.

Outputs the numbers and a figure: correlation-vs-f curve + the wide frame with
the located zoom box drawn on it (box width / frame width = 1 / zoom).
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from test_scale_warp import grab


def upscale_sweep(pre, post, factors):
    """Match the full zoomed frame (fixed-size template) against the wide frame
    upscaled by each f. No small-template bias: the template is always the whole
    post frame. Returns rows (f, peak_corr, loc_xy_in_pre, box_wh_in_pre).

    At the true zoom Z, upscaling pre by Z makes its content the same size as the
    zoomed frame, so the template matches strongly -> zoom = f at the peak.
    """
    pre_g = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_g = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
    th, tw = post_g.shape
    rows = []
    for f in factors:
        up = cv2.resize(pre_g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        if up.shape[0] < th or up.shape[1] < tw:
            continue
        res = cv2.matchTemplate(up, post_g, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        # back to original pre coords: divide by f
        rows.append((float(f), float(peak),
                     (loc[0] / f, loc[1] / f), (tw / f, th / f)))
    return rows


def fourier_mellin_scale(a, b):
    """Global scale of b relative to a via log-polar phase correlation of |FFT|."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = ga.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    fa = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(ga * win))))
    fb = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(gb * win))))
    center = (w / 2.0, h / 2.0)
    max_r = min(center)
    flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG
    pa = cv2.warpPolar(fa, (w, h), center, max_r, flags)
    pb = cv2.warpPolar(fb, (w, h), center, max_r, flags)
    (dx, _), resp = cv2.phaseCorrelate(pa, pb)
    klog = w / np.log(max_r)
    return float(np.exp(dx / klog)), float(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--pre", type=int, default=646)
    ap.add_argument("--post", type=int, default=647)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    pre = grab(args.video, args.pre)
    post = grab(args.video, args.post)

    coarse = upscale_sweep(pre, post, np.arange(1.0, 10.1, 0.5))
    f0 = max(coarse, key=lambda r: r[1])[0]
    fine = upscale_sweep(pre, post, np.arange(max(1.0, f0 - 0.5), f0 + 0.55, 0.1))
    f_ref, peak, loc, box = max(fine, key=lambda r: r[1])
    zoom = f_ref

    fm, fm_resp = fourier_mellin_scale(pre, post)
    fm = max(fm, 1.0 / fm)

    print(f"\nZoom across frame {args.pre} -> {args.post}")
    print(f"  upscale template match: peak corr {peak:.3f} at f={f_ref:.2f}")
    print(f"  ==> measured zoom     : {zoom:.2f}x")
    print(f"  Fourier-Mellin (xchk) : {fm:.2f}x  (response {fm_resp:.3f})")

    # ---- figure: curve | box on wide frame | post vs the matched wide region ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = [r[0] for r in coarse]
    cs = [r[1] for r in coarse]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(fs, cs, "-o", ms=3)
    ax[0].axvline(zoom, color="r", ls="--", label=f"peak {zoom:.2f}x")
    ax[0].set_xlabel("upscale factor f on wide frame")
    ax[0].set_ylabel("peak normalised cross-correlation")
    ax[0].set_title("Correlation peaks at the true zoom")
    ax[0].legend()

    x, y = loc
    bw, bh = box
    canvas = pre.copy()
    cv2.rectangle(canvas, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 0, 255), 2)
    ax[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"wide frame {args.pre}: red box = footprint of frame {args.post}")
    ax[1].axis("off")

    crop = pre[int(y):int(y + bh), int(x):int(x + bw)]
    crop = cv2.resize(crop, (post.shape[1], post.shape[0])) if crop.size else post * 0
    ax[2].imshow(cv2.cvtColor(np.hstack([post, crop]), cv2.COLOR_BGR2RGB))
    ax[2].set_title(f"frame {args.post}  |  wide box upscaled {zoom:.1f}x (should match)")
    ax[2].axis("off")
    fig.tight_layout()
    path = f"{args.out}/zoom_measure.png"
    fig.savefig(path, dpi=120)
    print(f"  figure -> {path}")


if __name__ == "__main__":
    main()
