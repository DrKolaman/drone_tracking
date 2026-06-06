"""Blur profile across the whole clip, per segment.

Sharpness = variance of the Laplacian (high = sharp, low = blurred) -- the same
metric build_map uses to skip blurred frames. Measured on the colour-normalised
(to_bw) frame so red-thermal frames are compared on their real signal, not on a
luminance-crushed grayscale. Also reports lapvar normalised by image variance
(relative sharpness), which is less sensitive to per-scene contrast.

    python3 src/blur_profile.py
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from colorfix import to_bw

SEGMENTS = [
    ("0-609 continuous", 0, 609),
    ("610-614 motion blur", 610, 614),
    ("615-646 static", 615, 646),
    ("647-744 zoom", 647, 744),
    ("745-956 jump (seg2)", 745, 956),
    ("957-1029 revisit (red)", 957, 1029),
    ("1030-1193 revisit (b/w)", 1030, 1193),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--hi", type=int, default=1193)
    ap.add_argument("--out", default="out/blur_profile.png")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    lap, rel, idx = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok or idx > args.hi:
            break
        g = cv2.cvtColor(to_bw(f), cv2.COLOR_BGR2GRAY).astype(np.float64)
        lv = cv2.Laplacian(g, cv2.CV_64F).var()
        lap.append(lv)
        rel.append(lv / (g.var() + 1.0))
        idx += 1
    lap = np.array(lap)
    rel = np.array(rel)

    print(f"{'segment':26} {'frames':>7} {'lapvar med':>11} {'lapvar mean':>12} "
          f"{'min':>7} {'rel med':>8}")
    for name, lo, hi in SEGMENTS:
        s = lap[lo:hi + 1]
        r = rel[lo:hi + 1]
        if len(s) == 0:
            continue
        print(f"{name:26} {len(s):>7} {np.median(s):>11.1f} {s.mean():>12.1f} "
              f"{s.min():>7.1f} {np.median(r):>8.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    colors = ["#dddddd", "#ff9999", "#cce5cc", "#cce5ff", "#ffe0b3", "#ffcccc", "#e0e0e0"]
    for (name, lo, hi), c in zip(SEGMENTS, colors):
        for a in ax:
            a.axvspan(lo, hi, color=c, alpha=0.6)
        ax[0].text((lo + hi) / 2, lap.max() * 0.95, name.split()[0], rotation=90,
                   va="top", ha="center", fontsize=7)
    ax[0].plot(lap, lw=0.7, color="k")
    bw = np.concatenate([lap[:957], lap[1030:]])      # B/W-comparable frames only
    med = np.median(bw)
    ax[0].axhline(0.35 * med, color="r", ls="--", lw=1,
                  label=f"blur-skip thresh (0.35*median={0.35*med:.0f})")
    ax[0].set_ylim(0, 170)                             # red revisit is off-scale (colormap)
    ax[0].text(990, 160, "957-1029 red:\noff-scale (colormap)", fontsize=7, ha="center", color="r")
    ax[0].set_ylabel("Laplacian variance (sharpness)")
    ax[0].set_title("Blur profile across the clip (shaded = segments; lower = blurrier)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(rel, lw=0.7, color="b")
    ax[1].set_ylabel("relative sharpness\n(lapvar / image var)")
    ax[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nplot -> {args.out}")


if __name__ == "__main__":
    main()
