"""Homography from DINOv3 dense features, benchmarked against ORB.

Classic homography matches hand-crafted keypoints (ORB) then fits with RANSAC.
That chain breaks when appearance changes (thermal<->colour) or scale exceeds the
detector's pyramid (~3.5x for ORB, measured in test_scale_warp.py). DINOv3 patch
features are learned and semantic, so they correspond across those gaps.

Pipeline (the DINOv3 half):
  1. Resize each frame to a multiple of the patch size; run the ViT backbone.
  2. Take the patch tokens (drop CLS + register tokens) -> a (gh, gw, D) grid of
     L2-normalised descriptors, one per 16x16 px cell.
  3. Mutual nearest-neighbour in cosine space + a similarity floor -> putative
     correspondences (patch centres mapped back to original pixel coords).
  4. cv2.findHomography(RANSAC) -- identical to the ORB tail.

Caveat: correspondences live on a 16 px grid, so the homography is coarse
(robust, not pixel-perfect). Raise --longside for a finer grid.

Usage:
    python3 src/dinov3_match.py --i 632 --j 1094
    python3 src/dinov3_match.py --i 632 --j 1094 --scale-j 4.0   # synthetic stress
    python3 src/dinov3_match.py --i 632 --j 990  --gray-j        # appearance flip
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from transformers import AutoModel

from scene_analysis import register_pair
from test_scale_warp import grab, implied_scale

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DenseMatcher:
    def __init__(self, repo: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
                 longside: int = 512, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(repo, dtype=self.dtype).to(self.device).eval()
        self.patch = getattr(self.model.config, "patch_size", 16)
        self.longside = longside
        self._mean = torch.tensor(MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

    def _snap(self, n: int) -> int:
        return max(self.patch, round(n / self.patch) * self.patch)

    @torch.inference_mode()
    def features(self, frame_bgr: np.ndarray):
        """Return (feat[gh*gw, D] normalised, gh, gw, sx, sy) for one frame.

        sx, sy map a resized pixel back to original-image pixels.
        """
        h0, w0 = frame_bgr.shape[:2]
        s = self.longside / max(h0, w0)
        W, H = self._snap(int(w0 * s)), self._snap(int(h0 * s))
        rgb = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (W, H),
                         interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        x = (x.to(self.device, self.dtype) - self._mean) / self._std
        lhs = self.model(pixel_values=x).last_hidden_state[0]   # (T, D)
        gh, gw = H // self.patch, W // self.patch
        feat = lhs[-gh * gw:]                                   # drop CLS + registers
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.cpu().numpy(), gh, gw, w0 / W, h0 / H

    def match(self, a: np.ndarray, b: np.ndarray, sim_floor: float = 0.5):
        """Mutual-NN patch correspondences in original-image pixel coords."""
        fa, gha, gwa, sxa, sya = self.features(a)
        fb, ghb, gwb, sxb, syb = self.features(b)
        sims = fa @ fb.T                                        # (Na, Nb) cosine
        a2b = sims.argmax(axis=1)
        b2a = sims.argmax(axis=0)
        pts_a, pts_b = [], []
        for i, j in enumerate(a2b):
            if b2a[j] == i and sims[i, j] >= sim_floor:
                ra, ca = divmod(i, gwa)
                rb, cb = divmod(int(j), gwb)
                pts_a.append([(ca + 0.5) * self.patch * sxa, (ra + 0.5) * self.patch * sya])
                pts_b.append([(cb + 0.5) * self.patch * sxb, (rb + 0.5) * self.patch * syb])
        return np.float32(pts_a), np.float32(pts_b)

    def homography(self, a: np.ndarray, b: np.ndarray):
        pa, pb = self.match(a, b)
        if len(pa) < 4:
            return None, len(pa), 0
        H, mask = cv2.findHomography(pa.reshape(-1, 1, 2), pb.reshape(-1, 1, 2),
                                     cv2.RANSAC, 3.0)
        return H, len(pa), (int(mask.sum()) if mask is not None else 0)


def _overlay(a: np.ndarray, b: np.ndarray, H, label: str) -> np.ndarray:
    out = b.copy()
    if H is not None:
        warped = cv2.warpPerspective(a, H, (b.shape[1], b.shape[0]))
        valid = cv2.warpPerspective(np.ones(a.shape[:2], np.uint8), H,
                                    (b.shape[1], b.shape[0])) > 0
        out[valid] = (0.5 * b[valid] + 0.5 * warped[valid]).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--i", type=int, default=632)
    ap.add_argument("--j", type=int, default=1094)
    ap.add_argument("--scale-j", type=float, default=1.0, help="shrink frame j by this")
    ap.add_argument("--gray-j", action="store_true", help="grayscale frame j (appearance flip)")
    ap.add_argument("--invert-j", action="store_true",
                    help="photometric negative of frame j (thermal/visible polarity stand-in)")
    ap.add_argument("--longside", type=int, default=512)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    a = grab(args.video, args.i)
    b = grab(args.video, args.j)
    if args.scale_j != 1.0:
        h, w = b.shape[:2]
        small = cv2.resize(b, (int(w / args.scale_j), int(h / args.scale_j)))
        b = cv2.copyMakeBorder(small, 0, h - small.shape[0], 0, w - small.shape[1],
                               cv2.BORDER_CONSTANT)
    if args.gray_j:
        b = cv2.cvtColor(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if args.invert_j:
        b = cv2.bitwise_not(b)

    tag = f"frame {args.i} -> frame {args.j}"
    if args.scale_j != 1.0:
        tag += f"  (j shrunk {args.scale_j}x)"
    if args.gray_j:
        tag += "  (j grayscaled)"
    if args.invert_j:
        tag += "  (j inverted)"
    print(f"\n{tag}\n{'-'*len(tag)}")

    Ho, no, io = register_pair(a, b)
    so = implied_scale(Ho, b.shape[1], b.shape[0]) if Ho is not None else float("nan")
    print(f"ORB     : matches={no:5d}  inliers={io:5d}  scale={so:5.2f}  "
          f"{'OK' if io >= 15 else 'FAIL'}")

    dm = DenseMatcher(longside=args.longside)
    Hd, nd, idl = dm.homography(a, b)
    sd = implied_scale(Hd, b.shape[1], b.shape[0]) if Hd is not None else float("nan")
    print(f"DINOv3  : matches={nd:5d}  inliers={idl:5d}  scale={sd:5.2f}  "
          f"{'OK' if idl >= 15 else 'FAIL'}")

    panel = np.hstack([_overlay(a, b, Ho, f"ORB  {io} inl"),
                       _overlay(a, b, Hd, f"DINOv3  {idl} inl")])
    path = f"{args.out}/dinov3_vs_orb.png"
    cv2.imwrite(path, panel)
    print(f"overlays [ORB | DINOv3] -> {path}")


if __name__ == "__main__":
    main()
