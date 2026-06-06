"""Video discontinuity + loop-closure analysis.

Three complementary questions about a clip, each answered by a different signal:

1. "Did the camera *jump*?"  -> consecutive-frame homography.
   We match ORB features between frame t-1 and t and fit a homography with
   RANSAC. If the frames are continuous, most matches are geometric inliers and
   the previous frame warps onto the current one with high correlation. A hard
   cut or a large jump makes the homography collapse: few inliers, low post-warp
   correlation. This is geometry-based and survives ordinary panning/zoom that a
   raw pixel diff would over-trigger on.

2. "Did the colour change?" -> HSV-histogram correlation + channel spread
   (delegated to scene_cut.SceneCutDetector). Catches the thermal->colour flip
   and abrupt global tone changes that homography alone is blind to (a grayscale
   and a colour frame of the *same* scene still align geometrically).

3. "Does the end revisit the middle?" -> DINOv3 global descriptors.
   We embed every frame with a frozen DINOv3 backbone, L2-normalise, and compare
   late frames against earlier ones by cosine similarity (with a temporal gap so
   adjacency is not mistaken for a loop). A bright off-diagonal in the
   self-similarity matrix = a place that was visited mid-clip and came back at
   the end. This is appearance-based place recognition, not tracking.

Run:
    python3 src/scene_analysis.py --video data/source.mp4 --out out/
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from transformers import AutoModel

from scene_cut import SceneCutDetector


# --------------------------------------------------------------------------- #
# 1. Consecutive-frame homography                                             #
# --------------------------------------------------------------------------- #
@dataclass
class PairAlign:
    """Geometric relationship between frame t-1 and t."""

    frame_idx: int          # index of the *current* frame t
    n_matches: int          # good ORB matches found
    n_inliers: int          # RANSAC geometric inliers
    inlier_ratio: float     # inliers / matches (0 if no matches)
    ncc_raw: float          # corr of grays before alignment
    ncc_aligned: float      # corr after warping t-1 onto t (NaN if no homography)
    align_ok: bool          # a usable homography was found
    H: "np.ndarray | None" = None   # homography mapping prev-image -> curr-image coords


class FrameAligner:
    """Estimates a homography between consecutive frames and scores the fit.

    Parameters
    ----------
    downscale:
        Long edge (px) frames are resized to before feature extraction. ORB is
        cheap; this just keeps it bounded and stable across resolution changes.
    n_features:
        ORB keypoint budget per frame.
    min_inliers:
        Fewer RANSAC inliers than this => treat the homography as failed
        (a jump/cut), regardless of ratio.
    """

    def __init__(self, downscale: int = 480, n_features: int = 1500,
                 min_inliers: int = 12) -> None:
        self.downscale = downscale
        self.min_inliers = min_inliers
        self._orb = cv2.ORB_create(nfeatures=n_features)
        # crossCheck gives symmetric, low-false-positive matches without a ratio test.
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: np.ndarray | None = None
        self._prev_kp = None
        self._prev_des = None

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.downscale / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Zero-mean normalised cross-correlation over valid pixels."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        if mask is not None:
            sel = mask > 0
            if sel.sum() < 64:
                return float("nan")
            a, b = a[sel], b[sel]
        a -= a.mean()
        b -= b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 1e-6 else float("nan")

    def update(self, frame: np.ndarray, frame_idx: int) -> PairAlign | None:
        """Feed the next frame; returns the alignment vs the previous one."""
        small = self._resize(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)

        result: PairAlign | None = None
        if self._prev_gray is not None:
            n_matches = n_inliers = 0
            ncc_aligned = float("nan")
            H = None
            if des is not None and self._prev_des is not None:
                matches = self._bf.match(self._prev_des, des)
                n_matches = len(matches)
                if n_matches >= 4:
                    src = np.float32([self._prev_kp[m.queryIdx].pt for m in matches])
                    dst = np.float32([kp[m.trainIdx].pt for m in matches])
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                    if H is not None and mask is not None:
                        n_inliers = int(mask.sum())
            ratio = n_inliers / n_matches if n_matches else 0.0
            align_ok = H is not None and n_inliers >= self.min_inliers
            if align_ok:
                warped = cv2.warpPerspective(self._prev_gray, H,
                                             (gray.shape[1], gray.shape[0]))
                valid = cv2.warpPerspective(np.ones_like(self._prev_gray), H,
                                            (gray.shape[1], gray.shape[0]))
                ncc_aligned = self._ncc(warped, gray, valid)
            ncc_raw = self._ncc(self._prev_gray, gray)
            result = PairAlign(frame_idx, n_matches, n_inliers, ratio,
                               ncc_raw, ncc_aligned, align_ok,
                               H if align_ok else None)

        self._prev_gray, self._prev_kp, self._prev_des = gray, kp, des
        return result


def stitch_segments(frames: list[np.ndarray], aligns: list[PairAlign],
                    downscale: int = 480, min_coverage: float = 0.6
                    ) -> tuple[list[np.ndarray], list[int]]:
    """Register every frame into its continuous segment's anchor coordinates.

    Within a continuous run of frames we chain the consecutive homographies to
    map each frame back onto the segment's first frame (the anchor), warping it
    onto that shared canvas -- i.e. we *stitch* the segment. The returned warped
    frames are what the embedder sees, so the DINOv3 match is geometry-normalised
    within a segment.

    A segment ends -- and the anchor resets to the current (raw) frame -- on
    either of two conditions:
      * the consecutive homography failed (a jump/cut), or
      * accumulated drift pushed valid content below `min_coverage` of the
        canvas. Without this guard a long pan chains homographies until the
        warped frame is almost entirely off-canvas (black), and black frames
        embed identically (a spurious cos~1.0). Re-anchoring keeps every
        embedded frame mostly real pixels.

    The aligner estimated H on a `downscale`-px frame, so H is rescaled to the
    full-resolution pixel grid before warping (S^-1 H S with S the downscale).
    """
    h, w = frames[0].shape[:2]
    s = min(1.0, downscale / max(h, w))
    S = np.diag([s, s, 1.0])
    S_inv = np.diag([1.0 / s, 1.0 / s, 1.0])
    ones = np.ones((h, w), np.float32)
    by_idx = {a.frame_idx: a for a in aligns}

    warped = [frames[0].copy()]
    anchors = [0]
    M = np.eye(3)          # anchor-image -> current-image, in full-res coords
    anchor = 0
    for j in range(1, len(frames)):
        a = by_idx.get(j)
        H_full = S_inv @ a.H @ S if (a is not None and a.align_ok and a.H is not None) else None
        reset = H_full is None
        if not reset:
            M_try = H_full @ M
            try:
                inv = np.linalg.inv(M_try)
            except np.linalg.LinAlgError:
                reset = True
            else:
                wj = cv2.warpPerspective(frames[j], inv, (w, h))
                coverage = float(cv2.warpPerspective(ones, inv, (w, h)).mean())
                if coverage < min_coverage:
                    reset = True
                else:
                    M = M_try
        if reset:                                 # new segment: anchor on raw frame
            M, anchor, wj = np.eye(3), j, frames[j].copy()
        warped.append(wj)
        anchors.append(anchor)
    return warped, anchors


# --------------------------------------------------------------------------- #
# 2. DINOv3 global descriptors                                                #
# --------------------------------------------------------------------------- #
class Dinov3Embedder:
    """Frozen DINOv3 backbone producing one L2-normalised vector per frame.

    Default is the ConvNeXt-Tiny variant: 768-d descriptors at ~600 crops/s on a
    4 GB laptop GPU (see reid_bench.py), the best speed/quality point measured.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 repo: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                 size: int = 224, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.size = size
        self.model = AutoModel.from_pretrained(repo, dtype=self.dtype).to(self.device).eval()
        self._mean = torch.tensor(self.IMAGENET_MEAN, device=self.device,
                                  dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD, device=self.device,
                                 dtype=self.dtype).view(1, 3, 1, 1)

    def _pool(self, out) -> torch.Tensor:
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        h = out.last_hidden_state
        return h.mean(dim=(2, 3)) if h.dim() == 4 else h[:, 0]

    @torch.inference_mode()
    def embed(self, frames_bgr: list[np.ndarray], batch: int = 32) -> np.ndarray:
        """Embed a list of BGR frames -> (N, D) L2-normalised float32 array."""
        vecs = []
        for i in range(0, len(frames_bgr), batch):
            chunk = frames_bgr[i:i + batch]
            t = np.stack([
                cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (self.size, self.size),
                           interpolation=cv2.INTER_AREA)
                for f in chunk
            ]).astype(np.float32) / 255.0
            x = torch.from_numpy(t).permute(0, 3, 1, 2).to(self.device, self.dtype)
            x = (x - self._mean) / self._std
            v = self._pool(self.model(pixel_values=x)).float()
            v = torch.nn.functional.normalize(v, dim=1)
            vecs.append(v.cpu().numpy())
        return np.concatenate(vecs, axis=0)


# --------------------------------------------------------------------------- #
# 3. Loop closure                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class FrameMatch:
    """Result of matching frame `j` against its entire history (frames < j)."""

    j: int           # the frame being matched
    nn_i: int        # most similar earlier frame, any gap (-1 for j==0)
    nn_sim: float    # its cosine similarity
    loop_i: int      # most similar earlier frame at least min_gap back (-1 if none)
    loop_sim: float  # its cosine similarity (NaN if no eligible earlier frame)


@dataclass
class LoopMatch:
    """A frame `j` re-observing an earlier, non-adjacent frame `i`."""

    i: int          # earlier (revisited) frame
    j: int          # later frame that revisits it
    sim: float      # cosine similarity of DINOv3 descriptors


def causal_match_log(emb: np.ndarray, fps: float,
                     min_gap_s: float = 5.0) -> tuple[list[FrameMatch], np.ndarray]:
    """Match every frame against all the frames before it.

    Returns one FrameMatch per frame plus the full NxN cosine similarity matrix
    (descriptors are already L2-normalised, so the Gram matrix *is* cosine).
    Two earlier-neighbours are reported per frame:

      * `nn_*`  -- nearest earlier frame with no gap constraint. This is the
        running "have I seen anything like this just now" signal.
      * `loop_*`-- nearest earlier frame at least `min_gap_s` back. A strong
        value here is a genuine loop closure (a revisit), not mere persistence
        of a near-static scene across a few frames.

    This is the per-frame, against-all-history log: causal (frame j only ever
    looks backwards), so it is exactly what an online pass would accumulate.
    """
    n = len(emb)
    gap = int(min_gap_s * fps)
    sims = emb @ emb.T
    rows: list[FrameMatch] = []
    for j in range(n):
        if j == 0:
            rows.append(FrameMatch(0, -1, float("nan"), -1, float("nan")))
            continue
        nn_i = int(np.argmax(sims[j, :j]))
        nn_sim = float(sims[j, nn_i])
        hi = j - gap
        if hi > 0:
            loop_i = int(np.argmax(sims[j, :hi]))
            loop_sim = float(sims[j, loop_i])
        else:
            loop_i, loop_sim = -1, float("nan")
        rows.append(FrameMatch(j, nn_i, nn_sim, loop_i, loop_sim))
    return rows, sims


def find_loop_closures(matches: list[FrameMatch], fps: float,
                       sim_thresh: float = 0.6, min_gap_s: float = 5.0,
                       top_k: int = 8) -> list[LoopMatch]:
    """Strongest distinct loop closures over the *whole* clip.

    Scans every frame's `loop_*` match (not just the tail), keeps those above
    `sim_thresh`, and de-duplicates so a revisit lasting many frames reports
    once rather than for every frame it spans.
    """
    gap = int(min_gap_s * fps)
    cands = [LoopMatch(m.loop_i, m.j, m.loop_sim) for m in matches
             if m.loop_i >= 0 and m.loop_sim >= sim_thresh]
    cands.sort(key=lambda m: m.sim, reverse=True)
    kept: list[LoopMatch] = []
    for m in cands:
        if all(abs(m.i - k.i) > gap or abs(m.j - k.j) > gap for k in kept):
            kept.append(m)
        if len(kept) >= top_k:
            break
    return kept


# --------------------------------------------------------------------------- #
# Orchestration + reporting                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    fps: float
    aligns: list[PairAlign] = field(default_factory=list)
    color_events: list = field(default_factory=list)
    emb: np.ndarray | None = None
    sims: np.ndarray | None = None
    matches: list[FrameMatch] = field(default_factory=list)
    loops: list[LoopMatch] = field(default_factory=list)


def analyze(video: str, embed_stride: int = 1,
            jump_ncc_thresh: float = 0.3
            ) -> tuple[Analysis, list[np.ndarray], list[np.ndarray]]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    aligner = FrameAligner()
    cutter = SceneCutDetector()
    out = Analysis(fps=fps)
    frames: list[np.ndarray] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        a = aligner.update(frame, idx)
        if a is not None:
            out.aligns.append(a)
        ev = cutter.update(frame, idx)
        if ev is not None:
            out.color_events.append(ev)
        idx += 1
    cap.release()

    warped, _ = stitch_segments(frames, out.aligns, aligner.downscale)
    embedder = Dinov3Embedder()
    out.emb = embedder.embed(warped[::embed_stride])
    fps_eff = fps / max(1, embed_stride)
    out.matches, out.sims = causal_match_log(out.emb, fps_eff)
    out.loops = find_loop_closures(out.matches, fps_eff)
    return out, frames, warped


def _is_jump(a: PairAlign, ncc_thresh: float) -> bool:
    return (not a.align_ok) or (not np.isnan(a.ncc_aligned) and a.ncc_aligned < ncc_thresh)


def register_pair(a: np.ndarray, b: np.ndarray, n_features: int = 4000,
                  ratio: float = 0.75) -> tuple["np.ndarray | None", int, int]:
    """Direct ORB+RANSAC homography mapping image `a` -> image `b`.

    Unlike the consecutive-frame aligner this matches two arbitrary frames, so it
    can register a loop-closure pair across the scale/viewpoint gap between them.
    8 pyramid levels give ORB ~3.6x of scale invariance. Returns
    (H, n_matches, n_inliers); H is None if too few matches survive.
    """
    orb = cv2.ORB_create(nfeatures=n_features, nlevels=8, scaleFactor=1.2)
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None, 0, 0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for pr in knn if len(pr) == 2 for m, n in [pr] if m.distance < ratio * n.distance]
    if len(good) < 4:
        return None, len(good), 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, len(good), (int(mask.sum()) if mask is not None else 0)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a caption onto a dark strip at the top of a tile."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def report(out: Analysis, frames: list[np.ndarray],
           warped: list[np.ndarray], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fps = out.fps

    # ---- persist embeddings + the per-frame, against-all-history match log ----
    emb_path = os.path.join(out_dir, "embeddings.npy")
    sims_path = os.path.join(out_dir, "similarity_matrix.npy")
    log_path = os.path.join(out_dir, "match_log.csv")
    np.save(emb_path, out.emb)
    np.save(sims_path, out.sims)
    align_by_idx = {a.frame_idx: a for a in out.aligns}
    with open(log_path, "w") as fh:
        fh.write("frame,time_s,nn_prev,nn_sim,loop_prev,loop_sim,"
                 "homography_ok,inlier_ratio,aligned_ncc\n")
        for m in out.matches:
            a = align_by_idx.get(m.j)
            ok = "" if a is None else int(a.align_ok)
            ratio = "" if a is None else f"{a.inlier_ratio:.4f}"
            ncc = "" if a is None else f"{a.ncc_aligned:.4f}"
            fh.write(f"{m.j},{m.j/fps:.3f},{m.nn_i},{m.nn_sim:.4f},"
                     f"{m.loop_i},{m.loop_sim:.4f},{ok},{ratio},{ncc}\n")
    print(f"\nEmbeddings -> {emb_path}  shape={out.emb.shape}")
    print(f"Similarity matrix -> {sims_path}  shape={out.sims.shape}")
    print(f"Per-frame match log (each frame vs all previous) -> {log_path}")

    idx = np.array([a.frame_idx for a in out.aligns])
    inlier = np.array([a.inlier_ratio for a in out.aligns])
    ncc = np.array([a.ncc_aligned for a in out.aligns])
    jumps = [a for a in out.aligns if _is_jump(a, 0.3)]

    print(f"\n=== {len(out.aligns)+1} frames @ {fps:.1f} fps "
          f"({(len(out.aligns)+1)/fps:.1f}s) ===")
    print(f"\nLarge jumps / cuts (homography): {len(jumps)}")
    for a in jumps:
        why = "no homography" if not a.align_ok else f"aligned NCC={a.ncc_aligned:.2f}"
        print(f"  frame {a.frame_idx:4d} (t={a.frame_idx/fps:5.2f}s)  "
              f"matches={a.n_matches:3d} inliers={a.n_inliers:3d} "
              f"ratio={a.inlier_ratio:.2f}  {why}")

    print(f"\nColour events: {len(out.color_events)}")
    for ev in out.color_events:
        print(f"  frame {ev.frame_idx:4d} (t={ev.frame_idx/fps:5.2f}s)  "
              f"{ev.kind:16s} hist_corr={ev.hist_corr:.2f} is_color={ev.is_color}")

    print(f"\nLoop closures (end revisits earlier scene): {len(out.loops)}")
    for m in out.loops:
        print(f"  end frame {m.j:4d} (t={m.j/fps:5.2f}s)  ==  "
              f"earlier frame {m.i:4d} (t={m.i/fps:5.2f}s)   cos={m.sim:.3f}")

    # ---- figure: signals + self-similarity matrix ----
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    ax[0].plot(idx / fps, inlier, label="inlier ratio", lw=1)
    ax[0].plot(idx / fps, ncc, label="aligned NCC", lw=1, alpha=0.8)
    for a in jumps:
        ax[0].axvline(a.frame_idx / fps, color="r", ls="--", alpha=0.5)
    ax[0].set_title("Homography alignment (red = detected jump)")
    ax[0].set_ylabel("score"); ax[0].set_ylim(-0.1, 1.1); ax[0].legend(loc="lower left")

    ax[1].plot(idx / fps, ncc, color="gray", lw=0.8, label="aligned NCC (ref)")
    for ev in out.color_events:
        c = "m" if ev.kind == "color_mode_flip" else "orange"
        ax[1].axvline(ev.frame_idx / fps, color=c, ls="--", alpha=0.7)
    ax[1].set_title("Colour events (magenta = thermal/colour flip, orange = hard cut)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("score"); ax[1].legend(loc="lower left")

    im = ax[2].imshow(out.sims, cmap="viridis", origin="lower",
                      extent=[0, len(out.emb) / fps, 0, len(out.emb) / fps])
    for m in out.loops:
        ax[2].plot(m.i / fps, m.j / fps, "rx", ms=10, mew=2)
    ax[2].set_title("DINOv3 self-similarity (bright off-diagonal = revisited scene; "
                    "red x = loop closure)")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("time (s)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "scene_analysis.png")
    fig.savefig(fig_path, dpi=110)
    print(f"\nFigure -> {fig_path}")

    # ---- labelled triptych of the strongest loop closure ----
    # earlier frame | later frame | earlier registered onto later (scale removed),
    # so the revisit is verifiable at a single, common zoom.
    if out.loops:
        m = out.loops[0]
        a, b = frames[m.i], frames[m.j]
        h = min(a.shape[0], b.shape[0])

        def fit(img):
            return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))

        H, _, inl = register_pair(a, b)
        if H is not None:
            reg = cv2.warpPerspective(a, H, (b.shape[1], b.shape[0]))
            reg_tile = _label(fit(reg), f"earlier->later, scale removed ({inl} inliers)")
        else:
            reg_tile = _label(fit(np.zeros_like(b)), "no pairwise homography")

        panel = np.hstack([
            _label(fit(a), f"earlier  t={m.i/fps:.2f}s  frame {m.i}"),
            _label(fit(b), f"later  t={m.j/fps:.2f}s  frame {m.j}"),
            reg_tile,
        ])
        pair_path = os.path.join(out_dir, "loop_closure_pair.png")
        cv2.imwrite(pair_path, panel)
        print(f"Top loop-closure pair (t={m.i/fps:.2f}s vs t={m.j/fps:.2f}s, "
              f"cos={m.sim:.3f}, {inl} pairwise inliers) -> {pair_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/source.mp4")
    ap.add_argument("--out", default="out")
    ap.add_argument("--stride", type=int, default=1, help="embed every Nth frame")
    args = ap.parse_args()
    out, frames, warped = analyze(args.video, embed_stride=args.stride)
    report(out, frames, warped, args.out)


if __name__ == "__main__":
    main()
