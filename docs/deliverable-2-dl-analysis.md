# Deliverable 2 — Deep-Learning Analysis & Literature Survey

**Project:** Detection & Precision Tracking System (`data/source.mp4`, top-down
aerial/drone footage; the target is a *small, low-contrast person*, a few pixels
across, under heavy camera motion).
**Scope of this document:** survey the candidate approaches for the project's
**deep-learning block**, justify a model choice, and document an empirical trial
of a transformer single-object tracker (**SGLATrack**) — including, in detail,
**why it did not work** on this footage.

---

## 1. The DL-block problem, stated honestly

This clip is adversarial in ways that dictate which DL approaches can work:

| Property of the footage | Consequence for the DL block |
|---|---|
| Target is **~20–35 px**, low-contrast, top-down | Generic COCO-trained detectors don't fire; appearance is information-poor. |
| Near-grayscale **thermal**, then a B/W↔red switch | Off-domain vs. RGB training data; colour cues unreliable. |
| **Camera pans/zooms/jumps** (610 blur, 647 ~7× zoom, 745 jump, 957 jump+colour) | Frame-to-frame motion is large; background is non-stationary; homography breaks at the discontinuities. |
| **Bright static landmark** (a "ladder"-shaped structure) | A strong appearance distractor that a template tracker happily locks onto. |
| **No ground-truth labels** | Quantitative accuracy (MOTA/IDF1/HOTA) cannot be measured; we rely on proxies (see Deliverable 3). |

So the DL block must do **spatial reasoning** (find a near-invisible target) and
**temporal reasoning** (hold identity across motion and occlusion) on a target
that is small, off-domain, and surrounded by distractors.

---

## 2. Approaches surveyed

We considered three families. Each is the standard answer to a different framing
of the task.

### 2.1 Approach A — Detection-based tracking-by-detection (the spec baseline)
**Detector (YOLO) + multi-object tracker (ByteTrack / BoT-SORT).**

- **Detector:** Ultralytics YOLOv8 (Jocher et al., 2023) — one-stage CNN detector,
  COCO `person` class.
- **Association:** ByteTrack (Zhang et al., ECCV 2022) associates *every* detection
  box, high- and low-score, via IoU + a Kalman motion model. BoT-SORT (Aharon et
  al., 2022) adds camera-motion compensation and appearance ReID on top.
- **Why it's attractive:** modular, real-time, the reference design for MOT, and
  exactly what the requirements name (FR2/FR3). Detect-and-reset at the
  discontinuities (FR4/FR5) fits naturally.
- **Pros:** re-detects every frame (no unbounded drift); mature tooling; swappable.
- **Cons / limitation here:** it is only as good as the detector. **Empirically,
  YOLOv8-n (COCO) returns _zero_ `person` detections on this target even at
  `conf=0.05`** across the first 60 frames — the target is too small and too
  off-domain (top-down thermal) for a COCO detector. Tracking-by-detection
  therefore **cannot start** without a detector that can see the target. The
  realistic fix is fine-tuning / a small-object regime (see §5), which the brief
  lists as out-of-scope.

### 2.2 Approach B — Transformer single-object tracking (SOT) — *the trial*
**A template-matching ViT tracker initialised with one box on frame 1.**

- **Family:** one-stream ViT trackers — OSTrack (Ye et al., ECCV 2022), MixFormer
  (Cui et al., CVPR 2022) — and the **lightweight UAV line** to which **SGLATrack**
  (GXNU-ZhongLab; checkpoint dir references *CVPR'25*) belongs. SGLATrack uses a
  **distilled DeiT-tiny** backbone (Touvron et al., ICML 2021), template 128², search
  256², a centre head; ~5.7 M params — designed for efficient UAV tracking
  (benchmarks like UAV123, Mueller et al., ECCV 2016).
- **Why it's attractive:** needs no detector (sidesteps §2.1's blocker); a single
  init box starts it; built for UAV/aerial motion; small enough for the 4 GB GPU.
- **Pros:** strong short-term appearance lock; real-time-class; UAV-tuned.
- **Cons / limitation here:** pure SOT has **no re-detection** — once it loses the
  target it cannot recover, and on a low-contrast target it is prone to **locking
  onto a high-contrast distractor**. This is precisely what we observed (§3).

### 2.3 Approach C — Classical ego-motion-compensated motion detection
**Stabilise camera motion, then track what moves independently.**

- **Method:** estimate global motion (ORB + RANSAC homography), warp the previous
  frame into the current one, take the residual = independent motion; pick the
  moving warm blob near a constant-velocity prediction. (Conceptually MOG2 /
  optical-flow foreground detection with ego-motion compensation.)
- **Why it's attractive:** the target's *defining* property here is that it **moves
  independently of the ground**, while the worst distractor (the structure) is
  **static**. Motion is the one cue that separates them — appearance does not.
- **Pros:** reliably *located* the target in our stabilised analysis (the
  independent-motion peak agreed with the visually-confirmed target at frame 0).
- **Cons / limitation here:** not a "deep-learning block"; and as a standalone
  tracker it is **unstable** — homography is poor across the 647 zoom / 745 jump,
  velocity integration can run away, and it wanders onto edges (our offline
  implementation diverged until clamped, and still drifted off-target after the
  clean segment).

---

## 3. The SGLATrack trial — setup and result

We ran SGLATrack end-to-end on the clip to test Approach B empirically.

### 3.1 Setup (reproducible)
- **Model/config:** `sglatrack` / `deit_distilled`, checkpoint `sglatrack_ep0297.pth.tar`
  (≈101 MB) + DeiT-tiny backbone; weights from the authors' Google Drive.
- **Environment:** the upstream repo pins `torch 1.12+cu102`, which **will not run on
  the RTX 3050 (Ampere, CUDA cap 8.6)**; we rebuilt on `torch 2.2.2+cu118`, Python
  3.12, and patched three torch-1.x breakages (`torch._six`, optional `visdom`/
  `tensorboardX`) plus the authors' hardcoded paths. (Headless WSL → custom runner
  instead of the repo's `video_demo.py`, which needs a GUI.)
- **Init box:** YOLO could not provide one (§2.1), so the target was localised by
  ego-motion-compensated motion saliency → `[172, 358, 22, 28]`, visually confirmed
  on the warm blob. The init **target identity was confirmed correct.**
- **Scope:** frames 0–956 (we initially trimmed at 957 on a colour-flip; note the
  *real* clean segment is 0–609 per the updated requirements timeline).
- **Runners:** `docs/sglatrack-trial/run_headless.py` (plain), `run_redetect.py`
  (brightness re-detect), `run_redetect2.py` (motion+warmth re-detect),
  `motion_track.py` (Approach C reference). Confidence dump: `plain_conf.txt`.

### 3.2 Headline result
SGLATrack achieves a **strong lock for only ~5 seconds (≈150 frames)**, then loses
the target and does not recover. Throughput ≈ **14 FPS** on the RTX 3050 (below the
≥25 FPS NFR1 target).

**Peak-response (confidence) over frames 1→956**, 60 bins (`@`=high … `.`=lost):

```
%%%%%#*##-=-::------------=-+=::-:::::-==---=--==---==++----
```

| Confidence stat (frames 1–956) | Value |
|---|---|
| max / mean / median | 0.966 / 0.479 / 0.428 |
| first drop below 0.5 | **frame 107** |
| first sustained drop below 0.3 | frame 155 |
| frames with conf ≥ 0.45 ("locked") | **400 / 956 = 42 %** |

The confidence collapse at ~frame 107–155 **exactly matches** the visual moment the
box leaves the target.

### 3.3 What failed, across configurations

| Configuration | Behaviour |
|---|---|
| Plain SGLATrack, init `[172,358,22,28]` | Lock ~150 frames, then box **explodes** to full-frame (target lost). |
| Bigger init `[168,354,38,42]` | **Identical** failure (~150 frames) — not an init-size problem. |
| + brightness re-detect (`run_redetect.py`) | Box stays compact but drifts onto bright clutter. |
| + motion+warmth re-detect (`run_redetect2.py`) | Holds to ~frame 160, then **locks onto the static "ladder" structure** with *high* confidence. |
| Ego-motion motion tracker (Approach C) | Wanders / hits frame edge; can't hold it either. |

---

## 4. Why SGLATrack did not work — root-cause analysis

1. **Target appearance is information-poor and off-domain.** A ~30 px, low-contrast,
   top-down *thermal* blob gives a 128² template that is mostly upscaled noise.
   SGLATrack was trained on UAV123-style targets (clearly-visible vehicles/people in
   RGB); the distribution gap means its matching head has little discriminative
   signal to lock onto. Enlarging the init box did **not** help — confirming the
   problem is signal content, not crop size.

2. **Appearance-distractor lock with *false confidence*.** The scene contains a
   bright, high-edge static structure. SGLATrack drifts onto it and reports **high
   peak response** there — because it *is* a strong, stable appearance feature. This
   is the critical finding: **a confidence-gated re-detect can never fire**, because
   the tracker is confidently wrong. Only an *independent-motion* check distinguishes
   the moving target from the static distractor — and appearance-only SOT has no such
   check.

3. **The "mean-of-boxes" head amplifies loss into blow-up.** SGLATrack returns the
   *mean* of all predicted boxes. When the score map is peaked (locked) this is fine;
   when it goes diffuse (lost) the mean spreads and the box balloons to full-frame —
   which is the explosion we see after frame ~520.

4. **No re-detection by design.** Pure SOT propagates from frame *t-1*; it has no
   mechanism to re-acquire after loss. On a target that is intermittently invisible,
   that guarantees unbounded drift.

5. **Camera discontinuities defeat naive recovery.** The 647 zoom and 745 jump break
   frame-to-frame correspondence, so even a motion-based re-detector (our attempted
   fix) is unreliable exactly where it is needed most.

**One-line conclusion:** SGLATrack is the *wrong tool* for this clip not because of a
wiring error (its setup, init, and confidence were all verified) but because an
**appearance-based single-object tracker cannot separate a faint moving target from a
bright static distractor**, and has no way to recover once lost.

---

## 5. Model selection & recommendation

Given the evidence, no single off-the-shelf model is sufficient. Ranked recommendation:

1. **Detection-led, motion-validated pipeline (recommended).** A **fine-tuned /
   small-object detector** (e.g. YOLOv8 with **tiled / sliced inference** — SAHI,
   Akyon et al., 2022 — to recover tiny targets) feeding **BoT-SORT** (camera-motion
   compensation matters here), with an **independent-motion gate** to reject static
   distractors and a **detect-and-reset** at the documented discontinuities. This is
   the spec's intended architecture (Approach A) made viable by addressing the
   detector's small-target blind spot.
2. **SGLATrack (or any ViT-SOT) as a short-term *refiner only*,** re-seeded every few
   frames by the detector/motion module — never as the standalone tracker.
3. **Classical ego-motion motion detection (Approach C) as the re-acquisition cue** —
   it is the only thing that reliably *found* the target, even if it is a poor
   standalone tracker.

The honest blocker for (1) is **labels / fine-tuning** (out-of-scope per the brief).
Deliverable 4 (Failure Analysis) and Deliverable 5 (Improvements) carry this forward:
the concrete next step is annotating a few hundred frames of this target to fine-tune
a small-object detector, then validating tracks against independent motion.

---

## 6. References

- Mueller, Smith, Ghanem. *A Benchmark and Simulator for UAV Tracking* (UAV123). ECCV 2016.
- Touvron et al. *Training data-efficient image transformers & distillation through attention* (DeiT). ICML 2021.
- Ye et al. *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework* (OSTrack). ECCV 2022.
- Cui et al. *MixFormer: End-to-End Tracking with Iterative Mixed Attention.* CVPR 2022.
- Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
- Aharon, Orfaig, Bobrovsky. *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* 2022.
- Jocher et al. *Ultralytics YOLOv8.* 2023.
- Akyon et al. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection* (SAHI). ICIP 2022.
- SGLATrack — GXNU-ZhongLab. https://github.com/GXNU-ZhongLab/SGLATrack (checkpoint dir references CVPR'25).

> Note: citation venues for the well-established methods are given as commonly cited;
> the SGLATrack reference is the source repository (verify the formal paper/venue
> before final submission).

---

*Reproduction artifacts:* `docs/sglatrack-trial/` (runner scripts, `plain_conf.txt`
confidence trace, box trajectories). Demo videos (`source_locked_0-155.mp4`,
`source_annotated.mp4`, `source_redetect_full.mp4`) were produced in the throwaway
`sglatrack` worktree and shared separately; they are not committed (size).
