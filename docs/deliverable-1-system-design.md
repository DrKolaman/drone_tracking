# Deliverable 1 — System Design

**Project:** Detection & Precision Tracking System (`data/source.mp4`, top-down
aerial/drone footage; target is a *small, low-contrast person*, ~20–35 px across,
seen from above in near-grayscale thermal video).
**Scope of this document:** high-level architecture, all functional blocks with
inputs/outputs, algorithms, assumptions, trade-offs, and risks/limitations.

---

## 1. Problem statement and design constraints

The clip is adversarial in ways that determine architecture choices:

| Property | Design constraint |
|---|---|
| Target ~20–35 px, low-contrast, top-down nadir | Standard COCO-trained detectors produce zero detections (confirmed empirically — see Deliverable 2). Motion is the reliable detection cue. |
| Near-grayscale thermal; B/W↔red colour switch at frames 957 and 1030 | A naive `BGR2GRAY` crushes red-thermal contrast. Colour normalisation must precede every other step. |
| Camera pans/rotates continuously | Background subtraction requires ego-motion compensation; the tracker must operate in stabilised coordinates. |
| Hard discontinuities: motion-blur burst (610–614), ~3× discrete FOV switch (647), area jump (745), jump-back + colour switch (957), colour switch back (1030) | Frame-to-frame registration collapses. The pipeline must detect these, segment the clip, and handle identity across segment boundaries. |
| No ground-truth labels | Quantitative accuracy (MOTA/IDF1) cannot be verified. Architecture must produce measurable proxies. |

The architecture therefore combines:
1. **Classical ego-motion compensation** (the only reliable way to isolate independent motion at this scale).
2. **Deep-learning appearance embedding** (DINOv3 ViT-S/16) to distinguish the correct target from other moving blobs and re-acquire it across gaps and segment boundaries.
3. **ByteTrack** as a lightweight multi-object bootstrapper / new-object spawner, not as the primary identity manager.

---

## 2. High-level architecture

### 2.1 Primary pipeline: full-clip tracker (`src/track_full_clip.py`)

The tracker is causal (online, left-to-right), operates in **stabilised map (canvas) coordinates**, and produces an annotated MP4 plus an optional per-frame CSV log.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         PASS 1  (chain_segments)                         │
│                                                                          │
│  Raw frames ─► [1] Colour normalisation ─► [2] Global motion estimation │
│                                               │                          │
│                                               │  ok=False → new segment  │
│                                               ▼                          │
│                                        Cumulative H, segment id,         │
│                                        per-frame blur score              │
│                                                                          │
│  [5] Per-segment canvas fit (fit_canvas, two-pass over segment Hs)       │
│      → T_seg, canvas (w × h)  per segment                               │
└──────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         PASS 2  (main loop, per frame)                   │
│                                                                          │
│  Raw frame                                                               │
│      │                                                                   │
│      ▼                                                                   │
│  [1] Colour normalisation                                                │
│      │                                                                   │
│      ▼                                                                   │
│  [4] Blur detection ─── too blurred ──► skip motion detection            │
│      │ sharp                                                             │
│      ▼                                                                   │
│  warpPerspective (T_seg × H_cum) → aligned canvas frame                 │
│      │                                                                   │
│      ▼                                                                   │
│  [6] Coverage-age validity mask update                                   │
│      │                                                                   │
│      ▼                                                                   │
│  [6] MOG2 background subtraction + morphology → candidate blobs         │
│      │                                                                   │
│      ▼                                                                   │
│  [7] DINOv3 appearance embedding  ◄─── DEEP-LEARNING BLOCK              │
│      │  (N × 384 L2-normalised vectors)                                  │
│      │                                                                   │
│      ├────────────────────────────────────────────────────┐             │
│      ▼                                                     ▼             │
│  [10] ByteTrack (bootstrap + spawn)              [9] Multi-identity     │
│       → track_ids, age counters                     association rules   │
│                                                     RULE 1 motion-near  │
│                                                     RULE 2 DINOv3 REACQ │
│  [8] Identity model                                 RULE 3 global re-ID  │
│      TargetMemory (per id)                          RULE 4 spawn NEW id  │
│      BackgroundMemory (shared)                              │             │
│      discriminative margin                                  ▼             │
│                                                    active id, state,     │
│                                                    canvas box            │
│                                                         │                │
│      ▼                                                  │                │
│  [11] Back-project to raw frame + render + write MP4 + log CSV          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Secondary subsystem: scene-map pipeline

A parallel pipeline builds a geographic mosaic for scene understanding across discontinuities. It is not used by the tracker at runtime; it characterises the clip geometry and validates the discontinuity taxonomy. Described briefly in §4 (Block 12).

---

## 3. Functional blocks — detail

### Block 1: Colour normalisation

**Source:** `src/colorfix.py` — `to_bw(bgr)`

| | |
|---|---|
| **Input** | Raw BGR frame from `cv2.VideoCapture` |
| **Output** | 3-channel grayscale frame (dtype uint8, shape H×W×3) |

**Algorithm.** Compute the mean per-pixel channel spread
`spread = mean(|R−G| + |G−B| + |R−B|) / 3`. If `spread ≤ 12` the frame is already
near-grayscale; convert via `BGR2GRAY`. Otherwise (colour-mapped thermal) select the
channel with the highest variance and replicate it to 3 channels. This recovers the
full thermal contrast that `BGR2GRAY` would crush (measured luminance variance drops
from ~4700 to ~100 on a red-thermal frame under `BGR2GRAY`).

**Assumptions.**
- Thermal colourmaps are single-channel intensity mapped to a colour palette, so one
  channel dominates the signal. This holds for the standard red false-colour overlay
  present in this clip.
- Threshold `spread_thresh=12` is a fixed hyperparameter; unusual colourmap choices
  could mis-classify a frame.

**Trade-offs.** The highest-variance-channel heuristic is robust to the standard red
thermal palette but is not a calibrated inverse-colormap. Chromatic aberration or
multi-channel sensors would require a different approach.

**Risks & limitations.** If the thermal sensor switches to a green or blue palette,
the detection heuristic still works (it picks the right channel), but a three-band
additive palette (e.g. RGB gradient) could pick the wrong channel. No such palette
is present in this clip.

---

### Block 2: Global motion estimation / stabilisation

**Source:** `src/registration.py` — `GlobalMotionEstimator.estimate(prev_gray, cur_gray)`

| | |
|---|---|
| **Input** | Two consecutive grayscale frames (prev, current) |
| **Output** | `MotionResult`: 3×3 homography H (current → previous coords), `n_inliers`, `n_matches`, `ok` flag |

**Algorithm.**
1. Detect Shi-Tomasi corners in `prev_gray` (up to 600, quality 0.01, min distance 7 px).
2. Track them into `cur_gray` with Lucas-Kanade pyramidal optical flow (3 pyramid levels, 21×21 window).
3. Fit a homography (8-DOF) from `(cur_pts, prev_pts)` pairs using RANSAC (reprojection threshold 3 px). Optionally a 4-DOF similarity (translation + rotation + uniform scale) can be selected when a nadir flat-ground assumption holds.
4. If fewer than `min_inliers=25` RANSAC inliers survive, return `ok=False` (signals a genuine discontinuity, not a gradual drift).

Cumulative product `H_cum = H_{t-1,cum} × H_{t,t-1}` chains each frame back to the
segment-start reference. A new segment re-anchors `H_cum` to identity.

**Assumptions.**
- The scene is approximately planar (nadir view of flat terrain), so a homography
  or similarity exactly models the camera-induced motion with no parallax.
- Feature-trackable corners exist in both frames. Very uniform terrain or
  dense smoke/haze would leave too few corners.

**Trade-offs.**
- Homography (8-DOF) can absorb shear and perspective — useful for any camera attitude
  deviation from true nadir. The downside is that shear can accumulate over a long pan,
  warping the map into a parallelogram. The similarity (4-DOF) model prevents this but
  assumes strictly uniform scale.
- RANSAC inlier threshold 3 px is tight enough to reject outliers from independently
  moving objects, but loose enough not to reject valid matches from sub-pixel
  registration noise.

**Risks & limitations.**
- The homography model has no depth. For a scene with tall structures (buildings,
  trees) at non-nadir camera angles, parallax would degrade the fit.
- Lucas-Kanade is a local gradient method; it fails when consecutive frames lack
  overlap (large pan) or when motion blur makes gradient computation unreliable.
  This is the intended failure mode that produces `ok=False` and triggers
  segmentation.

---

### Block 3: Discontinuity detection and segmentation

**Source:** `src/track_full_clip.py` — `chain_segments()`

| | |
|---|---|
| **Input** | All frames of the clip (via `VideoCapture`), colour-normalised to grayscale before estimation |
| **Output** | Per-frame cumulative homography list `Hs`, per-frame segment index `segs` (0-based), per-frame blur variance `blur` |

**Algorithm.** `GlobalMotionEstimator.estimate` runs on each consecutive frame pair.
When `ok=False` (inlier count below `min_inliers`), a new segment is started:
`seg += 1`, `H_cum` resets to identity. A new segment does NOT start on canvas overflow
(unlike simpler designs). Result: 4 segments on this clip, separated at the genuine
hard discontinuities.

**Assumptions.**
- Every genuine discontinuity will cause registration collapse (`ok=False`). This
  holds for the ~3× FOV switch at 647 (scale jump too large for LK to bridge) and
  the area jumps at 745 and 957. It may not hold for a very slow zoom or a brief
  cross-dissolve.
- Colour normalisation is applied before registration so the 957 B/W→red switch does
  not cause an artificial feature-tracking failure.

**Trade-offs.** The single criterion (`ok=False`) is simple and avoids false splits
from canvas overflow or short occlusions. The downside is that a very soft transition
(cross-dissolve over 5+ frames) could register weakly and produce one large wrong
segment instead of a clean split.

**Risks & limitations.** No explicit scene-content classifier (histogram correlation,
etc.) is used. A cut that is spectrally abrupt but geometrically coincident with a
large pan could be missed or misfired. No such case appears in this clip.

---

### Block 4: Blur detection and frame skip

**Source:** `src/track_full_clip.py` — inline in `chain_segments` and main loop

| | |
|---|---|
| **Input** | Grayscale frame |
| **Output** | Per-frame variance-of-Laplacian score `blur[i]`; boolean skip flag when `blur[i] < 0.19 × median(blur)` |

**Algorithm.** Variance of the Laplacian (`cv2.Laplacian(g, cv2.CV_64F).var()`) is a
standard sharpness measure. Frames below 19% of the clip median are classified as
motion-blurred and skipped for motion detection (MOG2 is not updated, no blobs are
returned). On this clip this correctly suppresses the 610–612 burst.

**Assumptions.** The typical frame is sharp; the blur threshold 0.19 is well below
the bimodal gap between sharp frames and the motion-blur burst. A clip that is
uniformly slightly blurry would have a low median and the threshold could admit bad
frames.

**Trade-offs.** The threshold is relative to the per-clip median, making it adaptive
to sensor characteristics. It introduces no per-frame tuning but has no per-scene
calibration.

**Risks & limitations.** Fine-grained temporal blur (e.g. rolling-shutter artefacts on
a single row) would not be caught. The variance-of-Laplacian score aggregates over the
whole frame, so a locally-blurred centre but sharp periphery still passes.

---

### Block 5: Per-segment canvas fitting

**Source:** `src/track_dino_reid.py` — `fit_canvas(Hs, w, h)` (also used in `track_full_clip.py`)

| | |
|---|---|
| **Input** | List of cumulative homographies for one segment; frame dimensions `(w, h)` |
| **Output** | Translation matrix `T` (3×3), canvas width `cw`, canvas height `ch` |

**Algorithm.** Project all four corners of the frame through every cumulative homography
in the segment. The canvas bounding box is the union of all projected corners. `T` is
a pure translation that shifts the minimum corner to the canvas origin. A long pan that
would overflow a fixed-size canvas is automatically accommodated; no re-anchoring is
needed mid-segment.

**Assumptions.** The canvas fits in memory. For a 1200-frame clip with a gentle pan
the canvas grows at most ~3–4× the frame dimension. Very long clips with large pans
would require incremental canvas management.

**Trade-offs.** The two-pass design (pass 1 collects all Hs, pass 2 uses the
pre-computed canvas) means the canvas size is exact and no reallocation occurs at
runtime. The cost is that pass 1 must complete before pass 2 starts — acceptable for
a finite-length clip, not suitable for a true real-time unbounded stream.

**Risks & limitations.** Canvas overflow is not possible by construction within a
segment, but memory grows with segment length. No canvas tiling or out-of-core
processing is implemented.

---

### Block 6: Motion detection (MOG2 + coverage-age validity mask)

**Source:** `src/track_dino_reid.py` — `detect(aligned, validity, mog2, k3, min_area, max_area)`

| | |
|---|---|
| **Input** | Stabilised (warped) canvas frame `aligned`; binary validity mask; MOG2 model; structuring elements; area thresholds |
| **Output** | List of candidate bounding boxes `[[x1,y1,x2,y2], ...]`; list of centroids `[[cx,cy], ...]` |

**Algorithm.**

1. **MOG2 background subtraction.** `cv2.createBackgroundSubtractorMOG2` (history=20,
   varThreshold=50, no shadows) models each pixel's background distribution. The
   foreground mask is thresholded at 200.
2. **Coverage-age validity mask.** A pixel counter `coverage[y,x]` is incremented each
   frame the pixel is inside the camera footprint (via the warped frame), reset to zero
   when it leaves. The validity mask accepts only pixels where `coverage ≥ 12`, then
   erodes the mask by a 21×21 elliptical kernel. This suppresses freshly-revealed
   ground at the moving frame edge where MOG2 has not yet built a background model.
   An optional relaxed disc around the predicted target position uses a looser coverage
   threshold (3 frames) to allow detection to survive at the leading edge of a
   follow-pan when the target is briefly missed.
3. **Morphological cleanup.** Open (3×3, 1 iter) to remove noise, close (3×3, 2 iter)
   to merge adjacent blobs, dilate (1 iter) to smooth contours.
4. **Connected-component filtering.** Retain components with `min_area_px=5 ≤ area ≤ 0.05×WH`
   and aspect ratio `0.25 ≤ h/w ≤ 4.0`.

**Assumptions.**
- The dominant scene motion is camera-induced; after stabilisation the only residual
  foreground is genuinely independently moving objects. This holds for a nadir drone
  over terrain with no wind-blown vegetation or other distributed motion.
- The target moves at least occasionally. A stationary target for the entire clip would
  produce no MOG2 foreground.
- MOG2's short history (20 frames, ~0.67 s at 30 fps) means it adapts quickly. A
  target that stops for more than ~20 frames is absorbed into the background model.
  This is why the HOLD mechanism (Block 9) is essential.

**Trade-offs.**
- The 21×21 erosion kernel is generous (masks ~10 px inward) to be conservative.
  This means detection can miss the target at the very edge of a pan even when
  coverage is adequate. The relaxed-disc override partially mitigates this.
- Short MOG2 history improves responsiveness to scene changes but increases
  false-positive rate on slow-varying surfaces (thermal AGC gradients).

**Risks & limitations.**
- A stopped target (HOLD) that the camera pans back to after >20 frames will not
  produce a MOG2 foreground response; re-acquisition then relies entirely on
  DINOv3 (Block 7 / Block 9 Rule 2).
- The 21×21 erosion kernel means a narrow or diagonal pan continuously exposes new
  ground and the valid region may be a thin strip, reducing the detection area.

---

### Block 7: DINOv3 appearance embedding (DEEP-LEARNING BLOCK)

**Source:** `src/dino_embedder.py` — `DinoEmbedder`

| | |
|---|---|
| **Input** | Stabilised canvas frame (BGR); list of bounding boxes `[[x1,y1,x2,y2], ...]` from Block 6 |
| **Output** | `(N, 384)` float32 array of L2-normalised appearance descriptors, one per blob |

**Model.** `facebook/dinov3-vits16-pretrain-lvd1689m` — DINOv3 ViT-S/16, 384-dim
hidden state, patch size 16. Loaded via HuggingFace `transformers.AutoModel` in fp16
on GPU. Access requires `HF_TOKEN` (gated repository). Weights are frozen; no
fine-tuning is performed.

**Algorithm.**
1. **Context padding.** Each bounding box is expanded to a square centred on the box,
   with half-side `max(box_w, box_h, 64) / 2`. This ensures the backbone receives
   enough context around the ~20 px target blob.
2. **Resize and normalise.** Crop is resized to `112×112` (a multiple of patch size 16),
   converted from BGR to RGB, divided by 255, normalised with ImageNet mean/std.
3. **Forward pass.** A batch of all crops is embedded in one GPU call.
   `model(pixel_values=batch).last_hidden_state` → shape `(N, T, 384)`.
4. **Patch-token extraction.** The last `(112/16)² = 49` tokens are taken (CLS token
   and any registers dropped) to get pure spatial patch embeddings.
5. **Mean-pooling + L2 normalisation.** Mean over the 49 patch tokens → `(N, 384)`.
   L2 normalisation makes cosine similarity equal to the dot product.

**Why DINOv3 for ReID.** The target is too small and off-domain for a COCO-trained
object detector (confirmed: YOLOv8-n returns zero detections at conf=0.05). DINOv3
self-supervised pre-training produces strong general-purpose features even for tiny,
off-domain crops; the mean-pooled patch tokens act as a compact global descriptor
for re-identification. This matches the recommendation from Deliverable 2's model
selection analysis.

**Assumptions.**
- The ViT-S/16 backbone produces discriminative descriptors for the target, even
  though it was never trained on aerial thermal targets. Empirically the margin
  between target and background is sufficient (`disc ≥ ~0.096` for the real target,
  `≤ 0` for false blobs — calibrated from this clip).
- The `HF_TOKEN` environment variable is set. The model weights are loaded once at
  startup; no per-frame model loading.
- GPU is available (`cuda`). On CPU the model runs in float32 with a significant
  latency penalty.

**Trade-offs.**
- ViT-S/16 (21 M parameters) is the smallest DINOv3 variant. Using ViT-B or ViT-L
  would give richer embeddings at the cost of more GPU memory and latency.
- Mean-pooling discards spatial token structure. For a larger target (e.g. dense
  matching for loop closure) the full token grid is used (`src/dinov3_match.py`), but
  for ReID of a small blob the global vector is both faster and more robust.
- fp16 inference on the RTX 3050 is fast enough that the DINOv3 embedding call is
  not the pipeline bottleneck at this resolution.

**Risks & limitations.**
- The model is off-domain (pre-trained on natural RGB images; the target is thermal
  near-grayscale). The 3-channel B/W input fed to the model is legitimate but the
  features are less informative than on a camera for which the model was trained.
- Fine-tuning on aerial thermal targets would improve discrimination but is out of
  scope (no annotations available).
- The gated HuggingFace repository requires an accepted access request. If the token
  is absent or expired the pipeline fails at startup.

---

### Block 8: Identity model

**Source:** `src/target_memory.py` — `TargetMemory`, `BackgroundMemory`

| | |
|---|---|
| **Input** | L2-normalised embedding vector `(384,)` per query or update call |
| **Output (score)** | Scalar: `max cosine similarity` over stored bank vectors |
| **Output (disc)** | `target.score(e) − background.score(e)` (discriminative margin) |

**TargetMemory.**
A fixed-capacity bank (default 24) of diverse appearance views of the tracked target.
New embeddings are added only when their best cosine match to the existing bank is
below `add_thresh=0.92` (novelty gate). When the bank is full, the oldest view is
evicted (FIFO). The `consolidate()` method rebuilds the bank greedily from a larger
pool: start with the medoid (most representative view), then iteratively add the most
unique remaining view until all remaining views are near-duplicates (`dedup_thresh=0.95`)
or capacity is reached. The bank is seeded from the best ByteTrack track's embeddings
at bootstrap.

**BackgroundMemory.**
A shared FIFO bank (default 120) of non-target blob embeddings. Light deduplication
avoids redundant storage of identical static-background patches. Score is the same
max-cosine. The background bank reflects the current scene's distractors.

**Discriminative margin.** `disc(e) = target.score(e) − background.score(e)`. A raw
cosine threshold cannot separate the ~20 px target from competing blobs; the margin
adds the contrastive signal. Empirically on this clip: real target `disc ≥ ~0.096`,
false blobs `disc ≤ 0`.

**Assumptions.**
- The target's appearance is stable enough that a bank of 24 views spans its
  variability across the clip. This holds for a near-grayscale thermal target with
  stable viewpoint.
- The background bank is a representative sample of current distractors. If a new
  distractor with appearance similar to the target enters the scene without first
  populating the background bank, a false positive is possible.

**Trade-offs.**
- A larger bank captures more appearance variability but dilutes the score by
  including less representative views. The novelty gate avoids near-duplicate storage.
- FIFO eviction on the target bank could discard early views that are relevant after
  a revisit; the `consolidate()` step mitigates this by re-selecting the most
  diverse subset.

**Risks & limitations.**
- If the background bank is empty (early in a new segment), `background.score` returns
  0 and the margin equals the raw target score. This is a looser criterion that could
  cause a false re-acquisition.
- The `add_thresh=0.92` is calibrated for this sensor and target; it should be
  validated for any new domain.

---

### Block 9: Multi-identity association

**Source:** `src/track_full_clip.py` — main loop, `best_identity()` function

| | |
|---|---|
| **Input** | Per-frame blob list (boxes, centroids, embeddings from Blocks 6–7); identity list `identities`; `BackgroundMemory`; current active id, last position, velocity |
| **Output** | Assigned identity index, state label (`TRACK`/`REACQ`/`NEW`/`HOLD`/searching), updated position and velocity |

**Algorithm (four rules, evaluated in order).**

| Rule | Trigger | Criterion | Outcome |
|---|---|---|---|
| RULE 1: motion-near | Active id exists, `last_pos` set | Nearest blob within `gate_radius=40 px` of velocity-predicted position | `TRACK`: position + velocity EMA-updated |
| RULE 2: DINOv3 re-acquire active | RULE 1 missed | Best discriminative margin blob; near reacq: `disc ≥ reid_margin=0.10`; far (> 60 px) reacq: `disc ≥ strong_margin=0.15`; during HOLD: near-only gate enforced | `REACQ`: active id maintained, velocity reset |
| RULE 3: re-acquire any id | RULE 1 and 2 missed | Global best margin over all identities; `disc ≥ reid_margin` | `REACQ`: active id switches to best-matching existing id |
| RULE 4: spawn new id | All above missed, limit not reached | Blob has a persistent ByteTrack track (age ≥ `warmup_views=4`), matches no existing id (`disc < reid_margin`) | `NEW`: new `TargetMemory` seeded from that ByteTrack track's embeddings |

**HOLD mechanism.** When all rules fail and the target was recently tracked, the
last-known box is frozen for up to `hold_seconds=2.0` (60 frames at 30 fps). HOLD
covers brief gaps (stopped target, momentary occlusion). After the hold limit, the
tracker enters `searching` state.

**Velocity model.** EMA velocity (`alpha=0.4` on the displacement per frame) is updated
only on consecutive RULE 1 TRACK frames. It is not updated across a gap or after a
re-acquisition, preventing runaway extrapolation.

**Identity permanence.** Identities are never deleted. An old id can always be
re-acquired by RULE 3. On this clip: 3 distinct identities are spawned (id1 = original
target, id2 = the ~frames 745–955 object, id3 = target in post-957 B/W segment).

**Assumptions.**
- A single primary target of interest per segment. Multiple simultaneously active
  targets would require a separate tracking state per identity, which is not
  implemented.
- The discriminative margin calibration (`disc ≥ 0.096` for the real target) holds
  across segments. The threshold is verified on this clip but should be recalibrated
  for a different sensor.
- 2 s HOLD is sufficient to bridge typical gaps (stopped target, short occlusion).
  Longer occlusions require a strong DINOv3 re-acquisition signal when the target
  reappears.

**Trade-offs.**
- The ordered rule priority means RULE 1 (cheap distance check) is applied before
  RULE 2 (GPU embedding comparison). This avoids unnecessary GPU calls on frames
  where motion is sufficient.
- The HOLD-protected re-acquisition in RULE 2 (near-only while holding) prevents
  false jumps to a distractor that appears near the frozen box. The cost is that a
  re-appearing target that is slightly farther than `max_jump=60 px` from the hold
  position needs `strong_margin` — a stricter criterion.

**Risks & limitations.**
- Two simultaneously moving objects with similar appearance could cause RULE 3 to
  switch the active id unexpectedly.
- The velocity model has no acceleration term; fast non-linear motion (sudden
  direction changes) can move the target outside the gate before the next frame.
- RULE 4 requires the ByteTrack track to persist for `warmup_views=4` frames before
  a new id is spawned, introducing a brief latency for genuinely novel targets.

---

### Block 10: ByteTrack (bootstrap and spawn)

**Source:** `src/bytetrack_shim.py` — `make_tracker(fps)`, `detections_from_boxes(boxes, frame_area)`

| | |
|---|---|
| **Input** | Per-frame blob list (from Block 6); frame area `W×H` |
| **Output** | Track assignment rows `[x1, y1, x2, y2, track_id, score, cls, det_idx]` |

**Algorithm.** `BYTETracker` (Ultralytics) is driven with a minimal shim that adapts
MOG2-sourced bounding boxes into the YOLO-Results interface the tracker expects.
Confidence is synthesised from blob area: `conf = 0.70 + 0.25 × clip(area / (0.02 × WH), 0, 1)`,
so all motion-confirmed blobs clear the `new_track_thresh=0.6` required by ByteTrack.
ByteTrack is not the primary identity source; it is used for two purposes only:
(a) bootstrap identity 1 — the first ByteTrack track to accumulate `warmup_views=4`
frames seeds the initial `TargetMemory`; and (b) spawn new identities (RULE 4) when a
persistent novel ByteTrack track matches no existing identity.

**Assumptions.** ByteTrack's IoU-based association is adequate for bootstrap over a
short warmup window. It is not relied upon for long-term identity — it is documented
to fragment the primary target into 117 ids over the full clip without the DINOv3
re-ID layer.

**Trade-offs.** Using an off-the-shelf tracker as a bootstrapper avoids reimplementing
temporal smoothing. The drawback is that it introduces the Ultralytics dependency and
its internal state is not checkpointed across segment boundaries (ByteTrack resets
implicitly when a new segment starts a new Python object via `make_tracker`).

**Risks & limitations.** ByteTrack's Kalman motion model is in raw-frame coordinates
while the primary tracker operates in canvas coordinates. The shim feeds canvas-space
boxes to ByteTrack, which is a deliberate approximation — the canvas is a stabilised
view, so ByteTrack's motion model is more reliable there than in the raw frame.

---

### Block 11: Output and visualisation

**Source:** `src/track_full_clip.py` — render section of main loop

| | |
|---|---|
| **Input** | Active canvas box; assigned id; state label; raw frame (BGR); cumulative homography `Hc` |
| **Output** | Annotated MP4 video; optional per-frame CSV log `(frame, seg, id, state, cx, cy)` |

**Algorithm.** The canvas bounding box is back-projected to raw-frame coordinates via
`perspectiveTransform(H_inv)`. Box size is EMA-filtered (`size_alpha=0.15`) to prevent
the MOG2 blob from causing visible box breathing; the centre is taken raw. The HUD
overlays frame index, segment index, id, and state. The CSV log enables golden-master
regression testing.

---

### Block 12: Scene-map pipeline (secondary subsystem)

| Script | Purpose |
|---|---|
| `src/build_map.py` | LK similarity chain + blur-skip + feather blending → stitched mosaic of a continuous segment |
| `src/map_with_zoom.py` | DINOv3 sliding-window match links the 647 zoom segment; recovered link scale ~0.39 ≈ 1/2.5 confirming ~2.5× FOV switch |
| `src/map_segments.py` | A jump (745) spawns a new segment placed beside; no spatial overlap |
| `src/loop_closure.py` | The 957 jump-back is relinked to a Segment-1 keyframe (~frame 300) by DINOv3 global embedding (cross-modal: red thermal matched to B/W); cosine ~0.765 |

**Block inputs/outputs (summary).**

| Script | Input | Output |
|---|---|---|
| `build_map.py` | Frame range of a continuous segment | Feather-blended mosaic PNG |
| `map_with_zoom.py` | Wide segment frames + zoom segment frames | Combined wide+zoom mosaic; link homography + inlier count |
| `map_segments.py` | All frames through the 957 cut | Two-segment side-by-side map |
| `loop_closure.py` | Segment-1 frames + revisit frames (957–1029, colour-normalised) | Loop-closed composite with 957 segment re-merged into Segment-1 canvas |

**Role of colour normalisation.** `colorfix.to_bw` is mandatory before any
cross-modal matching (B/W → red → B/W). Without it, DINOv3 cosine similarity between
the B/W and red-thermal views of the same ground collapses due to channel distribution
mismatch.

---

## 4. Architecture summary table

| # | Block | Module(s) | DL? | Primary output |
|---|---|---|---|---|
| 1 | Colour normalisation | `colorfix.to_bw` | No | 3-ch B/W frame |
| 2 | Global motion estimation | `registration.GlobalMotionEstimator` | No | H (3×3), ok flag |
| 3 | Discontinuity detection + segmentation | `track_full_clip.chain_segments` | No | Hs list, segs list |
| 4 | Blur detection / skip | Inline (`cv2.Laplacian.var`) | No | blur scores, skip flag |
| 5 | Per-segment canvas fit | `track_dino_reid.fit_canvas` | No | T, canvas (cw, ch) |
| 6 | Motion detection | `track_dino_reid.detect` + MOG2 + coverage mask | No | boxes, centroids |
| 7 | Appearance embedding | `dino_embedder.DinoEmbedder` | **Yes (DINOv3 ViT-S/16)** | (N, 384) embeddings |
| 8 | Identity model | `target_memory.TargetMemory`, `BackgroundMemory` | No | disc margin scores |
| 9 | Multi-identity association | `track_full_clip` main loop | No | active id, state, position |
| 10 | ByteTrack | `bytetrack_shim.make_tracker` | No | track assignments |
| 11 | Output / visualisation | `track_full_clip` render section | No | annotated MP4, CSV |
| 12 | Scene-map pipeline (secondary) | `build_map`, `map_with_zoom`, `map_segments`, `loop_closure` | Yes (DINOv3 dense match) | mosaic images |

---

## 5. Key design decisions and rationale

**Why motion detection instead of a detector.** YOLOv8-n (COCO) returns zero `person`
detections on this target even at `conf=0.05`. The target is too small (sub-2% of frame
area) and too off-domain (top-down thermal) for any COCO-trained detector. Motion
detection in stabilised coordinates is the only reliable per-frame localisation cue.
See Deliverable 2 for the full detector evaluation.

**Why DINOv3 for appearance instead of a re-ID model.** Automotive or person-ReID
models (MARS, Market1501 baselines) expect side-view, clearly-visible pedestrians.
DINOv3 self-supervised features are domain-general; the mean-pooled patch token vector
acts as a compact descriptor for a target the model was never trained to recognise.
The discriminative margin (target minus background score) is the key calibration that
makes raw cosine similarity usable.

**Why segment-based stabilisation instead of a global accumulating canvas.** A single
canvas accumulating across all 1200 frames would grow to an impractical size as the
camera pans widely. More critically, the ~3× zoom and the area jumps cause the
homography chain to degenerate. Resetting the canvas per segment keeps it bounded
and numerically stable.

**Why identity permanence.** A target that leaves the frame and re-enters (e.g., the
1170 reappearance) should receive the same id as before. Permanent identities allow
RULE 3 to match the re-appearing target to its stored appearance bank regardless of
how much time has elapsed.

**What is not solved.** Cross-segment re-identification relies on the DINOv3 margin
remaining above threshold after a colour switch, zoom, or area change. For the most
severe cuts (745 area jump, 957 colour+area change) the margin is unreliable and a
new identity is spawned instead of reliably re-acquiring the original. This is
documented in the out-of-scope section of REQUIREMENTS.md and is addressed in
Deliverable 4 (Failure Analysis) and Deliverable 5 (Improvements).

---

## 6. Measured system behaviour (no ground-truth labels)

All figures are from the standard run (`python3 src/track_full_clip.py`):

| Metric | Value | Note |
|---|---|---|
| Total frames processed | 1200 | Full clip |
| Segments detected | 4 | At genuine discontinuities only |
| Distinct identities spawned | 3 | id1 original target, id2 ~745–955 object, id3 post-957 |
| Frames with box shown | ~76% | TRACK + REACQ + HOLD states combined |
| First-segment box coverage | ~83% | Continuous clean segment (frames 0–609) |
| Re-acquisitions | logged per run | Counted in terminal output |

The remaining ~24% is dominated by the 745–956 other-object segment and the post-1052
exit periods, not by mid-run tracking failures in the clean first segment.

---

*Code cross-references:* `/project/src/track_full_clip.py`, `/project/src/registration.py`,
`/project/src/colorfix.py`, `/project/src/dino_embedder.py`, `/project/src/target_memory.py`,
`/project/src/bytetrack_shim.py`, `/project/src/track_dino_reid.py` (functions `detect`,
`fit_canvas`), `/project/src/build_map.py`, `/project/src/map_with_zoom.py`,
`/project/src/loop_closure.py`, `/project/src/map_segments.py`.
