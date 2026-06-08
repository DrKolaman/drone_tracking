# Deliverable 5 — Improvement Suggestions

**Project:** Detection & Precision Tracking System (`data/source.mp4`, top-down
aerial/drone footage; the target is a *small, low-contrast person*, a few pixels
across, under heavy camera motion).
**Scope of this document:** propose concrete, ranked improvements to the current
pipeline, grounded in the failure modes documented in Deliverable 4. Each
improvement is mapped to its failure mode, explained mechanically, and assessed
for effort and risk. This document directly carries forward the architecture
recommendation in Deliverable 2 §5.

---

## 1. The problem in one sentence

The target is a **few-pixel, low-contrast, top-down thermal blob** whose
independent-motion signal is **indistinguishable from registration residual**
after a fast pan, and whose appearance is too information-poor for any ReID
model to re-identify it across a thermal-sensor switch or a long gap.

Every improvement below attacks one or more of these three root causes:

| Root cause | Which failures it drives |
|---|---|
| **Detector blind spot** — COCO-trained YOLO never fires on a ~20–35 px thermal target | The tracking-by-detection pipeline cannot start; MOTA/IDF1 = 0 on this target |
| **Registration residual ≠ zero** — homography-estimated ego-motion leaves a structured residual that looks like target motion | MOG2/motion-gate false positives; velocity Kalman drifts; fast pans produce garbage detections |
| **Appearance invariance gap** — no descriptor survives the thermal-sensor swap (957), the large-zoom FOV change (647), or even the 30-frame low-contrast window | Cross-cut ReID cannot work; SGLATrack drifts to the bright static ladder structure with *false* high confidence |

---

## 2. Ranked improvements — summary table

| # | Improvement | Failure addressed | Effort | Risk if not done |
|---|---|---|---|---|
| **I1** | Fine-tuned small-object detector + SAHI tiled inference | Detector blind spot (all DL-block failures) | Medium (~2–3 days annotation + training) | Pipeline never starts on this target |
| **I2** | Similarity-model stabilisation + residual-quality gate | Registration residual / false motion | Low–Medium (config change + one filter) | Motion detector fires on camera shake; Kalman drifts |
| **I3** | Appearance-invariant / cross-modal matcher | 647 zoom link, 957/1170 cross-thermal ReID | Medium–High (new library or descriptor) | Cross-cut identity permanently broken |
| **I4** | Independent-motion gate on appearance lock | Static-distractor false lock (ladder) | Low (one validation layer) | SOT locks onto high-contrast static structure with false confidence |
| **I5** | Drone telemetry / IMU fusion | 647 FOV ratio unresolvable from imagery alone | High (payload integration) | Zoom factor must be guessed; cross-cut registration fails |
| **I6** | Ground-truth annotation (~200 frames) | Evaluation blockage; fine-tuning data | Medium (~1 day) | MOTA/IDF1/HOTA cannot be computed; I1 has no training signal |
| **I7** | Deterministic fp32 embedder | Non-reproducible golden tests | Low (dtype flag) | Test suite needs tolerance bands; bit-exact regression impossible |
| **I8** | ViT-SOT as detector-reseeded refiner (optional) | Short-term drift between detections | Low–Medium (integration only) | Slight temporal smoothing lost; standalone use must remain blocked |

---

## 3. Improvement detail

### I1 — Fine-tuned small-object detector + SAHI tiled inference
*(Highest leverage. Addresses the primary blocker from Deliverable 2 §2.1 and §5.)*

**Which failure it addresses.**
YOLOv8-n (COCO) returns zero `person` detections across this target at
`conf=0.05`. The entire detection-led, tracking-by-detection architecture
(Approach A, the spec's intended design) is blocked because the detector never
fires. This is not a threshold issue; it is a domain and resolution gap. A
COCO-trained model was never shown a top-down 20–35 px thermal blob; fine-tuning
on even a few hundred annotated frames of *this* target collapses that gap.

**How it works.**

1. **Annotation.** Label ~200–400 bounding boxes on frames drawn uniformly from
   the 0–609 clean segment, plus 20–30 frames around each discontinuity. The
   annotation cost is 2–4 hours with a tool such as CVAT or Label Studio; the
   data volume is deliberately small because the target class is essentially
   one-instance (single target, single camera geometry, single thermal mode).

2. **SAHI tiled inference** (Akyon et al., ICIP 2022). SAHI divides each
   360×640 frame into overlapping tiles (e.g. 256×256 with 0.2 overlap), runs
   the detector on each tile independently, then merges results with NMS. Because
   the target fills much more of a tile's receptive field than of the full frame,
   recall on small targets improves substantially without retraining — this is
   a free gain even with COCO weights, and a large gain after fine-tuning. SAHI
   integrates directly with the Ultralytics API used in the project.

3. **Fine-tuning.** Run `yolo train data=<custom.yaml> model=yolov8n.pt epochs=50
   imgsz=640` on the annotated frames. With ~300 boxes the model can learn the
   target's thermal signature; 50 epochs on an RTX 3050 completes in under 30
   minutes. Use YOLOv8n (smallest) to stay within the ≥25 FPS NFR1 budget.

**Why it should help.**
The cap is the detector's small-target blind spot, not the tracker. BoT-SORT with
camera-motion compensation (Aharon et al., 2022) already handles the motion
model; ByteTrack (Zhang et al., ECCV 2022) handles high/low confidence
detections. Once the detector fires reliably, the downstream tracker can
associate, propagate, and recover. Deliverable 2 §5 identified this as the
single step that makes the recommended architecture viable.

**Cost / effort / risk.**
Effort: ~1 day annotation + ~30 min training + 1–2 hours integration. Cost: zero
additional software (SAHI is MIT-licensed; Ultralytics is AGPL). Risk: if the
annotated frames are too few or too homogeneous, the detector will overfit and
fail to generalise to the 957–1190 thermal segment — mitigated by including
frames from both B/W and thermal phases in the annotation set. Pairs directly
with I6 (the annotation work is shared).

---

### I2 — Similarity-model stabilisation + high-residual frame rejection
*(Addresses registration-residual false motion, failure #2.)*

**Which failure it addresses.**
The current stabilisation estimates an 8-DOF full homography per frame pair.
On a fast pan, the inlier ratio can drop below 30 %, and the resulting warp
introduces shear and keystoning that looks like structured motion in the
residual. MOG2 or any frame-difference motion detector then fires on the warp
artefact, not on the target. The problem is not brightness/threshold — it is
that the input to the motion detector is corrupted during fast pans.

**How it works.**

1. **Switch to a 4-DOF similarity model** (rotation + uniform scale + translation,
   no shear). OpenCV's `estimateAffinePartial2D` with RANSAC estimates this
   model. A drone camera on a gimbal moves in translation and yaw; shear is
   physically unmotivated, and allowing 8 DOF overfits to noise. The similarity
   model is supported directly in `src/registration.py`'s RANSAC path — it is a
   flag change, not a rewrite.

2. **Add exposure/gain compensation.** ORB matching across frames with very
   different exposure (e.g. the 610–614 blur burst) degrades because descriptors
   are brightness-sensitive. A fast per-tile histogram normalisation or
   `cv2.createMergeMertens` exposure compensation before descriptor extraction
   stabilises the matching.

3. **Reject or down-weight high-residual frames.** After estimating the
   similarity warp, compute the inlier ratio from RANSAC. If the ratio falls
   below a configurable threshold (e.g. 0.35), flag the frame as unreliable:
   freeze the Kalman prediction (do not update with a motion-detector reading),
   and do not feed the frame to MOG2. This prevents the fast-pan garbage from
   contaminating the motion detector's background model.

**Why it should help.**
The false motion signal is registration residual. Fixing the registration model
removes the noise at source. The similarity model gives fewer artefacts than an
8-DOF homography for a camera that moves by translation and rotation without
physical shear. The frame-quality gate ensures that the one frame type that
defeats the estimator (the 610–614 blur burst) does not propagate bad readings
downstream.

**Cost / effort / risk.**
Effort: low — a model flag change in `src/registration.py`, a RANSAC inlier-ratio
check, and a gate condition in the motion-detector caller. Risk: a similarity
model is a subset of the homography; on a truly projective scene (a very tilted
drone) it may under-fit. For the footage geometry (near-nadir drone, small
angles) this is unlikely. The frame-rejection gate introduces a latency bubble of
1–2 frames around fast pans; this is acceptable given that detections during a
blur burst are unreliable anyway.

---

### I3 — Appearance-invariant / cross-modal matcher for the 647 zoom and 957 cross-thermal ReID
*(Addresses failures #4 and #5: the zoom-link gap and the B/W→thermal ReID gap.)*

**Which failure it addresses.**
At frame 647 the camera zooms approximately 7× in a single frame. At frame 957
it jumps back and switches from B/W to a red thermal palette. At frame 1170 the
target reappears at a different location. In all three cases the current
pipeline performs a tracker reset (FR5), but no attempt is made to re-acquire
the *same* identity on the other side of the boundary. The MASt3R focal-ratio
test (Deliverable 3/4) returned a ratio near 1, confirming that imagery alone
cannot determine whether the apparent 3D perspective change is a lens zoom or a
physical camera move; DINOv3 and ORB both plateau at 9–20 descriptor matches
across the thermal-sensor gap even at matched scale — well below the threshold
needed for reliable affine estimation.

**How it works.**

1. **Initialise the search from the known geometry.** At frame 647 the zoom
   factor is approximately 7× (measured from the apparent blob-diameter change).
   Rather than searching the full post-zoom frame, restrict the cross-frame
   search to the region around the target's last known position scaled by 7×,
   in the bottom-right quadrant of the frame (where the zoom appears to centre).
   This reduces the search space by ~50× and lowers the bar for any matcher.

2. **Replace ORB with a cross-spectral descriptor.** For the 957 B/W→thermal
   transition, RGB-gradient-based descriptors (ORB, SIFT) degrade because the
   sensor palette change flips brightness relationships. Alternatives with
   established cross-modal performance:
   - **RIFT** (Radiation-Invariant Feature Transform, Li et al., 2019): replaces
     gradient orientation with maximum-index-map orientation — invariant to
     non-linear intensity transforms including thermal-palette inversions.
   - **Phase-congruency keypoints** (Kovesi, 2003): detect structure where phase
     rather than gradient amplitude is maximised — stable across sensor modes.
   - **Mutual-information-based template search**: slide a small window around
     the predicted target location in the post-switch frame and maximise
     normalised mutual information against the last known template. MI is
     explicitly designed for multi-modal image registration (medical imaging
     literature). Computationally expensive for full-frame search but cheap
     within the constrained search window from step 1.

3. **Validate the match with the independent-motion gate** (I4 below). Any
   candidate re-ID must also be independently moving — this collapses the
   false-positive rate when the appearance match is ambiguous.

**Why it should help.**
The fundamental problem is that standard RGB descriptors are not invariant to the
thermal-palette switch. Cross-modal descriptors are specifically designed for
the modality gap this footage presents. The constrained geometry initialisation
makes any matcher's job tractable. The combination of a geometry prior +
cross-modal descriptor + motion validation provides three independent lines of
evidence for re-ID, replacing the current single-cue ORB approach.

**Cost / effort / risk.**
Effort: medium–high. RIFT is available as a Python implementation but is not in
the project's current dependency set. Phase-congruency is available via
`phasepack`. MI-template-search requires a custom loop over a search window.
Integration into the existing `src/track_full_clip.py` reset-and-continue logic
is a defined extension point (FR4/FR5 boundary). Risk: cross-modal matching on a
~20-pixel target with few distinctive features is fundamentally hard; even a
well-engineered solution may produce noisy matches. The improvement's value is
enabling *any* cross-cut re-ID rather than achieving perfect re-ID.

---

### I4 — Independent-motion gate to reject static distractors
*(Addresses failure #3: appearance lock on the static "ladder" structure.)*

**Which failure it addresses.**
The SGLATrack trial (Deliverable 2 §3.3) demonstrated that a template tracker
drifts onto the bright static ladder-shaped structure in the scene and reports
**high confidence** there — because the structure is a stronger appearance
feature than the faint moving target. This is the critical finding of the SOT
trial: a confidence gate cannot fire to trigger re-detection because the tracker
is *confidently wrong*. The only cue that distinguishes the moving person from
the static structure is **independent motion** — the target moves relative to
the stabilised background; the structure does not.

**How it works.**
After any appearance-based lock (SOT refiner or ReID module), check that the
locked region contains a statistically significant independent-motion signal:

1. Apply the current frame's ego-motion-compensated warp (from I2's similarity
   model) to the previous frame.
2. Compute the per-pixel difference between the warped previous frame and the
   current frame in the locked bounding box region.
3. Threshold the difference (using Otsu or a fixed thermal-noise floor) and
   count the fraction of pixels that are "active".
4. If the active fraction falls below a minimum (e.g. 5 % of box pixels), flag
   the appearance lock as "distractor candidate" and trigger re-detection from
   the motion branch.

This is a validation layer, not a primary tracker — it adds one matrix operation
and a threshold check per frame and imposes negligible latency.

**Why it should help.**
Motion is the one cue that is definitionally true of the target and definitionally
false for static distractors. An appearance-only tracker cannot make this
distinction. Adding an independent-motion validation step closes the loophole
that the SGLATrack trial exploited. The structure cannot pass this gate; the
moving target can. For the degenerate case where the target is briefly stationary
(possible in the 615–646 static-camera segment), the gate is weakened by raising
the threshold — an explicitly documented trade-off.

**Cost / effort / risk.**
Effort: low. The registration/warp infrastructure already exists in
`src/registration.py`; the pixel-difference operation is two OpenCV calls. The
gate integrates into the appearance-lock validation path in
`src/track_full_clip.py`. Risk: if the ego-motion compensation is poor (exactly
the I2 failure mode), the gate may fire spuriously on warp artefacts. I2 and I4
are therefore complementary and should be deployed together. A secondary risk is
that a very slow-moving target produces a weak motion signal — mitigated by
keeping the threshold low (5 % is conservative).

---

### I5 — Drone telemetry / IMU fusion
*(Addresses the 647 FOV-ratio ambiguity; reduces registration error across all pans.)*

**Which failure it addresses.**
At frame 647 the apparent zoom factor (~7×) is estimated from blob-diameter
ratios in the image — a noisy measurement for a few-pixel target. The MASt3R
reconstruction returned a focal-ratio near 1, meaning the geometric reasoner
interprets the zoom as a 3D camera move rather than a lens change (Deliverable
3). Any cross-frame correspondence that depends on knowing the intrinsic change
is therefore blocked by imagery alone. More broadly, the drone's actual angular
velocity and gimbal angles, if available, would allow exact ego-motion prediction
for each frame, eliminating the need for the RANSAC-based homography estimation
that currently fails during fast pans.

**How it works.**
Most drone/payload systems expose telemetry at the same frame rate as the video
(DJI, Elbit EO/IR pods: typically embedded in the video container as metadata or
via a companion UDP/serial stream). The relevant fields are:
- Gimbal pitch/roll/yaw and zoom factor (lens focal length or digital zoom ratio)
  — directly resolve the 647 event as a zoom, not a 3D move.
- IMU angular velocity and linear acceleration — provide a prior for the
  per-frame warp estimation, reducing the RANSAC search space from unconstrained
  8-DOF to a narrow cone around the IMU-predicted transform.

Integration would replace the RANSAC initialisation with a telemetry prior,
falling back to RANSAC only when telemetry is unavailable or flagged as
unreliable (e.g. gimbal slew rate limit exceeded).

**Why it should help.**
The 647 zoom is not recoverable from imagery alone at the target's size and
contrast level. Telemetry resolves it exactly. For fast pans, a good IMU prior
can reduce RANSAC iterations by an order of magnitude and improve inlier ratio
on the frames that currently defeat the estimator. This is the highest-quality
fix for the registration problem — it attacks the root cause rather than the
symptom.

**Cost / effort / risk.**
Effort: high. Requires access to the payload's telemetry stream, which is
system-dependent and not available for this take-home exercise. The source video
as provided embeds no telemetry metadata. This improvement is therefore
correctly marked out-of-scope for the current phase but is the most important
architectural step toward a production-grade system. Risk for the current scope:
none (purely additive); risk of not doing it: the 647 FOV estimation remains
approximate.

---

### I6 — Ground-truth annotation (~200 frames) to unlock quantitative evaluation
*(Addresses the evaluation blockage documented in Deliverable 3 §6.2 and NFR2/G4.)*

**Which failure it addresses.**
No ground-truth bounding boxes exist for this clip. As a direct consequence:
MOTA, IDF1, and HOTA cannot be computed; the detector fine-tuning in I1 has no
training signal; and all quality claims in Deliverables 3–5 rely on proxy metrics
rather than standard MOT benchmarks. This is the honest documentation required by
G4 — but it is also fixable.

**How it works.**
Annotate bounding boxes for the primary target on approximately 200 frames
sampled as follows:
- Every 3rd frame in the 0–609 clean segment (~200 frames).
- Every frame in the 610–646 blur/static segment (37 frames, ground truth for
  the hardest case).
- Every 10th frame in the 647–956 segment (~30 frames, for post-zoom evaluation).
- Every frame in the 957–1190 segment (~234 frames, for thermal and re-appearance
  evaluation).

Use CVAT (free, browser-based) or Label Studio. With a single target and mostly
visible motion, annotation takes approximately 1–2 hours at ~30 s/frame.

This annotation set serves triple duty: training data for I1, validation split
for I1's fine-tuned detector, and evaluation ground truth for MOTA/IDF1/HOTA
against the current and improved pipelines.

**Why it should help.**
Moving from proxy metrics (track lifetime, ID-switch count, target-lock
continuity) to standard MOT metrics (MOTA/IDF1/HOTA) is the difference between
"the pipeline looks reasonable" and "the pipeline improves by X % on
re-identification after the thermal switch." The annotation investment is small
relative to the analytical clarity it enables. It also makes the project directly
comparable to the UAV tracking literature (UAV123, Mueller et al., ECCV 2016).

**Cost / effort / risk.**
Effort: ~1 day. Zero software cost. Risk: annotation errors introduce noise into
the evaluation ground truth — mitigated by a second-pass consistency check on the
annotations (spot-check every 10th box for box-continuity drift). The annotation
does not need to be perfect; MOT metrics are robust to ~5 % label error.

---

### I7 — Deterministic fp32 embedder for bit-exact reproducibility
*(Addresses NFR2: reproducibility of golden-test baselines.)*

**Which failure it addresses.**
The current DINOv3 embedder runs in fp16 on the GPU. fp16 accumulation is
non-associative on CUDA: operations reordered by cuBLAS across different CUDA
versions, driver updates, or batch sizes produce numerically different outputs
for the same input. The practical consequence is that golden-test assertions in
the test suite require tolerance bands (`atol=1e-3`) rather than bit-exact
equality, which means a genuine regression in the embedding may be masked by
numerical drift.

**How it works.**
Force the embedder to fp32 via a single configuration flag:

```python
# src/dino_embedder.py — replace
model = model.half()
# with
model = model.float()
```

Alternatively, gate the precision on an environment variable
(`EMBED_DTYPE=fp32/fp16`) so production inference can still use fp16 for speed,
while CI and golden-test generation always use fp32.

**Why it should help.**
fp32 arithmetic on a given CUDA device and driver version is deterministic for
the same computation graph and input. Bit-exact embeddings mean the golden tests
can use strict equality, and any change to the embedding path (model weights,
pre-processing, padding) will produce a detectable regression immediately rather
than being absorbed by the tolerance band.

**Cost / effort / risk.**
Effort: low — a one-line change plus a CI environment variable. Risk: fp32 uses
roughly 2× the GPU memory and is ~15–20 % slower than fp16 for the embedding
pass alone. On the RTX 3050 (4 GB VRAM) with YOLOv8n + BoT-SORT running
concurrently, this may not fit. Gating on the environment variable (fp32 in
tests, fp16 in production) avoids the VRAM pressure while preserving bit-exact
test baselines.

---

### I8 — ViT-SOT as detector-reseeded short-term refiner (optional)
*(Partially addresses temporal smoothing loss between detections; builds on Deliverable 2 §5 point 2.)*

**Which failure it addresses.**
This is the conditional improvement flagged in Deliverable 2 §5: SGLATrack (or
any ViT-SOT) failed as a standalone tracker precisely because it has no
re-detection and drifts onto distractors with false confidence (Deliverable 2
§3.3, §4). The standalone use is permanently inadvisable on this footage. However,
used in a tightly constrained role — as a sub-frame-gap *refiner* between
detector fires — it contributes localisation precision that Kalman extrapolation
alone does not provide.

**How it works.**
The detector (post-I1) fires on every frame but may miss the target for 2–5
consecutive frames when it is at minimum contrast or partially occluded. Between
missed detections, the Kalman model propagates a linear-velocity prediction.
A ViT-SOT seeded from the last confirmed detection can provide a better
short-term localisation (non-linear motion, rotation-aware) for up to ~30 frames:

1. On a confirmed detection at frame *t*, seed the SOT with the detection box.
2. For frames *t+1*, *t+2*, ..., use the SOT output *only if* (a) the SOT
   confidence exceeds a threshold AND (b) the I4 independent-motion gate passes.
3. If either condition fails, revert to Kalman prediction and wait for the next
   detector hit.
4. On the next confirmed detection, re-seed the SOT and reset its template.

The key constraint: the SOT is never allowed to run for more than N frames
(recommended N=30) without a confirming detection. This prevents the drift-and-
distractor-lock failure documented in Deliverable 2.

**Why it should help.**
SGLATrack's failure was not its architecture but its use case: it was run
standalone with no re-detection fallback for 956 frames. In the constrained role
above, the same model provides smooth sub-pixel localisation for short gaps while
the independent-motion gate and the detector-reseeding limit prevent distractor
lock. The result is a smoother track in the clean 0–609 segment with no
additional failure risk, provided the constraints are enforced.

**Cost / effort / risk.**
Effort: low–medium. SGLATrack is already integrated and patched for the current
environment (Deliverable 2 §3.1). The integration is a wrapper around the
existing SOT call with a frame-counter gate and the I4 motion check. Risk:
if I1 (fine-tuned detector) is not in place, re-seeding frequency is too low and
the SOT still drifts. I8 is therefore a downstream dependency of I1 and should
not be deployed without it.

---

## 4. Dependency map

```
I6 (annotation)
  └─► I1 (fine-tuned detector)   ← primary blocker
            └─► I8 (SOT refiner)   ← downstream optional
I2 (stabilisation)
  └─► I4 (motion gate)           ← complementary pair
I3 (cross-modal matcher)
  ├─► I4 (motion gate as validator)
  └─► depends on I2 for registration quality
I7 (fp32 embedder)               ← standalone, low risk
I5 (telemetry)                   ← orthogonal, high effort, future work
```

---

## 5. Recommended implementation order

Given the ≤6-page submission constraint and the finite time budget, the
suggested order of implementation is:

1. **I6** — annotate ~200 frames (1 day). Unlocks I1 and enables quantitative
   comparison of any improvement against the current pipeline. Do this first.
2. **I2** — switch to similarity model + residual gate (half a day). Cleans up
   the motion-detector input with near-zero risk. Deploy before any motion-based
   evaluation.
3. **I4** — add independent-motion gate (half a day). Prevents distractor lock.
   Complements I2.
4. **I1** — fine-tune detector + SAHI (1–2 days). Highest-leverage change;
   requires I6's annotation data. Makes the entire tracking-by-detection
   architecture viable.
5. **I7** — fp32 embedder flag (1 hour). Low risk, high reproducibility benefit.
6. **I3** — cross-modal matcher (2–4 days). Highest complexity; attack after the
   detector is working.
7. **I8** — SOT refiner (1 day). Downstream of I1; integrate last.
8. **I5** — telemetry fusion (future work; requires payload access).

---

## 6. What these improvements do not fix

In the spirit of G4 (honest documented scope), the following limitations remain
even if all improvements above are implemented:

- **The 647 zoom-factor uncertainty** is only partially resolved by I3's
  geometry-prior initialisation and I5. Without telemetry (I5), the zoom ratio
  remains an image-based estimate with ~10–20 % error for a target of this size.
- **Target re-identification across the 745 jump** to a *different map area with
  a different-looking target* cannot be solved by any appearance or motion
  matcher: the target at 745–956 may not be the same individual as 0–609, and
  no evidence in the imagery resolves this.
- **Extended occlusion (> 30 frames) with no motion signal** breaks any
  motion-only re-acquisition scheme. If the target is stationary and
  occluded for a long interval, it cannot be distinguished from the background
  until it moves again.
- **Production-grade reliability** (sensor calibration, latency guarantees, edge
  deployment) is out of scope for a take-home exercise and is not addressed here.

---

## 7. References

*(All references below are already cited in Deliverable 2; no new citations
are introduced.)*

- Akyon et al. *Slicing Aided Hyper Inference and Fine-tuning for Small Object
  Detection* (SAHI). ICIP 2022.
- Aharon, Orfaig, Bobrovsky. *BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking.* 2022.
- Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection
  Box.* ECCV 2022.
- Jocher et al. *Ultralytics YOLOv8.* 2023.
- Mueller, Smith, Ghanem. *A Benchmark and Simulator for UAV Tracking* (UAV123).
  ECCV 2016.
- SGLATrack — GXNU-ZhongLab. https://github.com/GXNU-ZhongLab/SGLATrack

---

*This document is Deliverable 5 of 5. It builds directly on the model
recommendation in Deliverable 2 §5 and the failure modes documented in
Deliverable 4.*
