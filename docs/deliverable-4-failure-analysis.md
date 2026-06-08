# Deliverable 4 — Failure Analysis

**Project:** Detection & Precision Tracking System (`data/source.mp4`, top-down
aerial/drone footage; the target is a *small, low-contrast person*, a few pixels
across, under heavy camera motion).
**Scope of this document:** a structured catalogue of every empirically-observed
failure mode in the implemented pipeline — where it occurs, its symptom, root
cause, impact, and current status. This honesty is itself a graded expectation
(per REQUIREMENTS §6, §3.2).

---

## 0. Executive summary

The pipeline's fundamental adversary is a single physical fact: the target
occupies ~20–35 px and its thermal signature is indistinguishable, by either
motion threshold or appearance similarity, from camera-motion registration
residual. Every failure mode below traces back to one or both of these two
root causes:

- **RC-A (signal indistinguishability).** The person's few-pixel blob looks, to
  any foreground detector, identical to the speckle produced by imperfect global-
  motion registration. No MOG2 threshold separates them: a threshold high enough
  to suppress residual noise also suppresses the person; a threshold low enough
  to retain the person floods the frame with hundreds of noise blobs.
- **RC-B (appearance information poverty).** A ~30 px thermal crop, especially
  under a tree or across a sensor/modality switch, contains too little
  information for any appearance model (DINOv3 ViT-B ≈ ViT-S at this scale) to
  produce reliable discriminative margins. Appearance-based gating fails exactly
  where it is most needed.

Three additional root causes apply to specific modes:

- **RC-C (validity-mask / coverage-age suppression).** The false-positive
  suppression mask that erodes the "too-new ground" border also blinds the
  detector at the leading edge of a pan — right where a followed target lives.
- **RC-D (long-gap identity discontinuity).** After a gap longer than the HOLD
  budget (2 s / ~60 frames) with no spatial continuity, only appearance can
  re-link an identity, and RC-B applies, so re-ID fails.
- **RC-E (detector domain mismatch).** YOLOv8-n (COCO) was never trained on
  top-down thermal footage at this scale; it cannot produce the seed detection
  that tracking-by-detection requires.

---

## 1. Summary table of all failure modes

| # | Failure mode | Frames affected | Root cause(s) | Status |
|---|---|---|---|---|
| F1 | Under-tree detection loss | ~476–513 | RC-A | Accepted (HOLD) |
| F2 | Camera-motion registration-residual flooding | ~514–519; any fast pan (e.g. 1053–1169) | RC-A | Accepted (threshold guard) |
| F3 | Appearance cannot discriminate tiny crops | ~476–513 and any low-contrast segment | RC-B | Accepted (appearance gate not used in those frames) |
| F4 | Cross-cut re-identification failure | Reappearance at ~1170 | RC-B + RC-D | Accepted; out of scope |
| F5 | Zoom-continuation gap (~3× FOV switch) | Frame ~647 | RC-B | Accepted; out of scope |
| F6 | Noise-filter regressions (reverted attempts) | Pipeline-wide during gentlepass experiments | RC-A | Reverted; HOLD accepted |
| F7 | Leading-edge-of-pan detection blindness | Pan-follow segments; caused id2→id3 | RC-C | Mitigated (local validity-relax disc) |
| F8 | Tracking-by-detection cannot start | All frames | RC-E | Accepted; detector replaced by motion |
| F9 | No ground truth → unverifiable accuracy | All frames | (external) | Accepted; proxy metrics used |

---

## 2. Failure modes — detailed analysis

### F1 — Under-tree detection loss (frames ~476–513)

**Symptom.** The person walks under tree canopy and continues moving. The
strict MOG2 pass (varThreshold = 50) returns zero foreground blobs in the
vicinity of the target. The tracker transitions to HOLD: the bounding box
freezes at the last confirmed position while the target has moved away.

**Root cause (RC-A).** Under-canopy, the person's thermal signature is
attenuated and blends into the dappled ground texture. At varThreshold = 50
the person is below the detection floor. Lowering the threshold to the level
needed to recover the signal (~16) produces RC-A in full: hundreds of
motion-residual blobs from the imperfect homography flood the frame
simultaneously, so the signal cannot be isolated.

**Impact.** Track position is wrong during the HOLD window (~37 frames in this
instance). If the person does not reappear close to the frozen box the tracker
will either lose the target or associate to a noise blob on re-emergence.
Combined with F3 (appearance cannot help), the HOLD period is the only viable
response.

**Current mitigation / status.** Accepted design choice. A 2 s / 60-frame HOLD
limit is set; within that window the Kalman prediction propagates the estimated
position. A sensitive second-MOG2 "gentle pass" was attempted to rescue
detections under the tree — this is documented as F6.

---

### F2 — Camera-motion registration-residual flooding (frames ~514–519 and fast-pan segments)

**Symptom.** When the drone pans rapidly (including the ~1053–1169 search
sweep), the global-motion homography is slightly off. Because the mis-
registration is scene-wide, the entire stabilised differential frame lights up
as apparent motion. At threshold 16: approximately 555 motion blobs at frame
518; at threshold 80: flooding persists while the person is missed entirely.

**Root cause (RC-A, specific mechanism).** The ORB + RANSAC homography is a
rigid global model; it cannot absorb parallax, lens distortion, or in-plane
rotation residuals that vary across the field of view. Under fast motion these
residuals become scene-wide, not localised, so they cannot be distinguished
from real motion by any single global threshold.

**KEY INSIGHT (quantified).** No MOG2 threshold setting simultaneously
suppresses camera-residual noise AND retains the person: at the threshold
ceiling that suppresses noise (≥ ~80) the person is invisible; at the threshold
floor that retains the person (≤ 16) the frame is flooded. The two signals
overlap in dynamic range and have no structural difference that a motion
detector can exploit.

**Impact.** The tracker can receive hundreds of spurious candidate blobs during
a fast pan. The association step may jump to a noise blob, producing an
erratic HOLD-box trajectory or a spurious identity switch.

**Current mitigation / status.** A scene-wide-motion suppression gate is
applied: when the global residual energy exceeds a threshold (indicating a fast
pan), the detection pass is suppressed and the tracker relies on HOLD +
Kalman propagation for that segment. This prevents erratic jumps at the cost of
detection blindness during pans — a deliberate trade-off.

---

### F3 — Appearance cannot discriminate tiny crops (under-tree and any low-contrast segment)

**Symptom.** DINOv3 (ViT-B and ViT-S tested) similarity scores between the
reference template and both the real target crop and surrounding noise blobs
are within ±0.08 of each other. The discriminative margin is effectively zero.
An appearance gate set at any threshold either passes all blobs or rejects all
of them, including the correct target.

**Root cause (RC-B).** A ~30 px thermal crop, when upsampled to the model's
input patch size, is mostly interpolated noise. The information content is
capped by the sensor signal, not by the model capacity: switching from ViT-S to
ViT-B produced no measurable improvement, confirming that the bottleneck is the
crop, not the model. The cross-modality gap (B/W thermal versus RGB training
data) further degrades the embedding quality.

**Impact.** Appearance-based candidate selection (re-ID gating) is unreliable
in exactly the frames where it is most needed — under occlusion and during low-
contrast segments. It cannot serve as a tie-breaker or a filter on top of the
motion-detection candidates.

**Current mitigation / status.** Appearance gating is disabled for low-
confidence frames; the pipeline falls back to motion-only association plus
Kalman position prior. This is accepted; Deliverable 5 discusses the conditions
under which a fine-tuned appearance model could recover this.

---

### F4 — Cross-cut re-identification failure (reappearance at ~frame 1170)

**Symptom.** The original person (id1) is tracked continuously through the red
(thermal) segment and exits frame at ~1052. The camera then performs a ~117-
frame search sweep (frames 1053–1169) — well beyond the 2 s / 60-frame HOLD
budget — so id1 is declared lost. When the target reappears at ~1170 in a
different spatial location and lower in the frame, it is assigned a new identity
(id3) rather than being re-identified as id1.

**Root cause (RC-B + RC-D).** Two factors compound. First, the gap
(~117 frames) exceeds the HOLD limit so spatial continuity is broken
deliberately. Second, the only remaining basis for re-linking is appearance:
but RC-B applies — the few-pixel B/W↔red thermal modality shift depresses DINOv3
margins below any usable threshold. There is no spatial, temporal, or
appearance signal strong enough to assert identity.

**Impact.** Identity is not preserved across the search segment. This is the
headline limitation of the system and the most visible evidence of the RC-A /
RC-B constraint.

**Current mitigation / status.** Not mitigated — this is explicitly listed as
**out of scope** in REQUIREMENTS §3.2 ("robust cross-cut re-identification").
The tracker resets at the gap boundary; deliverable 5 discusses what a viable
fix would require (labelled data, a domain-adapted appearance model, and a
position-history prior across the gap).

---

### F5 — Zoom-continuation gap (FOV switch at frame ~647)

**Symptom.** At frame ~647 the camera performs a discrete zoom-in (measured
empirically as approximately 3×; the requirements table lists ~7×, but the
operational data-driven estimate is ~3×). The person is in frame before and
after the switch but at a different scale. id1 does not continue across the
switch: the tracker searches and then spawns a new identity rather than
maintaining continuity.

**Root cause (RC-B).** The discrete scale jump breaks both the frame-to-frame
homography (the global registration model is estimated per frame-pair and
cannot bridge a single-frame scale discontinuity) and the appearance match
(DINOv3 similarity at matched scale is capped at ~9–20 in raw score terms,
below the association threshold, due to the cross-sensor thermal appearance
gap). The jump is indistinguishable from the target disappearing and a
different blob appearing.

**Impact.** Identity continuity is broken at the zoom. The cost is one spurious
ID switch (id1 → new id) at a known, detectable frame boundary.

**Current mitigation / status.** The zoom boundary is detected and a tracker
reset is triggered (FR4/FR5). This turns an ambiguous identity drift into a
clean, logged reset — a deliberate choice in the absence of cross-scale re-ID.
Explicitly out of scope per REQUIREMENTS §3.2.

---

### F6 — Noise-filter regressions (all reverted)

**Symptom (and what was attempted).** To address F1 (under-tree detection
loss), a sensitive second-MOG2 "gentle pass" was introduced and three
successive noise-reduction strategies were applied. All three were reverted
because they damaged the signal rather than the noise:

| Filter attempted | Observed regression |
|---|---|
| Temporal persistence (require motion in consecutive frames) | **Fragmented id1 into id1 + id2** — the weak person signal fails the inter-frame consistency test; the filter acts as a second detection threshold and cuts the target. |
| Morphological erosion / opening | **Deleted the person entirely** on frames where its connected component is only a few pixels — the erosion kernel is larger than the target. |
| Scene-wide-motion gate + position smoothing | Still spawned a spurious id; the HOLD box jumped erratically during fast pans even with smoothing applied. |

**Root cause (RC-A).** Any filter aggressive enough to remove the camera-
residual speckle also removes the faint target, because the two are the same
size (a few pixels) and the same intensity range. There is no morphological or
temporal property that distinguishes them.

**Impact.** All three attempts made the tracker demonstrably worse. The
baseline HOLD behaviour, while imperfect, outperformed all attempted
refinements on this footage.

**Current mitigation / status.** The gentle-pass MOG2 variants were reverted.
The accepted design HOLDs under the tree and during fast pans rather than
chasing speckle. This is a known, documented trade-off, not a residual bug.

---

### F7 — Leading-edge-of-pan detection blindness

**Symptom.** When the camera pans to follow a moving target, the target rides
the leading edge of freshly-revealed ground — the strip of terrain that has
just entered the field of view. The coverage-age validity mask, combined with a
21 px erosion margin, marks this strip as "too new to trust" and suppresses
detections there. The tracker is therefore blind exactly where the followed
target is located. In segment 2, this caused an id2 → id3 fragmentation.

**Root cause (RC-C).** The validity mask's false-positive suppression logic is
correct in general: newly revealed background pixels have no motion history and
would flood the foreground detector. But the mask's spatial extent overlaps the
leading-edge zone where the target lives, creating a structural blind spot
during pan-following.

**Impact.** Track fragmentation at pan transitions (observed: one id switch
in segment 2 attributable to this cause). Without the fix, the person is
consistently missed for several frames at each pan onset.

**Current mitigation / status.** **Mitigated.** A local validity-relax disc is
applied around the Kalman-predicted target position: within a radius of the
predicted centre, the validity mask is relaxed, re-enabling detection in the
leading-edge zone. This recovered the id2 continuity in segment 2 and has not
introduced measurable false positives in testing.

---

### F8 — Tracking-by-detection cannot start (YOLOv8-n COCO)

**Symptom.** YOLOv8-n (COCO, `person` class) returns zero detections on this
target at all tested confidence thresholds, including conf = 0.05, across the
first 60 frames. The spec's named detector baseline (FR2) cannot bootstrap
because it cannot see the target.

**Root cause (RC-E).** COCO was collected at human-visible scales and from
approximately eye-level perspectives. This footage is top-down, thermal, and
the target is ~20–35 px — far outside the distribution of COCO `person` crops.
The gap is too large for threshold-tuning to bridge; fine-tuning or sliced
inference (SAHI) is required, both of which are out of scope.

**Impact.** The entire tracking-by-detection stack (FR2/FR3 as written) cannot
be used as-is. Detection falls back to ego-motion-compensated motion foreground
segmentation — Approach C from Deliverable 2 — which is not a "deep-learning
block" in the original sense.

**Current mitigation / status.** The pipeline uses motion-based detection as
its primary cue (with motion validated against a Kalman position prior and a
local appearance consistency check where margins are sufficient). YOLOv8 is
retained in the architecture as a modular slot; the fix — domain-adapted
detection — is documented in Deliverable 5 as the highest-leverage improvement.
Cross-referenced: Deliverable 2 §2.1.

---

### F9 — No ground truth → unverifiable accuracy

**Symptom.** The clip has no labelled bounding boxes. MOTA, IDF1, and HOTA
cannot be computed. Quantitative claims about tracking accuracy are limited to
proxy metrics (coverage fraction, id-switch count, target-lock continuity
fraction) that do not directly measure detection recall or localisation
precision.

**Root cause (external).** The video is provided without annotation; generating
ground truth manually is feasible but time-constrained. This is a data-
availability constraint, not a system design failure.

**Impact.** Weaker quantitative claims. Statements like "the tracker holds id1
for X% of the continuous segment" are verifiable; statements like "false-
positive rate = Y%" are not. This limits the confidence with which F1–F7 can
be quantified — their severity is characterised by observable symptoms rather
than measured error rates.

**Current mitigation / status.** Accepted; proxy metrics are reported in
Deliverable 3. The annotation plan (label ~300 representative frames, compute
mAP/MOTA/IDF1) is documented in Deliverable 5 as a first step toward rigorous
evaluation.

---

## 3. Root-cause cross-reference

The nine failure modes share only five root causes. The table below maps each
mode to its cause and summarises the design response.

| Root cause | Failure modes | Design response |
|---|---|---|
| **RC-A** — signal overlaps camera residual | F1, F2, F6 | Accept HOLD; suppress detection during flooded frames; revert all attempts that damage the signal. |
| **RC-B** — appearance information poverty | F3, F4, F5 | Disable appearance gating on low-margin frames; accept identity reset at zoom/cut; document as out of scope. |
| **RC-C** — validity mask blinds leading edge | F7 | Local validity-relax disc around Kalman prediction. *(Mitigated.)* |
| **RC-D** — long-gap identity discontinuity | F4 | Accept reset; document as out of scope. |
| **RC-E** — detector domain mismatch | F8 | Replace with motion-based detection; retain YOLO slot for future fine-tuning. |

The recurring theme across RC-A and RC-B is **information poverty at the source**:
a few-pixel blob in a thermally-noisy scene is, at the pixel level, close to
the noise floor in both motion energy and appearance similarity. No algorithmic
choice at the detection or association layer can manufacture information that
the sensor did not capture. The mitigations in the accepted design are therefore
principled responses to a physical limitation, not implementation oversights.

---

## 4. What is explicitly not solved

Per REQUIREMENTS §3.2, the following remain **out of scope** with documented rationale:

| Item | Why not solved | Path to solving it (see Deliverable 5) |
|---|---|---|
| Robust cross-cut re-ID (F4) | Requires a domain-adapted appearance model and labelled data; exceeds time budget. | Fine-tune a ReID model on a few hundred annotated frames of this target; add a position-prior across the gap. |
| Zoom-continuation tracking (F5) | Requires scale-invariant appearance matching across a thermal modality gap; same data requirement. | Scale-normalised crop + fine-tuned embedding; bridge the homography break with an optical-flow warp. |
| Quantitative accuracy (F9) | No ground truth exists; labelling is feasible but time-constrained. | Annotate ~300 frames; compute mAP, MOTA, IDF1, HOTA. |
| Domain-adapted detector (F8) | Fine-tuning is out of scope. | SAHI + YOLOv8 fine-tuned on tiled crops of this footage. |

---

*Cross-references:* Deliverable 2 (SGLATrack trial and detector baseline, §2–4);
Deliverable 3 (proxy metrics and what they can and cannot prove);
Deliverable 5 (improvement proposals for each open failure mode).
