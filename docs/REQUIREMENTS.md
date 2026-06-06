# Detection & Precision Tracking System — Project Requirements

**Author:** Dr. Kolaman
**Repository:** `git@github.com:DrKolaman/drone_tracking.git`
**Status:** Draft · 2026-06-06

---

## 1. Context

We are given a single short video (`data/source.mp4`,
https://youtube.com/shorts/wXDXFysTJIk): **top-down aerial/drone footage** in
which the tracked target is a **small moving person** (a few pixels), seen from
above. 360×640, 30 fps, 1200 frames (~40 s). The task is to design and implement
a system that **detects** and **continuously tracks** that target in real time,
keeping a stable identity even when it is hard to see.

The footage is deliberately adversarial:

- The target may be **partially or fully occluded** and is **small / low-contrast
  against cluttered terrain** — hard to distinguish from the background.
- The **camera moves** (pan/rotate, with motion-blur bursts), so detection must
  **compensate global motion**, not assume a static background.
- The clip contains **hard discontinuities**: a fast move + motion blur, a large
  optical **zoom**, **jumps** to other map areas, and a **B/W↔red colour switch**.

Therefore the system must combine **spatial reasoning** (per-frame detection)
with **temporal reasoning** (track propagation across frames).

Guidance from the assignment-giver: *"That's part of the real world. Handle it
the best way you see fit — be creative. This is not expected to be
production-grade. Your time is finite. Importantly: explicitly document what you
chose not to solve, why, and how you would solve it."*

### 1.1 Source-video content timeline

Frame-by-frame (1-indexed); the cleanly-trackable segment is ~frames **0–609**.

| Frames | What happens |
|--------|--------------|
| 0 – 609 | Camera tracks the person (continuous, stable footage). |
| 610 – 614 | Camera suddenly drops/moves **down fast**, creating **large motion blur**. |
| 615 – 646 | Camera **stays in one place** (roughly static). |
| 647 – 744 | At **frame 647** a **large zoom-in** (~7×), then moves across the map. |
| 745 – 956 | At **frame 745** it **switches to another (unknown) area** of the map and follows the person. |
| 957 – 1029 | At **frame 957** it **jumps back** to the earlier area **and the scene colour changes B/W → red** (thermal). Person starts moving at **1010**. |
| 1030 – 1053 | At **frame 1030** the scene **switches back to B/W**; the person **leaves the frame at 1053**. |
| 1054 – 1193 | Camera **searches** (moving) **without seeing the person** until 1169; the person **leaves the scene at 1193**. |

**Discontinuities that matter for registration/tracking:** the 610–614 blur, the
**647 zoom** (single-frame scale jump that breaks frame-to-frame homography), the
**745 jump**, the **957 jump + colour switch**, and the **1030 colour switch**.

---

## 2. Goals

| # | Goal | Meaning |
|---|------|---------|
| G1 | **Real-time detection** | Immediate per-frame identification of the target under occlusion and background clutter. |
| G2 | **Real-time tracking** | Maintain continuous tracking and preserve target identity despite occlusions and visibility changes. |
| G3 | **Robustness to discontinuity** | Behave sensibly across the scene cut and thermal→colour switch — without producing nonsense identities across the boundary. |
| G4 | **Honest, documented scope** | Clearly state what is and isn't solved, with reasoning and a path to solving the rest. |

---

## 3. Scope

### 3.1 In scope

- A runnable **detection + tracking pipeline** over the input video.
- **Deep-learning detector block** (object detection, person class) — the
  required DL component.
- **BoT-SORT tracker with appearance ReID** to preserve identity through
  occlusion and appearance/scale change (motion-only ByteTrack as documented
  baseline).
- **Discontinuity handling = detect-and-reset-continue**: cheaply detect hard
  cuts and the thermal→colour flip, **reset the tracker** at each boundary so
  identities do not bleed across, then keep processing the full clip.
- **Target-lock**: highlight the single most persistent track so the demo shows
  one tracked subject rather than every passer-by.
- An **annotated demo video** of the detection + tracking results.
- The **written deliverables** (Section 5) packaged as a **≤6-page PDF or
  ≤15-slide deck**.

### 3.2 Out of scope (documented, not solved — see deliverable #4/#5)

- **Robust cross-cut re-identification** — re-acquiring the *same* target
  identity after the hard cut / thermal→colour switch. We reset instead.
- **Custom model training / fine-tuning** on this domain — we use off-the-shelf
  weights given finite time.
- **Quantitative MOT scoring against ground truth** — no labelled bounding
  boxes exist for this clip (see Section 6).
- **Multi-camera / multi-target hand-off**, deployment hardening, and any
  production-grade reliability work.

---

## 4. Functional requirements

| ID | Requirement |
|----|-------------|
| FR1 | Ingest the input video frame-by-frame at native resolution and FPS. |
| FR2 | Detect persons per frame with a deep-learning detector; expose a confidence threshold. |
| FR3 | Associate detections across frames into stable tracks with persistent IDs (BoT-SORT + ReID). |
| FR4 | Detect scene discontinuities: hard cut (appearance) and thermal↔colour mode flip. |
| FR5 | On a detected discontinuity, reset tracker state so IDs do not propagate across it. |
| FR6 | Select and visually highlight a single "locked" target (most persistent track). |
| FR7 | Render an annotated output video (boxes, IDs, locked-target emphasis) and write a run log. |
| FR8 | Report per-run performance and quality signals (Section 6). |
| FR9 | Be configurable via CLI: source, model, tracker, confidence, output path, device. |

## 4.1 Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR1 | **Real-time-capable** on the available NVIDIA RTX 3050 laptop GPU (target ≥ ~25 FPS at the clip's resolution with the chosen model size). |
| NFR2 | Reproducible: pinned dependencies, single documented run command. |
| NFR3 | Modular blocks (detector / tracker / discontinuity / visualisation) so any block is swappable. |
| NFR4 | Readable, maintainable code suitable for review on GitHub. |

---

## 5. Deliverables (assignment-mandated)

1. **System Design** — high-level architecture; every functional block with
   inputs/outputs, algorithms, assumptions, trade-offs, risks/limitations.
   Includes ≥1 deep-learning block.
2. **Deep-Learning Analysis** — literature review of 2–3 approaches for the DL
   block, model selection + justification, pros/cons, limitations.
3. **Success Criteria** — measurable metrics per block + end-to-end (Section 6).
4. **Failure Analysis** — structured failure modes, root causes, impact.
5. **Improvement Suggestions** — proposed improvements with reasoning.

**Submission package:** ≤6-page PDF or ≤15-slide deck · running code (this
GitHub repo) · demo video.

---

## 6. Success criteria & metrics

Because **no ground-truth labels exist** for this clip, we separate metrics that
can be **genuinely measured** from those we can only **define** (and would
measure given annotations). This honesty is itself a deliverable expectation.

### 6.1 Measurable now (no labels needed)

| Metric | Block | Why it matters |
|--------|-------|----------------|
| Detector + tracker latency (ms) / FPS | Detector, Tracker | G1/G2 real-time goal (NFR1). |
| Detection confidence distribution | Detector | Detector health under clutter/occlusion. |
| Active track count over time | Tracker | Clutter / over-segmentation signal. |
| ID-switch count at discontinuities | Discontinuity | Validates reset behaviour (FR5). |
| Track fragmentation (mean track lifetime) | Tracker | Stability through occlusion. |
| Target-lock continuity (fraction of continuous segment the locked ID persists) | Target-lock | End-to-end proxy for "preserve identity". |

### 6.2 Defined but not measured here (require ground-truth labels)

| Metric | What it would tell us |
|--------|-----------------------|
| Precision / Recall / mAP@0.5 | Detection accuracy. |
| MOTA | Combined FP/FN/ID-switch tracking accuracy. |
| IDF1 | Identity preservation quality. |
| HOTA | Balanced detection + association quality. |

---

## 7. Constraints & assumptions

- **Time-boxed**, single-developer take-home; off-the-shelf over custom where
  possible.
- **Environment:** Python 3.12, venv at `/project/.venv`, NVIDIA RTX 3050
  laptop GPU, Ultralytics (YOLO + built-in BoT-SORT/ByteTrack), OpenCV.
- **Assumption:** a single primary target of interest per continuous segment.
- **Assumption:** the continuous first half is where tracking quality is judged;
  the post-cut content is handled gracefully but not tracked as the same target.

---

## 8. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Discontinuity detector mis-fires (false cut) | Spurious tracker resets, ID churn | Conservative thresholds; log every event for inspection. |
| Discontinuity detector misses a soft transition | IDs bleed across scenes | Use two complementary signals (histogram correlation + colour-mode flip). |
| Occlusion longer than ReID memory | Lost/Swapped identity | Tune BoT-SORT track-buffer; document residual limitation. |
| No labels → can't prove accuracy | Weaker quantitative claims | Report measurable proxies; define the label-based metrics and the eval plan. |
| Real-time target not met at full model size | Misses G1/G2 | Use smaller YOLO size; report the FPS/accuracy trade-off. |

---

## 9. Acceptance

The project is acceptable when: the pipeline runs end-to-end on the clip via a
single documented command (FR1–FR9), produces the annotated demo video with a
stable locked target across the continuous segment, resets cleanly at the
discontinuity, reports the Section 6.1 metrics, and the five written
deliverables are complete within the page/slide limit — with out-of-scope items
explicitly documented per G4.
