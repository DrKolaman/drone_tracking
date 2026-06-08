# Detection & Precision Tracking System — Submission

**Repo:** `github.com/DrKolaman/drone_tracking` · **Code:** `src/` · **Detail:** `docs/deliverable-1..5`
**Demo videos:** tracker → <https://youtu.be/rEoeuRza6Y4> · scene-map build → <https://youtu.be/1hKg0ZwIBbQ>

**Problem.** Track one person in `data/source.mp4` — 360×640, 30 fps, ~40 s of **top-down aerial thermal** footage. The target is a **few-pixel (~20–35 px), low-contrast** blob, frequently occluded, hard to separate from background. The camera moves and the clip has hard **discontinuities**: a motion-blur burst (610–614), a **~3× FOV switch** (647), a **jump** to another area with a *different* object (745–956), a **jump-back + B/W→red thermal colour switch** (957), back to B/W (1030), target leaves (1052) and reappears elsewhere (1170). **No ground-truth labels exist.** The system must combine *spatial* (per-frame detection) and *temporal* (identity over time) reasoning.

---

## 1. System Design

A static-background assumption fails (the camera moves), so we **compensate global motion** into a stabilized map, detect **independent motion** there, and preserve identity with a **deep-learning appearance** model.

```
frame ─▶ colour-norm(red→B/W) ─▶ global-motion registration ─▶ segment @ discontinuity
                                       (LK + RANSAC H)            (registration collapse)
                                   stabilized MAP coords
                        ┌──────────────┴───────────────┐
                MOG2 motion + coverage mask     DINOv3 ViT-S/16 embeddings ◀─ DL block
                        └──────────────┬───────────────┘
            multi-id association (motion-near · appearance re-ID · spawn · HOLD) ─▶ video + log
```

| # | Block | In → Out | Algorithm | Key assumption / risk |
|---|---|---|---|---|
| 1 | Colour-normalise | BGR → B/W | `to_bw`: highest-variance channel | Red thermal & B/W are the *same* scene; unifies modality |
| 2 | Global-motion reg. | 2 grays → 3×3 H | Shi-Tomasi + LK + RANSAC (homography/similarity) | Nadir, ~flat ground; fails on fast pans / FOV switch |
| 3 | Segmentation | H-stream → segment ids | new segment only on registration collapse | A real cut vs. a hard but continuous move |
| 4 | Blur-skip | frame → keep/skip | var-of-Laplacian < 0.19×median | Skips the 610–612 burst |
| 5 | Motion detect | map → blobs | MOG2 + coverage-age validity mask (≥12 frames, 21px erode) | Suppresses freshly-revealed ground; can blind the leading edge |
| **6** | **Appearance embed (DL)** | crop → 384-d vec | **DINOv3 ViT-S/16**, fp16; mean-pool 7×7 patch tokens, L2-norm | Tiny thermal crop is information-poor |
| 7 | Identity model | vec → margin | TargetMemory − BackgroundMemory = **discriminative margin** (real ≥0.096, false ≤0) | Raw cosine can't separate; margin can |
| 8 | Association | blobs+vecs → track | R1 motion-near (40px gate) · R2 re-ID active (HOLD-protected) · R3 re-ID any id · R4 spawn novel; HOLD ≤2 s | Single primary target per segment |
| 9 | Output | → mp4 + CSV | per-id boxes, state, `--log-csv` | — |

A parallel **scene-map** subsystem stitches each segment (`build_map`, feather-blended mosaic), links the 647 zoom via DINOv3 sliding-window (`map_with_zoom`), and loop-closes the 957 jump-back to a Segment-1 keyframe (`loop_closure`). *(Full per-block I/O, trade-offs, risks: `docs/deliverable-1-system-design.md`.)*

---

## 2. Deep-Learning Analysis

**Chosen DL block: DINOv3 (ViT-S/16) appearance embeddings for ReID** — not a detector. Rationale: the target's defining cue is *independent motion*, not appearance; detection is done by ego-motion-compensated MOG2, and the DL model supplies *identity*.

| Approach | Idea | Verdict here |
|---|---|---|
| A. YOLO + ByteTrack/BoT-SORT (spec baseline) | detect `person` per frame, associate | **Cannot start** — YOLOv8-n (COCO) returns **0** detections even at conf 0.05 (target too small/off-domain) |
| B. ViT single-object tracker (OSTrack/SGLATrack) | template-match from one init box | **Trialled & failed** — locks ~150 frames then drifts onto a bright **static distractor** with *high* confidence; no re-detection |
| C. Ego-motion motion detection + **DINOv3 ReID** *(used)* | stabilize → motion blobs → DINOv3 identity | Motion *finds* the target; DINOv3 margin *holds* identity through the clean segment |

DINOv3 over a supervised ReID net: it is **self-supervised / domain-agnostic**, needs no `person`-ReID labels, and ViT-B ≈ ViT-S here (the cap is the few-pixel crop, not model size). *(Survey + SGLATrack trial: `docs/deliverable-2-dl-analysis.md`.)*

---

## 3. Success Criteria

No labels exist, so we split **measurable-now** from **defined-but-needs-labels**.

| Measurable now (proxy) | Block | Value (this clip) |
|---|---|---|
| Registration inliers: continuous vs jump | 2/3 | 588 (742→743) vs 10 (744→745) — clean separator |
| Blur-skip set | 4 | {610, 611, 612} |
| Zoom-link scale / loop-closure cosine | map | 0.39 (≈2.5×) / 0.765 to kf≈300 |
| Discriminative-margin separation | 7 | real ≥0.096, false ≤0 |
| Target-lock continuity (id1, 0–646) | e2e | ~83% box-shown, **no false far-jumps** |
| Full-clip coverage / identities / re-acq | e2e | 916/1200 ≈ **76%** · 3 ids · 2 re-acq |
| Throughput (real-time NFR ≥25 FPS) | e2e | DINOv3 fp16 on RTX 3050 4 GB *(SGLATrack trial ~14 FPS)* |

| Needs labels (defined, eval plan) | Tells us |
|---|---|
| Precision / Recall / mAP@0.5 | detection accuracy |
| MOTA / IDF1 / HOTA | tracking + identity quality |

Plan: annotate ~200 frames → score with `py-motmetrics`. *(Full tables: `docs/deliverable-3-success-criteria.md`.)*

---

## 4. Failure Analysis

Two root causes explain almost everything: **(RC-A)** the target's signal is indistinguishable from camera-motion/registration residual; **(RC-B)** a ~30 px thermal crop is too information-poor for appearance to discriminate.

| Failure | Where | Root cause | Impact | Status |
|---|---|---|---|---|
| Under-tree detection loss | 476–513 | RC-A: low-contrast under canopy; MOG2 (thr 50) sees nothing | tracker HOLDs (frozen) though target moves | open (HOLD) |
| Camera-motion / reg. residual noise | fast pans (514–519, 1053–1169) | RC-A: imperfect H floods map as false motion (f518 ≈555 blobs @thr16) | erratic jumps; no threshold separates target from flood | open |
| Appearance can't discriminate tiny crops | under tree | RC-B: DINOv3 margins ≈0 for target *and* noise | appearance gate unreliable where most needed | inherent |
| **Cross-cut re-ID** (headline) | 1170 reappearance | long gap (117 f ≫ 60-f HOLD) + B/W→red gap → re-ID below margin | returning target gets a **new id** (id3≠id1) | out of scope (brief) |
| Zoom-continuation | 647 | ~3× FOV/sensor switch breaks H + appearance | id1 not continued across zoom | out of scope |
| Noise-filter attempts (negative result) | — | any filter strong enough to cut noise cuts the faint target too (temporal split id1→id2; erosion deleted it) | reverted to HOLD-under-tree | reverted |
| Leading-edge-of-pan blindness | follow-pans | validity mask suppresses detection where a followed target rides the edge | earlier id2→id3 split | mitigated (relax disc) |
| Detector can't start | all | YOLO-COCO blind to the target | no tracking-by-detection baseline | see §2 |
| No ground truth | all | no labels | MOTA/IDF1/HOTA unmeasurable | see §3 |

*(Per-mode detail: `docs/deliverable-4-failure-analysis.md`.)*

---

## 5. Improvement Suggestions (ranked by leverage)

| # | Improvement | Addresses | Effort |
|---|---|---|---|
| 1 | **Fine-tuned small-object detector** (YOLOv8 + **SAHI** tiled inference) on ~200 annotated frames | detector-can't-start → unlocks YOLO+BoT-SORT | med (needs labels) |
| 2 | **Better stabilization**: 4-DOF similarity (no shear), exposure comp, **reject high-residual frames** | camera-motion noise *at source* | low–med |
| 3 | **Cross-modal matcher** (mutual-information / RIFT / phase-congruency), init from ~3× + footprint | 647 zoom link, 957/1170 cross-thermal re-ID | med |
| 4 | **Independent-motion gate** validating any appearance lock | static-distractor lock (SGLATrack) | low |
| 5 | **Telemetry / IMU fusion** (exact FOV, ego-motion) | zoom-factor ambiguity, fast-pan reg. | low (if data) |
| 6 | **Annotate ~200 frames** | enables #1 + MOTA/IDF1/HOTA eval | med |
| 7 | **fp32 / deterministic embedder** | bit-exact test reproducibility | trivial |

*(Reasoning per item: `docs/deliverable-5-improvements.md`.)*

---

## Results & Demo

- **Demo video — tracker:** <https://youtu.be/rEoeuRza6Y4> (full-clip annotated run, `output/track_full_clip.mp4`, 1200 frames, all discontinuities; clean-segment highlight `output/track_dino_reid.mp4`, 0–646).
- **Demo video — scene-map build:** <https://youtu.be/1hKg0ZwIBbQ> (progressive stabilized mosaic with the 647-zoom link).
- **Run:** `HF_TOKEN=… python3 src/track_full_clip.py --output output/track_full_clip.mp4` (see `README.md`).
- **Tests:** `pytest -m "not slow"` (fast) and `pytest -m slow` (golden-master) pin current behaviour for safe refactoring.
- **Honest scope:** the continuous segment is tracked well; robust cross-cut re-identification after a long gap / thermal switch is documented as **not solved** (Failure §4) with the path to solve it (Improvements §5).
