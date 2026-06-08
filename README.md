# Detection & Precision Tracking System

Single-target **detection and precision tracking** of a small, low-contrast person in
**top-down aerial / drone thermal footage**, under heavy camera motion and hard scene
discontinuities (motion-blur burst, a ~3× FOV switch, map jumps, and a B/W↔red thermal
colour change). Built for the Elbit Systems "Detection and Precision Tracking System"
take-home exercise.

> **Source clip:** `data/source.mp4` — 360×640, 30 fps, 1200 frames (~40 s); from
> <https://youtube.com/shorts/wXDXFysTJIk>. The target is a **few-pixel** person seen
> from above in near-grayscale thermal, frequently occluded and hard to separate from
> the background. The clip is not redistributed here (see [Getting the data](#getting-the-data)).

---

## Approach in one picture

The camera moves, so a static-background assumption fails. The system **compensates
global camera motion** (registers each frame into a stabilized map), detects
**independent motion** there, and preserves **identity** with a deep-learning
appearance model — combining *spatial* reasoning (per-frame detection) with *temporal*
reasoning (track propagation, re-acquisition, and a short freeze through occlusion).

```
frame ─▶ colour-normalise ─▶ global-motion registration ─▶ segment @ discontinuities
            (red→B/W)            (LK + RANSAC homography)        (registration collapse)
                                          │
                         stabilized "map" coordinates
                                          │
        ┌──────────────┴───────────────┐
   MOG2 motion detection            DINOv3 ViT-S/16            ◀── deep-learning block
   (+ coverage validity mask)       appearance embeddings
        └──────────────┬───────────────┘
                                          ▼
        multi-identity association  (motion-near · appearance re-ID · spawn-new · HOLD)
                                          ▼
                       annotated video  +  per-frame track log
```

The **deep-learning block** is the **DINOv3 (ViT-S/16) appearance embedder** used for
discriminative-margin re-identification. See
[`docs/deliverable-2-dl-analysis.md`](docs/deliverable-2-dl-analysis.md) for the model
survey and the (negative) SGLATrack single-object-tracker trial.

---

## Results

**Demo videos:** tracker → <https://youtu.be/rEoeuRza6Y4> · scene-map build → <https://youtu.be/1hKg0ZwIBbQ>
(or run `src/track_full_clip.py` to regenerate `output/track_full_clip.mp4` locally).

| Metric (full clip, 1200 frames) | Value |
|---|---|
| Segments detected at discontinuities | 4 |
| Target identities | 3 (id1 first-half person · id2 the post-745 object · id3 the 1170 reappearance) |
| Box-shown coverage | 916 / 1200 ≈ **76 %** |
| Continuous first half (`track_dino_reid`, 0–646) | ~83 % box-shown, no false far-jumps |
| Appearance embedder | DINOv3 ViT-S/16, fp16 on a 4 GB RTX 3050 |

Honest scope: the *continuous* segment is tracked well; cross-cut **re-identification**
of the *same* identity after a long search gap / thermal switch is **not** solved (a
returning target is given a fresh id). See the deliverables below for the full,
documented limitations.

---

## Repository layout

```
src/
  # core tracker
  track_full_clip.py    full-clip multi-identity tracker (the main deliverable)
  track_dino_reid.py    continuous-segment tracker (perfect on the clean 0–646 half)
  registration.py       GlobalMotionEstimator: Shi-Tomasi + LK + RANSAC homography/similarity
  colorfix.py           to_bw: red-thermal → B/W (highest-variance channel) — one modality
  motion_detector.py    MOG2 foreground in stabilized coords
  dino_embedder.py      DINOv3 ViT-S/16 appearance embeddings (the DL block)
  target_memory.py      TargetMemory / BackgroundMemory + discriminative-margin ReID
  bytetrack_shim.py     Ultralytics BYTETracker adapter (bootstrap + novel-object spawn)
  scene_cut.py          discontinuity detection (HSV-hist correlation + colour-mode flip)

  # scene map / mosaic (scene understanding across the discontinuities)
  build_map.py map_with_zoom.py map_segments.py loop_closure.py
  build_map_video.py full_map.py full_map_video.py
  scene_analysis.py dinov3_match.py zoom_geometry.py

  # exploratory / debug (not required to run the tracker or map)
  debug_motion_mask.py debug_match.py debug_sliding.py mask_debug.py
  mast3r_zoom.py measure_zoom.py test_scale_warp.py reid_bench.py blur_profile.py
  stitch_preview.py stabilize_preview.py pipeline.py detect_track.py track_stabilized.py

docs/      requirements + the five assignment deliverables (see below)
tests/     pytest regression / golden-master safety net (see tests/README.md)
```

---

## Deliverables (assignment)

| # | Deliverable | Document |
|---|---|---|
| ★ | **Submission** (≤6-page distillation of all five) | [`docs/SUBMISSION.md`](docs/SUBMISSION.md) |
| — | Requirements, video timeline, scope | [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) |
| 1 | System Design | [`docs/deliverable-1-system-design.md`](docs/deliverable-1-system-design.md) |
| 2 | Deep-Learning Analysis | [`docs/deliverable-2-dl-analysis.md`](docs/deliverable-2-dl-analysis.md) |
| 3 | Success Criteria | [`docs/deliverable-3-success-criteria.md`](docs/deliverable-3-success-criteria.md) |
| 4 | Failure Analysis | [`docs/deliverable-4-failure-analysis.md`](docs/deliverable-4-failure-analysis.md) |
| 5 | Improvement Suggestions | [`docs/deliverable-5-improvements.md`](docs/deliverable-5-improvements.md) |

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
```

The DINOv3 weights are **gated** on Hugging Face. Request access to
`facebook/dinov3-vits16-pretrain-lvd1689m` and export a token:

```bash
export HF_TOKEN=hf_xxx        # required for any run that uses the appearance block
```

GPU: tested on an RTX 3050 (4 GB), DINOv3 in fp16. CPU-only works for the map/registration
parts and the fast tests; the DINOv3 tracker is slow on CPU.

### Getting the data

The clip is not redistributed. Download the source short
(<https://youtube.com/shorts/wXDXFysTJIk>) and place it at `data/source.mp4`
(360×640, 30 fps). `data/`, `output/`, `*.mp4` and `*.pt` are git-ignored.

---

## Usage

```bash
# Full-clip multi-identity tracker -> output/track_full_clip.mp4
HF_TOKEN=$HF_TOKEN python3 src/track_full_clip.py --output output/track_full_clip.mp4

#   per-frame track log (frame,seg,id,state,cx,cy) for analysis / regression:
HF_TOKEN=$HF_TOKEN python3 src/track_full_clip.py --log-csv output/track_log.csv

# Clean continuous-segment tracker (0–646), the strongest result
HF_TOKEN=$HF_TOKEN python3 src/track_dino_reid.py --output output/track_dino_reid.mp4

# Scene map / mosaic with the 647-zoom link
HF_TOKEN=$HF_TOKEN python3 src/map_with_zoom.py --out output

# Motion-mask diagnostic video (no DINO): masks at several MOG2 thresholds
python3 src/debug_motion_mask.py --start 460 --end 646 --thresholds 16,30,50 \
    --out output/motion_mask_debug.mp4
```

All generated artifacts go to `output/` (git-ignored).

## Tests

A regression / golden-master safety net so the code can be refactored and reordered
while staying behaviour-equivalent on the clip. See [`tests/README.md`](tests/README.md).

```bash
python3 -m pytest -m "not slow"                  # fast unit/characterization (CPU, ~15 s)
HF_TOKEN=$HF_TOKEN python3 -m pytest -m slow     # full tracker golden (GPU + token)
RUN_DINOV3=1 HF_TOKEN=$HF_TOKEN python3 -m pytest # + gated DINOv3 map goldens
```

---

## License

See [`LICENSE`](LICENSE).
