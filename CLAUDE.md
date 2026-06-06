# drone_tracking

Detection & precision tracking of a small moving target in top-down aerial/drone
footage (`data/source.mp4`), under a moving camera and scene discontinuities.

See **`docs/REQUIREMENTS.md`** for the full requirements, the source-video content
timeline, and the key discontinuities (the 647 zoom, the 745/957 jumps, the
B/W↔red colour switches).

---

## Scene registration → homography stitching → map

Goal: register frames into a common coordinate frame to build a stitched map/mosaic,
robust to the clip's discontinuities. **Matching is the foundation** — feed good
correspondences to `cv2.findHomography` and chain them into a map.

### Matching strategy (use this)

**Within a continuous segment** (consecutive frames, same FOV/sensor):
ORB or DINOv3 homography — `FrameAligner` / `register_pair` in `scene_analysis.py`.
Cheap, dense, reliable. This is the backbone of the per-segment mosaic.

**Across a discrete FOV switch** (the 647 "zoom" — a dual thermal-sensor / step-zoom,
~3×, **not** a dolly): a single global homography on raw frames FAILS. Working recipe,
validated in `debug_match.py` / `debug_sliding.py`:
1. **Scale-correct first** — downscale the narrow (zoomed-in) frame toward the wide
   frame's scale (~1/3). This alone tripled inliers (9 → ~39). Scale, not appearance,
   was the dominant blocker.
2. **Sliding-window localization** — slide a window across the wide frame at a 10%
   stride; for each window, upscale to the narrow size and match; the best-scoring
   window is the footprint (`debug_sliding.py`; also `zoom_geometry.locate_narrow_in_wide`).
   For 647 the footprint is **middle-right** (center ≈ (250,320)).
3. **Match** the scale-corrected, localized window ↔ narrow frame with DINOv3
   (`dinov3_match.DenseMatcher.match`).
4. **Filter by parallel-bundle / translation consistency** — after scale-correction a
   correct match set is a near-pure translation, so keep only matches whose
   displacement agrees with the dominant one (parallel, equal-length lines). This is
   more robust than homography RANSAC, which over-fits scattered points.
5. **Estimate the homography / similarity** from the consistent matches.

**Residual limit:** the two thermal modes have a genuine cross-sensor appearance gap,
so appearance matchers (ORB, DINOv3 ViT-S/B, MASt3R) cap at **~9 consistent matches**
across 647↔646 even at matched scale. For clean cross-mode registration, add an
**appearance-invariant** matcher — mutual-information registration (init from the ~3×
scale + middle-right center) or edge/cross-spectral descriptors (RIFT / phase-congruency).

### Stitching & map building
- **Map of a continuous segment:** `build_map.py` — LK global-motion homography
  (`registration.GlobalMotionEstimator`, current->previous) chained to the first
  frame, canvas fit to the trajectory, then **feather blending** (each frame weighted
  by distance-to-border) so frame centres stay sharp while seams *and* per-frame
  thermal-AGC brightness jumps blend out. Use feathering, NOT naive averaging (blurry)
  or hard last-wins (visible frame-edge seams). Stitches frames 0->609 (start->zoom)
  into one coherent mosaic. Residual: large-scale AGC brightness shifts across a long
  pan need gain/exposure compensation (`cv2.detail` ExposureCompensator).
- **Per continuous segment (stabilisation):** `scene_analysis.stitch_segments`
  (coverage-gated anchor reset).
- **Use a SIMILARITY model (4-DOF), not a homography (8-DOF), for the chain.** The
  drone is nadir (straight down) over ~flat ground, so inter-frame motion is
  translation + rotation + scale. An 8-DOF homography lets noise/parallax inject
  *shear* that accumulates over hundreds of frames and visibly deforms the map (a
  footprint that should be a 90-degree rectangle comes out a parallelogram).
  `GlobalMotionEstimator(model="similarity")` (cv2.estimateAffinePartial2D) fixes it.
- **Skip motion-blurred frames in the chain.** Blurred frames (the 610-614 burst)
  yield bad transforms that tilt everything after them. Score each frame by
  variance-of-Laplacian and skip frames below ~0.35x the median (`build_map.chain`),
  bridging from the last sharp frame to the next sharp one.
- **Link the 647 zoom segment into the map** with the DINOv3 sliding-window matcher
  (`map_with_zoom.py`): slide a window over the last sharp wide frame, DINOv3-match
  each against 647, fit a *similarity* 647->wide from the best window's
  correspondences, and anchor the zoomed segment there. The link scale (~0.39 =>
  ~2.5x) independently confirms the zoom. Limited by the thermal gap (~20 matches);
  MI/edge matching would firm it up.
- **A JUMP spawns a new segment placed away** (`map_segments.py`): a jump (744->745)
  lands on a non-overlapping area, so it can't be stitched in. Detected by the
  classifier (registration collapse + sharp frames + low DINOv3 global cos, e.g.
  inliers 10 / cos 0.56), the current segment is finalised and a fresh one is stitched
  and laid out beside it (they share no coordinates). Segment 2 (745-956) auto-stops at
  the next discontinuity (957). If a jumped-to area is a *revisit* of earlier ground,
  loop-closure (DINOv3 global descriptors) would relink it instead of placing it apart.
- **Cross-segment / cross-FOV links:** register one segment's frame to another's using
  the scale-corrected sliding-window matching above, to tie segments into one map.
- **Loop closure:** DINOv3 global descriptors (`scene_analysis` causal loop-closure) to
  detect revisits and close the map.

### Key empirical findings (don't re-walk these dead ends)
- **647 is a discrete FOV/camera switch** (dual thermal sensor or step-zoom), **~3×**
  (data-driven: 2.4–3.5× across methods), centered **middle-right**. NOT a moving-camera
  dolly, NOT the doc's ~7×. MASt3R's focal ratio ≈ 1 ⇒ it reads the switch as 3D
  geometry, not a lens focal change.
- A single global 2D homography/similarity **cannot** register 646↔647 (parallax +
  cross-sensor appearance). Raw-match consistency: ORB 4 inliers, DINOv3 ~9, MASt3R 37/214.
- **Scale gap was the dominant blocker** (fix by downscaling the narrow frame ~3×);
  the **thermal appearance gap is the residual blocker** (costs ~90% of matches even at
  matched scale — same-sensor self-test gives ~574 inliers vs ~39 cross-mode).
- DINOv3 ViT-B beats ViT-S only marginally; the appearance gap is the real cap, not model size.
- Precise zoom is **not** recoverable from imagery alone; payload spec / drone telemetry
  would give the exact FOV ratio.
- SfM is the WRONG tool for the FOV switch (no motion baseline). Pure-2D template /
  Fourier-Mellin / homography on raw frames all fail — see git history / `measure_zoom.py`.

### Modules (`src/`)
- `build_map.py` — stitched map of a continuous segment (LK similarity + blur-skip + feather blending).
- `map_with_zoom.py` — map that links the 647 zoom segment in via DINOv3 sliding-window.
- `map_segments.py` — multi-segment map; a detected jump spawns a new segment placed away.
- `build_map_video.py` — progressive map-build video (accumulate / skip-blur / link-zoom).
- `registration.py` — `GlobalMotionEstimator` (Shi-Tomasi + LK + RANSAC; `model="homography"|"similarity"`).
- `detect_track.py` — YOLO + BoT-SORT detection/tracking (the deliverable pipeline).
- `scene_cut.py` — discontinuity detection (HSV-hist correlation + colour-mode flip).
- `scene_analysis.py` — `FrameAligner` (consecutive homography), `stitch_segments`
  (per-segment mosaic), `Dinov3Embedder`, causal loop-closure, `register_pair` (ORB).
- `dinov3_match.py` — `DenseMatcher`: DINOv3 dense patch correspondences (cross-scale).
- `zoom_geometry.py` — `characterize_from_correspondences` (H-vs-F verdict, scale@FoE),
  `locate_narrow_in_wide` (embedding localization).
- `debug_sliding.py` — sliding-window cross-mode match debug → video.
- `debug_match.py` — controlled match decomposition (isolates scale vs appearance).
- `mast3r_zoom.py` — MASt3R two-view matching + zoom (similarity scale + focal ratio).
- `measure_zoom.py`, `test_scale_warp.py`, `reid_bench.py` — zoom measurement, helpers,
  DINOv3 ReID-backbone benchmark.

### Environment / running
- DINOv3 weights are **gated** — run with `HF_TOKEN` set in the shell.
- GPU: RTX 3050 **4 GB** — DINOv3 fp16; MASt3R fp16@384 on GPU or CPU@512 fallback.
- MASt3R lives at `/project/mast3r_repo` (+ checkpoint `/project/mast3r/checkpoints/`);
  import via `sys.path`. The global git config rewrites https→ssh, so clone with
  `GIT_CONFIG_GLOBAL=/dev/null`.
- Common runs:
  - Map/stitch + loop-closure: `HF_TOKEN=... python3 src/scene_analysis.py --video data/source.mp4 --out out`
  - Cross-mode match debug video: `HF_TOKEN=... python3 src/debug_sliding.py --win 0.33 --stride 0.1`
  - Zoom (MASt3R): `python3 src/mast3r_zoom.py --pairs 646:647`
