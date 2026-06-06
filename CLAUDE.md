# drone_tracking

## Source video — content timeline (`data/source.mp4`)

Top-down aerial/drone footage; the tracked **target is a small moving person**
(a few pixels), not an eye-level figure. 360×640, 30 fps, 1200 frames (~40 s).
Frame-by-frame description of what the camera/scene does (1-indexed by frame):

| Frames | What happens |
|--------|--------------|
| 0 – 609 | Camera tracks the person (continuous, stable footage). |
| 610 – 614 | Camera suddenly drops/moves **down fast**, creating **large motion blur**. |
| 615 – 646 | Camera **stays in one place** (roughly static). |
| 647 – 744 | At **frame 647** a **large zoom-in**, then moves across the map. |
| 745 – 956 | At **frame 745** it **switches to another area** of the map (an unknown location, elsewhere) and follows the person. |
| 957 – 1029 | At **frame 957** it **jumps back to the previous location** (the pre-jump area) **and the scene colour changes from B/W to red** (thermal palette). Stays red through 1029; the **person starts moving at frame 1010**. |
| 1030 – 1053 | At **frame 1030** the scene **switches back to B/W**; the **person goes out of the frame at frame 1053**. |
| 1054 – 1193 | Camera **searches** (moving) **without seeing the person** until frame 1169; the **person goes out of the scene at frame 1193**. |

### Key discontinuities (matter for registration / tracking)
- **610–614** — fast downward move + motion blur (registration stress).
- **647** — ~7× **zoom-in** (single-frame scale jump; breaks frame-to-frame homography → must re-anchor).
- **745** — hard **jump to a different map area** (discontinuity → re-anchor).
- **957** — **jump back** to the earlier area **and B/W→red colour switch**.
- **1030** — **red→B/W** switch.
- Target leaves view at **1053**; finally leaves the scene at **1193**.

The continuous, cleanly-trackable segment is roughly **frames 0–609**.
