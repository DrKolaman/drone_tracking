"""DINOv3 matching / zoom tests. Gated: run with RUN_DINOV3=1 (GPU + HF_TOKEN).

These pin the cross-scale matching + zoom-localisation behaviour the map relies on.
"""
import cv2
import pytest

from conftest import needs_dinov3, needs_video

pytestmark = [needs_dinov3, needs_video, pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def dm():
    from dinov3_match import DenseMatcher
    return DenseMatcher(longside=1024)


def test_dense_match_handles_4x_self_scale(dm, grab):
    # DINOv3 should match a frame to its own 4x downscale (scale-invariance sanity).
    f = grab(647)
    h, w = f.shape[:2]
    small = cv2.resize(f, (w // 4, h // 4))
    pa, pb = dm.match(f, small, sim_floor=0.5)
    assert len(pa) >= 100


def test_locate_narrow_in_wide_scale_range(dm, grab):
    from scene_analysis import Dinov3Embedder
    from zoom_geometry import locate_narrow_in_wide
    loc = locate_narrow_in_wide(grab(646), grab(647), Dinov3Embedder())
    assert loc is not None
    scale = loc[0]
    assert 2.0 <= scale <= 5.5            # the 647 FOV switch is ~3x (eyeballed ~4x)


def test_zoom_link_scale_plausible(dm, grab):
    from map_with_zoom import link_via_sliding
    L, inl, box = link_via_sliding(grab(646), grab(647), dm)
    assert L is not None and inl >= 8
    import numpy as np
    scale = float(np.sqrt(L[0, 0] ** 2 + L[0, 1] ** 2))   # narrow->wide shrink
    assert 0.18 <= scale <= 0.5          # => zoom ~2-5.5x
    # GOLDEN (seeded): link ~20 inliers, scale ~0.39 (zoom ~2.5x), footprint middle-right
    assert box[0] > grab(646).shape[1] * 0.4             # footprint x in right half-ish


def test_golden_loop_closure_957(dm, grab):
    """957 jump-back loop-closes to a Segment-1 keyframe (cross-modal, colour-filtered)."""
    import numpy as np
    from colorfix import to_bw
    from scene_analysis import Dinov3Embedder
    from build_map import chain
    emb = Dinov3Embedder()
    wide = [to_bw(grab(i)) for i in range(0, 647, 20)]   # coarse keyframes
    Hs, kept = chain([to_bw(grab(i)) for i in range(0, 647)], 25)  # for indices only
    kfs = [to_bw(grab(k)) for k in kept[::20]]
    sims = emb.embed(kfs) @ emb.embed([to_bw(grab(957))])[0]
    best_kf = kept[::20][int(sims.argmax())]
    assert best_kf == pytest.approx(300, abs=60)          # GOLDEN keyframe ~300
    assert 0.70 <= float(sims.max()) <= 0.82              # GOLDEN cos ~0.765
