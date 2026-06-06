"""Shared fixtures + markers for the map / DINOv3 pipeline tests.

Most tests are pure/fast (no GPU). DINOv3 tests are gated behind RUN_DINOV3=1
(they need the gated weights + GPU and are slow).
"""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
VIDEO = os.path.join(ROOT, "data", "source.mp4")

RUN_DINOV3 = os.environ.get("RUN_DINOV3") == "1"
needs_dinov3 = pytest.mark.skipif(not RUN_DINOV3, reason="set RUN_DINOV3=1 (GPU + HF_TOKEN) to run")
needs_video = pytest.mark.skipif(not os.path.exists(VIDEO), reason="data/source.mp4 not present")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs DINOv3 weights + GPU (RUN_DINOV3=1)")
    config.addinivalue_line("markers", "slow: slower integration test")


@pytest.fixture(autouse=True)
def _determinism():
    """Seed OpenCV's RANSAC + numpy so registration/map outputs are reproducible.

    This is what makes the golden-master tests meaningful: an optimisation that
    preserves behaviour produces identical results; one that changes it fails.
    """
    cv2.setRNGSeed(0)
    np.random.seed(0)
    yield


@pytest.fixture(scope="session")
def read_frames():
    """Factory: sequentially read raw BGR frames [lo, hi] from the clip."""
    def _read(lo, hi):
        cap = cv2.VideoCapture(VIDEO)
        out, i = [], 0
        while True:
            ok, f = cap.read()
            if not ok or i > hi:
                break
            if i >= lo:
                out.append(f)
            i += 1
        cap.release()
        return out
    return _read


@pytest.fixture(scope="session")
def make_texture():
    """Factory: a deterministic, corner-rich BGR image (good for LK / ORB)."""
    def _make(h=320, w=480, seed=0, n=350):
        rng = np.random.default_rng(seed)
        img = np.full((h, w, 3), 40, np.uint8)
        for _ in range(n):
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.circle(img, c, int(rng.integers(3, 16)),
                       tuple(int(x) for x in rng.integers(60, 255, 3)), -1)
        return img
    return _make


@pytest.fixture(scope="session")
def grab():
    """Factory: read a specific frame index from the committed clip."""
    def _grab(i):
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        cap.release()
        assert ok, f"could not read frame {i}"
        return f
    return _grab


def sim_matrix(scale=1.0, deg=0.0, tx=0.0, ty=0.0):
    """A 3x3 similarity (scale * rotation + translation)."""
    th = np.deg2rad(deg)
    c, s = scale * np.cos(th), scale * np.sin(th)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], np.float64)
