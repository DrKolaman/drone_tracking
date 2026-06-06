"""Shared fixtures for the tracker test suite.

Fast unit tests use synthetic data (no video, no DINOv3). The slow
characterization + golden tests run the real tracker once (session-scoped) and
need HF_TOKEN + a GPU; they skip cleanly if HF_TOKEN is unset.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
VIDEO = "/project/data/source.mp4"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs the full tracker on the video (needs HF_TOKEN + GPU)")


@pytest.fixture
def textured():
    """Factory: a feature-rich grayscale image (good for goodFeaturesToTrack)."""
    import cv2

    def _make(seed=0, h=240, w=320):
        rng = np.random.default_rng(seed)
        img = np.full((h, w), 127, np.uint8)
        for _ in range(80):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.rectangle(img, (x, y),
                          (x + int(rng.integers(6, 34)), y + int(rng.integers(6, 34))),
                          int(rng.integers(0, 255)), -1)
        return img
    return _make


@pytest.fixture
def unitvec():
    """Factory: a deterministic L2-normalised float32 vector (a DINOv3-like embedding)."""
    def _make(seed, dim=384):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
    return _make


@pytest.fixture(scope="session")
def tracker_rows():
    """Run the real tracker once over frames 0-475 and return the per-frame log rows."""
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set; skipping slow tracker run")
    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    cmd = [sys.executable, str(SRC / "track_dino_reid.py"), "--source", VIDEO,
           "--max-frames", "475", "--log-csv", out, "--output", "/tmp/_chartest.mp4"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"tracker run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return list(csv.DictReader(open(out)))
