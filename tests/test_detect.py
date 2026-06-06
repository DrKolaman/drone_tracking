"""Unit tests for the MOG2 blob detector (detect())."""

import cv2
import numpy as np

from track_dino_reid import detect

H, W = 120, 160


def _primed_mog2():
    mog2 = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=16, detectShadows=False)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    validity = np.full((H, W), 255, np.uint8)
    bg = np.zeros((H, W, 3), np.uint8)
    for _ in range(15):                       # learn the empty background
        detect(bg, validity, mog2, k3, 5, 0.5 * W * H)
    return mog2, k3, validity, bg


def test_detect_finds_moving_blob():
    mog2, k3, validity, bg = _primed_mog2()
    fr = bg.copy()
    cv2.rectangle(fr, (70, 50), (82, 68), (255, 255, 255), -1)   # ~12x18 blob
    boxes, cents = detect(fr, validity, mog2, k3, 5, 0.5 * W * H)
    assert len(boxes) >= 1
    cx, cy = min(cents, key=lambda c: (c[0] - 76) ** 2 + (c[1] - 59) ** 2)
    assert abs(cx - 76) < 15 and abs(cy - 59) < 15


def test_detect_min_area_filters():
    mog2, k3, validity, bg = _primed_mog2()
    fr = bg.copy()
    cv2.rectangle(fr, (70, 50), (82, 68), (255, 255, 255), -1)
    # impossibly large min_area -> the blob is filtered out
    boxes, _ = detect(fr, validity, mog2, k3, 100000, 0.5 * W * H)
    assert len(boxes) == 0


def test_detect_validity_masks_outside():
    mog2, k3, validity, bg = _primed_mog2()
    validity = np.zeros((H, W), np.uint8)     # nothing valid
    fr = bg.copy()
    cv2.rectangle(fr, (70, 50), (82, 68), (255, 255, 255), -1)
    boxes, _ = detect(fr, validity, mog2, k3, 5, 0.5 * W * H)
    assert len(boxes) == 0
