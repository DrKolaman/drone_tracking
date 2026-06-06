"""Unit tests for the ByteTrack detection shim."""

import numpy as np

from bytetrack_shim import Detections, detections_from_boxes, make_tracker


def test_detections_len_and_xywh():
    d = Detections([[0, 0, 10, 20], [100, 100, 110, 140]], [0.9, 0.8], [0, 0])
    assert len(d) == 2
    assert np.allclose(d.xywh[0], [5, 10, 10, 20])      # cx,cy,w,h
    assert np.allclose(d.xywh[1], [105, 120, 10, 40])


def test_detections_boolean_slice():
    d = Detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.7], [0, 0])
    sub = d[np.array([True, False])]
    assert len(sub) == 1
    assert np.allclose(sub.xyxy[0], [0, 0, 10, 10])


def test_no_xywhr_attr():
    # ByteTrack picks the rotated path if .xywhr exists; we must NOT expose it.
    d = Detections([[0, 0, 10, 10]], [0.9], [0])
    assert not hasattr(d, "xywhr")


def test_detections_from_boxes_conf_band():
    d = detections_from_boxes([[0, 0, 6, 6], [0, 0, 300, 300]], frame_area=360 * 640)
    assert len(d) == 2
    assert (d.conf >= 0.70).all() and (d.conf <= 0.95).all()
    assert (d.cls == 0).all()


def test_detections_from_boxes_empty():
    d = detections_from_boxes([], frame_area=1.0)
    assert len(d) == 0


def test_make_tracker():
    t = make_tracker(30.0)
    assert hasattr(t, "update")
