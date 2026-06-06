"""Adapter to drive Ultralytics' ByteTrack with non-YOLO (MOG2) detections.

Ultralytics' `BYTETracker.update(results)` expects a YOLO `Results`-like object:
it reads `results.conf`, slices `results[bool_mask]`, and `init_track` reads
`results.xywh` (center x,y,w,h) and `results.cls`. We never produce rotated boxes
so we deliberately do NOT expose `xywhr` (its presence would change the path).

MOG2 blobs have no confidence, so we synthesise one from blob area: bigger, more
confident. `update` returns rows `[x1, y1, x2, y2, track_id, score, cls, idx]`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class Detections:
    """Minimal YOLO-Results stand-in that ByteTrack can consume."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, idx) -> "Detections":
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])


def detections_from_boxes(boxes, frame_area: float) -> Detections:
    """Build Detections from MOG2 [x1,y1,x2,y2] boxes with area-based confidence."""
    if not boxes:
        return Detections(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))
    xyxy = np.asarray(boxes, dtype=np.float32)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    # map area -> [0.5, 0.95]; saturates at 5% of the frame
    conf = 0.5 + 0.45 * np.clip(area / (0.05 * frame_area), 0.0, 1.0)
    cls = np.zeros(len(xyxy), dtype=np.float32)
    return Detections(xyxy, conf, cls)


def make_tracker(fps: float = 30.0, **overrides) -> BYTETracker:
    args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.8,
        track_buffer=30,
        fuse_score=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return BYTETracker(args, frame_rate=int(round(fps)))
