"""Appearance memory bank for the tracked target (DINOv3 ReID).

The target frequently stops or creeps, so motion detection loses it for stretches
(see mask_debug.py). Instead of a single drifting EMA vector, we keep a small
**bank of distinct target views**: while the target is visibly tracked we add a
new embedding only when it is *novel* (so the bank spans the target's appearances
rather than collapsing to one), and the bank simply persists through gaps. When a
new motion detection appears, its similarity to the target is the **max cosine
over the bank** — so a target that stopped and reappeared is re-matched to a
stored view and keeps the same id.

All embeddings are L2-normalised, so cosine = dot product.
"""

from __future__ import annotations

import numpy as np


class TargetMemory:
    def __init__(self, capacity: int = 12, add_thresh: float = 0.92) -> None:
        # add_thresh: only store a new view if its best match to the bank is BELOW
        # this (i.e. it is novel). Higher => store more (more diverse) views.
        self.capacity = capacity
        self.add_thresh = add_thresh
        self._bank: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._bank)

    def seed(self, embeddings: np.ndarray) -> None:
        """Initialise the bank from bootstrap embeddings (greedy novelty subset)."""
        for e in embeddings:
            self._consider(e, force_first=True)

    def score(self, emb: np.ndarray) -> float:
        """Max cosine similarity of `emb` against the bank (0 if empty)."""
        if not self._bank:
            return 0.0
        return float(max(b @ emb for b in self._bank))

    def _consider(self, emb: np.ndarray, force_first: bool = False) -> bool:
        """Add `emb` if novel (or if bank empty). Returns True if stored."""
        if not self._bank:
            self._bank.append(emb.astype(np.float32))
            return True
        best = self.score(emb)
        if best < self.add_thresh:
            self._bank.append(emb.astype(np.float32))
            if len(self._bank) > self.capacity:
                self._bank.pop(0)        # evict oldest (FIFO)
            return True
        return False

    def update(self, emb: np.ndarray) -> bool:
        """Called when the target is confidently matched this frame."""
        return self._consider(emb)

    def consolidate(self, embeddings, dedup_thresh: float = 0.95,
                    capacity: int | None = None) -> None:
        """Rebuild the bank from current views + a track's worth of new features,
        keeping a DIVERSE set of UNIQUE views and pruning near-duplicates.

        Start from the medoid (most representative), then greedily add the most
        unique remaining feature (lowest max-cosine to the kept set) until what's
        left is all near-duplicate (>= dedup_thresh) or capacity is reached.
        """
        cap = capacity or self.capacity
        pool = list(self._bank) + [np.asarray(e, np.float32) for e in embeddings]
        if not pool:
            return
        P = np.stack(pool)
        S = P @ P.T
        kept = [int(S.sum(1).argmax())]                     # medoid
        while len(kept) < min(cap, len(pool)):
            closeness = S[:, kept].max(1)                   # each item's closeness to kept
            c = int(closeness.argmin())                     # most unique remaining
            if closeness[c] >= dedup_thresh:                # rest are duplicates
                break
            kept.append(c)
        self._bank = [P[i] for i in kept]


class BackgroundMemory:
    """Capped bank of NON-target (distractor / terrain) features.

    Used for discriminative-margin matching: a candidate is the target only if it
    is markedly more similar to the target bank than to this background. Stores
    recent rejected blobs (light novelty dedup, FIFO eviction) so it reflects the
    current scene's distractors.
    """

    def __init__(self, capacity: int = 100, dedup_thresh: float = 0.97) -> None:
        self.capacity = capacity
        self.dedup_thresh = dedup_thresh
        self._bank: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._bank)

    def score(self, emb: np.ndarray) -> float:
        if not self._bank:
            return 0.0
        return float(max(b @ emb for b in self._bank))

    def add(self, emb: np.ndarray) -> None:
        if self._bank and self.score(emb) >= self.dedup_thresh:
            return                                          # already represented
        self._bank.append(np.asarray(emb, np.float32))
        if len(self._bank) > self.capacity:
            self._bank.pop(0)
