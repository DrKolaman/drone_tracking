"""Unit tests for the target memory bank, consolidation and background bank."""

from target_memory import BackgroundMemory, TargetMemory


def test_score_empty_is_zero(unitvec):
    assert TargetMemory().score(unitvec(1)) == 0.0


def test_score_identical_is_one(unitvec):
    m = TargetMemory()
    v = unitvec(1)
    m.update(v)
    assert m.score(v) > 0.99


def test_update_novelty(unitvec):
    m = TargetMemory(capacity=12, add_thresh=0.92)
    v = unitvec(1)
    assert m.update(v) is True            # first view stored
    assert m.update(v) is False           # duplicate rejected
    assert m.update(unitvec(2)) is True   # distinct view stored
    assert len(m) == 2


def test_consolidate_prunes_to_unique(unitvec):
    a, b, c = unitvec(1), unitvec(2), unitvec(3)
    pool = [a] * 20 + [b] * 5 + [c]       # mostly duplicates of a
    m = TargetMemory(capacity=24, add_thresh=0.92)
    m.consolidate(pool, dedup_thresh=0.95, capacity=24)
    assert len(m) <= 6                    # duplicates pruned
    assert m.score(a) > 0.99 and m.score(b) > 0.99 and m.score(c) > 0.99  # all represented


def test_consolidate_respects_capacity(unitvec):
    m = TargetMemory(capacity=5)
    m.consolidate([unitvec(i) for i in range(50)], dedup_thresh=0.99, capacity=5)
    assert len(m) <= 5


def test_background_score_and_cap(unitvec):
    bg = BackgroundMemory(capacity=5, dedup_thresh=0.97)
    for i in range(30):
        bg.add(unitvec(i))
    assert len(bg) <= 5
    v = unitvec(0)
    bg2 = BackgroundMemory()
    bg2.add(v)
    assert bg2.score(v) > 0.99


def test_discriminative_margin_separates(unitvec):
    # target bank around one cluster, background around another -> the target
    # vector has a clearly positive margin, a background vector a negative one.
    tgt = TargetMemory(capacity=12)
    bg = BackgroundMemory(capacity=12)
    t = unitvec(10)
    b = unitvec(20)
    tgt.update(t)
    bg.add(b)
    assert tgt.score(t) - bg.score(t) > 0.5
    assert tgt.score(b) - bg.score(b) < 0.0
