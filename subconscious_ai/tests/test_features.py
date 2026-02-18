from __future__ import annotations

import time

from features.aggregator import FEATURE_NAMES, SlidingWindowAggregator


def test_feature_ranges() -> None:
    ag = SlidingWindowAggregator()
    now = time.time()
    for i in range(20):
        ag.add_event({"type": "keyboard", "ts": now + i * 0.1, "payload": {"is_keydown": True, "keymeta": "character"}})
    v = ag.extract_feature_vector(now + 3)
    assert len(v) == len(FEATURE_NAMES)
    assert ((v >= 0) & (v <= 1)).all()
