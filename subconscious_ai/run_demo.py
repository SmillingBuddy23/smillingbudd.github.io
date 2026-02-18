from __future__ import annotations

import argparse
import logging
import queue
import time

import numpy as np

from explain.explainer import OcclusionExplainer
from features.aggregator import FEATURE_NAMES, SlidingWindowAggregator
from features.synthetic_generator import SyntheticConfig, scenario_events
from model.net import IntentModel
from system.monitor import cpu_percent, rss_mb
from system.throttler import AdaptiveThrottler


def run_demo(duration_s: int = 10, synthetic: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    q: "queue.Queue[dict]" = queue.Queue()
    ag = SlidingWindowAggregator()
    model = IntentModel(input_size=len(FEATURE_NAMES))
    explainer = OcclusionExplainer(model)
    throttler = AdaptiveThrottler()

    if synthetic:
        for evt in scenario_events(SyntheticConfig(preset="distracted_browsing", duration_s=float(duration_s))):
            q.put(evt)

    seq = []
    start = time.time()
    while time.time() - start < duration_s:
        while not q.empty():
            ag.add_event(q.get())
        f = ag.extract_feature_vector(time.time())
        # optimization 1: feature caching/early exit heuristic
        if f[17] > 0.85 and f[1] < 0.1:
            pred = {"predicted_action": "idle", "probability": 0.9, "explanation": {"features": [("short_term_confidence_score", 0.4)], "text": "Low activity and high consistency."}, "timestamp": time.time()}
        else:
            seq.append(f)
            seq = seq[-16:]  # optimization 2: batch recent steps
            s = np.stack(seq) if seq else np.zeros((16, len(FEATURE_NAMES)), dtype=np.float32)
            if s.shape[0] < 16:
                s = np.vstack([np.zeros((16 - s.shape[0], len(FEATURE_NAMES)), dtype=np.float32), s])
            pred = model.predict(s)
            pred["explanation"] = explainer.explain(s)
        tstate = throttler.update()
        print(f"pred={pred['predicted_action']} p={pred['probability']:.2f} cpu={cpu_percent():.1f}% mem={rss_mb():.1f}MB throttle={tstate}")
        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()
    run_demo(duration_s=args.duration, synthetic=not args.no_synthetic)
