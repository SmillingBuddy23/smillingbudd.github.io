from __future__ import annotations

import numpy as np
import time

from features.aggregator import FEATURE_NAMES
from model.net import IntentModel

if __name__ == "__main__":
    model = IntentModel(input_size=len(FEATURE_NAMES))
    lat = []
    for _ in range(100):
        x = np.random.rand(16, len(FEATURE_NAMES)).astype(np.float32)
        t0 = time.perf_counter()
        _ = model.predict(x)
        lat.append((time.perf_counter() - t0) * 1000)
    print("metric,p50_ms,p90_ms,p99_ms")
    print(f"inference_latency,{np.percentile(lat,50):.3f},{np.percentile(lat,90):.3f},{np.percentile(lat,99):.3f}")
