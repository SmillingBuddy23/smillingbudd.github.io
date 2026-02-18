from __future__ import annotations

import time

from run_demo import run_demo
from system.monitor import cpu_percent, rss_mb


if __name__ == "__main__":
    print("metric,value")
    run_demo(duration_s=5, synthetic=True)
    time.sleep(0.5)
    print(f"avg_cpu_percent,{cpu_percent():.2f}")
    print(f"rss_mb,{rss_mb():.2f}")
