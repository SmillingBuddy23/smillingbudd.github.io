from __future__ import annotations

import os

import psutil


def cpu_percent() -> float:
    return float(psutil.cpu_percent(interval=None))


def rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss / (1024 * 1024))
