from __future__ import annotations

from run_demo import run_demo


def test_demo_runs_10s() -> None:
    run_demo(duration_s=2, synthetic=True)
