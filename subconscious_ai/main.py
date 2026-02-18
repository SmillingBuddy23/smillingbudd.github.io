from __future__ import annotations

import argparse
import logging
import queue

from run_demo import run_demo
from system.service import BackgroundService


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Subconscious AI - Edge Edition")
    p.add_argument("--mode", choices=["service", "demo"], default="demo")
    p.add_argument("--enable-audio", action="store_true")
    p.add_argument("--enable-webcam", action="store_true")
    p.add_argument("--online-learning", action="store_true")
    p.add_argument("--cpu-threshold", type=float, default=20.0)
    p.add_argument("--duration", type=int, default=15)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    if args.mode == "service":
        q: "queue.Queue[dict]" = queue.Queue()
        BackgroundService(q).run_forever()
    else:
        run_demo(duration_s=args.duration, synthetic=True)


if __name__ == "__main__":
    main()
