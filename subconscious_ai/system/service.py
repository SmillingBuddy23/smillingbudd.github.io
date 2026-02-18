from __future__ import annotations

import logging
import queue
import time

from capture.keyboard import KeyboardCapture
from capture.mouse import MouseCapture

logger = logging.getLogger(__name__)


class BackgroundService:
    def __init__(self, event_queue: "queue.Queue[dict]") -> None:
        self.event_queue = event_queue
        self.keyboard = KeyboardCapture(event_queue)
        self.mouse = MouseCapture(event_queue)

    def start(self) -> None:
        self.keyboard.start()
        self.mouse.start()
        logger.info("Background service started")

    def stop(self) -> None:
        self.keyboard.stop()
        self.mouse.stop()
        logger.info("Background service stopped")

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
