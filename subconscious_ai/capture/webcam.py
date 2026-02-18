from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WebcamCaptureConfig:
    enabled: bool = False


class WebcamCapture:
    """Optional webcam sampler stub. Emits low-cost eye focus proxy only."""

    def __init__(self, event_queue, config: WebcamCaptureConfig | None = None) -> None:
        self.event_queue = event_queue
        self.config = config or WebcamCaptureConfig()

    def sample_once(self) -> None:
        if not self.config.enabled:
            return
        self.event_queue.put(
            {
                "type": "webcam",
                "ts": time.time(),
                "payload": {"eye_focus_confidence": random.uniform(0.0, 1.0)},
            }
        )
