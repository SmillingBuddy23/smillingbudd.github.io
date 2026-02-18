from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioCaptureConfig:
    enabled: bool = False


class AudioCapture:
    """Optional audio sampler stub. Emits feature-level summaries only."""

    def __init__(self, event_queue, config: AudioCaptureConfig | None = None) -> None:
        self.event_queue = event_queue
        self.config = config or AudioCaptureConfig()

    def sample_once(self) -> None:
        if not self.config.enabled:
            return
        self.event_queue.put(
            {
                "type": "audio",
                "ts": time.time(),
                "payload": {
                    "energy": random.uniform(0.0, 1.0),
                    "zcr": random.uniform(0.0, 1.0),
                    "pitch_mean": random.uniform(80, 240),
                    "pitch_std": random.uniform(5, 40),
                },
            }
        )
