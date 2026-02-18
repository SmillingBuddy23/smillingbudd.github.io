from __future__ import annotations

from dataclasses import dataclass

from system.monitor import cpu_percent


@dataclass
class ThrottleState:
    keyboard_scale: float = 1.0
    mouse_scale: float = 1.0
    audio_scale: float = 1.0
    webcam_scale: float = 1.0


class AdaptiveThrottler:
    def __init__(self, cpu_threshold: float = 20.0) -> None:
        self.cpu_threshold = cpu_threshold
        self.state = ThrottleState()

    def update(self) -> ThrottleState:
        cpu = cpu_percent()
        if cpu > self.cpu_threshold:
            self.state = ThrottleState(0.7, 0.5, 0.4, 0.4)
        else:
            self.state = ThrottleState()
        return self.state
