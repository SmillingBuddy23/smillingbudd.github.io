from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pynput import mouse
except Exception:  # pragma: no cover
    mouse = None


@dataclass
class MouseCaptureConfig:
    movement_threshold: float = 2.0
    max_hz: float = 20.0
    enabled: bool = True


class MouseCapture:
    def __init__(self, event_queue: "queue.Queue[dict]", config: Optional[MouseCaptureConfig] = None) -> None:
        self.event_queue = event_queue
        self.config = config or MouseCaptureConfig()
        self.listener: Optional[object] = None
        self._last_emit = 0.0
        self._last_pos: Optional[tuple[int, int]] = None

    def _emit_move(self, x: int, y: int) -> None:
        now = time.time()
        if now - self._last_emit < 1 / max(self.config.max_hz, 1.0):
            return
        if self._last_pos is not None:
            dx, dy = x - self._last_pos[0], y - self._last_pos[1]
            if (dx * dx + dy * dy) ** 0.5 < self.config.movement_threshold:
                return
        self._last_emit = now
        self._last_pos = (x, y)
        self.event_queue.put({"type": "mouse", "ts": now, "payload": {"event": "move", "x": x, "y": y}})

    def _emit_click(self, x: int, y: int, button: object, pressed: bool) -> None:
        self.event_queue.put(
            {
                "type": "mouse",
                "ts": time.time(),
                "payload": {"event": "click", "x": x, "y": y, "button": str(button), "pressed": bool(pressed)},
            }
        )

    def start(self) -> None:
        if not self.config.enabled:
            logger.info("Mouse capture disabled")
            return
        if mouse is None:
            logger.warning("pynput mouse unavailable; mouse capture disabled")
            return
        self.listener = mouse.Listener(on_move=self._emit_move, on_click=self._emit_click)
        self.listener.start()
        logger.info("Mouse capture started")

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            logger.info("Mouse capture stopped")
