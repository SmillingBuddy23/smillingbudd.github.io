from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pynput import keyboard
except Exception:  # pragma: no cover
    keyboard = None


@dataclass
class KeyboardCaptureConfig:
    enabled: bool = True


class KeyboardCapture:
    """Keyboard listener that emits privacy-preserving metadata events."""

    def __init__(self, event_queue: "queue.Queue[dict]", config: Optional[KeyboardCaptureConfig] = None) -> None:
        self.event_queue = event_queue
        self.config = config or KeyboardCaptureConfig()
        self.listener: Optional[object] = None

    @staticmethod
    def _normalize_key(key: object) -> str:
        key_s = str(key)
        if "Key.backspace" in key_s:
            return "backspace"
        if "Key." in key_s:
            return key_s.replace("Key.", "")
        return "character"

    def _emit(self, is_keydown: bool, key: object) -> None:
        evt = {
            "type": "keyboard",
            "ts": time.time(),
            "payload": {
                "is_keydown": is_keydown,
                "keymeta": self._normalize_key(key),
            },
        }
        self.event_queue.put(evt)

    def start(self) -> None:
        if not self.config.enabled:
            logger.info("Keyboard capture disabled")
            return
        if keyboard is None:
            logger.warning("pynput keyboard unavailable; keyboard capture disabled")
            return

        self.listener = keyboard.Listener(
            on_press=lambda k: self._emit(True, k),
            on_release=lambda k: self._emit(False, k),
        )
        self.listener.start()
        logger.info("Keyboard capture started")

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            logger.info("Keyboard capture stopped")
