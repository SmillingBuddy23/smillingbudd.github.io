from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

FEATURE_NAMES = [
    "typing_speed_3s",
    "typing_speed_7s",
    "avg_interkey_interval",
    "std_interkey_interval",
    "backspace_rate",
    "burst_count",
    "avg_burst_length",
    "pause_count",
    "longest_pause",
    "mouse_speed_mean",
    "mouse_jitter",
    "click_rate",
    "avg_dwell_time",
    "voice_energy_mean",
    "pitch_mean",
    "pitch_std",
    "eye_focus_confidence",
    "short_term_confidence_score",
]


@dataclass
class AggregatorConfig:
    max_history_seconds: float = 15.0


class SlidingWindowAggregator:
    def __init__(self, config: AggregatorConfig | None = None) -> None:
        self.config = config or AggregatorConfig()
        self.events: Deque[dict] = deque()
        self._last_feature: np.ndarray | None = None

    def add_event(self, event: dict) -> None:
        self.events.append(event)
        cutoff = event["ts"] - self.config.max_history_seconds
        while self.events and self.events[0]["ts"] < cutoff:
            self.events.popleft()

    def _slice(self, now: float, seconds: float) -> List[dict]:
        start = now - seconds
        return [e for e in self.events if e["ts"] >= start]

    def _keyboard_stats(self, events: List[dict]) -> Dict[str, float]:
        keydowns = [e for e in events if e["type"] == "keyboard" and e["payload"].get("is_keydown")]
        times = [e["ts"] for e in keydowns]
        intervals = np.diff(times).astype(np.float32) if len(times) > 1 else np.array([0.0], dtype=np.float32)
        backspaces = sum(1 for e in keydowns if e["payload"].get("keymeta") == "backspace")
        pauses = [float(i) for i in intervals if i > 0.8]
        bursts = []
        current = 0
        for i in intervals:
            current += 1
            if i > 0.8:
                bursts.append(current)
                current = 0
        if current > 0:
            bursts.append(current)
        active_typing_seconds = max(float(np.sum(intervals[intervals <= 0.8])) if len(intervals) else 0.0, 0.1)
        return {
            "keys": float(len(keydowns)),
            "avg_iki": float(np.mean(intervals)) if len(intervals) else 0.0,
            "std_iki": float(np.std(intervals)) if len(intervals) else 0.0,
            "backspace_rate": float(backspaces / max(len(keydowns), 1)),
            "burst_count": float(len(bursts)),
            "avg_burst_length": float(np.mean(bursts)) if bursts else 0.0,
            "pause_count": float(len(pauses)),
            "longest_pause": float(max(pauses) if pauses else 0.0),
            "active_typing_seconds": active_typing_seconds,
        }

    def _mouse_stats(self, events: List[dict]) -> Dict[str, float]:
        moves = [e for e in events if e["type"] == "mouse" and e["payload"].get("event") == "move"]
        clicks = [e for e in events if e["type"] == "mouse" and e["payload"].get("event") == "click" and e["payload"].get("pressed")]
        vels = []
        for a, b in zip(moves, moves[1:]):
            dt = max(b["ts"] - a["ts"], 1e-3)
            dx = b["payload"]["x"] - a["payload"]["x"]
            dy = b["payload"]["y"] - a["payload"]["y"]
            vels.append(((dx * dx + dy * dy) ** 0.5) / dt)
        dwell = [b["ts"] - a["ts"] for a, b in zip(clicks, clicks[1:])]
        return {
            "mouse_speed_mean": float(np.mean(vels)) if vels else 0.0,
            "mouse_jitter": float(np.std(vels)) if vels else 0.0,
            "click_rate": float(len(clicks)),
            "avg_dwell_time": float(np.mean(dwell)) if dwell else 0.0,
        }

    def _audio_stats(self, events: List[dict]) -> Tuple[float, float, float]:
        audio = [e for e in events if e["type"] == "audio"]
        if not audio:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean([a["payload"].get("energy", 0.0) for a in audio])),
            float(np.mean([a["payload"].get("pitch_mean", 0.0) for a in audio])),
            float(np.mean([a["payload"].get("pitch_std", 0.0) for a in audio])),
        )

    def _webcam_focus(self, events: List[dict]) -> float:
        webcam = [e for e in events if e["type"] == "webcam"]
        if not webcam:
            return 0.0
        return float(np.mean([w["payload"].get("eye_focus_confidence", 0.0) for w in webcam]))

    def extract_feature_vector(self, now: float) -> np.ndarray:
        events_3 = self._slice(now, 3.0)
        events_7 = self._slice(now, 7.0)
        kb_3 = self._keyboard_stats(events_3)
        kb_7 = self._keyboard_stats(events_7)
        mouse_7 = self._mouse_stats(events_7)
        energy, pitch_mean, pitch_std = self._audio_stats(events_7)
        eye_focus = self._webcam_focus(events_7)
        consistency = 1.0 / (1.0 + kb_7["std_iki"] + kb_7["backspace_rate"] + mouse_7["mouse_jitter"] / 200.0)

        vec = np.array(
            [
                kb_3["keys"] / kb_3["active_typing_seconds"],
                kb_7["keys"] / kb_7["active_typing_seconds"],
                kb_7["avg_iki"],
                kb_7["std_iki"],
                kb_7["backspace_rate"],
                kb_7["burst_count"],
                kb_7["avg_burst_length"],
                kb_7["pause_count"],
                kb_7["longest_pause"],
                mouse_7["mouse_speed_mean"],
                mouse_7["mouse_jitter"],
                mouse_7["click_rate"] / 7.0,
                mouse_7["avg_dwell_time"],
                energy,
                pitch_mean,
                pitch_std,
                eye_focus,
                consistency,
            ],
            dtype=np.float32,
        )
        vec = self.normalize(vec)
        self._last_feature = vec
        return vec

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        scales = np.array([8, 8, 1, 1, 1, 20, 20, 20, 5, 800, 600, 2, 5, 1, 300, 80, 1, 1], dtype=np.float32)
        out = np.clip(vec / np.maximum(scales, 1e-6), 0.0, 1.0)
        return out.astype(np.float32)
