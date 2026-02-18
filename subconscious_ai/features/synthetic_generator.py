from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

PRESETS = {
    "calm_study": {"burstiness": 0.4, "error_rate": 0.05, "pause_mean": 1.2, "mouse_jitter": 5.0},
    "distracted_browsing": {"burstiness": 0.2, "error_rate": 0.1, "pause_mean": 0.8, "mouse_jitter": 20.0},
    "exam_stress": {"burstiness": 0.7, "error_rate": 0.2, "pause_mean": 0.5, "mouse_jitter": 35.0},
    "idle": {"burstiness": 0.05, "error_rate": 0.0, "pause_mean": 3.0, "mouse_jitter": 1.0},
}

ACTIONS = ["switch_app", "open_browser", "start_typing", "click", "idle"]


@dataclass
class SyntheticConfig:
    preset: str = "calm_study"
    duration_s: float = 60.0


def _keyboard_events(t0: float, cfg: Dict[str, float], duration_s: float) -> List[dict]:
    ts, events = t0, []
    while ts < t0 + duration_s:
        burst_len = max(1, int(random.gauss(4 + 10 * cfg["burstiness"], 2)))
        for _ in range(burst_len):
            keymeta = "backspace" if random.random() < cfg["error_rate"] else "character"
            events.append({"type": "keyboard", "ts": ts, "payload": {"is_keydown": True, "keymeta": keymeta}})
            ts += max(0.03, random.gauss(0.13, 0.04))
        ts += max(0.1, random.gauss(cfg["pause_mean"], 0.2))
    return events


def _mouse_events(t0: float, cfg: Dict[str, float], duration_s: float) -> List[dict]:
    ts, x, y, events = t0, 200.0, 120.0, []
    while ts < t0 + duration_s:
        for _ in range(3):
            theta = random.uniform(0, 2 * math.pi)
            step = random.uniform(5, 40)
            x += math.cos(theta) * step + random.gauss(0, cfg["mouse_jitter"])
            y += math.sin(theta) * step + random.gauss(0, cfg["mouse_jitter"])
            events.append({"type": "mouse", "ts": ts, "payload": {"event": "move", "x": int(x), "y": int(y)}})
            ts += 0.05
        if random.random() < 0.3:
            events.append({"type": "mouse", "ts": ts, "payload": {"event": "click", "x": int(x), "y": int(y), "pressed": True}})
        ts += random.uniform(0.1, 1.0)
    return events


def _audio_events(t0: float, cfg: Dict[str, float], duration_s: float) -> List[dict]:
    ts, events = t0, []
    while ts < t0 + duration_s:
        stress = min(1.0, cfg["error_rate"] * 4 + cfg["burstiness"])
        events.append(
            {
                "type": "audio",
                "ts": ts,
                "payload": {"energy": 0.3 + 0.5 * stress + random.uniform(-0.05, 0.05), "pitch_mean": 120 + 60 * stress, "pitch_std": 8 + 30 * stress},
            }
        )
        ts += random.uniform(8, 12)
    return events


def scenario_events(config: SyntheticConfig) -> List[dict]:
    cfg = PRESETS[config.preset]
    t0 = time.time()
    events = _keyboard_events(t0, cfg, config.duration_s) + _mouse_events(t0, cfg, config.duration_s) + _audio_events(t0, cfg, config.duration_s)
    events.sort(key=lambda e: e["ts"])
    return events


def label_for_preset(preset: str) -> str:
    return {
        "calm_study": "start_typing",
        "distracted_browsing": "open_browser",
        "exam_stress": "switch_app",
        "idle": "idle",
    }.get(preset, "click")
