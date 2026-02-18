from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["switch_app", "open_browser", "start_typing", "click", "idle"]


class IntentGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 5) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


@dataclass
class PredictResult:
    predicted_action: str
    probability: float
    explanation: dict
    timestamp: float


class IntentModel:
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1) -> None:
        self.net = IntentGRU(input_size, hidden_size, num_layers)
        self.net.eval()
        self.device = torch.device("cpu")
        self.net.to(self.device)

    def predict_proba(self, seq: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.net(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            return probs

    def predict(self, seq: np.ndarray, explanation: dict | None = None) -> dict:
        probs = self.predict_proba(seq)
        idx = int(np.argmax(probs))
        return {
            "predicted_action": ACTIONS[idx],
            "probability": float(probs[idx]),
            "explanation": explanation or {"features": [], "text": "No explanation provided."},
            "timestamp": time.time(),
        }

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
