from __future__ import annotations

from typing import List, Tuple

import numpy as np

from features.aggregator import FEATURE_NAMES


class OcclusionExplainer:
    def __init__(self, model) -> None:
        self.model = model

    def explain(self, sequence: np.ndarray, top_k: int = 3) -> dict:
        base = self.model.predict_proba(sequence)
        pred_idx = int(np.argmax(base))
        impacts: List[Tuple[str, float]] = []
        for i, name in enumerate(FEATURE_NAMES):
            modified = sequence.copy()
            modified[:, i] = 0.0
            p = self.model.predict_proba(modified)
            impacts.append((name, float(base[pred_idx] - p[pred_idx])))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top = impacts[:top_k]
        text = " + ".join([f"{n}({v:+.2f})" for n, v in top])
        rationale = f"Top contributors suggest '{self.model.predict(sequence)['predicted_action']}' due to {text}."
        return {"features": top, "text": rationale}
