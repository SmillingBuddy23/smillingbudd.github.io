from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

from features.aggregator import FEATURE_NAMES, SlidingWindowAggregator
from features.synthetic_generator import PRESETS, SyntheticConfig, label_for_preset, scenario_events
from model.net import ACTIONS, IntentModel


def build_dataset(n_per_preset: int = 40):
    X, y = [], []
    for preset in PRESETS.keys():
        for _ in range(n_per_preset):
            ag = SlidingWindowAggregator()
            evs = scenario_events(SyntheticConfig(preset=preset, duration_s=12.0))
            for e in evs:
                ag.add_event(e)
            now = evs[-1]["ts"] if evs else 0.0
            f = ag.extract_feature_vector(now)
            X.append(np.tile(f, (16, 1)))
            y.append(ACTIONS.index(label_for_preset(preset)))
    return np.stack(X), np.array(y)


def train(out_path: str) -> None:
    X, y = build_dataset()
    model = IntentModel(input_size=len(FEATURE_NAMES))
    net = model.net
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    net.train()
    tx, ty = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    for _ in range(20):
        logits = net(tx)
        loss = loss_fn(logits, ty)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    probs = torch.softmax(net(tx), dim=-1).detach().numpy()
    pred = probs.argmax(axis=1)
    print("precision", precision_score(y, pred, average="macro"))
    print("recall", recall_score(y, pred, average="macro"))
    print("f1", f1_score(y, pred, average="macro"))
    try:
        print("roc_auc", roc_auc_score(y, probs, multi_class="ovr"))
    except ValueError:
        print("roc_auc", "n/a")
    print(classification_report(y, pred, target_names=ACTIONS))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/demo_model.pt")
    args = parser.parse_args()
    train(args.out)
