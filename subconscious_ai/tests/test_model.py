from __future__ import annotations

import numpy as np
import torch

from features.aggregator import FEATURE_NAMES
from model.net import IntentModel


def test_model_forward_and_train_step() -> None:
    model = IntentModel(input_size=len(FEATURE_NAMES))
    x = torch.tensor(np.random.rand(4, 16, len(FEATURE_NAMES)).astype(np.float32))
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()(model.net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    pred = model.predict(np.random.rand(16, len(FEATURE_NAMES)).astype(np.float32))
    assert "predicted_action" in pred
