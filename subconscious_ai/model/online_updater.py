from __future__ import annotations

import torch


class OnlineUpdater:
    def __init__(self, model, lr: float = 1e-4) -> None:
        self.model = model
        self.opt = torch.optim.Adam(self.model.net.parameters(), lr=lr)
        self.loss = torch.nn.CrossEntropyLoss()

    def update(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.model.net.train()
        logits = self.model.net(x_batch)
        l = self.loss(logits, y_batch)
        self.opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.model.net.parameters(), 0.5)
        self.opt.step()
        self.model.net.eval()
        return float(l.item())
