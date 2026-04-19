from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..models.autograd import Parameter


class SGD:
    def __init__(self, parameters: Iterable[Parameter], lr: float, weight_decay: float = 0.0) -> None:
        self.parameters: List[Parameter] = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def clip_grad_norm(self, max_norm: float, eps: float = 1e-6) -> float:
        if max_norm <= 0.0:
            return 0.0

        total_norm_sq = 0.0
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            total_norm_sq += float(np.sum(parameter.grad * parameter.grad))

        total_norm = float(np.sqrt(total_norm_sq))
        if total_norm <= max_norm:
            return total_norm

        scale = max_norm / (total_norm + eps)
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.grad = parameter.grad * scale
        return total_norm

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            update = parameter.grad
            if self.weight_decay > 0.0:
                update = update + self.weight_decay * parameter.data
            parameter.data = parameter.data - self.lr * update


class StepLRScheduler:
    def __init__(self, optimizer: SGD, decay: float = 0.5, decay_every: int = 10, min_lr: float = 1e-5) -> None:
        self.optimizer = optimizer
        self.decay = decay
        self.decay_every = decay_every
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        if self.decay_every > 0 and epoch > 0 and epoch % self.decay_every == 0:
            self.optimizer.lr = max(self.min_lr, self.optimizer.lr * self.decay)
        return self.optimizer.lr
