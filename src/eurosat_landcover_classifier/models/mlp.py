from __future__ import annotations

from typing import Dict, List

import numpy as np

from .autograd import Parameter, Tensor


def _activation_tensor(name: str, x: Tensor) -> Tensor:
    key = name.lower()
    if key == "relu":
        return x.relu()
    if key == "sigmoid":
        return x.sigmoid()
    if key == "tanh":
        return x.tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _activation_array(name: str, x: np.ndarray) -> np.ndarray:
    key = name.lower()
    if key == "relu":
        return np.maximum(x, 0.0)
    if key == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if key == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unsupported activation: {name}")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = softmax(logits.data)
    indices = np.arange(targets.shape[0])
    losses = -np.log(probabilities[indices, targets] + 1e-12)
    loss = Tensor(
        np.array(losses.mean(), dtype=np.float32),
        requires_grad=logits.requires_grad,
        _prev=(logits,),
        _op="cross_entropy",
    )

    def _backward() -> None:
        if not logits.requires_grad:
            return
        grad = probabilities.copy()
        grad[indices, targets] -= 1.0
        grad /= targets.shape[0]
        logits.grad = grad if logits.grad is None else logits.grad + grad

    loss._backward = _backward
    return loss


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> List[Parameter]:
        raise NotImplementedError

    def train(self) -> "Module":
        self.training = True
        return self

    def eval(self) -> "Module":
        self.training = False
        return self

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, init_scale: float, name: str) -> None:
        super().__init__()
        weight = rng.standard_normal((in_features, out_features)).astype(np.float32) * init_scale
        bias = np.zeros(out_features, dtype=np.float32)
        self.weight = Parameter(weight, name=f"{name}.weight")
        self.bias = Parameter(bias, name=f"{name}.bias")

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.data + self.bias.data

    def parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"weight": self.weight.data.copy(), "bias": self.bias.data.copy()}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.weight.data = state_dict["weight"].astype(np.float32)
        self.bias.data = state_dict["bias"].astype(np.float32)


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        super().__init__()
        activation_key = activation.lower()
        hidden_scale = np.sqrt(2.0 / input_dim) if activation_key == "relu" else np.sqrt(1.0 / input_dim)
        output_scale = np.sqrt(1.0 / hidden_dim)
        rng = np.random.default_rng(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation = activation_key

        self.fc1 = Linear(input_dim, hidden_dim, rng=rng, init_scale=hidden_scale, name="fc1")
        self.fc2 = Linear(hidden_dim, num_classes, rng=rng, init_scale=output_scale, name="fc2")

    def forward(self, x: Tensor) -> Tensor:
        hidden = _activation_tensor(self.activation, self.fc1.forward(x))
        return self.fc2.forward(hidden)

    def forward_array(self, x: np.ndarray) -> np.ndarray:
        hidden = _activation_array(self.activation, self.fc1.forward_array(x))
        return self.fc2.forward_array(hidden)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward_array(x)
        return np.argmax(logits, axis=1)

    def parameters(self) -> List[Parameter]:
        return self.fc1.parameters() + self.fc2.parameters()

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "fc1.weight": self.fc1.weight.data.copy(),
            "fc1.bias": self.fc1.bias.data.copy(),
            "fc2.weight": self.fc2.weight.data.copy(),
            "fc2.bias": self.fc2.bias.data.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.fc1.weight.data = state_dict["fc1.weight"].astype(np.float32)
        self.fc1.bias.data = state_dict["fc1.bias"].astype(np.float32)
        self.fc2.weight.data = state_dict["fc2.weight"].astype(np.float32)
        self.fc2.bias.data = state_dict["fc2.bias"].astype(np.float32)
