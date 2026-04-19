from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union["Tensor", np.ndarray, Sequence[float], float, int]


def _to_array(data: ArrayLike) -> np.ndarray:
    if isinstance(data, Tensor):
        return data.data
    array = np.asarray(data, dtype=np.float32)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return array


def _sum_to_shape(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _prev: Iterable["Tensor"] = (),
        _op: str = "",
        name: Optional[str] = None,
    ) -> None:
        self.data = _to_array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev = tuple(_prev)
        self._op = _op
        self.name = name

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op={self._op!r})"

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False, name=self.name)

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise ValueError("Gradient must be provided for non-scalar tensors.")
            grad_array = np.ones_like(self.data, dtype=np.float32)
        else:
            grad_array = _to_array(grad)

        topo = []
        visited = set()

        def build(node: "Tensor") -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = grad_array

        for node in reversed(topo):
            node._backward()

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        out = Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad.T
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            divisor = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            divisor = 1
            for ax in axes:
                divisor *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad * (self.data > 0.0)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad, _prev=(self,), _op="sigmoid")

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad * sig * (1.0 - sig)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_value = np.tanh(self.data)
        out = Tensor(tanh_value, requires_grad=self.requires_grad, _prev=(self,), _op="tanh")

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad * (1.0 - tanh_value ** 2)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = _sum_to_shape(out.grad, self.data.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = _sum_to_shape(out.grad, other.data.shape)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -float(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return (-self) + other

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = _sum_to_shape(out.grad * other.data, self.data.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = _sum_to_shape(out.grad * self.data, other.data.shape)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="div",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = _sum_to_shape(out.grad / other.data, self.data.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = _sum_to_shape((-out.grad * self.data) / (other.data ** 2), other.data.shape)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out


class Parameter(Tensor):
    def __init__(self, data: ArrayLike, name: Optional[str] = None) -> None:
        super().__init__(data, requires_grad=True, name=name)
