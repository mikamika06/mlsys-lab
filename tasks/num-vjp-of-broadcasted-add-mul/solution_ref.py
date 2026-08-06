"""Reference solution for `num-vjp-of-broadcasted-add-mul`."""
from __future__ import annotations

import numpy as np


def _sum_axis_0(arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    new_shape = shape[1:]
    out = np.zeros(new_shape, dtype=np.float64)
    for idx in np.ndindex(new_shape):
        s = 0.0
        for d in range(shape[0]):
            s += arr[(d,) + idx]
        out[idx] = s
    return out


def _sum_axis_i_keepdims(arr: np.ndarray, i: int) -> np.ndarray:
    shape = arr.shape
    new_shape = list(shape)
    new_shape[i] = 1
    new_shape = tuple(new_shape)
    out = np.zeros(new_shape, dtype=np.float64)
    for idx in np.ndindex(new_shape):
        s = 0.0
        idx_list = list(idx)
        for d in range(shape[i]):
            idx_list[i] = d
            s += arr[tuple(idx_list)]
        out[idx] = s
    return out


def _sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` (some broadcast-result shape) down to `shape` by
    summing exactly the dimensions broadcasting introduced:
    extra leading dims, and dims that were size-1 in `shape`.
    """
    grad = np.asarray(grad, dtype=np.float64)

    extra = grad.ndim - len(shape)
    for _ in range(extra):
        grad = _sum_axis_0(grad)

    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = _sum_axis_i_keepdims(grad, i)

    return grad.reshape(shape)


def add_vjp(a: np.ndarray, b: np.ndarray, grad_out: np.ndarray):
    """VJP of `out = a + b` (NumPy broadcasting): returns (grad_a, grad_b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    grad_out = np.asarray(grad_out, dtype=np.float64)

    grad_a = _sum_to_shape(grad_out, a.shape)
    grad_b = _sum_to_shape(grad_out, b.shape)
    return grad_a, grad_b


def mul_vjp(a: np.ndarray, b: np.ndarray, grad_out: np.ndarray):
    """VJP of `out = a * b` (NumPy broadcasting): returns (grad_a, grad_b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    grad_out = np.asarray(grad_out, dtype=np.float64)

    grad_a = _sum_to_shape(grad_out * b, a.shape)
    grad_b = _sum_to_shape(grad_out * a, b.shape)
    return grad_a, grad_b
