import numpy as np


def _fused_op(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    t = np.tanh(x)
    return s * (x ** 2) + t


def _fused_op_grad(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    t = np.tanh(x)
    ds = s * (1.0 - s)
    dt = 1.0 - t ** 2
    return ds * (x ** 2) + 2.0 * x * s + dt


def fused_elementwise_forward(x: np.ndarray, index_map: np.ndarray) -> np.ndarray:
    """Compute forward fused activation map."""
    return _fused_op(x[index_map])


def fused_elementwise_backward(
    grad_output: np.ndarray, x: np.ndarray, index_map: np.ndarray, use_atomic: bool = True
) -> np.ndarray:
    """Compute backward gradient for input x."""
    grad_x = np.zeros(len(x), dtype=np.float64)
    ds = _fused_op_grad(x)
    if use_atomic:
        for i in range(len(index_map)):
            j = index_map[i]
            grad_x[j] += grad_output[i] * ds[j]
    else:
        for i in range(len(index_map)):
            j = index_map[i]
            grad_x[j] = grad_output[i] * ds[j]
    return grad_x


def finite_difference_grad_x(
    x: np.ndarray, index_map: np.ndarray, grad_output: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Compute numerical gradient of x using central finite differences."""
    n = len(x)
    fd_grad = np.zeros(n, dtype=np.float64)
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        y_plus = fused_elementwise_forward(x_plus, index_map)
        l_plus = float(np.sum(grad_output * y_plus))

        x_minus = x.copy()
        x_minus[j] -= eps
        y_minus = fused_elementwise_forward(x_minus, index_map)
        l_minus = float(np.sum(grad_output * y_minus))

        fd_grad[j] = (l_plus - l_minus) / (2.0 * eps)
    return fd_grad
