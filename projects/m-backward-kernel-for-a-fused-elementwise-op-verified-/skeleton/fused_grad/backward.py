import numpy as np


def fused_elementwise_forward(x: np.ndarray, index_map: np.ndarray) -> np.ndarray:
    """Compute forward fused activation map."""
    raise NotImplementedError


def fused_elementwise_backward(
    grad_output: np.ndarray, x: np.ndarray, index_map: np.ndarray, use_atomic: bool = True
) -> np.ndarray:
    """Compute backward gradient for input x."""
    raise NotImplementedError


def finite_difference_grad_x(
    x: np.ndarray, index_map: np.ndarray, grad_output: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Compute numerical gradient of x using central finite differences."""
    raise NotImplementedError
