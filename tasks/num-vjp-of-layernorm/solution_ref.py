import numpy as np


def layernorm_vjp(x: np.ndarray, grad_y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grad_y = np.asarray(grad_y, dtype=np.float64)

    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std

    mean_grad = np.mean(grad_y, axis=1, keepdims=True)
    mean_grad_hat = np.mean(grad_y * x_hat, axis=1, keepdims=True)

    return inv_std * (grad_y - mean_grad - x_hat * mean_grad_hat)
