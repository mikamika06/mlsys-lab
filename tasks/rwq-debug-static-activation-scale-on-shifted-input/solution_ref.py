import numpy as np


def quantized_linear_dynamic(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        x_hat = np.zeros_like(x)
    else:
        q = np.clip(np.round(x / scale), -127, 127)
        x_hat = q * scale

    return x_hat @ W.T + b
