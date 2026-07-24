import numpy as np

_cached_scale = None


def quantized_linear_dynamic(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    global _cached_scale

    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # TODO: this incorrectly freezes the first batch's activation scale.
    if _cached_scale is None:
        _cached_scale = np.max(np.abs(x)) / 127.0

    scale = _cached_scale
    if scale == 0:
        x_hat = np.zeros_like(x)
    else:
        q = np.clip(np.round(x / scale), -127, 127)
        x_hat = q * scale

    return x_hat @ W.T + b
