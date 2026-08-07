import numpy as np


def compute_minmax_scales(x: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """Compute per-channel min-max scales."""
    qmax = (1 << (num_bits - 1)) - 1
    max_val = np.max(np.abs(x), axis=1, keepdims=True)
    return np.where(max_val == 0, 1.0, max_val / qmax)


def quantize_dequantize_channel(x: np.ndarray, scale: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """Fake-quantize and dequantize per channel."""
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    q = np.clip(np.round(x / scale), qmin, qmax)
    return q * scale


def channel_mse(x: np.ndarray, deq: np.ndarray) -> np.ndarray:
    """Compute mean squared error per channel."""
    return np.mean((x - deq) ** 2, axis=1)
